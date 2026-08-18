import os
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict
import imagehash
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import defaultdict
import random
import json

from .config import DEFAULT_DATA_ROOT, DEFAULT_HF_CACHE_ROOT


class LLaVAPretrainDataset(Dataset):

    def __init__(self, hf_dataset, clip_image_processor, qwen_tokenizer,max_caption_token_len: int, images_root_dir: str):
        self.hf_dataset = hf_dataset
        self.clip_image_processor = clip_image_processor
        self.qwen_tokenizer = qwen_tokenizer
        self.max_caption_token_len = max_caption_token_len
        self.images_root_dir = Path(images_root_dir)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, sample_index: int) -> Dict[str, torch.Tensor]:
        sample = self.hf_dataset[sample_index]
        caption_text = sample['conversations'][1]['value']
        image_path = self.images_root_dir / sample['image']
        raw_pil_image = Image.open(image_path).convert('RGB')
    
        processed_image = self.clip_image_processor(images=raw_pil_image, return_tensors="pt")['pixel_values'].squeeze(0)
    
        # # Wrap caption in chat template to match inference format
        # messages = [{"role": "user", "content": caption_text}]
        # formatted_caption = self.qwen_tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
    
        # tokenised_caption = self.qwen_tokenizer(
        #     formatted_caption,
        #     max_length=self.max_caption_token_len,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt',
        #     add_special_tokens=False,  # chat template already added special tokens
        # )

        tokenised_caption = self.qwen_tokenizer(
            caption_text,
            max_length=self.max_caption_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,  # let tokeniser add BOS naturally
        )
    
        return {
            'processed_image': processed_image,
            'caption_token_ids': tokenised_caption['input_ids'].squeeze(0),
            'caption_attention_mask': tokenised_caption['attention_mask'].squeeze(0),
        }


def download_and_prepare_llava(config):
    cache_root = Path(getattr(config, "cache_dir", DEFAULT_HF_CACHE_ROOT))

    raw_dataset = load_dataset(
        "json",
        data_files="hf://datasets/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        split="train",
    )

    snapshot_dir = (
        cache_root
        / "hub"
        / "datasets--liuhaotian--LLaVA-Pretrain"
        / "snapshots"
        / "70f9d1e5e1a697fe35830875cfc7de1dd590d727"
    )

    zip_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="images.zip",
        repo_type="dataset",
        cache_dir=str(cache_root),
    )
    print(f"Downloaded to: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(snapshot_dir)
    print("Done. Contents:", os.listdir(snapshot_dir)[:5])

    images_root_dir = (
        str(snapshot_dir / "images")
        if (snapshot_dir / "images").is_dir()
        else snapshot_dir
    )

    print(f"Dataset loaded: {len(raw_dataset):,} samples")
    print(f"Images root dir: {images_root_dir}")

    return raw_dataset, images_root_dir


def build_dataloader(raw_dataset, clip_image_processor, qwen_tokenizer, images_root_dir, config):

    train_dataset = LLaVAPretrainDataset(
        hf_dataset=raw_dataset,
        clip_image_processor=clip_image_processor,
        qwen_tokenizer=qwen_tokenizer,
        max_caption_token_len=config.max_caption_token_len,
        images_root_dir=images_root_dir,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
    )

    print(f"Dataset: {len(train_dataset):,} samples")
    print(f"Batch size: {config.per_device_batch_size}")
    print(f"Steps per epoch: {len(train_dataloader):,}")
    print(f"Effective batch: {config.per_device_batch_size * config.gradient_accumulation_steps}")

    return train_dataloader


# Data decontamination

PHASH_SIZE = 16
HAMMING_THRESH = 4
HAMMING_BATCH = 256
SSCD_THRESH = 0.75
EMBED_BATCH = 64
NUM_WORKERS = 4

popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def compute_phash_bits(path):
    try:
        h = imagehash.phash(Image.open(path).convert("RGB"), hash_size=PHASH_SIZE)
        return h.hash.flatten()
    except Exception:
        return None


def build_hash_matrix(paths, desc):
    results = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(compute_phash_bits, p): idx for idx, p in enumerate(paths)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx = futures[fut]
            v = fut.result()
            if v is not None:
                results[idx] = v
    sorted_items = sorted(results.items())
    valid_indices = [idx for idx, _ in sorted_items]
    matrix = np.packbits(np.array([v for _, v in sorted_items], dtype=bool), axis=1)
    return matrix, valid_indices


class SSCDDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), True
        except Exception:
            return torch.zeros(3, 288, 288), False


@torch.no_grad()
def embed_sscd(paths, sscd, desc, device):
    sscd_transform = T.Compose([
        T.Resize(288),
        T.CenterCrop(288),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SSCDDataset(paths, sscd_transform)
    loader = DataLoader(dataset, batch_size=EMBED_BATCH, num_workers=NUM_WORKERS, pin_memory=True)
    parts = []
    flags = []
    for batch, batch_flags in tqdm(loader, desc=desc):
        batch = batch.to(device)
        feats = sscd(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        parts.append(feats.cpu().numpy())
        flags.extend(batch_flags.numpy().tolist())
    return np.concatenate(parts, axis=0), np.array(flags, dtype=bool)
    

def collect_vqav2_test2015_paths(vqav2_root: str):
    """All COCO test2015 JPG paths — shared reference set for VQAv2 test-dev and test-standard."""
    test_dir = Path(vqav2_root) / "test2015"
    if not test_dir.exists():
        print(
            f"VQAv2 test2015 directory not found: {test_dir}. "
            "Download COCO test2015 and place images under that path."
        )
        return []
    paths = sorted(test_dir.glob("*.jpg"))
    print(f"VQAv2 test2015: {len(paths):,} images from {test_dir}")
    return paths


def collect_vqav2_testdev_paths(vqav2_root: str):
    """Alias: test-dev eval images live in COCO test2015 (same as test-standard)."""
    return collect_vqav2_test2015_paths(vqav2_root)


def load_vqav2_eval_image_ids(vqav2_root: str):
    """Unique image_ids in VQAv2 test-dev and test-standard question files."""
    import json
    import re

    vqav2_root = Path(vqav2_root)
    splits = {
        "test-dev": vqav2_root / "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "test-std": vqav2_root / "v2_OpenEnded_mscoco_test2015_questions.json",
    }
    out = {}
    for split_name, q_path in splits.items():
        if not q_path.exists():
            print(f"Missing {split_name} questions: {q_path}")
            out[split_name] = set()
            continue
        with open(q_path) as f:
            questions = json.load(f)["questions"]
        out[split_name] = {int(q["image_id"]) for q in questions}
        print(f"VQAv2 {split_name}: {len(questions):,} questions, {len(out[split_name]):,} unique images")
    union = set().union(*out.values()) if out else set()
    if not union:
        test_dir = vqav2_root / "test2015"
        if test_dir.is_dir():
            pat = re.compile(r"COCO_test2015_(\d+)\.jpg$", re.I)
            for jpg in test_dir.glob("*.jpg"):
                m = pat.match(jpg.name)
                if m:
                    union.add(int(m.group(1)))
            if union:
                print(
                    f"VQAv2 test2015 fallback (JPG filenames under {test_dir}): "
                    f"{len(union):,} unique image ids"
                )
        if not union:
            print(
                "WARNING: No VQAv2 test2015 image ids found. "
                "Download test question JSONs (v2_Questions_Test_mscoco.zip) or COCO test2015 JPGs."
            )
    else:
        print(f"VQAv2 test-dev ∪ test-std unique images: {len(union):,}")
    return out, union


VQAV2_TEST_QUESTIONS_URL = (
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
)


def download_vqav2_test_questions(vqav2_root: str) -> Path:
    """Download VQAv2 test-dev + test-standard question JSONs if missing."""
    vqav2_root = Path(vqav2_root)
    vqav2_root.mkdir(parents=True, exist_ok=True)
    dev = vqav2_root / "v2_OpenEnded_mscoco_test-dev2015_questions.json"
    std = vqav2_root / "v2_OpenEnded_mscoco_test2015_questions.json"
    if dev.exists() and std.exists():
        print(f"VQAv2 test question JSONs already present under {vqav2_root}")
        return vqav2_root

    zip_path = vqav2_root / "v2_Questions_Test_mscoco.zip"
    if not zip_path.exists():
        print(f"Downloading VQAv2 test questions -> {zip_path}")
        urllib.request.urlretrieve(VQAV2_TEST_QUESTIONS_URL, zip_path)

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(vqav2_root)
    print(f"VQAv2 test question JSONs ready under {vqav2_root}")
    return vqav2_root


# MMBench + SEED helpers live in benchmark_data.py (no faiss import).
from .benchmark_data import (  # noqa: E402
    MMBENCH_TSV_URLS,
    SEED_BENCH_HF_REPO,
    SEED_BENCH_IMAGE_ZIP,
    SEED_BENCH_JSON_FILE,
    SEED_BENCH_VIDEO_ZIP_PARTS,
    build_mmbench_image_map as _build_mmbench_image_map,
    collect_mmbench_reference_paths,
    decode_base64_image_to_jpg as _decode_base64_image_to_jpg,
    download_mmbench_tsv,
    download_seed_bench_images,
    download_seed_bench_json,
    extract_mmbench_images_from_tsv,
    load_seed_bench_questions,
    prepare_mmbench_reference_images,
    resolve_mmbench_b64 as _resolve_mmbench_b64,
    resolve_seed_image_path as _resolve_seed_image_path,
)

def _find_seed_video_frames_root(seed_root: Path):
    """Return directory containing extracted v1_video frames, if present."""
    seed_root = Path(seed_root)
    candidates = [
        seed_root / "v1_video",
        seed_root / "SEED-Bench-video",
        seed_root / "SEED-Bench-video-image",
    ]
    for root in candidates:
        if root.is_dir() and any(root.rglob("*.jpg")):
            return root
        if root.is_dir() and any(root.rglob("*.png")):
            return root
    return None


def collect_seed_bench_reference_paths( seed_root: Path, hf_cache=DEFAULT_HF_CACHE_ROOT, include_video_frames: bool = True, ):
    """Collect SEED-Bench v1 reference image paths for pHash decontamination."""
    seed_root = Path(seed_root)
    json_path = download_seed_bench_json(seed_root, hf_cache=hf_cache)
    images_root = download_seed_bench_images(seed_root, hf_cache=hf_cache)
    questions = load_seed_bench_questions(json_path)

    image_paths = []
    missing = 0
    for q in questions:
        if q.get("data_type") != "image":
            continue
        path = _resolve_seed_image_path(images_root, q["data_id"])
        if path is None:
            missing += 1
            continue
        image_paths.append(path)

    seen = set()
    unique_images = []
    for path in image_paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_images.append(path)

    stats = {
        "total_questions": len(questions),
        "image_questions": sum(1 for q in questions if q.get("data_type") == "image"),
        "video_questions": sum(1 for q in questions if q.get("data_type") == "video"),
        "unique_image_files": len(unique_images),
        "missing_image_files": missing,
        "video_frame_files": 0,
    }

    all_paths = list(unique_images)
    if include_video_frames:
        video_root = _find_seed_video_frames_root(seed_root)
        if video_root is not None:
            frame_paths = sorted(
                p
                for p in video_root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            stats["video_frame_files"] = len(frame_paths)
            all_paths.extend(frame_paths)
        else:
            print(
                "SEED-Bench video frames not found. Image-only reference set for dims 1–9. "
                "Run Section 0 video download (~22 GB) to include dims 10–12."
            )

    print(
        f"SEED-Bench reference: {len(all_paths):,} files "
        f"({stats['unique_image_files']:,} CC3M images, "
        f"{stats['video_frame_files']:,} video frames)"
    )
    return all_paths, stats


def prepare_seed_bench_reference_images( seed_root: Path, hf_cache=DEFAULT_HF_CACHE_ROOT, include_video_frames: bool = True, ):
    """Download SEED-Bench assets and return reference paths + summary stats."""
    return collect_seed_bench_reference_paths(
        seed_root,
        hf_cache=hf_cache,
        include_video_frames=include_video_frames,
    )


def collect_gqa_testdev_paths(gqa_root: str):
    """Return on-disk paths for GQA **testdev** split images only (eval reference)."""
    return collect_gqa_testdev_reference_paths(gqa_root)


# Previous implementation (kept for reference): globbed *all* GQA JPGs under images/,
# not just testdev eval images — too broad for pHash decontamination reference.
# def collect_gqa_testdev_paths(gqa_root: str):
#     images_dir = Path(gqa_root) / "images"
#     if not images_dir.exists():
#         images_dir = Path(gqa_root)
#     paths = sorted(images_dir.glob("*.jpg"))
#     if not paths:
#         print(f"No GQA images found under {images_dir}. Download GQA images and place them there.")
#         return []
#     print(f"GQA testdev: {len(paths):,} images from {images_dir}")
#     return paths


def collect_gqa_testdev_reference_paths(gqa_root: str):
    """Map GQA testdev question JSON imageIds -> local JPG paths for pHash reference."""
    gqa_root = Path(gqa_root)
    images_dir = gqa_root / "images"
    if not images_dir.exists():
        images_dir = gqa_root / "images" / "images"
    if not images_dir.exists():
        images_dir = gqa_root / "gqa" / "images"

    questions_dir = None
    for candidate in (
        gqa_root,
        gqa_root / "questions1.2",
        gqa_root / "questions",
        gqa_root / "gqa" / "questions1.2",
        gqa_root / "gqa" / "questions",
    ):
        if not candidate.is_dir():
            continue
        if (candidate / "testdev_balanced_questions.json").exists() or (
            candidate / "testdev_all_questions.json"
        ).exists():
            questions_dir = candidate
            break

    if questions_dir is None:
        print(f"GQA testdev questions not found under {gqa_root}")
        return []

    testdev_json = questions_dir / "testdev_balanced_questions.json"
    if not testdev_json.exists():
        testdev_json = questions_dir / "testdev_all_questions.json"

    with open(testdev_json) as f:
        questions = json.load(f)

    image_ids = sorted({str(q["imageId"]) for q in questions.values()})
    paths = []
    missing = 0
    for image_id in image_ids:
        candidate = images_dir / f"{image_id}.jpg"
        if candidate.exists():
            paths.append(candidate)
        else:
            missing += 1

    print(
        f"GQA testdev reference: {len(paths):,} images "
        f"({len(image_ids):,} unique ids, {missing:,} missing on disk)"
    )
    return paths


POPE_JSON_URLS = {
    "random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    "popular": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
}


def load_pope_image_filenames(pope_json_paths):
    """Parse POPE JSONL files -> unique COCO val2014 filenames."""
    filenames = set()
    per_split = {}
    for jp in pope_json_paths:
        jp = Path(jp)
        split_names = []
        with open(jp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fname = row.get("image", "")
                if fname:
                    filenames.add(fname)
                    split_names.append(fname)
        per_split[jp.name] = len(split_names)
    return filenames, per_split


def collect_pope_reference_paths(pope_json_paths, coco_val2014_dir):
    """Map POPE image filenames to local COCO val2014 file paths."""
    coco_val2014_dir = Path(coco_val2014_dir)
    filenames, per_split = load_pope_image_filenames(pope_json_paths)
    print(f"POPE splits (question rows): {per_split}")

    ref_paths = []
    missing = []
    for fname in sorted(filenames):
        candidate = coco_val2014_dir / fname
        if candidate.exists():
            ref_paths.append(candidate)
        else:
            missing.append(fname)

    print(f"POPE val2014 found on disk: {len(ref_paths):,}")
    print(f"POPE val2014 missing: {len(missing):,}")
    if missing:
        print(f"  example missing: {missing[0]}")
    return ref_paths


def clear_phash_hash_cache(cache_root: Path = DEFAULT_DATA_ROOT):
    """Delete cached pHash matrices so the next scan rebuilds reference + train hashes."""
    cache_root = Path(cache_root)
    removed = []
    for name in ("ref_packed.npy", "train_packed.npy", "train_valid_indices.npy"):
        path = cache_root / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    if removed:
        print("Cleared pHash cache files:")
        for p in removed:
            print(f"  {p}")
    else:
        print("No pHash cache files to clear.")
    return removed


def generate_curriculum_decontamination_log(curriculum_samples: list, all_ref_paths: list, config, output_json_path: str = "curriculum_vs_benchmarks.json", type: str = None):

    if type != "phash" and type != "sscd" and type != "both":
        raise Exception("Enter a valid type for decontamination: 'phash' or 'sscd' or 'both'.")

    
    train_paths = [Path(s["image_path"]) for s in curriculum_samples]
    all_ref_paths = [Path(p) for p in all_ref_paths]

    verification_log = {
        "phash_matches": [],
        "sscd_top5_matches": [],
        "corrupt_indices": [],
    }

    GPU_BATCH = 64
    device = config.device
    
    if type == "phash" or type == "both":
        
        print(f"Stage 1: pHash Hamming <= {HAMMING_THRESH}")
    
        ref_cache = DEFAULT_DATA_ROOT / "ref_packed.npy"
        train_cache = DEFAULT_DATA_ROOT / "train_packed.npy"
        idx_cache = DEFAULT_DATA_ROOT / "train_valid_indices.npy"
    
        if ref_cache.exists() and train_cache.exists() and idx_cache.exists():
            print("Loading cached hash matrices ...")
            ref_packed = np.load(ref_cache)
            train_packed = np.load(train_cache)
            train_valid_indices = np.load(idx_cache).tolist()
            print(f"Loaded: ref={ref_packed.shape}, train={train_packed.shape}")
        else:
            ref_packed, _ = build_hash_matrix(all_ref_paths, "hashing reference (benchmarks)")
            train_packed, train_valid_indices = build_hash_matrix(train_paths, "hashing curriculum")
            np.save(ref_cache, ref_packed)
            np.save(train_cache, train_packed)
            np.save(idx_cache, np.array(train_valid_indices))
            print("Hash matrices saved to cache.")
    
        corrupt_indices = set(range(len(curriculum_samples))) - set(train_valid_indices)
        verification_log["corrupt_indices"] = [int(i) for i in corrupt_indices]
        print(f"corrupt/unreadable: {len(corrupt_indices):,}")
    
        ref_bits = torch.tensor(ref_packed, dtype=torch.uint8, device=device)
        train_bits = torch.tensor(train_packed, dtype=torch.uint8, device=device)
    
        phash_removed = []
        for i in tqdm(range(0, len(train_bits), GPU_BATCH), desc="hamming scan (GPU)"):
            chunk = train_bits[i : i + GPU_BATCH]
            xor = chunk.unsqueeze(1) ^ ref_bits.unsqueeze(0)
            counts = torch.zeros(xor.shape[:2], dtype=torch.int32, device=device)
            for bit in range(8):
                counts += ((xor >> bit) & 1).sum(dim=-1).to(torch.int32)
            matches = (counts <= HAMMING_THRESH).nonzero(as_tuple=False)
    
            for local_idx, ref_idx in matches.tolist():
                actual_local_idx = i + local_idx
                if actual_local_idx >= len(train_valid_indices):
                    continue
                global_train_idx = train_valid_indices[actual_local_idx]
                phash_removed.append(global_train_idx)
                verification_log["phash_matches"].append({
                    "train_idx": int(global_train_idx),
                    "train_image": str(train_paths[global_train_idx]),
                    "reference_image": str(all_ref_paths[ref_idx]),
                    "hamming_distance": int(counts[local_idx, ref_idx].item()),
                })
    
        phash_removed_set = set(phash_removed)
        print(f"pHash matches identified: {len(phash_removed_set):,}")

    if type == "sscd" or type == "both":
        import faiss

        print("Stage 2: SSCD model setup")
        sscd_path = DEFAULT_DATA_ROOT / "sscd_disc_mixup.torchscript.pt"
        if not sscd_path.exists():
            print("Downloading SSCD model ...")
            urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt", sscd_path)
        sscd = torch.jit.load(str(sscd_path)).to(config.device)
        sscd.eval()
    
        print("Embedding reference images ...")
        ref_embeds, ref_flags = embed_sscd(all_ref_paths, sscd, "embedding reference", config.device)
        valid_ref_paths = [all_ref_paths[idx] for idx, flag in enumerate(ref_flags) if flag]
        ref_embeds = ref_embeds[ref_flags]
        ref_embeds = ref_embeds / np.linalg.norm(ref_embeds, axis=1, keepdims=True)
    
        DIM = ref_embeds.shape[1]
        index = faiss.IndexFlatIP(DIM)
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(ref_embeds.astype(np.float32))
    
        sscd_scan_indices = [i for i in range(len(curriculum_samples)) if i not in phash_removed_set and i not in corrupt_indices]
        sscd_scan_paths = [train_paths[i] for i in sscd_scan_indices]
    
        print(f"Embedding {len(sscd_scan_paths):,} remaining curriculum images ...")
        train_embeds, train_flags = embed_sscd(sscd_scan_paths, sscd, "embedding curriculum", config.device)
    
        valid_indices = []
        valid_embeds_list = []
        for local_idx, is_valid in enumerate(train_flags):
            global_idx = sscd_scan_indices[local_idx]
            if not is_valid:
                verification_log["corrupt_indices"].append(int(global_idx))
            else:
                valid_indices.append(global_idx)
                valid_embeds_list.append(train_embeds[local_idx])
    
        valid_embeds = np.array(valid_embeds_list)
        valid_embeds = valid_embeds / np.linalg.norm(valid_embeds, axis=1, keepdims=True)
    
        print("Executing FAISS Top-5 index search ...")
        D, I = index.search(valid_embeds.astype(np.float32), k=5)
    
        for i in range(len(valid_indices)):
            global_train_idx = valid_indices[i]
            neighbors_data = [
                {
                    "rank": rank + 1,
                    "reference_image": str(valid_ref_paths[I[i, rank]]),
                    "cosine_similarity": float(D[i, rank]),
                }
                for rank in range(5)
            ]
            verification_log["sscd_top5_matches"].append({
                "train_idx": int(global_train_idx),
                "train_image": str(train_paths[global_train_idx]),
                "source": curriculum_samples[global_train_idx].get("source", "unknown"),
                "neighbors": neighbors_data,
            })

    print(f"Saving log to {output_json_path} ...")
    with open(output_json_path, "w") as f:
        json.dump(verification_log, f, indent=4)
    print("Log generation complete.")
    return output_json_path


def filter_curriculum_from_log(curriculum_samples: list, json_log_path: str, sscd_threshold: float = SSCD_THRESH):

    print(f"Loading log from {json_log_path} ...")
    with open(json_log_path) as f:
        log_data = json.load(f)

    to_remove: set = set()

    corrupt_count = len(log_data["corrupt_indices"])
    to_remove.update(log_data["corrupt_indices"])

    phash_count = 0
    for match in log_data["phash_matches"]:
        to_remove.add(match["train_idx"])
        phash_count += 1

    sscd_count = 0
    for item in log_data["sscd_top5_matches"]:
        top1 = next(n for n in item["neighbors"] if n["rank"] == 1)
        if top1["cosine_similarity"] >= sscd_threshold:
            to_remove.add(item["train_idx"])
            sscd_count += 1

    n_total = len(curriculum_samples)
    clean_samples = [s for idx, s in enumerate(curriculum_samples) if idx not in to_remove]

    from collections import Counter
    removed_sources = Counter(
        curriculum_samples[idx].get("source", "unknown")
        for idx in to_remove
        if idx < n_total
    )

    print("\nDecontamination Summary")
    print(f"Original curriculum size: {n_total:,}")
    print(f"pHash matches removed: {phash_count:,}")
    print(f"SSCD Top-1 breaches: {sscd_count:,}")
    print(f"Corrupt/unreadable: {corrupt_count:,}")
    print(f"Total removed: {len(to_remove):,}")
    print(f"Clean curriculum size: {len(clean_samples):,} ({len(clean_samples) / n_total * 100:.3f}%)")
    if removed_sources:
        print("Removed by source:", dict(removed_sources))

    return clean_samples


# class RegionAlignmentDataset(Dataset):
#     """
#     Each sample: one image, one bounding box, one text description.
#     Training objective: given region token, predict the description.
#     Loss is computed on description tokens only — region token is masked.
#     """
#     def __init__(self, samples, image_processor, tokenizer, max_tokens=64):
#         self.samples = samples
#         self.image_processor = image_processor
#         self.tokenizer = tokenizer
#         self.max_tokens = max_tokens

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         import logging
#         logger = logging.getLogger(__name__)
#         sample = self.samples[idx]
#         try:
#             pil_image = Image.open(sample['image_path']).convert('RGB')
#             img_w, img_h = pil_image.size

#             # Clamp box to image bounds and ensure minimum size
#             x1, y1, x2, y2 = sample['box']
#             x1 = max(0.0, min(float(x1), img_w - 1))
#             y1 = max(0.0, min(float(y1), img_h - 1))
#             x2 = max(x1 + 1, min(float(x2), img_w))
#             y2 = max(y1 + 1, min(float(y2), img_h))

#             # Scale box to 336x336 space — CLIPImageProcessor resizes to 336x336
#             scale_x = 336.0 / img_w
#             scale_y = 336.0 / img_h
#             box_336 = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

#             pixel_values = self.image_processor(
#                 images=pil_image, return_tensors="pt"
#             )['pixel_values'].squeeze(0)  # (3, 336, 336)

#             token_ids = self.tokenizer(
#                 sample['description'].strip(),
#                 max_length=self.max_tokens,
#                 truncation=True,
#                 padding='max_length',
#                 return_tensors='pt',
#                 add_special_tokens=True,
#             )

#             return {
#                 'pixel_values': pixel_values,                           # (3, 336, 336)
#                 'box': torch.tensor(box_336, dtype=torch.float32),      # (4,) in 336px space
#                 'input_ids': token_ids['input_ids'].squeeze(0),         # (max_tokens,)
#                 'attention_mask': token_ids['attention_mask'].squeeze(0), # (max_tokens,)
#             }
#         except Exception as e:
#             logger.warning(f"Sample {idx} failed: {e}")
#             return self.__getitem__((idx + 1) % len(self.samples))


# def collate_region_batch(batch):
#     return {
#         'pixel_values': torch.stack([b['pixel_values'] for b in batch]),   # (B, 3, 336, 336)
#         'boxes': torch.stack([b['box'] for b in batch]),                    # (B, 4)
#         'input_ids': torch.stack([b['input_ids'] for b in batch]),          # (B, max_tokens)
#         'attention_mask': torch.stack([b['attention_mask'] for b in batch]),# (B, max_tokens)
#     }
class RegionAlignmentDataset(Dataset):
    """
    Each sample: one image, one bounding box, one text description.
    Training objective: given region token, predict the description.
    Loss is computed on description tokens only — region token is masked.
    """
    def __init__(self, samples, image_processor, tokenizer, max_tokens=64):
        self.samples = samples
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import logging
        logger = logging.getLogger(__name__)
        sample = self.samples[idx]
        try:
            pil_image = Image.open(sample['image_path']).convert('RGB')
            img_w, img_h = pil_image.size

            # Clamp and scale description box to 336px space
            x1, y1, x2, y2 = sample['box']
            x1 = max(0.0, min(float(x1), img_w - 1))
            y1 = max(0.0, min(float(y1), img_h - 1))
            x2 = max(x1 + 1, min(float(x2), img_w))
            y2 = max(y1 + 1, min(float(y2), img_h))
            scale_x = 336.0 / img_w
            scale_y = 336.0 / img_h
            box_336 = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

            pixel_values = self.image_processor(
                images=pil_image, return_tensors="pt"
            )['pixel_values'].squeeze(0)  # (3, 336, 336)

            token_ids = self.tokenizer(
                sample['description'].strip(),
                max_length=self.max_tokens,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                add_special_tokens=True,
            )

            return {
                'pixel_values': pixel_values,
                'box': torch.tensor(box_336, dtype=torch.float32),        # (4,)
                'input_ids': token_ids['input_ids'].squeeze(0),
                'attention_mask': token_ids['attention_mask'].squeeze(0),
                'source': sample.get('source', 'other'),
            }

        except Exception as e:
            logger.warning(f"Sample {idx} failed: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))


def collate_region_batch(batch):
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'boxes': torch.stack([b['box'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'sources': [b.get('source', 'other') for b in batch],
    }


def load_coco_detection_samples(data_dir, max_samples=None):
    import json
    coco_dir = data_dir / "coco"
    ann_path = coco_dir / "annotations" / "instances_train2017.json"
    images_dir = coco_dir / "train2017"

    if not ann_path.exists():
        print(f"COCO annotations not found at {ann_path}")
        print("Download from https://cocodataset.org/")
        return []

    with open(ann_path) as f:
        coco_data = json.load(f)

    cat_map = {c['id']: c['name'] for c in coco_data['categories']}
    img_map = {img['id']: img['file_name'] for img in coco_data['images']}

    samples = []
    for ann in coco_data['annotations']:
        img_file = images_dir / img_map[ann['image_id']]
        if not img_file.exists():
            continue
        x, y, w, h = ann['bbox']
        if w < 10 or h < 10:
            continue
        samples.append({
            'image_path': str(img_file),
            'box': [x, y, x + w, y + h],
            'description': cat_map[ann['category_id']],  # single category name — simple Stage 2a target
            'source': 'coco',
        })
        if max_samples and len(samples) >= max_samples:
            break

    print(f"COCO samples: {len(samples):,}")
    return samples


def load_refcoco_samples(data_dir, splits=['train'], max_samples=None):
    """
    Load RefCOCO, RefCOCO+ and RefCOCOg samples from jxu124/* HuggingFace repos.

    jxu124/refcoco, jxu124/refcocoplus, jxu124/refcocog are auto-converted to
    Parquet — no trust_remote_code needed. Each has train/val/testA/testB splits
    with 42k+ train rows.

    Field mapping from jxu124/* repos:
      image_path  : relative path e.g. 'coco/train2014/COCO_train2014_XXXXXX.jpg'
                    — joined with data_dir to get the full path
      bbox        : [x1, y1, x2, y2] in pixel coords (NOT [x, y, w, h])
      sentences   : list of dicts with 'raw' and 'sent' keys
      split       : 'train' / 'val' / 'testA' / 'testB'

    Requires COCO train2014 images at data_dir/coco/train2014/.
    Download from http://images.cocodataset.org/zips/train2014.zip (~13GB).
    """
    all_samples = []

    # Three separate jxu124 repos — refcoco+ uses 'refcocoplus' (no '+' in repo ID)
    repos = [
        ("jxu124/refcoco",     "refcoco"),
        ("jxu124/refcocog",    "refcocog"),
        ("jxu124/refcocoplus", "refcoco+"),
    ]

    coco_train2014 = data_dir / "coco" / "train2014"
    if not coco_train2014.exists():
        print(f"WARNING: COCO train2014 not found at {coco_train2014}")
        print("Download from http://images.cocodataset.org/zips/train2014.zip")

    for hf_id, label in repos:
        for split in splits:
            try:
                ds = load_dataset(hf_id, split=split)

                dataset_samples = []
                for item in ds:
                    # split field in item matches the HuggingFace split parameter,
                    # but double-check since some repos include all splits in one file
                    if item.get('split') not in splits:
                        continue

                    # image_path is relative — join with data_dir to resolve
                    rel_path = item.get('image_path', '')
                    if not rel_path:
                        continue
                    img_path = data_dir / rel_path
                    if not img_path.exists():
                        continue

                    bbox = item.get('bbox')
                    if bbox is None:
                        continue
                    # bbox is [x1, y1, x2, y2] — not [x, y, w, h]
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    sentences = item.get('sentences', [])
                    for sent in sentences:
                        if isinstance(sent, dict):
                            text = sent.get('raw', sent.get('sent', '')).strip()
                        else:
                            text = str(sent).strip()
                        if not text:
                            continue
                        dataset_samples.append({
                            'image_path': str(img_path),
                            'box': [x1, y1, x2, y2],
                            'description': text,
                            'source': label,
                        })

                print(f"  {label}/{split}: {len(dataset_samples):,} samples")
                all_samples.extend(dataset_samples)

            except Exception as e:
                print(f"  {label}/{split} failed: {e}")

    if max_samples:
        all_samples = all_samples[:max_samples]
    print(f"RefCOCO/+/g total: {len(all_samples):,}")
    return all_samples


def load_visual_genome_samples(data_dir, max_per_image=15, max_samples=None):
    """
    Load Visual Genome region description samples.

    Downloads region_descriptions.json from Stanford (~170MB) and VG images
    from Stanford (~15GB total across VG_100K and VG_100K_2) on first run.
    Both are cached under data_dir/visual_genome/ for subsequent runs.

    HuggingFace visual_genome repos use loading scripts which are no longer
    supported by newer datasets versions — direct Stanford download used instead.

    region_descriptions.json format per entry:
      id      : image_id
      regions : list of {phrase, x, y, width, height, region_id}
    Images stored as {image_id}.jpg in VG_100K/ or VG_100K_2/.
    """
    import json
    import random
    import urllib.request

    vg_dir = data_dir / "visual_genome"
    vg_dir.mkdir(exist_ok=True)

    regions_path = vg_dir / "region_descriptions.json"
    if not regions_path.exists():
        print("Downloading Visual Genome region_descriptions.json (~170MB)...")
        urllib.request.urlretrieve(
            "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip",
            vg_dir / "region_descriptions.json.zip"
        )
        import zipfile
        with zipfile.ZipFile(vg_dir / "region_descriptions.json.zip") as zf:
            zf.extractall(vg_dir)
        (vg_dir / "region_descriptions.json.zip").unlink()
        print("region_descriptions.json downloaded.")

    vg_100k   = vg_dir / "VG_100K"
    vg_100k_2 = vg_dir / "VG_100K_2"
    # Stanford zips embed a VG_100K/ prefix; unzipping into VG_100K/ nests one level.
    image_roots = [vg_100k, vg_100k_2, vg_100k / "VG_100K", vg_100k_2 / "VG_100K_2"]
    if not vg_100k.exists() or not any(vg_100k.iterdir()):
        print("Downloading VG_100K images (~9GB)...")
        urllib.request.urlretrieve(
            "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
            vg_dir / "images.zip"
        )
        import zipfile
        with zipfile.ZipFile(vg_dir / "images.zip") as zf:
            zf.extractall(vg_dir)
        (vg_dir / "images.zip").unlink()
        print("VG_100K downloaded.")

    if not vg_100k_2.exists() or not any(vg_100k_2.iterdir()):
        print("Downloading VG_100K_2 images (~5GB)...")
        urllib.request.urlretrieve(
            "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            vg_dir / "images2.zip"
        )
        import zipfile
        with zipfile.ZipFile(vg_dir / "images2.zip") as zf:
            zf.extractall(vg_dir)
        (vg_dir / "images2.zip").unlink()
        print("VG_100K_2 downloaded.")

    with open(regions_path) as f:
        all_regions = json.load(f)

    samples = []
    for image_entry in all_regions:
        image_id = image_entry['id']
        img_path = None
        for subdir in image_roots:
            candidate = subdir / f"{image_id}.jpg"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        regions = image_entry.get('regions', [])
        # Cap per image for dataset diversity — without capping, high-annotation
        # images dominate and reduce effective image diversity
        if len(regions) > max_per_image:
            regions = random.sample(regions, max_per_image)

        for region in regions:
            desc = region.get('phrase', '').strip()
            if not desc or len(desc) < 3:
                continue
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            if w < 10 or h < 10:
                continue
            samples.append({
                'image_path': str(img_path),
                'box': [x, y, x + w, y + h],  # convert [x,y,w,h] -> [x1,y1,x2,y2]
                'description': desc,            # rich attribute description — Stage 2b target
                'source': 'visual_genome',
            })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"Visual Genome samples: {len(samples):,}  (cap={max_per_image}/image)")
    return samples


def _download_grit_metadata(data_dir, shard=0):
    """Download a single GRIT-20M metadata shard (a Parquet of grounded captions).

    Each shard 'coyo_{shard}_snappy.parquet' holds ~1M rows of
    {url, caption, width, height, noun_chunks, ref_exps, clip_similarity_*}.
    Boxes inside noun_chunks/ref_exps are NORMALISED [0,1]; the phrase text is a
    character span into 'caption'. We only fetch the shard we need (~300MB) rather
    than cloning the full 6.6GB repo.
    """
    grit_dir = data_dir / "grit"
    meta_dir = grit_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    fname = f"coyo_{shard}_snappy.parquet"
    out_path = meta_dir / "grit-20m" / fname
    if not out_path.exists():
        print(f"Downloading GRIT metadata shard {shard} ({fname})...")
        hf_hub_download(
            repo_id="zzliang/GRIT",
            repo_type="dataset",
            subfolder="grit-20m",
            filename=fname,
            local_dir=str(meta_dir),
        )
    return out_path


def _prepare_grit_download_list(meta_path, grit_dir, max_images, min_clip_l14):
    """Filter the raw shard down to the rows we actually want images for.

    Filtering BEFORE downloading keeps disk/network bounded: we drop low
    image-text alignment rows and rows with no grounded boxes, then cap to
    max_images. img2dataset reads the resulting Parquet via its 'url' column.
    """
    import pandas as pd

    filtered_path = grit_dir / "filtered.parquet"
    if filtered_path.exists():
        return filtered_path

    cap_str = "none (full shard)" if max_images is None else f"{max_images:,}"
    print(f"Filtering GRIT shard (clip_l14 >= {min_clip_l14}, has boxes, cap={cap_str})...")
    df = pd.read_parquet(meta_path)
    df = df[df["clip_similarity_vitl14"] >= min_clip_l14]
    df = df[df["noun_chunks"].map(lambda x: x is not None and len(x) > 0)]
    if max_images is not None:
        df = df.head(max_images)
    df = df.reset_index(drop=True)
    df.to_parquet(filtered_path)
    print(f"Kept {len(df):,} rows -> {filtered_path}")
    return filtered_path


def _download_grit_images(filtered_parquet, images_dir):
    """Download GRIT images with img2dataset into a flat files layout.

    'files' output writes per-image NNN.jpg + NNN.json (the json carries the
    saved additional columns: noun_chunks, ref_exps, etc.). Dead/blocked URLs are
    skipped gracefully by img2dataset, so the realised count is < requested.
    """
    import subprocess

    images_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "img2dataset",
        "--url_list", str(filtered_parquet),
        "--input_format", "parquet",
        "--url_col", "url",
        "--caption_col", "caption",
        "--output_format", "files",
        "--output_folder", str(images_dir),
        "--processes_count", "4",
        "--thread_count", "64",
        "--image_size", "336",
        "--resize_only_if_bigger", "True",
        "--resize_mode", "keep_ratio",
        "--skip_reencode", "True",
        "--save_additional_columns",
        '["noun_chunks","ref_exps","clip_similarity_vitl14"]',
        "--enable_wandb", "False",
    ]
    print("Running img2dataset (this can take a while)...")
    subprocess.run(cmd, check=True)
    return images_dir


def load_grit_samples(data_dir, shard=0, max_images=50000, min_clip_l14=0.30, min_box_frac=0.05, max_boxes_per_image=16, use_ref_exps=True, max_samples=None):
    """Load open-vocabulary region samples from one GRIT-20M shard.

    GRIT pairs web captions with GLIP-grounded boxes, so its phrases span a far
    wider vocabulary than COCO/VG (e.g. 'wire hanger', 'paper cover') — this is
    what broadens Projection-A beyond the COCO/VG closed set.

    On first run this downloads the shard metadata, filters it, and fetches images
    via img2dataset (requires `pip install img2dataset`). Subsequent runs reuse the
    cached images under data_dir/grit/images/.

    Returns the same schema as the other loaders:
      {'image_path': str, 'box': [x1,y1,x2,y2] pixel, 'description': str, 'source': 'grit'}

    Boxes in GRIT are normalised [0,1]; we scale them by the ACTUAL on-disk image
    size (img2dataset may have resized), so they land in the same pixel space the
    region collator expects. The phrase is caption[start:end].
    """
    grit_dir = data_dir / "grit"
    images_dir = grit_dir / "images"

    if not images_dir.exists() or not any(images_dir.glob("**/*.jpg")):
        meta_path = _download_grit_metadata(data_dir, shard)
        filtered = _prepare_grit_download_list(meta_path, grit_dir, max_images, min_clip_l14)
        _download_grit_images(filtered, images_dir)

    span_key = "ref_exps" if use_ref_exps else "noun_chunks"
    samples = []
    for json_path in images_dir.glob("**/*.json"):
        jpg_path = json_path.with_suffix(".jpg")
        if not jpg_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("status") not in (None, "success"):
            continue

        try:
            W, H = Image.open(jpg_path).size
        except Exception:
            continue

        caption = meta.get("caption", "") or ""
        spans = meta.get(span_key) or []
        kept = 0
        for item in spans:
            # item = [start, end, x1, y1, x2, y2, score]  (coords normalised 0-1)
            start, end = int(item[0]), int(item[1])
            x1, y1, x2, y2 = float(item[2]), float(item[3]), float(item[4]), float(item[5])
            phrase = caption[start:end].strip()
            if len(phrase) < 3:
                continue
            if (x2 - x1) < min_box_frac or (y2 - y1) < min_box_frac:
                continue
            samples.append({
                "image_path": str(jpg_path),
                "box": [x1 * W, y1 * H, x2 * W, y2 * H],
                "description": phrase,
                "source": "grit",
            })
            kept += 1
            if kept >= max_boxes_per_image:
                break

        if max_samples and len(samples) >= max_samples:
            break

    print(f"GRIT samples: {len(samples):,}  (shard={shard}, source=ref_exps={use_ref_exps})")
    return samples


# Map a sample's fine-grained 'source' to a coarse training family.
# RefCOCO/+/g are collapsed into one family so they aren't triple-weighted.
_FAMILY_MAP = {
    'coco':           'coco',
    'refcoco':        'refcoco',
    'refcocog':       'refcoco',
    'refcoco+':       'refcoco',
    'visual_genome':  'vg',
    'grit':           'grit',
}


def region_family_of(source):
    """Coarse family label for balancing (unknown sources fall back to 'other')."""
    return _FAMILY_MAP.get(source, 'other')


class FamilyBalancedBatchSampler(Sampler):
    """Yield batches with (near-)equal representation from each dataset family.

    Standard concatenation makes a sample's gradient exposure proportional to its
    family's raw size, so VG (~1.6M) and COCO (~0.77M) swamp RefCOCO (~0.32M) and
    GRIT (~0.5M). This sampler instead gives each family an equal share of every
    batch: with B samples and F families present, each family contributes ~B/F
    indices per batch (remainder slots rotate randomly for fairness).

    Smaller families are cycled (oversampled) and larger families are subsampled
    within an epoch, equalising exposure without duplicating data on disk. Each
    epoch reshuffles deterministically from (seed + epoch); call set_epoch(epoch).

    Args:
        sources: list of per-sample 'source' strings, aligned with dataset indices.
        batch_size: micro-batch size (same value you'd pass to DataLoader).
        num_batches: batches per epoch; defaults to len(sources) // batch_size so
                     the step/scheduler estimate matches the concatenated baseline.
        seed: base RNG seed; combined with epoch for per-epoch shuffling.
    """

    def __init__(self, sources, batch_size, num_batches=None, seed=42):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        family_indices = defaultdict(list)
        for idx, src in enumerate(sources):
            family_indices[region_family_of(src)].append(idx)

        # Keep only non-empty families, in a stable order
        self.families = sorted(f for f, idxs in family_indices.items() if idxs)
        self.family_indices = {f: family_indices[f] for f in self.families}
        self.num_families = len(self.families)
        if self.num_families == 0:
            raise ValueError("FamilyBalancedBatchSampler: no samples provided")

        self.num_batches = (len(sources) // self.batch_size) if num_batches is None else int(num_batches)

        counts = {f: len(self.family_indices[f]) for f in self.families}
        print(f"FamilyBalancedBatchSampler | families={counts} | "
              f"per-batch≈{self.batch_size // self.num_families}/family | batches/epoch={self.num_batches:,}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        pools = {}
        cursors = {}
        for f in self.families:
            pool = self.family_indices[f][:]
            rng.shuffle(pool)
            pools[f] = pool
            cursors[f] = 0

        def draw(f, k):
            out = []
            pool = pools[f]
            cur = cursors[f]
            for _ in range(k):
                if cur >= len(pool):          # exhausted -> reshuffle and cycle
                    rng.shuffle(pool)
                    cur = 0
                out.append(pool[cur])
                cur += 1
            cursors[f] = cur
            return out

        base = self.batch_size // self.num_families
        remainder = self.batch_size - base * self.num_families

        for _ in range(self.num_batches):
            batch = []
            fam_order = self.families[:]
            rng.shuffle(fam_order)            # rotate who gets remainder slots
            for i, f in enumerate(fam_order):
                k = base + (1 if i < remainder else 0)
                batch.extend(draw(f, k))
            rng.shuffle(batch)                # mix families within the batch
            yield batch
