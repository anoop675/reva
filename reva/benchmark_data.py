"""Benchmark download/helpers for official eval (no faiss / decontamination deps)."""

import json
import shutil
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from .config import DEFAULT_HF_CACHE_ROOT

# Official MMBench TSV URLs (VLMEvalKit / open-compass/MMBench README).
MMBENCH_TSV_URLS = {
    "MMBench_DEV_EN": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv",
    "MMBench_TEST_EN": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv",
    "MMBench_DEV_EN_V11": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN_V11.tsv",
    "MMBench_TEST_EN_V11": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN_V11.tsv",
}

SEED_BENCH_HF_REPO = "AILab-CVC/SEED-Bench"
SEED_BENCH_JSON_FILE = "SEED-Bench.json"
SEED_BENCH_IMAGE_ZIP = "SEED-Bench-image.zip"
SEED_BENCH_VIDEO_ZIP_PARTS = 32


def _normalize_mmbench_index(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def build_mmbench_image_map(df) -> dict:
    image_map = {}
    for idx, img in zip(df["index"], df["image"]):
        key = _normalize_mmbench_index(idx)
        if not key:
            continue
        if img is None or (isinstance(img, float) and np.isnan(img)):
            continue
        image_map[key] = img
    return image_map


def resolve_mmbench_b64(image_map: dict, idx_key: str):
    raw = image_map.get(idx_key)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    if len(raw) <= 64:
        ref_key = _normalize_mmbench_index(raw)
        ref_raw = image_map.get(ref_key)
        if ref_raw is None or (isinstance(ref_raw, float) and np.isnan(ref_raw)):
            return None
        ref_raw = str(ref_raw).strip()
        if len(ref_raw) <= 64:
            return resolve_mmbench_b64(image_map, ref_key)
        return ref_raw
    return raw


def decode_base64_image_to_jpg(b64_str: str, out_path: Path):
    import base64

    b64_str = str(b64_str).strip()
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[-1]
    pad = (-len(b64_str)) % 4
    if pad:
        b64_str += "=" * pad

    raw = base64.b64decode(b64_str)
    img = Image.open(BytesIO(raw)).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def extract_mmbench_images_from_tsv(tsv_path: Path, images_dir: Path):
    import pandas as pd

    tsv_path = Path(tsv_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tsv_path, sep="\t")
    if "image" not in df.columns:
        raise ValueError(f"No 'image' column in {tsv_path}")

    image_map = build_mmbench_image_map(df)
    canonical_keys = sorted(
        (
            key
            for key, raw in image_map.items()
            if len(str(raw).strip()) > 64
        ),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    paths = []
    for idx_key in canonical_keys:
        b64 = resolve_mmbench_b64(image_map, idx_key)
        if not b64:
            continue
        out_path = images_dir / f"{idx_key}.jpg"
        if not out_path.exists():
            decode_base64_image_to_jpg(b64, out_path)
        paths.append(out_path)

    short_refs = sum(
        1 for raw in image_map.values() if 0 < len(str(raw).strip()) <= 64
    )
    print(
        f"{tsv_path.name}: {len(df):,} rows -> {len(paths):,} unique images "
        f"({short_refs:,} short refs, dir={images_dir})"
    )
    return paths


def download_mmbench_tsv(mmbench_root: Path, split_names=None):
    mmbench_root = Path(mmbench_root)
    tsv_dir = mmbench_root / "tsv"
    tsv_dir.mkdir(parents=True, exist_ok=True)

    split_names = split_names or list(MMBENCH_TSV_URLS.keys())
    local_paths = {}
    for split_name in split_names:
        url = MMBENCH_TSV_URLS.get(split_name)
        if url is None:
            raise ValueError(f"Unknown MMBench split: {split_name}")
        dest = tsv_dir / f"{split_name}.tsv"
        min_bytes = 1_000_000
        if dest.exists() and dest.stat().st_size < min_bytes:
            print(f"Removing incomplete {split_name}: {dest} ({dest.stat().st_size} bytes)")
            dest.unlink()
        if not dest.exists():
            print(f"Downloading {split_name} ...")
            urllib.request.urlretrieve(url, dest)
        else:
            print(f"Using cached {split_name}: {dest}")
        local_paths[split_name] = dest
    return local_paths


def prepare_mmbench_reference_images(mmbench_root: Path, split_names=None):
    mmbench_root = Path(mmbench_root)
    split_names = split_names or list(MMBENCH_TSV_URLS.keys())
    tsv_paths = download_mmbench_tsv(mmbench_root, split_names=split_names)

    all_paths = []
    per_split = {}
    for split_name, tsv_path in tsv_paths.items():
        img_dir = mmbench_root / "images" / split_name
        split_paths = extract_mmbench_images_from_tsv(tsv_path, img_dir)
        per_split[split_name] = split_paths
        all_paths.extend(split_paths)

    print(f"MMBench total reference images: {len(all_paths):,} across {len(split_names)} splits")
    return all_paths, per_split


def collect_mmbench_reference_paths(mmbench_root: Path, split_names=None):
    mmbench_root = Path(mmbench_root)
    split_names = split_names or list(MMBENCH_TSV_URLS.keys())
    images_root = mmbench_root / "images"

    existing = sorted(images_root.glob("*/*.jpg"))
    if existing:
        print(f"MMBench cached images: {len(existing):,} under {images_root}")
        return existing

    paths, _ = prepare_mmbench_reference_images(mmbench_root, split_names=split_names)
    return sorted(paths)


def download_seed_bench_json(seed_root: Path, hf_cache=DEFAULT_HF_CACHE_ROOT) -> Path:
    seed_root = Path(seed_root)
    seed_root.mkdir(parents=True, exist_ok=True)
    dest = seed_root / SEED_BENCH_JSON_FILE
    if dest.exists():
        print(f"Using cached {dest}")
        return dest

    cached = hf_hub_download(
        repo_id=SEED_BENCH_HF_REPO,
        filename=SEED_BENCH_JSON_FILE,
        repo_type="dataset",
        cache_dir=hf_cache,
    )
    shutil.copy2(cached, dest)
    print(f"Downloaded {SEED_BENCH_JSON_FILE} -> {dest}")
    return dest


def download_seed_bench_images(seed_root: Path, hf_cache=DEFAULT_HF_CACHE_ROOT) -> Path:
    seed_root = Path(seed_root)
    images_root = seed_root / "SEED-Bench-image"
    if images_root.is_dir():
        existing = [p for p in images_root.iterdir() if p.is_file()]
        if existing:
            print(f"Using cached SEED-Bench images: {len(existing):,} under {images_root}")
            return images_root

    zip_path = hf_hub_download(
        repo_id=SEED_BENCH_HF_REPO,
        filename=SEED_BENCH_IMAGE_ZIP,
        repo_type="dataset",
        cache_dir=hf_cache,
    )
    print(f"Extracting {SEED_BENCH_IMAGE_ZIP} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(seed_root)

    existing = [p for p in images_root.iterdir() if p.is_file()]
    print(f"Extracted {len(existing):,} images to {images_root}")
    return images_root


def load_seed_bench_questions(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if "questions" not in data:
        raise ValueError(f"No 'questions' key in {json_path}")
    return data["questions"]


def resolve_seed_image_path(images_root: Path, data_id: str):
    images_root = Path(images_root)
    data_id = str(data_id).strip()
    if not data_id:
        return None
    for candidate in (
        images_root / data_id,
        images_root / f"{data_id}.jpg",
        images_root / f"{data_id}.png",
        images_root / f"{data_id}.jpeg",
    ):
        if candidate.is_file():
            return candidate
    return None