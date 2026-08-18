import json
import math
import os
import re
import urllib.request
import zipfile
from collections import defaultdict, Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
from PIL import Image
from tqdm.auto import tqdm
from .config import DEFAULT_VQAV2_ROOT, EvalConfig, Stage3Config
from .inference import (
    run_vqa_inference,
    run_combined_inference,
    normalise_answer,
    vqa_soft_score,
)

def download_gqa_val(gqa_dir: Path):
    gqa_dir.mkdir(exist_ok=True, parents=True)
    questions_path = gqa_dir / "val_balanced_questions.json"
    gqa_images_dir = gqa_dir / "images"

    if not questions_path.exists():
        zip_path = gqa_dir / "questions1.2.zip"
        if not zip_path.exists():
            print("Downloading questions1.2.zip from Stanford...")
            urllib.request.urlretrieve(
                "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
                zip_path
            )
        print(f"Extracting {questions_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(questions_path.name, gqa_dir)

    if not gqa_images_dir.exists():
        img_zip = gqa_dir / "images.zip"
        if not img_zip.exists():
            print("Downloading GQA images.zip from Stanford...")
            urllib.request.urlretrieve(
                "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
                img_zip
            )
        print("Extracting images.zip...")
        with zipfile.ZipFile(img_zip, 'r') as zf:
            zf.extractall(gqa_dir)
    else:
        n_images = sum(1 for _ in gqa_images_dir.glob("*.jpg"))
        print(f"Images already present: {n_images:,} jpg files at {gqa_images_dir}")

    with open(questions_path) as f:
        gqa_raw = json.load(f)

    gqa_samples = [{
        "question_id": qid,
        "question": v["question"],
        "answer": v["answer"],
        "type": v.get("types", {}).get("semantic", "unknown"),
        "image_id": v["imageId"],
    } for qid, v in gqa_raw.items()]

    print(f"GQA val loaded: {len(gqa_samples):,} samples")
    return gqa_samples, gqa_images_dir


def download_gqa_testdev(gqa_dir: Path):
    gqa_dir.mkdir(exist_ok=True, parents=True)
    questions_path = gqa_dir / "testdev_balanced_questions.json"
    gqa_images_dir = gqa_dir / "images"

    if not questions_path.exists():
        zip_path = gqa_dir / "questions1.2.zip"
        if not zip_path.exists():
            print("Downloading questions1.2.zip from Stanford...")
            urllib.request.urlretrieve(
                "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
                zip_path
            )
        print(f"Extracting {questions_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(questions_path.name, gqa_dir)

    if not gqa_images_dir.exists():
        img_zip = gqa_dir / "images.zip"
        if not img_zip.exists():
            print("Downloading GQA images.zip from Stanford...")
            urllib.request.urlretrieve(
                "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
                img_zip
            )
        print("Extracting images.zip...")
        with zipfile.ZipFile(img_zip, 'r') as zf:
            zf.extractall(gqa_dir)
    else:
        n_images = sum(1 for _ in gqa_images_dir.glob("*.jpg"))
        print(f"Images already present: {n_images:,} jpg files at {gqa_images_dir}")

    with open(questions_path) as f:
        gqa_raw = json.load(f)

    gqa_samples = [{
        "question_id": qid,
        "question": v["question"],
        "answer": v["answer"],
        "type": v.get("types", {}).get("semantic", "unknown"),
        "image_id": v["imageId"],
    } for qid, v in gqa_raw.items()]

    print(f"GQA testdev loaded: {len(gqa_samples):,} samples")
    return gqa_samples, gqa_images_dir


def run_gqa_evaluation(gqa_samples, gqa_images_dir, frozen_vit, projection_head_b, frozen_qwen, clip_image_processor, qwen_tokenizer, eval_config, desc="GQA"):
    gqa_correct = 0
    gqa_total = 0
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    gqa_results = {"predictions": [], "overall_accuracy": 0.0, "by_type": {}}

    print("Running GQA zero-shot inference...")

    for sample in tqdm(gqa_samples, desc=desc):
        if 'image' in sample and not isinstance(sample['image'], str):
            pil_image = sample['image'].convert('RGB')
        else:
            image_id = sample.get('image_id') or sample.get('imageId')
            img_path = gqa_images_dir / f"{image_id}.jpg"
            pil_image = Image.open(img_path).convert('RGB')

        question = sample['question']
        ground_truth = normalise_answer(str(sample['answer']))
        question_type = sample.get('type', 'unknown')

        predicted = run_vqa_inference(
            pil_image=pil_image,
            question_text=question,
            format_prompt=eval_config.gqa_format_prompt,
            frozen_vit=frozen_vit,
            projection_head_b=projection_head_b,
            frozen_qwen=frozen_qwen,
            clip_image_processor=clip_image_processor,
            qwen_tokenizer=qwen_tokenizer,
            eval_config=eval_config,
        )

        predicted_norm = normalise_answer(predicted)
        is_correct = predicted_norm == ground_truth

        gqa_correct += int(is_correct)
        gqa_total += 1
        type_correct[question_type] += int(is_correct)
        type_total[question_type] += 1

        gqa_results["predictions"].append({
            "question": question,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "question_type": question_type,
        })

    gqa_results["overall_accuracy"] = gqa_correct / gqa_total
    gqa_results["by_type"] = {qt: type_correct[qt] / type_total[qt] for qt in type_total}

    with open(eval_config.gqa_results_path, 'w') as f:
        json.dump(gqa_results, f, indent=2)

    print(f"\nGQA Zero-Shot Results on global-only baseline")
    print(f"Overall accuracy: {gqa_results['overall_accuracy']*100:.2f}%")
    print("\nBy question type:")
    for qt, acc in sorted(gqa_results['by_type'].items()):
        print(f"{qt:}\t{acc*100:.2f}%  (n={type_total[qt]:,})")

    return gqa_results


def download_vqav2_val(vqav2_dir: Path):
    vqav2_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir = vqav2_dir / "val2014"
    val_images_dir.mkdir(exist_ok=True)

    q_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
    q_zip = vqav2_dir / "v2_Questions_Val_mscoco.zip"
    if not (vqav2_dir / "v2_OpenEnded_mscoco_val2014_questions.json").exists():
        print("Downloading val questions...")
        urllib.request.urlretrieve(q_url, q_zip)
        with zipfile.ZipFile(q_zip) as zf:
            zf.extractall(vqav2_dir)
        q_zip.unlink()

    a_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
    a_zip = vqav2_dir / "v2_Annotations_Val_mscoco.zip"
    if not (vqav2_dir / "v2_mscoco_val2014_annotations.json").exists():
        print("Downloading val annotations...")
        urllib.request.urlretrieve(a_url, a_zip)
        with zipfile.ZipFile(a_zip) as zf:
            zf.extractall(vqav2_dir)
        a_zip.unlink()

    img_url = "http://images.cocodataset.org/zips/val2014.zip"
    img_zip = vqav2_dir / "val2014.zip"
    if not any(val_images_dir.glob("*.jpg")):
        print("Downloading val2014 images...")
        urllib.request.urlretrieve(img_url, img_zip)
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(vqav2_dir)
        img_zip.unlink()

    with open(vqav2_dir / "v2_OpenEnded_mscoco_val2014_questions.json") as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}

    with open(vqav2_dir / "v2_mscoco_val2014_annotations.json") as f:
        annotations = json.load(f)["annotations"]

    samples = []
    for ann in annotations:
        qid = ann["question_id"]
        samples.append({
            "question_id": qid,
            "image_id": ann["image_id"],
            "question": questions[qid]["question"],
            "answers": [a["answer"] for a in ann["answers"]],
        })

    print(f"VQAv2 val: {len(samples)} samples loaded.")
    return samples, val_images_dir


def run_vqav2_val_inference( vqav2_samples, val_images_dir: Path, frozen_vit, projection_head_b, frozen_qwen, clip_image_processor, qwen_tokenizer, eval_config, results_path: Path, ):

    results_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(frozen_qwen.parameters()).device

    predictions = []
    correct = 0

    for sample in tqdm(vqav2_samples, desc="VQAv2 val inference"):
        img_file = val_images_dir / f"COCO_val2014_{sample['image_id']:012d}.jpg"
        image = Image.open(img_file).convert("RGB")

        predicted_answer = run_vqa_inference(
            pil_image=image,
            question_text=sample["question"],
            format_prompt=eval_config.vqa_format_prompt,
            frozen_vit=frozen_vit,
            projection_head_b=projection_head_b,
            frozen_qwen=frozen_qwen,
            clip_image_processor=clip_image_processor,
            qwen_tokenizer=qwen_tokenizer,
            eval_config=eval_config,
        )

        soft_score = vqa_soft_score(predicted_answer, sample["answers"])
        correct += soft_score

        predictions.append({
            "question_id": sample["question_id"],
            "image_id": sample["image_id"],
            "question": sample["question"],
            "predicted_answer": predicted_answer,
            "ground_truth": sample["answers"],
            "soft_score": soft_score,
        })

    accuracy = correct / len(predictions) * 100 if predictions else 0.0
    output = {
        "split": "val",
        "method": "concat576_coco_gt_boxes",
        "coco_instances_path": str(coco_instances_path),
        "max_boxes_per_image": max_boxes_per_image,
        "num_questions": len(predictions),
        "n": len(predictions),
        "accuracy": accuracy,
        "predictions": predictions,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"VQAv2 concat576 val accuracy: {accuracy:.2f}%  (n={len(predictions)})")
    print(f"Saved: {results_path}")
    return output


def download_vqav2_testdev(vqav2_dir: Path):
    """Download test2015 images and return test-dev questions (legacy subset)."""
    vqav2_dir = Path(vqav2_dir)
    ensure_vqav2_test_question_jsons(vqav2_dir)
    test_images_dir = _ensure_vqav2_test2015_images(vqav2_dir)
    samples = _load_vqav2_questions_from_json(
        vqav2_dir / "v2_OpenEnded_mscoco_test-dev2015_questions.json"
    )
    print(f"VQAv2 test-dev: {len(samples):,} questions loaded.")
    return samples, test_images_dir


VQAV2_TEST_QUESTIONS_URL = (
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
)


def ensure_vqav2_test_question_jsons(vqav2_dir: Path) -> tuple[Path, Path]:
    """Ensure both test-dev and full test2015 question JSONs exist."""
    vqav2_dir = Path(vqav2_dir)
    vqav2_dir.mkdir(parents=True, exist_ok=True)
    dev_path = vqav2_dir / "v2_OpenEnded_mscoco_test-dev2015_questions.json"
    std_path = vqav2_dir / "v2_OpenEnded_mscoco_test2015_questions.json"
    if dev_path.exists() and std_path.exists():
        return dev_path, std_path

    q_url = VQAV2_TEST_QUESTIONS_URL
    q_zip = vqav2_dir / "v2_Questions_Test_mscoco.zip"
    print("Downloading VQAv2 test question JSONs (test-dev + test-standard)...")
    urllib.request.urlretrieve(q_url, q_zip)
    with zipfile.ZipFile(q_zip) as zf:
        zf.extractall(vqav2_dir)
    q_zip.unlink(missing_ok=True)
    if not dev_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            "Expected VQAv2 test question JSONs under "
            f"{vqav2_dir} after extracting v2_Questions_Test_mscoco.zip"
        )
    return dev_path, std_path


def _ensure_vqav2_test2015_images(vqav2_dir: Path) -> Path:
    vqav2_dir = Path(vqav2_dir)
    test_images_dir = vqav2_dir / "test2015"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    n_jpg = sum(1 for _ in test_images_dir.glob("*.jpg"))
    if n_jpg:
        return test_images_dir

    alt_dir = DEFAULT_VQAV2_ROOT / "test2015"
    alt_n = sum(1 for _ in alt_dir.glob("*.jpg")) if alt_dir.is_dir() else 0
    if alt_n:
        raise FileNotFoundError(
            f"No COCO test2015 images under {test_images_dir}, but {alt_n:,} JPGs "
            f"already exist at {alt_dir}. Point VQAV2_ROOT to {DEFAULT_VQAV2_ROOT} "
            f"(or symlink: ln -s {alt_dir} {test_images_dir}). "
            "Do not re-download on each GPU — all shards share the same image folder."
        )

    img_url = "http://images.cocodataset.org/zips/test2015.zip"
    img_zip = vqav2_dir / "test2015.zip"
    need_gb = 13
    try:
        free_gb = os.statvfs(vqav2_dir).f_bavail * os.statvfs(vqav2_dir).f_frsize / 1e9
    except OSError:
        free_gb = None
    if free_gb is not None and free_gb < need_gb:
        raise OSError(
            f"Need ~{need_gb} GB free under {vqav2_dir} to download COCO test2015 "
            f"(zip + extract); only {free_gb:.1f} GB available. "
            f"Use an existing copy (e.g. {DEFAULT_VQAV2_ROOT}) or free disk space."
        )

    print(f"Downloading test2015 images -> {img_zip} (~6 GB zip, ~12 GB total)...")
    try:
        urllib.request.urlretrieve(img_url, img_zip)
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(vqav2_dir)
    except OSError as exc:
        img_zip.unlink(missing_ok=True)
        raise OSError(
            f"Failed while downloading/extracting test2015 under {vqav2_dir}: {exc}. "
            "If disk is full, delete any partial test2015.zip and use "
            f"{DEFAULT_VQAV2_ROOT} if images are already there."
        ) from exc
    finally:
        img_zip.unlink(missing_ok=True)

    if not any(test_images_dir.glob("*.jpg")):
        raise FileNotFoundError(
            f"Download finished but no JPGs found under {test_images_dir}."
        )
    return test_images_dir


def _load_vqav2_questions_from_json(question_json: Path) -> List[dict]:
    with open(question_json) as f:
        questions = json.load(f)["questions"]
    return [
        {
            "question_id": int(q["question_id"]),
            "image_id": int(q["image_id"]),
            "question": q["question"],
        }
        for q in questions
    ]


def load_vqav2_test_samples( vqav2_dir: Path, *, for_evalai: bool = True, ) -> tuple[List[dict], Path]:
    """
    Load VQAv2 test samples.

    EvalAI test-dev phase now requires predictions on the **full** test2015
    question set (``v2_OpenEnded_mscoco_test2015_questions.json``), not the
    smaller test-dev subset alone.
    """
    vqav2_dir = Path(vqav2_dir)
    dev_path, std_path = ensure_vqav2_test_question_jsons(vqav2_dir)
    test_images_dir = _ensure_vqav2_test2015_images(vqav2_dir)

    if for_evalai:
        samples = _load_vqav2_questions_from_json(std_path)
        label = "EvalAI full test2015"
    else:
        samples = _load_vqav2_questions_from_json(dev_path)
        label = "test-dev subset"

    print(f"VQAv2 {label}: {len(samples):,} questions loaded.")
    return samples, test_images_dir


def _normalize_vqav2_submission_answer(answer: str) -> str:
    text = str(answer).strip().split("\n")[0].strip().rstrip(".")
    return text if text else "unknown"


def export_vqav2_evalai_submission( vqav2_root: Path, output_dir: Path, *, for_evalai: bool = True, predictions_jsonl: Path = None, submission_path: Path = None, ) -> Path:
    """Build an EvalAI-compatible JSON from JSONL predictions."""
    output_dir = Path(output_dir)
    predictions_jsonl = (
        Path(predictions_jsonl)
        if predictions_jsonl is not None
        else output_dir / "vqav2_testdev_predictions.jsonl"
    )
    submission_path = (
        Path(submission_path)
        if submission_path is not None
        else output_dir / "vqav2_testdev_submission.json"
    )
    samples, _ = load_vqav2_test_samples(vqav2_root, for_evalai=for_evalai)
    required_qids = {int(s["question_id"]) for s in samples}
    by_qid = {
        int(row["question_id"]): _normalize_vqav2_submission_answer(row["answer"])
        for row in _read_jsonl(predictions_jsonl)
        if "question_id" in row and "answer" in row
    }
    missing = sorted(required_qids - set(by_qid.keys()))
    extra = sorted(set(by_qid.keys()) - required_qids)
    if missing:
        raise ValueError(
            f"Submission incomplete: missing {len(missing):,} / {len(required_qids):,} "
            f"required question_ids. First missing: {missing[:5]}. "
            "Re-run evaluate_vqav2_testdev_full(for_evalai=True, resume=True) to fill gaps."
        )
    if extra:
        print(
            f"Note: ignoring {len(extra):,} predictions for question_ids outside the "
            f"required EvalAI set."
        )
    submission = [
        {"question_id": qid, "answer": by_qid[qid]}
        for qid in sorted(required_qids)
    ]
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with open(submission_path, "w") as f:
        json.dump(submission, f)
    print(
        f"EvalAI submission saved -> {submission_path} "
        f"({len(submission):,} answers)"
    )
    return submission_path


def run_vqav2_testdev_inference( vqav2_samples, test_images_dir : Path, frozen_vit, projection_head_b, frozen_qwen, clip_image_processor, qwen_tokenizer, eval_config, submission_path: Path, ):
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(frozen_qwen.parameters()).device

    predictions = []

    for sample in tqdm(vqav2_samples, desc="VQAv2 test-dev inference"):
        img_file = test_images_dir / f"COCO_test2015_{sample['image_id']:012d}.jpg"
        image = Image.open(img_file).convert("RGB")

        predicted_answer = run_vqa_inference(
            pil_image=image,
            question_text=sample["question"],
            format_prompt=eval_config.vqa_format_prompt,
            frozen_vit=frozen_vit,
            projection_head_b=projection_head_b,
            frozen_qwen=frozen_qwen,
            clip_image_processor=clip_image_processor,
            qwen_tokenizer=qwen_tokenizer,
            eval_config=eval_config,
        )

        predictions.append({
            "question_id": sample["question_id"],
            "answer": predicted_answer,
        })

    with open(submission_path, "w") as f:
        json.dump(predictions, f)
    print(f"Test-dev submission saved -> {submission_path} ({len(predictions)} answers)")

    return predictions


def _vqav2_image_path(split: str, images_dir: Path, image_id: int) -> Path:
    if split == "val":
        return images_dir / f"COCO_val2014_{image_id:012d}.jpg"
    if split == "test-dev":
        return images_dir / f"COCO_test2015_{image_id:012d}.jpg"
    raise ValueError(f"split must be 'val' or 'test-dev', got {split!r}")


def download_coco_val2014_instances(coco_dir: Path) -> Path:
    """
    Download COCO instances_val2014.json (same images as VQAv2 val).

    Uses annotations_trainval2014.zip if the JSON is not already present.
    """
    coco_dir = Path(coco_dir)
    ann_dir = coco_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ann_dir / "instances_val2014.json"

    if ann_path.exists():
        return ann_path

    zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    zip_path = coco_dir / "annotations_trainval2014.zip"
    print("Downloading COCO annotations_trainval2014.zip (~241 MB)...")
    urllib.request.urlretrieve(zip_url, zip_path)
    print(f"Extracting instances_val2014.json...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2014.json", coco_dir)
    extracted = coco_dir / "annotations" / "instances_val2014.json"
    if extracted != ann_path and extracted.exists():
        extracted.rename(ann_path)
    if not ann_path.exists():
        raise FileNotFoundError(f"Expected {ann_path} after extracting COCO annotations zip")
    return ann_path


def download_coco_val2014_images(coco_dir: Path) -> Path:
    """Download and extract COCO val2014 images (~6.2 GB). POPE uses this split."""
    coco_dir = Path(coco_dir)
    val_images_dir = coco_dir / "val2014"
    val_images_dir.mkdir(parents=True, exist_ok=True)

    if not any(val_images_dir.glob("*.jpg")):
        img_url = "http://images.cocodataset.org/zips/val2014.zip"
        img_zip = coco_dir / "val2014.zip"
        print(f"Downloading COCO val2014 images to {val_images_dir} (~6.2 GB)...")
        urllib.request.urlretrieve(img_url, img_zip)
        print("Extracting val2014.zip...")
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(coco_dir)
        img_zip.unlink(missing_ok=True)

    n_images = sum(1 for _ in val_images_dir.glob("*.jpg"))
    print(f"COCO val2014 ready: {n_images:,} images at {val_images_dir}")
    return val_images_dir


def ensure_coco_val2014_images(coco_val2014_dir: Path) -> Path:
    """Ensure POPE/VQAv2-val COCO images exist under ``.../coco/val2014``."""
    coco_val2014_dir = Path(coco_val2014_dir)
    if any(coco_val2014_dir.glob("*.jpg")):
        return coco_val2014_dir

    # Common layout if val2014 was downloaded for VQAv2 val diagnostics.
    alt = coco_val2014_dir.parent.parent / "vqav2" / "val2014"
    if alt.is_dir() and any(alt.glob("*.jpg")):
        print(f"Using existing COCO val2014 images at {alt}")
        return alt

    return download_coco_val2014_images(coco_val2014_dir.parent)


def load_coco_val2014_boxes_for_vqav2( coco_instances_path: Path, *, min_box_size: float = 10.0, max_boxes_per_image: int = None, ) -> dict:
    """
    Load COCO val2014 GT boxes keyed by image_id for VQAv2 val.

    VQAv2 val images are COCO val2014 — this is the official object-level GT.
    Boxes are [x1, y1, x2, y2] in original image pixels (not question-specific).

    Returns:
      {
        image_id: {
          "boxes_px": [[x1, y1, x2, y2], ...],
          "width": int,
          "height": int,
        },
        ...
      }
    """
    coco_instances_path = Path(coco_instances_path)
    if not coco_instances_path.exists():
        raise FileNotFoundError(
            f"COCO instances not found: {coco_instances_path}. "
            "Call download_coco_val2014_instances() first."
        )

    with open(coco_instances_path) as f:
        coco_data = json.load(f)

    image_sizes = {
        img["id"]: (img["width"], img["height"])
        for img in coco_data["images"]
    }
    boxes_by_image = defaultdict(list)

    for ann in coco_data["annotations"]:
        x, y, w, h = ann["bbox"]
        if w < min_box_size or h < min_box_size:
            continue
        boxes_by_image[ann["image_id"]].append([x, y, x + w, y + h])

    out = {}
    for image_id, (img_w, img_h) in image_sizes.items():
        boxes = boxes_by_image.get(image_id, [])
        if max_boxes_per_image is not None and len(boxes) > max_boxes_per_image:
            boxes = sorted(
                boxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                reverse=True,
            )[:max_boxes_per_image]
        if not boxes:
            boxes = [[0.0, 0.0, float(img_w), float(img_h)]]
        out[image_id] = {
            "boxes_px": boxes,
            "width": img_w,
            "height": img_h,
        }

    print(
        f"COCO val2014 boxes loaded: {len(out):,} images, "
        f"{sum(len(v['boxes_px']) for v in out.values()):,} boxes total"
    )
    return out


def run_vqav2_combined_val_evaluation( vqav2_dir: Path, frozen_vit, projection_head_b, region_extractor, frozen_qwen, clip_image_processor, qwen_tokenizer, config, eval_config=None, results_path: Path = None, coco_instances_path: Path = None, num_samples: int = None, max_boxes_per_image: int = None, verbose: bool = False, ):
    """
    VQAv2 **val** eval: concat576 with COCO val2014 GT boxes
    (same images as VQAv2 val).

    No Grounding DINO. All questions on an image share that image's COCO
    instance boxes (object-level GT, not question-specific).

    test-dev is not supported here — COCO test2015 instance GT is not public.
    Use run_vqav2_testdev_inference (global-only) for test-dev submission.
    """
    if eval_config is None:
        eval_config = EvalConfig()

    vqav2_dir = Path(vqav2_dir)
    samples, val_images_dir = download_vqav2_val(vqav2_dir)

    if num_samples is None:
        num_samples = eval_config.vqav2_num_samples
    if num_samples is not None:
        samples = samples[:num_samples]

    if coco_instances_path is None:
        coco_dir = vqav2_dir.parent / "coco"
        coco_instances_path = download_coco_val2014_instances(coco_dir)
    else:
        coco_instances_path = Path(coco_instances_path)

    if max_boxes_per_image is None:
        max_boxes_per_image = getattr(config, "max_boxes_per_image", 20)

    box_lookup = load_coco_val2014_boxes_for_vqav2(
        coco_instances_path,
        max_boxes_per_image=max_boxes_per_image,
    )

    results_path = (
        Path(results_path)
        if results_path is not None
        else eval_config.results_dir / "vqav2_concat576_coco_gt_val.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    frozen_vit.eval()
    projection_head_b.eval()
    region_extractor.eval()
    frozen_qwen.eval()

    predictions = []
    correct = 0.0

    for sample in tqdm(samples, desc="VQAv2 val concat576 (COCO GT boxes)"):
        image_id = sample["image_id"]
        if image_id not in box_lookup:
            raise KeyError(f"No COCO boxes for image_id={image_id}")

        img_path = _vqav2_image_path("val", val_images_dir, image_id)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing VQAv2 val image: {img_path}")

        manual_boxes = box_lookup[image_id]["boxes_px"]

        result = run_combined_inference(
            str(img_path),
            sample["question"],
            frozen_vit=frozen_vit,
            projection_head_b=projection_head_b,
            region_extractor=region_extractor,
            frozen_qwen=frozen_qwen,
            qwen_tokenizer=qwen_tokenizer,
            clip_image_processor=clip_image_processor,
            config=config,
            manual_boxes=manual_boxes,
            grounding_dino=None,
            format_prompt=eval_config.vqa_format_prompt,
            show_boxes=False,
            max_new_tokens=eval_config.max_new_tokens,
            verbose=verbose,
        )

        predicted_answer = result["answer"]
        soft_score = vqa_soft_score(predicted_answer, sample["answers"])
        correct += soft_score
        predictions.append({
            "question_id": sample["question_id"],
            "image_id": image_id,
            "question": sample["question"],
            "predicted_answer": predicted_answer,
            "ground_truth": sample["answers"],
            "soft_score": soft_score,
            "num_regions": len(manual_boxes),
        })

    accuracy = correct / len(predictions) * 100 if predictions else 0.0
    output = {
        "split": "val",
        "method": "concat576_coco_gt_boxes",
        "coco_instances_path": str(coco_instances_path),
        "max_boxes_per_image": max_boxes_per_image,
        "num_questions": len(predictions),
        "n": len(predictions),
        "accuracy": accuracy,
        "predictions": predictions,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"VQAv2 concat576 val accuracy: {accuracy:.2f}%  (n={len(predictions)})")
    print(f"Saved: {results_path}")
    return output


# ---------------------------------------------------------------------------
# Stage 3 concat576 full-pipeline evaluation
# ---------------------------------------------------------------------------

SHORT_FORMAT = "Answer the question using a single word or phrase."
YES_NO_FORMAT = "Answer yes or no only."
MCQ_FORMAT = "Answer with the best option letter (A, B, C, or D) only."


@dataclass
class Stage1EvalStack:
    """ViT + Projection-B + global576 LoRA (no regions, RAM, or DINO)."""

    config: Stage3Config
    frozen_vit: Any
    projection_head_b: Any
    qwen: Any
    tokenizer: Any
    image_processor: Any


@dataclass
class Stage3EvalStack:
    config: Stage3Config
    frozen_vit: Any
    projection_head_b: Any
    region_extractor: Any
    qwen: Any
    tokenizer: Any
    image_processor: Any
    grounding_dino: Any
    ram_proposer: Any = None


EvalStack = Union[Stage1EvalStack, Stage3EvalStack]


def load_stage1_eval_stack( *, projection_b_path: Path, lora_path: Path, device: Optional[str] = None, compute_dtype: Optional[torch.dtype] = None, ) -> Stage1EvalStack:
    """Load ViT, Projection-B, and a global576-only LoRA adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .config import configure_hf_cache
    from .models import load_frozen_vit, load_projection_b_weights

    projection_b_path = Path(projection_b_path)
    lora_path = Path(lora_path)
    for path in (projection_b_path, lora_path):
        if not path.exists():
            raise FileNotFoundError(path)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
    config = Stage3Config(
        projection_b_path=str(projection_b_path),
        device=resolved_device,
        compute_dtype=compute_dtype,
    )
    cache_root = configure_hf_cache(config.cache_dir)
    cache_dir = str(cache_root)
    print(f"HF cache: {cache_root} (hub: {cache_root / 'hub'})")

    frozen_vit, image_processor = load_frozen_vit(config)
    projection_head_b = load_projection_b_weights(str(projection_b_path), config)

    print(f"Loading tokenizer + base LM: {config.language_model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.language_model_hf_id,
        use_fast=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_qwen = AutoModelForCausalLM.from_pretrained(
        config.language_model_hf_id,
        torch_dtype=config.compute_dtype,
        device_map={"": config.device},
        cache_dir=cache_dir,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter: {lora_path}")
    qwen = PeftModel.from_pretrained(base_qwen, str(lora_path), is_trainable=False)
    qwen.config.use_cache = True
    qwen.eval()

    frozen_vit.eval()
    projection_head_b.eval()
    return Stage1EvalStack(
        config,
        frozen_vit,
        projection_head_b,
        qwen,
        tokenizer,
        image_processor,
    )


def load_stage3_eval_stack( *, projection_b_path: Path, projection_a_path: Path, lora_path: Path, device: Optional[str] = None, compute_dtype: Optional[torch.dtype] = None, max_boxes: int = 20, box_source: str = "hybrid", load_ram: bool = True, ) -> Stage3EvalStack:
    """Load the frozen visual stack and exactly one concat576 LoRA adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .config import configure_hf_cache
    from .models import (
        build_region_feature_extractor,
        load_frozen_vit,
        load_grounding_dino,
        load_projection_b_weights,
        load_ram as load_ram_model,
        load_region_feature_extractor,
    )

    projection_b_path = Path(projection_b_path)
    projection_a_path = Path(projection_a_path)
    lora_path = Path(lora_path)
    for path in (projection_b_path, projection_a_path, lora_path):
        if not path.exists():
            raise FileNotFoundError(path)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
    config = Stage3Config(
        projection_b_path=str(projection_b_path),
        projection_a_weights_path=projection_a_path.name,
        checkpoint_dir=projection_a_path.parent,
        max_boxes_per_image=max_boxes,
        dino_max_boxes=max_boxes,
        box_source=box_source,
        device=resolved_device,
        compute_dtype=compute_dtype,
    )
    cache_root = configure_hf_cache(config.cache_dir)
    cache_dir = str(cache_root)
    print(f"HF cache: {cache_root} (hub: {cache_root / 'hub'})")

    frozen_vit, image_processor = load_frozen_vit(config)
    projection_head_b = load_projection_b_weights(str(projection_b_path), config)
    region_extractor = build_region_feature_extractor(config)
    load_region_feature_extractor(region_extractor, str(projection_a_path), config)

    print(f"Loading tokenizer + base LM: {config.language_model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.language_model_hf_id,
        use_fast=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_qwen = AutoModelForCausalLM.from_pretrained(
        config.language_model_hf_id,
        torch_dtype=config.compute_dtype,
        device_map={"": config.device},
        cache_dir=cache_dir,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter: {lora_path}")
    qwen = PeftModel.from_pretrained(base_qwen, str(lora_path), is_trainable=False)
    qwen.config.use_cache = True
    qwen.eval()

    grounding_dino = load_grounding_dino(config)
    ram_proposer = load_ram_model(config) if load_ram else None
    if box_source in {"ram", "hybrid"} and ram_proposer is None:
        raise ValueError(f"box_source={box_source!r} requires RAM++")
    if not load_ram and box_source == "question":
        print("RAM++ disabled: using spaCy question nouns only")

    frozen_vit.eval()
    projection_head_b.eval()
    region_extractor.eval()
    return Stage3EvalStack(
        config,
        frozen_vit,
        projection_head_b,
        region_extractor,
        qwen,
        tokenizer,
        image_processor,
        grounding_dino,
        ram_proposer,
    )


def _stage1_predict( stack: Stage1EvalStack, image_path: Path, question: str, *, format_prompt: str, max_new_tokens: int, ) -> dict:
    from .inference import run_concat576_inference

    pred, aux = run_concat576_inference(
        str(image_path),
        question,
        frozen_vit=stack.frozen_vit,
        projection_head_b=stack.projection_head_b,
        region_extractor=None,
        qwen=stack.qwen,
        qwen_tokenizer=stack.tokenizer,
        clip_image_processor=stack.image_processor,
        config=stack.config,
        boxes_px=[],
        format_prompt=format_prompt,
        max_new_tokens=max_new_tokens,
        include_regions=False,
    )
    answer = str(pred).strip().splitlines()[0].strip().rstrip(".")
    return {
        "answer": answer,
        "num_regions": 0,
        "num_global_tokens": aux.get("num_global_tokens", 576),
        "num_region_tokens": 0,
    }


def _stage3_predict( stack: Stage3EvalStack, image_path: Path, question: str, *, format_prompt: str, max_new_tokens: int, ) -> dict:
    result = run_combined_inference(
        str(image_path),
        question,
        frozen_vit=stack.frozen_vit,
        projection_head_b=stack.projection_head_b,
        region_extractor=stack.region_extractor,
        frozen_qwen=stack.qwen,
        qwen_tokenizer=stack.tokenizer,
        clip_image_processor=stack.image_processor,
        config=stack.config,
        grounding_dino=stack.grounding_dino,
        ram_proposer=stack.ram_proposer,
        box_source=stack.config.box_source,
        format_prompt=format_prompt,
        max_new_tokens=max_new_tokens,
        show_boxes=False,
        verbose=False,
    )
    result["answer"] = str(result["answer"]).strip().splitlines()[0].strip().rstrip(".")
    return result


def _benchmark_predict( stack: Stage1EvalStack | Stage3EvalStack, image_path: Path, question: str, *, format_prompt: str, max_new_tokens: int, ) -> dict:
    if isinstance(stack, Stage1EvalStack):
        return _stage1_predict(
            stack,
            image_path,
            question,
            format_prompt=format_prompt,
            max_new_tokens=max_new_tokens,
        )
    return _stage3_predict(
        stack,
        image_path,
        question,
        format_prompt=format_prompt,
        max_new_tokens=max_new_tokens,
    )


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_records(path: Path, key: str, resume: bool) -> set:
    if not resume and path.exists():
        path.unlink()
    return {str(row[key]) for row in _read_jsonl(path) if key in row}


def _eval_rows(rows: Sequence[dict], limit: Optional[int]) -> Sequence[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    return rows if limit is None else rows[:limit]


def _save_eval_summary(output_dir: Path, name: str, summary: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(path)
    return summary


def evaluate_gqa_testdev_full( stack: EvalStack, *, gqa_root: Path, output_dir: Path, limit: Optional[int] = None, resume: bool = True, ) -> dict:
    """Score GQA balanced test-dev using the automatic full pipeline."""
    samples, images_dir = download_gqa_testdev(Path(gqa_root))
    samples = _eval_rows(samples, limit)
    path = Path(output_dir) / "gqa_testdev_predictions.jsonl"
    done = _prepare_records(path, "question_id", resume)
    for sample in tqdm(samples, desc="GQA test-dev full pipeline"):
        qid = str(sample["question_id"])
        if qid in done:
            continue
        result = _benchmark_predict(
            stack,
            images_dir / f"{sample['image_id']}.jpg",
            sample["question"],
            format_prompt=SHORT_FORMAT,
            max_new_tokens=16,
        )
        target = str(sample["answer"])
        _append_jsonl(path, {
            "question_id": qid,
            "image_id": str(sample["image_id"]),
            "type": sample.get("type", "unknown"),
            "prediction": result["answer"],
            "answer": target,
            "correct": normalise_answer(result["answer"]) == normalise_answer(target),
            "num_regions": result.get("num_regions", 0),
        })
    allowed = {str(s["question_id"]) for s in samples}
    records = [r for r in _read_jsonl(path) if str(r["question_id"]) in allowed]
    by_type = defaultdict(list)
    for row in records:
        by_type[row["type"]].append(row["correct"])
    return _save_eval_summary(Path(output_dir), "gqa_testdev", {
        "benchmark": "gqa_testdev",
        "accuracy_pct": 100 * sum(r["correct"] for r in records) / max(1, len(records)),
        "n": len(records),
        "by_type_pct": {
            key: 100 * sum(values) / len(values)
            for key, values in sorted(by_type.items())
        },
        "predictions_path": str(path),
    })


def evaluate_vqav2_testdev_full( stack: EvalStack, *, vqav2_root: Path, output_dir: Path, limit: Optional[int] = None, resume: bool = True, for_evalai: bool = True, ) -> dict:
    """Generate a VQAv2 EvalAI submission JSON.

    ``for_evalai=True`` (default) predicts on the full test2015 question file
    required by the current EvalAI test-dev phase (~447k questions).
    """
    samples, images_dir = load_vqav2_test_samples(
        Path(vqav2_root), for_evalai=for_evalai
    )
    samples = _eval_rows(samples, limit)
    path = Path(output_dir) / "vqav2_testdev_predictions.jsonl"
    done = _prepare_records(path, "question_id", resume)

    for sample in tqdm(samples, desc="VQAv2 test full pipeline"):
        qid = str(sample["question_id"])
        if qid in done:
            continue
        image_path = images_dir / f"COCO_test2015_{int(sample['image_id']):012d}.jpg"
        result = _benchmark_predict(
            stack,
            image_path,
            sample["question"],
            format_prompt=SHORT_FORMAT,
            max_new_tokens=16,
        )
        _append_jsonl(path, {
            "question_id": int(sample["question_id"]),
            "image_id": int(sample["image_id"]),
            "answer": _normalize_vqav2_submission_answer(result["answer"]),
            "num_regions": result.get("num_regions", 0),
        })

    submission_path = export_vqav2_evalai_submission(
        vqav2_root,
        output_dir,
        for_evalai=for_evalai,
        predictions_jsonl=path,
    )
    submission = json.loads(submission_path.read_text())
    return _save_eval_summary(Path(output_dir), "vqav2_testdev", {
        "benchmark": "vqav2_testdev",
        "score": None,
        "n": len(submission),
        "for_evalai_full_test": for_evalai,
        "submission_path": str(submission_path),
        "predictions_path": str(path),
        "note": (
            "Upload submission_path to EvalAI VQAv2 test-dev phase. "
            "EvalAI requires the full test2015 question set, not test-dev subset only."
        ),
    })


def _yes_no(text: str) -> Optional[str]:
    match = re.search(r"\b(yes|no)\b", normalise_answer(text))
    return match.group(1) if match else None


def _pope_metrics(rows: Sequence[dict]) -> dict:
    tp = sum(r["label"] == "yes" and r["prediction"] == "yes" for r in rows)
    tn = sum(r["label"] == "no" and r["prediction"] == "no" for r in rows)
    fp = sum(r["label"] == "no" and r["prediction"] == "yes" for r in rows)
    fn = sum(r["label"] == "yes" and r["prediction"] == "no" for r in rows)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "n": len(rows),
        "accuracy_pct": 100 * (tp + tn) / max(1, len(rows)),
        "precision_pct": 100 * precision,
        "recall_pct": 100 * recall,
        "f1_pct": 100 * f1,
        "invalid": sum(r["prediction"] not in {"yes", "no"} for r in rows),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate_pope_full( stack: EvalStack, *, pope_root: Path, coco_val2014_dir: Path, output_dir: Path, limit: Optional[int] = None, resume: bool = True, ) -> dict:
    """Score POPE random, popular, and adversarial splits."""
    from .stage3_dataset import download_pope_jsons

    coco_val2014_dir = ensure_coco_val2014_images(Path(coco_val2014_dir))
    paths = download_pope_jsons(Path(pope_root))
    split_metrics = {}
    total = 0
    for source in paths:
        split = next((name for name in ("random", "popular", "adversarial") if name in source.name), source.stem)
        samples = _eval_rows(_read_jsonl(source), limit)
        path = Path(output_dir) / f"pope_{split}_predictions.jsonl"
        done = _prepare_records(path, "question_id", resume)
        for i, sample in enumerate(tqdm(samples, desc=f"POPE {split} full pipeline")):
            qid = str(sample.get("question_id", f"{split}-{i}"))
            if qid in done:
                continue
            result = _benchmark_predict(
                stack,
                Path(coco_val2014_dir) / sample["image"],
                sample.get("text", sample.get("question", "")),
                format_prompt=YES_NO_FORMAT,
                max_new_tokens=4,
            )
            _append_jsonl(path, {
                "question_id": qid,
                "prediction": _yes_no(result["answer"]),
                "raw_prediction": result["answer"],
                "label": str(sample["label"]).lower(),
                "num_regions": result.get("num_regions", 0),
            })
        allowed = {
            str(s.get("question_id", f"{split}-{i}")) for i, s in enumerate(samples)
        }
        records = [r for r in _read_jsonl(path) if r["question_id"] in allowed]
        split_metrics[split] = _pope_metrics(records)
        split_metrics[split]["predictions_path"] = str(path)
        total += len(records)
    return _save_eval_summary(Path(output_dir), "pope", {
        "benchmark": "pope",
        "macro_f1_pct": sum(m["f1_pct"] for m in split_metrics.values()) / len(split_metrics),
        "macro_accuracy_pct": sum(m["accuracy_pct"] for m in split_metrics.values()) / len(split_metrics),
        "splits": split_metrics,
        "n": total,
    })


def _mmbench_key(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _mcq_prompt(row: dict, *, seed: bool = False) -> tuple[str, Dict[str, str]]:
    if seed:
        options = {
            "A": str(row["choice_a"]), "B": str(row["choice_b"]),
            "C": str(row["choice_c"]), "D": str(row["choice_d"]),
        }
        parts = [str(row["question"])]
    else:
        options = {
            letter: str(row[letter]) for letter in ("A", "B", "C", "D")
            if str(row.get(letter, "")).strip().lower() not in {"", "nan", "none"}
        }
        hint = str(row.get("hint", ""))
        parts = ([] if hint.lower() in {"", "nan", "none"} else [hint]) + [str(row["question"])]
    parts.append("\n".join(f"{letter}. {value}" for letter, value in options.items()))
    return "\n".join(parts), options


def _option_letter(text: str, options: Dict[str, str]) -> Optional[str]:
    match = re.search(r"(?:^|\b|\()([A-D])(?:\b|\)|[\.:])", str(text).upper())
    if match and match.group(1) in options:
        return match.group(1)
    normalized = normalise_answer(text)
    exact = [
        letter for letter, value in options.items()
        if normalise_answer(value) == normalized
    ]
    return exact[0] if len(exact) == 1 else None


def evaluate_mmbench_full( stack: EvalStack, *, mmbench_root: Path, output_dir: Path, split: str = "MMBench_DEV_EN", limit: Optional[int] = None, resume: bool = True, ) -> dict:
    """Run MMBench EN dev and export predictions for canonical VLMEvalKit scoring."""
    import pandas as pd
    from .benchmark_data import (
        build_mmbench_image_map,
        decode_base64_image_to_jpg,
        download_mmbench_tsv,
        resolve_mmbench_b64,
    )

    tsv_path = download_mmbench_tsv(Path(mmbench_root), [split])[split]
    df = pd.read_csv(tsv_path, sep="\t")
    rows = list(_eval_rows(df.to_dict(orient="records"), limit))
    image_map = build_mmbench_image_map(df)
    image_dir = Path(mmbench_root) / "images" / split
    path = Path(output_dir) / f"{split.lower()}_predictions.jsonl"
    done = _prepare_records(path, "index", resume)
    for row in tqdm(rows, desc=f"{split} full pipeline"):
        index = _mmbench_key(row["index"])
        if index in done:
            continue
        image_path = image_dir / f"{index}.jpg"
        if not image_path.exists():
            encoded = resolve_mmbench_b64(image_map, index)
            if not encoded:
                raise ValueError(f"Missing MMBench image for index={index}")
            decode_base64_image_to_jpg(encoded, image_path)
        question, options = _mcq_prompt(row)
        result = _benchmark_predict(
            stack, image_path, question, format_prompt=MCQ_FORMAT, max_new_tokens=4
        )
        prediction = _option_letter(result["answer"], options)
        answer = str(row.get("answer", "")).strip().upper()
        _append_jsonl(path, {
            "index": index,
            "prediction": prediction,
            "raw_prediction": result["answer"],
            "answer": answer,
            "correct": prediction == answer if answer in options else None,
            "num_regions": result.get("num_regions", 0),
        })
    allowed = {_mmbench_key(row["index"]) for row in rows}
    records = [r for r in _read_jsonl(path) if r["index"] in allowed]
    scored = [r for r in records if r["correct"] is not None]
    predictions = {r["index"]: r["prediction"] or "" for r in records}
    export = df.copy()
    export["prediction"] = [
        predictions.get(_mmbench_key(index), "") for index in export["index"]
    ]
    export_path = Path(output_dir) / f"{split}_with_predictions.tsv"
    export.to_csv(export_path, sep="\t", index=False)
    return _save_eval_summary(Path(output_dir), split.lower(), {
        "benchmark": split,
        "local_accuracy_pct": 100 * sum(r["correct"] for r in scored) / max(1, len(scored)),
        "reportable_as_official_mmbench": False,
        "n": len(records),
        "invalid": sum(r["prediction"] is None for r in records),
        "vlmevalkit_input_path": str(export_path),
        "note": "Use VLMEvalKit output for the reportable MMBench score.",
    })


def evaluate_seed_image_full( stack: EvalStack, *, seed_root: Path, output_dir: Path, limit: Optional[int] = None, resume: bool = True, ) -> dict:
    """Score SEED-Bench v1 image questions (SEED-Image).

    Keeps rows with data_type == "image" and question_type_id in {1..9}.
    Do not report this as SEED All (image+video).
    """
    from .benchmark_data import (
        download_seed_bench_images,
        download_seed_bench_json,
        load_seed_bench_questions,
        resolve_seed_image_path,
    )

    seed_root = Path(seed_root)
    json_path = download_seed_bench_json(seed_root, str(stack.config.cache_dir))
    images_root = download_seed_bench_images(seed_root, str(stack.config.cache_dir))
    samples = []
    for row in load_seed_bench_questions(json_path):
        if row.get("data_type") != "image":
            continue
        try:
            qtype = int(row.get("question_type_id", -1))
        except (TypeError, ValueError):
            continue
        if qtype < 1 or qtype > 9:
            continue
        samples.append(row)
    samples = list(_eval_rows(samples, limit))
    path = Path(output_dir) / "seed_image_predictions.jsonl"
    done = _prepare_records(path, "question_id", resume)
    for row in tqdm(samples, desc="SEED-Bench image full pipeline"):
        qid = str(row["question_id"])
        if qid in done:
            continue
        image_path = resolve_seed_image_path(images_root, row["data_id"])
        if image_path is None:
            raise FileNotFoundError(f"SEED image missing: {row['data_id']}")
        question, options = _mcq_prompt(row, seed=True)
        result = _benchmark_predict(
            stack, image_path, question, format_prompt=MCQ_FORMAT, max_new_tokens=4
        )
        prediction = _option_letter(result["answer"], options)
        answer = str(row["answer"]).strip().upper()
        _append_jsonl(path, {
            "question_id": qid,
            "question_type_id": str(row.get("question_type_id", "unknown")),
            "prediction": prediction,
            "raw_prediction": result["answer"],
            "answer": answer,
            "correct": prediction == answer,
            "num_regions": result.get("num_regions", 0),
        })
    allowed = {str(row["question_id"]) for row in samples}
    records = [r for r in _read_jsonl(path) if r["question_id"] in allowed]
    by_type = defaultdict(list)
    for row in records:
        by_type[row["question_type_id"]].append(row["correct"])
    return _save_eval_summary(Path(output_dir), "seed_image", {
        "benchmark": "seed_bench_image",
        "accuracy_pct": 100 * sum(r["correct"] for r in records) / max(1, len(records)),
        "reportable_as_seed_all": False,
        "n": len(records),
        "invalid": sum(r["prediction"] is None for r in records),
        "by_question_type_pct": {
            key: 100 * sum(values) / len(values)
            for key, values in sorted(by_type.items())
        },
        "filter": "data_type==image and question_type_id in 1..9",
        "note": "SEED-Image only (dims 1–9); do not label as SEED All.",
    })
