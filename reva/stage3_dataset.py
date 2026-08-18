"""
Stage 3 fine-tuning pool builders and hard-ID exclusion sets.

Each exported sample:
  {
    "image_path": str,
    "boxes": [[x1, y1, x2, y2], ...],   # pixel coords, capped per image
    "question": str,
    "answer": str,
    "source": "gqa" | "visual7w" | "vqav2" | "vcr" | "aokvqa",
    "question_id": str,
    "image_id": str,
    "format_prompt": str,
  }
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Lazy-import dataset helpers inside functions that need them (avoids faiss at import time).

AOKVQA_HF_REPO = "HuggingFaceM4/A-OKVQA"

STAGE3_TARGET_MIX = {
    "visual7w": 0.35,
    "gqa": 0.30,
    "vqav2": 0.15,
    "vcr": 0.10,
    "aokvqa": 0.10,
}

STAGE3_MAX_BOXES_PER_IMAGE = 20

DEFAULT_FORMAT_PROMPT = "Answer the question using a single word or phrase."
MCQ_FORMAT_PROMPT = "Answer with the best option letter (A, B, C, or D) only."

POPE_JSON_URLS = {
    "random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    "popular": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
}


def _cap_boxes(boxes: List[List[float]], max_boxes: int) -> List[List[float]]:
    if max_boxes is None or len(boxes) <= max_boxes:
        return boxes
    return sorted(
        boxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True,
    )[:max_boxes]


def _full_frame_box(width: float, height: float) -> List[List[float]]:
    return [[0.0, 0.0, float(width), float(height)]]


def load_coco_instance_boxes( instances_json: Path, *, min_box_size: float = 10.0, max_boxes_per_image: int = STAGE3_MAX_BOXES_PER_IMAGE, ) -> Dict[int, dict]:
    """COCO instances keyed by image_id -> boxes_px, width, height."""
    instances_json = Path(instances_json)
    if not instances_json.exists():
        print(f"COCO instances not found: {instances_json}")
        return {}

    with open(instances_json) as f:
        coco_data = json.load(f)

    image_sizes = {
        img["id"]: (img["width"], img["height"], img["file_name"])
        for img in coco_data["images"]
    }
    boxes_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        x, y, w, h = ann["bbox"]
        if w < min_box_size or h < min_box_size:
            continue
        boxes_by_image[ann["image_id"]].append([x, y, x + w, y + h])

    out = {}
    for image_id, (img_w, img_h, file_name) in image_sizes.items():
        boxes = _cap_boxes(boxes_by_image.get(image_id, []), max_boxes_per_image)
        if not boxes:
            boxes = _full_frame_box(img_w, img_h)
        out[image_id] = {
            "boxes_px": boxes,
            "width": img_w,
            "height": img_h,
            "file_name": file_name,
        }
    print(
        f"COCO boxes from {instances_json.name}: {len(out):,} images, "
        f"{sum(len(v['boxes_px']) for v in out.values()):,} boxes"
    )
    return out


def _make_sample( *, image_path: Path, boxes: List[List[float]], question: str, answer: str, source: str, question_id: str, image_id: str, format_prompt: str = DEFAULT_FORMAT_PROMPT, ) -> dict:
    return {
        "image_path": str(image_path),
        "boxes": boxes,
        "question": question.strip(),
        "answer": answer.strip(),
        "source": source,
        "question_id": str(question_id),
        "image_id": str(image_id),
        "format_prompt": format_prompt,
    }


def _resolve_gqa_questions_dir(gqa_root: Path) -> Optional[Path]:
    gqa_root = Path(gqa_root)
    candidates = [
        gqa_root,  # evaluation.py layout: JSONs at repo root
        gqa_root / "questions1.2",
        gqa_root / "questions",
        gqa_root / "gqa" / "questions1.2",
        gqa_root / "gqa" / "questions",
    ]
    for path in candidates:
        if not path.is_dir():
            continue
        if (path / "train_balanced_questions.json").exists():
            return path
        if any(path.glob("*_questions.json")):
            return path
    return None


def _resolve_gqa_scene_dir(gqa_root: Path) -> Path:
    gqa_root = Path(gqa_root)
    for candidate in (
        gqa_root / "sceneGraphs",
        gqa_root / "gqa" / "sceneGraphs",
    ):
        if candidate.is_dir():
            return candidate
    return gqa_root / "sceneGraphs"


def load_gqa_exclusion_question_ids(gqa_root: Path) -> Set[str]:
    """Question ids from GQA eval splits that must never appear in training."""
    gqa_root = Path(gqa_root)
    questions_dir = _resolve_gqa_questions_dir(gqa_root)

    excluded = set()
    if questions_dir is None:
        print(
            f"WARNING: GQA questions directory not found under {gqa_root}. "
            "Expected questions1.2/ or questions/. Exclusion set will be empty."
        )
        print(f"GQA hard-excluded question ids: {len(excluded):,}")
        return excluded

    found_files = []
    for fname in (
        "testdev_balanced_questions.json",
        "testdev_all_questions.json",
        "test_balanced_questions.json",
        "test_all_questions.json",
        "challenge_balanced_questions.json",
        "challenge_all_questions.json",
        "val_balanced_questions.json",
        "val_all_questions.json",
    ):
        path = questions_dir / fname
        if not path.exists():
            continue
        found_files.append(fname)
        with open(path) as f:
            data = json.load(f)
        excluded.update(str(qid) for qid in data.keys())

    if not found_files:
        print(
            f"WARNING: GQA questions dir exists ({questions_dir}) but no eval JSON files found. "
            "Download GQA questions1.2 bundle from stanfordvlg.github.io/GQA/download/"
        )
    else:
        print(f"GQA eval JSON files loaded from {questions_dir}: {found_files}")
    print(f"GQA hard-excluded question ids: {len(excluded):,}")
    return excluded


def load_gqa_scene_graphs(scene_graphs_json: Path) -> dict:
    if not scene_graphs_json.exists():
        return {}
    with open(scene_graphs_json) as f:
        return json.load(f)


def load_gqa_stage3_samples( gqa_root: Path, *, max_boxes_per_image: int = STAGE3_MAX_BOXES_PER_IMAGE, max_samples: Optional[int] = None, ) -> List[dict]:
    gqa_root = Path(gqa_root)
    images_dir = gqa_root / "images"
    if not images_dir.exists():
        images_dir = gqa_root / "gqa" / "images"
    questions_dir = _resolve_gqa_questions_dir(gqa_root)
    scene_dir = _resolve_gqa_scene_dir(gqa_root)

    if questions_dir is None:
        print(f"GQA train questions not found under {gqa_root}")
        return []

    train_q_path = questions_dir / "train_balanced_questions.json"
    if not train_q_path.exists():
        train_q_path = questions_dir / "train_all_questions.json"
    if not train_q_path.exists():
        print(f"GQA train questions not found under {questions_dir}")
        return []

    excluded_qids = load_gqa_exclusion_question_ids(gqa_root)
    scene_graphs = load_gqa_scene_graphs(scene_dir / "train_sceneGraphs.json")
    if not scene_graphs:
        scene_graphs = load_gqa_scene_graphs(scene_dir / "train_all_sceneGraphs.json")

    with open(train_q_path) as f:
        questions = json.load(f)

    samples = []
    for qid, q in questions.items():
        if str(qid) in excluded_qids:
            continue
        image_id = str(q["imageId"])
        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue

        sg = scene_graphs.get(image_id, {})
        img_w = float(sg.get("width", q.get("width", 0)) or 0)
        img_h = float(sg.get("height", q.get("height", 0)) or 0)
        boxes = []
        for obj in (sg.get("objects") or {}).values():
            x, y, w, h = float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
            if w <= 0 or h <= 0:
                continue
            if max(x, y, w, h) <= 1.0:
                x1, y1 = x * img_w, y * img_h
                x2, y2 = (x + w) * img_w, (y + h) * img_h
            else:
                x1, y1, x2, y2 = x, y, x + w, y + h
            if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                boxes.append([x1, y1, x2, y2])
        boxes = _cap_boxes(boxes, max_boxes_per_image)
        if not boxes and img_w > 0 and img_h > 0:
            boxes = _full_frame_box(img_w, img_h)
        if not boxes:
            continue

        samples.append(
            _make_sample(
                image_path=img_path,
                boxes=boxes,
                question=q["question"],
                answer=q["answer"],
                source="gqa",
                question_id=qid,
                image_id=image_id,
            )
        )
        if max_samples and len(samples) >= max_samples:
            break

    print(f"GQA train_balanced samples: {len(samples):,}")
    return samples


def _resolve_visual7w_dataset_json(v7w_root: Path) -> Optional[Path]:
    v7w_root = Path(v7w_root)
    for rel in (
        "visual7w_pointing/data/dataset.json",
        "dataset_v7w_pointing.json",
        "visual7w/data/dataset.json",
        "dataset.json",
        "data/dataset.json",
    ):
        path = v7w_root / rel
        if path.exists():
            return path
    return None


def _visual7w_boxes_by_id(data: dict) -> Dict[int, dict]:
    return {int(box["box_id"]): box for box in data.get("boxes", [])}


def _visual7w_box_to_xyxy(box: dict) -> List[float]:
    x, y = float(box["x"]), float(box["y"])
    w, h = float(box["width"]), float(box["height"])
    return [x, y, x + w, y + h]


def _iter_visual7w_qa_pairs(data: dict, *, split: Optional[str] = None):
    """Yield (qa, image_meta) from flat or nested Visual7W JSON layouts."""
    images = data.get("images", [])
    if images and isinstance(images[0], dict) and "qa_pairs" in images[0]:
        for img in images:
            if split is not None and img.get("split") != split:
                continue
            for qa in img.get("qa_pairs", []):
                yield qa, img
        return

    img_by_id = {img["image_id"]: img for img in images}
    if split is None or split == "train":
        for qa in data.get("train", data.get("qa_pairs", [])):
            yield qa, img_by_id.get(qa.get("image_id"), {})
    elif split in data:
        for qa in data.get(split, []):
            yield qa, img_by_id.get(qa.get("image_id"), {})


def load_visual7w_exclusion_ids(v7w_root: Path) -> Set[str]:
    v7w_root = Path(v7w_root)
    excluded = set()
    dataset_json = _resolve_visual7w_dataset_json(v7w_root)
    if dataset_json is None:
        print(
            f"WARNING: Visual7W dataset.json not found under {v7w_root}. "
            "Exclusion set will be empty."
        )
        print(f"Visual7W hard-excluded qa ids: {len(excluded):,}")
        return excluded

    with open(dataset_json) as f:
        data = json.load(f)

    for qa, _img in _iter_visual7w_qa_pairs(data, split="val"):
        qid = qa.get("qa_id", qa.get("question_id", qa.get("id")))
        if qid is not None:
            excluded.add(str(qid))
    for qa, _img in _iter_visual7w_qa_pairs(data, split="test"):
        qid = qa.get("qa_id", qa.get("question_id", qa.get("id")))
        if qid is not None:
            excluded.add(str(qid))
    for split_name in ("val", "test", "valid", "validation"):
        for qa in data.get(split_name, []):
            qid = qa.get("qa_id", qa.get("question_id", qa.get("id")))
            if qid is not None:
                excluded.add(str(qid))

    print(f"Visual7W hard-excluded qa ids: {len(excluded):,} (from {dataset_json})")
    return excluded


def load_visual7w_stage3_samples( v7w_root: Path, *, max_boxes_per_image: int = STAGE3_MAX_BOXES_PER_IMAGE, max_samples: Optional[int] = None, ) -> List[dict]:
    """Visual7W pointing split — `which` (and legacy `where`) questions."""
    v7w_root = Path(v7w_root)
    dataset_json = _resolve_visual7w_dataset_json(v7w_root)
    if dataset_json is None:
        print(f"Visual7W dataset.json not found under {v7w_root}")
        return []

    excluded_qids = load_visual7w_exclusion_ids(v7w_root)
    with open(dataset_json) as f:
        data = json.load(f)

    boxes_by_id = _visual7w_boxes_by_id(data)
    samples = []
    for qa, img_meta in _iter_visual7w_qa_pairs(data, split="train"):
        qa_type = str(qa.get("type", qa.get("qa_type", ""))).lower()
        if qa_type and qa_type not in ("which", "where"):
            continue
        qid = str(qa.get("qa_id", qa.get("question_id", qa.get("id", len(samples)))))
        if qid in excluded_qids:
            continue

        image_id = qa.get("image_id", img_meta.get("image_id"))
        rel_path = img_meta.get("filename", img_meta.get("image_path", qa.get("image_path", "")))
        if not rel_path:
            continue
        img_path = v7w_root / rel_path
        if not img_path.exists():
            img_path = dataset_json.parent / rel_path
        if not img_path.exists():
            img_path = v7w_root / "visual7w_pointing/images" / Path(rel_path).name
        if not img_path.exists():
            continue

        img_w = float(img_meta.get("width", qa.get("width", 0)) or 0)
        img_h = float(img_meta.get("height", qa.get("height", 0)) or 0)
        boxes = []
        answer_box = boxes_by_id.get(int(qa["answer"])) if qa.get("answer") is not None else None
        if answer_box is not None:
            boxes = [_visual7w_box_to_xyxy(answer_box)]
            answer = str(answer_box.get("name", qa.get("answer", "")))
        else:
            box = qa.get("box_xywh") or qa.get("bbox") or qa.get("box")
            if box and len(box) == 4:
                x, y, w, h = [float(v) for v in box]
                if max(x, y, w, h) <= 1.0 and img_w > 0 and img_h > 0:
                    boxes = [[x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h]]
                else:
                    boxes = [[x, y, x + w, y + h]]
            answer = str(qa.get("answer", qa.get("multiple_choice_answer", "")))
        if not boxes and img_w > 0 and img_h > 0:
            boxes = _full_frame_box(img_w, img_h)
        if not boxes:
            continue

        samples.append(
            _make_sample(
                image_path=img_path,
                boxes=_cap_boxes(boxes, max_boxes_per_image),
                question=qa.get("question", qa.get("qp", "")),
                answer=answer,
                source="visual7w",
                question_id=qid,
                image_id=str(image_id),
            )
        )
        if max_samples and len(samples) >= max_samples:
            break

    print(f"Visual7W pointing samples: {len(samples):,}")
    return samples


def load_vqav2_train_exclusion_ids(vqav2_root: Path) -> Tuple[Set[int], Set[int]]:
    """Return (excluded_question_ids, excluded_image_ids)."""
    vqav2_root = Path(vqav2_root)
    excluded_qids = set()
    excluded_image_ids = set()

    val_q = vqav2_root / "v2_OpenEnded_mscoco_val2014_questions.json"
    if val_q.exists():
        with open(val_q) as f:
            for q in json.load(f)["questions"]:
                excluded_qids.add(int(q["question_id"]))
                excluded_image_ids.add(int(q["image_id"]))
    else:
        print(f"WARNING: VQAv2 val questions missing: {val_q}")

    from .dataset import load_vqav2_eval_image_ids
    _, test_image_ids = load_vqav2_eval_image_ids(str(vqav2_root))
    excluded_image_ids.update(test_image_ids)
    print(
        f"VQAv2 hard-excluded question ids: {len(excluded_qids):,}; "
        f"image ids: {len(excluded_image_ids):,} "
        f"(includes {len(test_image_ids):,} test2015 ids)"
    )
    return excluded_qids, excluded_image_ids


def load_vqav2_stage3_samples( vqav2_root: Path, coco_train2014_dir: Path, coco_boxes: Dict[int, dict], *, max_samples: Optional[int] = None, ) -> List[dict]:
    vqav2_root = Path(vqav2_root)
    coco_train2014_dir = Path(coco_train2014_dir)
    train_q = vqav2_root / "v2_OpenEnded_mscoco_train2014_questions.json"
    train_a = vqav2_root / "v2_mscoco_train2014_annotations.json"
    if not train_q.exists() or not train_a.exists():
        print(f"VQAv2 train JSON not found under {vqav2_root}")
        return []

    excluded_qids, excluded_image_ids = load_vqav2_train_exclusion_ids(vqav2_root)
    with open(train_q) as f:
        questions = {int(q["question_id"]): q for q in json.load(f)["questions"]}
    with open(train_a) as f:
        annotations = json.load(f)["annotations"]

    samples = []
    for ann in annotations:
        qid = int(ann["question_id"])
        if qid in excluded_qids:
            continue
        q = questions.get(qid)
        if q is None:
            continue
        image_id = int(q["image_id"])
        if image_id in excluded_image_ids:
            continue

        coco_info = coco_boxes.get(image_id)
        if coco_info is None:
            continue
        img_path = coco_train2014_dir / coco_info["file_name"]
        if not img_path.exists():
            continue

        answers = [a["answer"] for a in ann.get("answers", []) if a.get("answer")]
        answer = Counter(answers).most_common(1)[0][0] if answers else ""
        if not answer:
            continue

        samples.append(
            _make_sample(
                image_path=img_path,
                boxes=coco_info["boxes_px"],
                question=q["question"],
                answer=answer,
                source="vqav2",
                question_id=str(qid),
                image_id=str(image_id),
            )
        )
        if max_samples and len(samples) >= max_samples:
            break

    print(f"VQAv2 train samples: {len(samples):,}")
    return samples


def _iter_vcr_rows(path: Path) -> Iterable[dict]:
    path = Path(path)
    if not path.exists():
        return
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with open(path) as f:
        data = json.load(f)
    rows = data.get("questions", data.get("data", data))
    if isinstance(rows, dict):
        rows = list(rows.values())
    yield from rows


# Same gender-neutral names as rowanz/r2c (dataloaders/vcr.py).
VCR_GENDER_NEUTRAL_NAMES = (
    "Casey",
    "Riley",
    "Jessie",
    "Jackie",
    "Avery",
    "Jaime",
    "Peyton",
    "Kerry",
    "Jody",
    "Kendall",
    "Skyler",
    "Frankie",
    "Pat",
    "Quinn",
)


def _vcr_row_id(row: dict) -> str:
    return str(row.get("annot_id", row.get("question_id", row.get("qid", row.get("id", "")))))


def _vcr_object_names(objects: List[str]) -> List[str]:
    names: List[str] = []
    for idx, obj in enumerate(objects):
        obj_type = str(obj).strip().lower()
        if obj_type == "person":
            names.append(VCR_GENDER_NEUTRAL_NAMES[idx % len(VCR_GENDER_NEUTRAL_NAMES)])
        else:
            names.append(obj_type or f"object{idx + 1}")
    return names


def _vcr_label_for_object(obj_idx: int, objects: List[str]) -> str:
    if 0 <= obj_idx < len(objects):
        return _vcr_object_names(objects)[obj_idx]
    return str(obj_idx + 1)


def _vcr_resolve_detection_text(text: str, objects: List[str]) -> str:
    """Replace 1-based detection indices in VCR text with r2c-style names/classes."""
    if not text or not objects:
        return text

    def repl(match: re.Match) -> str:
        num = int(match.group(1))
        idx = num - 1
        if 0 <= idx < len(objects):
            return _vcr_label_for_object(idx, objects)
        return match.group(0)

    return re.sub(r"\b(\d+)\b", repl, text)


def _vcr_tokens_to_text(tokens, objects: List[str]) -> str:
    """Convert tokenized VCR question/answer (with [det_idx] lists) to natural language."""
    parts: List[str] = []
    for tok in tokens or []:
        if isinstance(tok, list):
            for idx in tok:
                parts.append(_vcr_label_for_object(int(idx), objects))
        else:
            parts.append(str(tok))
    return " ".join(parts)


def _vcr_row_question(row: dict) -> str:
    objects = [str(o) for o in row.get("objects") or []]
    if row.get("question_text"):
        return _vcr_resolve_detection_text(str(row["question_text"]), objects)
    if row.get("question_orig"):
        return _vcr_resolve_detection_text(str(row["question_orig"]), objects)
    question = row.get("question")
    if isinstance(question, list):
        return _vcr_tokens_to_text(question, objects)
    if isinstance(question, str):
        return _vcr_resolve_detection_text(question, objects)
    return ""


def _vcr_row_answer(row: dict) -> str:
    objects = [str(o) for o in row.get("objects") or []]
    if row.get("answer_orig"):
        return _vcr_resolve_detection_text(str(row["answer_orig"]), objects)
    choices = row.get("answer_choice_texts") or row.get("answer_choices")
    label = row.get("answer_label")
    if choices is not None and label is not None:
        try:
            choice = choices[int(label)]
            if isinstance(choice, list):
                return _vcr_tokens_to_text(choice, objects)
            return _vcr_resolve_detection_text(str(choice), objects)
        except (IndexError, TypeError, ValueError):
            pass
    answer = row.get("answer", row.get("label", ""))
    if isinstance(answer, list):
        return _vcr_tokens_to_text(answer, objects)
    return _vcr_resolve_detection_text(str(answer), objects)


def _vcr_image_path(vcr_root: Path, img_fn: str) -> Optional[Path]:
    vcr_root = Path(vcr_root)
    img_fn = str(img_fn).strip()
    if not img_fn:
        return None
    for base in (vcr_root / "vcr1images", vcr_root / "images", vcr_root):
        candidate = base / img_fn
        if candidate.is_file():
            return candidate
    return None


def _vcr_parse_bbox(bbox) -> Optional[List[float]]:
    """Parse VCR-style bbox as [x1, y1, x2, y2]; ignore class-name strings and bad values."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        inner = bbox.get("bbox") or bbox.get("box")
        return _vcr_parse_bbox(inner) if inner is not None else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 > x1 and y2 > y1:
        return [x1, y1, x2, y2]
    return None


def _vcr_metadata_boxes(vcr_root: Path, metadata_fn: str) -> List[List[float]]:
    if not metadata_fn:
        return []
    vcr_root = Path(vcr_root)
    for base in (vcr_root / "vcr1images", vcr_root):
        meta_path = base / metadata_fn
        if not meta_path.is_file():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        boxes: List[List[float]] = []
        # Official VCR format: metadata["boxes"] = [[x1, y1, x2, y2, score], ...]
        for box in meta.get("boxes", []):
            parsed = _vcr_parse_bbox(box)
            if parsed:
                boxes.append(parsed)
        if boxes:
            return boxes
        for region in meta.get("regions", []):
            parsed = _vcr_parse_bbox(region.get("bbox"))
            if parsed:
                boxes.append(parsed)
                continue
            box = region.get("box", region.get("bbox_xywh"))
            if box and len(box) >= 4:
                try:
                    x, y, w, h = (float(v) for v in box[:4])
                except (TypeError, ValueError):
                    continue
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
        return boxes
    return []


def load_vcr_exclusion_ids(vcr_root: Path) -> Set[str]:
    vcr_root = Path(vcr_root)
    excluded = set()
    found = []
    for fname in ("val.jsonl", "test.jsonl", "vcr1val.json", "vcr1test.json", "val.json", "test.json"):
        path = vcr_root / fname
        if not path.exists():
            continue
        found.append(fname)
        for row in _iter_vcr_rows(path):
            qid = _vcr_row_id(row)
            if qid:
                excluded.add(qid)
    if not found:
        print(
            f"WARNING: VCR val/test annotations not found under {vcr_root}. "
            "Expected val.jsonl / test.jsonl (HF Rowan/vcr export). Exclusion set will be empty."
        )
    else:
        print(f"VCR eval files loaded: {found}")
    print(f"VCR hard-excluded question ids: {len(excluded):,}")
    return excluded


def load_vcr_stage3_samples( vcr_root: Path, *, max_boxes_per_image: int = STAGE3_MAX_BOXES_PER_IMAGE, max_samples: Optional[int] = None, ) -> List[dict]:
    vcr_root = Path(vcr_root)
    train_path = None
    for candidate in (
        vcr_root / "train.jsonl",
        vcr_root / "vcr1train.json",
        vcr_root / "train.json",
    ):
        if candidate.exists():
            train_path = candidate
            break
    if train_path is None:
        print(f"VCR train annotations not found under {vcr_root} (expected train.jsonl)")
        return []

    excluded_qids = load_vcr_exclusion_ids(vcr_root)
    samples = []
    for row in _iter_vcr_rows(train_path):
        qid = _vcr_row_id(row)
        if qid in excluded_qids:
            continue
        rel_img = row.get("img_fn", row.get("image_name", row.get("image_path", "")))
        img_path = _vcr_image_path(vcr_root, rel_img)
        if img_path is None:
            continue

        boxes = _vcr_metadata_boxes(vcr_root, row.get("metadata_fn", ""))
        if not boxes:
            # row["objects"] is class labels (e.g. "person"), not bboxes — only use explicit bboxes.
            for bbox in row.get("bboxes", row.get("boxes", [])):
                parsed = _vcr_parse_bbox(bbox)
                if parsed:
                    boxes.append(parsed)
        boxes = _cap_boxes(boxes, max_boxes_per_image)
        if not boxes:
            w = float(row.get("img_width", row.get("width", 0)) or 0)
            h = float(row.get("img_height", row.get("height", 0)) or 0)
            if w > 0 and h > 0:
                boxes = _full_frame_box(w, h)
        if not boxes:
            continue

        question = _vcr_row_question(row)
        answer = _vcr_row_answer(row)
        if not question or not answer:
            continue

        samples.append(
            _make_sample(
                image_path=img_path,
                boxes=boxes,
                question=question,
                answer=answer,
                source="vcr",
                question_id=qid,
                image_id=str(rel_img),
            )
        )
        if max_samples and len(samples) >= max_samples:
            break

    print(f"VCR train samples: {len(samples):,}")
    return samples


def _format_aokvqa_mcq(question: str, choices: List[str], correct_idx: int) -> Tuple[str, str]:
    letters = "ABCD"
    lines = [question.strip(), "Options:"]
    for i, choice in enumerate(choices[:4]):
        lines.append(f"{letters[i]}) {choice}")
    prompt = "\n".join(lines)
    answer = letters[int(correct_idx)] if 0 <= int(correct_idx) < 4 else str(choices[int(correct_idx)])
    return prompt, answer


def load_aokvqa_exclusion_ids(aokvqa_root: Path) -> Set[str]:
    """Exclude A-OKVQA val/test question ids."""
    aokvqa_root = Path(aokvqa_root)
    excluded = set()
    found_local = []
    for fname in ("val.json", "validation.json", "test.json"):
        path = aokvqa_root / fname
        if not path.exists():
            continue
        found_local.append(fname)
        with open(path) as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = list(rows.values())
        for row in rows:
            qid = row.get("question_id", row.get("id"))
            if qid is not None:
                excluded.add(str(qid))

    if not found_local:
        try:
            from datasets import load_dataset

            ds = load_dataset(AOKVQA_HF_REPO, split="validation")
            for row in ds:
                excluded.add(str(row["question_id"]))
            print(f"A-OKVQA val question ids loaded from HuggingFace {AOKVQA_HF_REPO}: {len(excluded):,}")
        except Exception as exc:
            print(
                f"WARNING: A-OKVQA val JSON not found under {aokvqa_root} "
                f"and HF load failed ({exc}). Exclusion set will be empty."
            )
    else:
        print(f"A-OKVQA eval JSON files loaded: {found_local}")
    print(f"A-OKVQA hard-excluded question ids: {len(excluded):,}")
    return excluded


def load_aokvqa_stage3_samples( aokvqa_root: Path, coco_train2014_dir: Path, coco_boxes: Dict[int, dict], *, max_samples: Optional[int] = None, ) -> List[dict]:
    aokvqa_root = Path(aokvqa_root)
    coco_train2014_dir = Path(coco_train2014_dir)
    train_json = aokvqa_root / "train.json"
    excluded_qids = load_aokvqa_exclusion_ids(aokvqa_root)

    rows = []
    if train_json.exists():
        with open(train_json) as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else list(data.values())
    else:
        try:
            from datasets import load_dataset

            ds = load_dataset(AOKVQA_HF_REPO, split="train")
            rows = [dict(row) for row in ds]
            print(f"A-OKVQA loaded from HuggingFace {AOKVQA_HF_REPO} train: {len(rows):,}")
        except Exception as exc:
            print(f"A-OKVQA train data not found under {aokvqa_root} and HF load failed: {exc}")
            return []

    samples = []
    for row in rows:
        qid = str(row.get("question_id", row.get("id", len(samples))))
        if qid in excluded_qids:
            continue

        image_id = int(row["image_id"])
        coco_info = coco_boxes.get(image_id)
        if coco_info is None:
            continue
        img_path = coco_train2014_dir / coco_info["file_name"]
        if not img_path.exists():
            continue

        choices = row.get("choices")
        correct_idx = row.get("correct_choice_idx")
        if choices is not None and correct_idx is not None:
            question, answer = _format_aokvqa_mcq(row["question"], choices, correct_idx)
            fmt = MCQ_FORMAT_PROMPT
        else:
            question = row["question"]
            direct = row.get("direct_answers") or row.get("answers") or []
            answer = direct[0] if direct else ""
            fmt = DEFAULT_FORMAT_PROMPT
        if not answer:
            continue

        samples.append(
            _make_sample(
                image_path=img_path,
                boxes=coco_info["boxes_px"],
                question=question,
                answer=answer,
                source="aokvqa",
                question_id=qid,
                image_id=str(image_id),
                format_prompt=fmt,
            )
        )
        if max_samples and len(samples) >= max_samples:
            break

    print(f"A-OKVQA train samples: {len(samples):,}")
    return samples


def build_stage3_raw_pool( *, gqa_root: Path, v7w_root: Path, vqav2_root: Path, vcr_root: Path, aokvqa_root: Path, coco_train2014_dir: Path, coco_train_instances: Path, max_samples_per_source: Optional[int] = None, max_boxes_per_image: int = STAGE3_MAX_BOXES_PER_IMAGE, ) -> List[dict]:
    coco_boxes = load_coco_instance_boxes(
        coco_train_instances, max_boxes_per_image=max_boxes_per_image
    )
    loaders = [
        (
            "gqa",
            lambda: load_gqa_stage3_samples(
                gqa_root,
                max_boxes_per_image=max_boxes_per_image,
                max_samples=max_samples_per_source,
            ),
        ),
        (
            "visual7w",
            lambda: load_visual7w_stage3_samples(
                v7w_root,
                max_boxes_per_image=max_boxes_per_image,
                max_samples=max_samples_per_source,
            ),
        ),
        (
            "vqav2",
            lambda: load_vqav2_stage3_samples(
                vqav2_root, coco_train2014_dir, coco_boxes, max_samples=max_samples_per_source
            ),
        ),
        (
            "vcr",
            lambda: load_vcr_stage3_samples(
                vcr_root,
                max_boxes_per_image=max_boxes_per_image,
                max_samples=max_samples_per_source,
            ),
        ),
        (
            "aokvqa",
            lambda: load_aokvqa_stage3_samples(
                aokvqa_root, coco_train2014_dir, coco_boxes, max_samples=max_samples_per_source
            ),
        ),
    ]

    pool = []
    counts = Counter()
    for source_name, loader in loaders:
        rows = loader()
        pool.extend(rows)
        counts[source_name] = len(rows)

    print("\nStage 3 raw pool by source:")
    for source_name in STAGE3_TARGET_MIX:
        print(f"  {source_name:10s}: {counts[source_name]:,}")
    print(f"  {'total':10s}: {len(pool):,}")
    return pool


def build_hard_exclusion_sets( *, gqa_root: Path, v7w_root: Path, vqav2_root: Path, vcr_root: Path, aokvqa_root: Path, ) -> dict:
    vqav2_qids, vqav2_image_ids = load_vqav2_train_exclusion_ids(vqav2_root)
    return {
        "gqa_question_ids": load_gqa_exclusion_question_ids(gqa_root),
        "visual7w_question_ids": load_visual7w_exclusion_ids(v7w_root),
        "vqav2_question_ids": vqav2_qids,
        "vqav2_image_ids": vqav2_image_ids,
        "vcr_question_ids": load_vcr_exclusion_ids(vcr_root),
        "aokvqa_question_ids": load_aokvqa_exclusion_ids(aokvqa_root),
    }


def diagnose_stage3_paths( *, gqa_root: Path, v7w_root: Path, vqav2_root: Path, vcr_root: Path, aokvqa_root: Path, coco_train2014_dir: Path, coco_train_instances: Path, ) -> dict:
    """Print which expected files exist before building the pool."""
    gqa_qdir = _resolve_gqa_questions_dir(gqa_root)
    checks = {
        "gqa_questions_dir": gqa_qdir,
        "gqa_train_balanced": (gqa_qdir / "train_balanced_questions.json") if gqa_qdir else None,
        "gqa_scene_graphs": _resolve_gqa_scene_dir(gqa_root) / "train_sceneGraphs.json",
        "gqa_images": Path(gqa_root) / "images",
        "visual7w_dataset_json": None,
        "vqav2_train_questions": Path(vqav2_root) / "v2_OpenEnded_mscoco_train2014_questions.json",
        "vqav2_train_annotations": Path(vqav2_root) / "v2_mscoco_train2014_annotations.json",
        "vqav2_val_questions": Path(vqav2_root) / "v2_OpenEnded_mscoco_val2014_questions.json",
        "vqav2_test_dev_questions": Path(vqav2_root) / "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "vqav2_test_std_questions": Path(vqav2_root) / "v2_OpenEnded_mscoco_test2015_questions.json",
        "vqav2_test2015_dir": Path(vqav2_root) / "test2015",
        "vcr_train": Path(vcr_root) / "train.jsonl",
        "vcr_val": Path(vcr_root) / "val.jsonl",
        "aokvqa_train": Path(aokvqa_root) / "train.json",
        "aokvqa_val": Path(aokvqa_root) / "val.json",
        "coco_train2014_dir": Path(coco_train2014_dir),
        "coco_train_instances": Path(coco_train_instances),
    }
    gqa_qdir = checks["gqa_questions_dir"]
    if gqa_qdir is not None and checks["gqa_train_balanced"] is None:
        checks["gqa_train_balanced"] = gqa_qdir / "train_balanced_questions.json"

    for rel in (
        "visual7w_pointing/data/dataset.json",
        "dataset_v7w_pointing.json",
        "visual7w/data/dataset.json",
        "dataset.json",
    ):
        p = Path(v7w_root) / rel
        if p.exists():
            checks["visual7w_dataset_json"] = p
            break

    report = {}
    print("\nStage 3 path diagnostics:")
    for name, path in checks.items():
        if path is None:
            report[name] = False
            print(f"  {name:28s}: MISSING")
            continue
        path = Path(path)
        ok = path.exists()
        report[name] = ok
        suffix = ""
        if ok and path.is_dir():
            n = len(list(path.glob("*.jpg")))
            suffix = f" ({n:,} JPGs)" if n else " (dir)"
        print(f"  {name:28s}: {'OK' if ok else 'MISSING':7s} {path}{suffix}")

    missing = [k for k, v in report.items() if not v]
    if missing:
        print(f"\nWARNING: {len(missing)} required paths missing — exclusion sets or loaders may be empty.")
    return report


def apply_hard_id_exclusions(samples: List[dict], exclusion_sets: dict) -> Tuple[List[dict], Counter]:
    removed = Counter()
    clean = []
    for sample in samples:
        source = sample.get("source", "")
        qid = str(sample.get("question_id", ""))
        image_id = sample.get("image_id", "")

        if source == "gqa" and qid in exclusion_sets["gqa_question_ids"]:
            removed["gqa_question"] += 1
            continue
        if source == "visual7w" and qid in exclusion_sets["visual7w_question_ids"]:
            removed["visual7w_question"] += 1
            continue
        if source == "vqav2":
            if qid.isdigit() and int(qid) in exclusion_sets["vqav2_question_ids"]:
                removed["vqav2_question"] += 1
                continue
            if image_id.isdigit() and int(image_id) in exclusion_sets["vqav2_image_ids"]:
                removed["vqav2_image"] += 1
                continue
        if source == "vcr" and qid in exclusion_sets["vcr_question_ids"]:
            removed["vcr_question"] += 1
            continue
        if source == "aokvqa" and qid in exclusion_sets["aokvqa_question_ids"]:
            removed["aokvqa_question"] += 1
            continue
        clean.append(sample)

    return clean, removed


def download_pope_jsons(decontam_dir: Path) -> List[Path]:
    import urllib.request

    decontam_dir = Path(decontam_dir)
    pope_dir = decontam_dir / "pope_repo" / "output" / "coco"
    pope_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split_name, url in POPE_JSON_URLS.items():
        dest = pope_dir / f"coco_pope_{split_name}.json"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        paths.append(dest)
    return paths


def collect_aokvqa_val_reference_paths( aokvqa_root: Path, coco_val2017_dir: Path, *, coco_train2017_dir: Optional[Path] = None, ) -> List[Path]:
    """On-disk JPG paths for A-OKVQA val images (COCO 2017 val split)."""
    aokvqa_root = Path(aokvqa_root)
    coco_val2017_dir = Path(coco_val2017_dir)
    coco_train2017_dir = Path(coco_train2017_dir) if coco_train2017_dir else None
    val_json = aokvqa_root / "val.json"
    rows = []
    if val_json.exists():
        with open(val_json) as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else list(data.values())
    else:
        try:
            from datasets import load_dataset

            ds = load_dataset(AOKVQA_HF_REPO, split="validation")
            rows = [dict(row) for row in ds]
        except Exception as exc:
            print(f"A-OKVQA val reference unavailable: {exc}")
            return []

    def _candidate_paths(image_id: int) -> List[Path]:
        stem = f"{image_id:012d}.jpg"
        prefixed = f"COCO_val2017_{stem}"
        out = []
        for root in (coco_val2017_dir, coco_train2017_dir):
            if root is None or not root.exists():
                continue
            out.extend([root / stem, root / prefixed])
        return out

    paths = []
    seen = set()
    missing = 0
    for row in rows:
        image_id = int(row["image_id"])
        found = False
        for candidate in _candidate_paths(image_id):
            key = str(candidate)
            if candidate.exists() and key not in seen:
                seen.add(key)
                paths.append(candidate)
                found = True
                break
        if not found:
            missing += 1
    print(
        f"A-OKVQA val reference images: {len(paths):,} "
        f"({len(rows):,} val rows, {missing:,} missing on disk under val2017)"
    )
    return paths


def collect_stage3_eval_reference_paths( *, data_dir: Path, vqav2_root: Path, mmbench_root: Path, seed_root: Path, gqa_root: Path, aokvqa_root: Path, decontam_dir: Path, include_seed_video_frames: bool = False, ) -> Tuple[List[Path], dict]:
    """Union of all benchmark eval images for Stage 3 pHash decontamination."""
    from .dataset import (
        collect_gqa_testdev_reference_paths,
        collect_mmbench_reference_paths,
        collect_pope_reference_paths,
        collect_seed_bench_reference_paths,
        collect_vqav2_test2015_paths,
    )

    data_dir = Path(data_dir)
    vqav2_root = Path(vqav2_root)
    mmbench_root = Path(mmbench_root)
    seed_root = Path(seed_root)
    gqa_root = Path(gqa_root)
    aokvqa_root = Path(aokvqa_root)
    decontam_dir = Path(decontam_dir)

    per_benchmark = {}
    all_paths = []

    pope_jsons = download_pope_jsons(decontam_dir)
    pope_paths = collect_pope_reference_paths(pope_jsons, data_dir / "coco" / "val2014")
    per_benchmark["pope"] = pope_paths
    all_paths.extend(pope_paths)

    vqav2_paths = collect_vqav2_test2015_paths(str(vqav2_root))
    per_benchmark["vqav2_test2015"] = vqav2_paths
    all_paths.extend(vqav2_paths)

    mmbench_paths = collect_mmbench_reference_paths(mmbench_root)
    per_benchmark["mmbench"] = mmbench_paths
    all_paths.extend(mmbench_paths)

    seed_paths, seed_stats = collect_seed_bench_reference_paths(
        seed_root,
        include_video_frames=include_seed_video_frames,
    )
    per_benchmark["seed_bench"] = seed_paths
    all_paths.extend(seed_paths)

    gqa_paths = collect_gqa_testdev_reference_paths(gqa_root)
    per_benchmark["gqa_testdev"] = gqa_paths
    all_paths.extend(gqa_paths)

    aokvqa_paths = collect_aokvqa_val_reference_paths(
        aokvqa_root,
        data_dir / "coco" / "val2017",
        coco_train2017_dir=data_dir / "coco" / "train2017",
    )
    per_benchmark["aokvqa_val"] = aokvqa_paths
    all_paths.extend(aokvqa_paths)

    unique = []
    seen = set()
    for path in all_paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    summary = {k: len(v) for k, v in per_benchmark.items()}
    summary["union_unique"] = len(unique)
    summary["seed_stats"] = seed_stats if seed_paths else {}
    print("\nStage 3 eval reference union:")
    for name, count in summary.items():
        if name != "seed_stats":
            print(f"  {name:16s}: {count:,}")
    return unique, summary


def validate_stage3_sample(sample: dict) -> List[str]:
    errors = []
    required = ("image_path", "boxes", "question", "answer", "source", "question_id", "image_id")
    for key in required:
        if key not in sample:
            errors.append(f"missing key: {key}")
    if not sample.get("question", "").strip():
        errors.append("empty question")
    if not sample.get("answer", "").strip():
        errors.append("empty answer")
    boxes = sample.get("boxes") or []
    if not boxes:
        errors.append("no boxes")
    else:
        for box in boxes:
            if len(box) != 4:
                errors.append(f"invalid box length: {box}")
    return errors


def summarize_pool(samples: Iterable[dict]) -> dict:
    samples = list(samples)
    by_source = Counter(s.get("source", "unknown") for s in samples)
    unique_images = len({s["image_path"] for s in samples})
    return {
        "total_rows": len(samples),
        "unique_images": unique_images,
        "by_source": dict(by_source),
    }


def verify_stage3_modules() -> None:
    """Lightweight self-check with synthetic samples (no dataset downloads required)."""
    from .config import DEFAULT_DATA_ROOT
    from .dataset import HAMMING_THRESH, clear_phash_hash_cache, filter_curriculum_from_log, generate_curriculum_decontamination_log
    from .config import ProjectionAConfig
    from PIL import Image
    import numpy as np
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="stage3_verify_"))
    clear_phash_hash_cache(DEFAULT_DATA_ROOT)

    rng = np.random.default_rng(0)
    img_a = tmp / "a.jpg"
    img_b = tmp / "b.jpg"
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(img_a)
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(img_b)

    sample_ok = _make_sample(
        image_path=img_a,
        boxes=[[0, 0, 10, 10]],
        question="What color?",
        answer="red",
        source="gqa",
        question_id="q1",
        image_id="1",
    )
    sample_dup = _make_sample(
        image_path=img_b,
        boxes=[[0, 0, 10, 10]],
        question="What color?",
        answer="blue",
        source="vqav2",
        question_id="q2",
        image_id="2",
    )
    assert not validate_stage3_sample(sample_ok), "valid sample flagged invalid"

    exclusion_sets = {
        "gqa_question_ids": {"bad_q"},
        "visual7w_question_ids": set(),
        "vqav2_question_ids": set(),
        "vqav2_image_ids": set(),
        "vcr_question_ids": set(),
        "aokvqa_question_ids": set(),
    }
    bad = dict(sample_ok)
    bad["question_id"] = "bad_q"
    kept, removed = apply_hard_id_exclusions([sample_ok, bad], exclusion_sets)
    assert len(kept) == 1 and removed["gqa_question"] == 1

    config = ProjectionAConfig()
    log_path = tmp / "verify_phash.json"
    generate_curriculum_decontamination_log(
        [sample_ok, sample_dup],
        [img_a],
        config,
        output_json_path=str(log_path),
        type="phash",
    )
    clean = filter_curriculum_from_log([sample_ok, sample_dup], str(log_path))
    clean_paths = {s["image_path"] for s in clean}
    assert str(img_b) in clean_paths, f"distinct image should survive pHash; got {clean_paths}"
    assert str(img_a) not in clean_paths, "reference-identical image should be removed"
    assert HAMMING_THRESH == 4

    vcr_row = {
        "objects": ["person", "person", "car"],
        "question_orig": "Why is 1 smiling at 2?",
        "answer_orig": "2 has spinach in her teeth.",
    }
    assert _vcr_row_question(vcr_row) == "Why is Casey smiling at Riley?"
    assert _vcr_row_answer(vcr_row) == "Riley has spinach in her teeth."
    vcr_tok = {
        "objects": ["person", "person", "person", "car"],
        "question": ["Does", [2], "feel", "comfortable", "?"],
        "answer_orig": "No she does not",
    }
    assert _vcr_row_question(vcr_tok) == "Does Jessie feel comfortable ?"

    print("verify_stage3_modules: all checks passed")
