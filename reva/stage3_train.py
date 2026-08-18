"""
Stage 3 concat576 + LoRA — simple visual prefix:

  [576 Projection-B globals] + [K × 16 Projection-A region tokens] + format / Q / A

No QGCRF, no global-slot compression, no overview text.
Projection-A/B and ViT stay frozen; only LoRA is trainable.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .inference import _embed_text, manual_boxes_to_336
from .stage3_dataset import DEFAULT_FORMAT_PROMPT, STAGE3_MAX_BOXES_PER_IMAGE, STAGE3_TARGET_MIX

logger = logging.getLogger(__name__)


def load_stage3_clean_pool(pkl_path: Path, *, max_samples: Optional[int] = None) -> List[dict]:
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        pool = pickle.load(f)
    if max_samples is not None and max_samples < len(pool):
        rng = random.Random(42)
        pool = rng.sample(pool, max_samples)
    print(f"Loaded Stage 3 clean pool: {len(pool):,} rows from {pkl_path}")
    print(f"  by source: {dict(Counter(s['source'] for s in pool))}")
    return pool


def scale_boxes_to_336(boxes: List[List[float]], img_w: int, img_h: int) -> torch.Tensor:
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return manual_boxes_to_336(boxes, img_w, img_h, device="cpu").cpu()


def build_stage3_weighted_sampler(samples: List[dict]) -> WeightedRandomSampler:
    weights = [float(STAGE3_TARGET_MIX.get(s["source"], 0.1)) for s in samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(samples), replacement=True)


def _max_answer_tokens_for_sample(sample: dict, config) -> int:
    if sample.get("source") == "aokvqa":
        return int(getattr(config, "max_aokvqa_answer_tokens", 8))
    return int(getattr(config, "max_answer_tokens", 32))


def extract_region_tokens_by_box( vit_out, boxes_336, region_extractor, max_boxes: int, tokens_per_box: int = 16, ):
    device = boxes_336.device if boxes_336.numel() else "cpu"
    if boxes_336.numel() == 0:
        return torch.zeros(1, 0, tokens_per_box, 3584, device=device)
    boxes_336 = boxes_336[:max_boxes]
    chunks = []
    with torch.no_grad():
        for i in range(boxes_336.shape[0]):
            chunks.append(region_extractor(vit_out.hidden_states, boxes_336[i : i + 1]))
    return torch.stack([c.squeeze(0) for c in chunks], dim=0).unsqueeze(0)


def build_stage3_training_inputs( *, vit_out, boxes_336: torch.Tensor, question: str, answer: str, format_prompt: str, projection_head_b, region_extractor, qwen, qwen_tokenizer, config, max_answer_tokens: Optional[int] = None, include_regions: bool = True, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Prefix: [576 globals] + [K*16 regions] + format/Q/A + answer.

    Set include_regions=False for the global576 ablation (0 region tokens).
    """
    device = config.device
    max_answer_tokens = max_answer_tokens or config.max_answer_tokens
    max_boxes = int(getattr(config, "max_boxes_per_image", STAGE3_MAX_BOXES_PER_IMAGE))
    tokens_per_box = int(getattr(config, "tokens_per_box", 16))
    compute_dtype = config.compute_dtype

    assembled_embeds: List[torch.Tensor] = []
    assembled_attn: List[torch.Tensor] = []
    prefix_len = 0

    def append_prefix(embeds, mask):
        nonlocal prefix_len
        assembled_embeds.append(embeds)
        assembled_attn.append(mask)
        prefix_len += embeds.shape[1]

    with torch.no_grad():
        global_tokens = projection_head_b(vit_out.last_hidden_state[:, 1:, :]).to(
            device=device, dtype=compute_dtype
        )
    append_prefix(
        global_tokens,
        torch.ones(1, global_tokens.shape[1], dtype=torch.long, device=device),
    )

    region_flat = None
    num_boxes = 0
    if include_regions and boxes_336.numel():
        boxes_336 = boxes_336.to(device=device, dtype=torch.float32)
        region_by_box = extract_region_tokens_by_box(
            vit_out,
            boxes_336,
            region_extractor,
            max_boxes,
            tokens_per_box=tokens_per_box,
        ).to(device=device, dtype=compute_dtype)
        if region_by_box.shape[1] > 0:
            region_flat = region_by_box.reshape(1, -1, region_by_box.shape[-1])
            num_boxes = int(region_by_box.shape[1])
            append_prefix(
                region_flat,
                torch.ones(1, region_flat.shape[1], dtype=torch.long, device=device),
            )

    footer = f"\n{format_prompt}\n\nQuestion:\n{question}\n\nAnswer:\n"
    footer_embeds, footer_mask = _embed_text(footer, qwen_tokenizer, qwen, device)
    append_prefix(footer_embeds, footer_mask)

    answer_str = str(answer).strip()
    eos = qwen_tokenizer.eos_token or ""
    if eos and not answer_str.endswith(eos):
        answer_str += eos
    answer_tok = qwen_tokenizer(
        answer_str,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_answer_tokens,
    )
    answer_ids = answer_tok["input_ids"].to(device)
    answer_embeds = qwen.get_input_embeddings()(answer_ids)
    answer_mask = torch.ones(1, answer_embeds.shape[1], dtype=torch.long, device=device)
    assembled_embeds.append(answer_embeds)
    assembled_attn.append(answer_mask)

    input_embeds = torch.cat(assembled_embeds, dim=1)
    attention_mask = torch.cat(assembled_attn, dim=1)
    labels = torch.full((1, input_embeds.shape[1]), -100, dtype=torch.long, device=device)
    labels[0, prefix_len : prefix_len + answer_ids.shape[1]] = answer_ids[0]

    aux = {
        "num_global_tokens": int(global_tokens.shape[1]),
        "num_region_tokens": 0 if region_flat is None else int(region_flat.shape[1]),
        "num_boxes": num_boxes,
    }
    return input_embeds, attention_mask, labels, aux


def stage3_forward_pass( batch: dict, *, frozen_vit, projection_head_b, region_extractor=None, qwen, qwen_tokenizer, config, include_regions: bool = True, ) -> torch.Tensor:
    if include_regions and region_extractor is None:
        raise ValueError("region_extractor is required when include_regions=True")

    pixel_values = batch["pixel_values"].to(config.device, dtype=config.compute_dtype)
    losses = []

    for i in range(pixel_values.shape[0]):
        with torch.no_grad():
            vit_out = frozen_vit(
                pixel_values=pixel_values[i : i + 1],
                output_hidden_states=True,
            )

        input_embeds, attention_mask, labels, _ = build_stage3_training_inputs(
            vit_out=vit_out,
            boxes_336=batch["boxes_336"][i],
            question=batch["question"][i],
            answer=batch["answer"][i],
            format_prompt=batch["format_prompt"][i],
            projection_head_b=projection_head_b,
            region_extractor=region_extractor,
            qwen=qwen,
            qwen_tokenizer=qwen_tokenizer,
            config=config,
            max_answer_tokens=_max_answer_tokens_for_sample(
                {"source": batch["source"][i]},
                config,
            ),
            include_regions=include_regions,
        )
        outputs = qwen(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        if outputs.loss is not None:
            losses.append(outputs.loss)

    if not losses:
        return torch.tensor(0.0, device=config.device, requires_grad=True)
    return torch.stack(losses).mean()


class Stage3VQADataset(Dataset):
    def __init__( self, samples: List[dict], image_processor, *, max_boxes_per_image: int = STAGE3_MAX_BOXES_PER_IMAGE, ):
        self.samples = samples
        self.image_processor = image_processor
        self.max_boxes_per_image = max_boxes_per_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        try:
            pil_image = Image.open(sample["image_path"]).convert("RGB")
            img_w, img_h = pil_image.size
            boxes = list(sample.get("boxes") or [])[: self.max_boxes_per_image]
            boxes_336 = scale_boxes_to_336(boxes, img_w, img_h)
            pixel_values = self.image_processor(
                images=pil_image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)
            return {
                "pixel_values": pixel_values,
                "boxes_336": boxes_336,
                "question": sample["question"],
                "answer": sample["answer"],
                "format_prompt": sample.get("format_prompt") or DEFAULT_FORMAT_PROMPT,
                "source": sample["source"],
                "question_id": sample["question_id"],
                "image_path": sample["image_path"],
            }
        except Exception as exc:
            logger.warning("Stage3 sample %s failed: %s", idx, exc)
            return self.__getitem__((idx + 1) % len(self.samples))


def collate_stage3_batch(batch: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "boxes_336": [b["boxes_336"] for b in batch],
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "format_prompt": [b["format_prompt"] for b in batch],
        "source": [b["source"] for b in batch],
        "question_id": [b["question_id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def resolve_stage3_weight_paths(project_dir: Path, config) -> Tuple[Path, Path]:
    project_dir = Path(project_dir)
    proj_b = Path(config.projection_b_path)
    if not proj_b.is_file():
        candidate = project_dir / proj_b.name
        if candidate.is_file():
            proj_b = candidate
    proj_a = Path(config.projection_a_weights_path)
    if not proj_a.is_file():
        candidate = project_dir / proj_a.name
        if candidate.is_file():
            proj_a = candidate
    return proj_b, proj_a


def write_stage3_training_manifest( config, *, pool_size: int, include_regions: bool = True, extra: Optional[dict] = None, ) -> Path:
    tokens_per_box = int(getattr(config, "tokens_per_box", 16))
    max_boxes = int(getattr(config, "max_boxes_per_image", STAGE3_MAX_BOXES_PER_IMAGE))
    num_region_tokens = max_boxes * tokens_per_box if include_regions else 0
    manifest = {
        "pool_size": pool_size,
        "target_mix": STAGE3_TARGET_MIX,
        "fusion_mode": "concat576_simple" if include_regions else "global576_only",
        "num_global_tokens": 576,
        "tokens_per_box": tokens_per_box,
        "max_boxes_per_image": max_boxes if include_regions else 0,
        "max_visual_prefix_tokens": 576 + num_region_tokens,
        "lora_lr": config.peak_learning_rate,
        "max_optimizer_steps": config.max_optimizer_steps,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "clean_pkl_path": str(config.clean_pkl_path),
        "lora_output_dir": str(config.lora_output_dir),
    }
    if extra:
        manifest.update(extra)
    manifest_name = (
        "stage3_training_manifest.json"
        if include_regions
        else "stage1_global_lora_training_manifest.json"
    )
    path = Path(config.lora_output_dir) / manifest_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
