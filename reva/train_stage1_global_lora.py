from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import DEFAULT_CHECKPOINT_ROOT, DEFAULT_DECONTAMINATION_ROOT, Stage3Config
from .models import load_frozen_vit, load_projection_b_weights, load_qwen_with_lora
from .stage3_train import load_stage3_clean_pool
from .training import train_stage3_lora

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a global576-only LoRA on the exact Stage 3 fine-tuning pool.")
    parser.add_argument("--clean-pkl", type=Path, default=DEFAULT_DECONTAMINATION_ROOT / "stage3_eval_clean.pkl")
    parser.add_argument("--projection-b", type=Path, default=DEFAULT_CHECKPOINT_ROOT / "projection_b_best_weights.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_CHECKPOINT_ROOT}/global576_stage1_<timestamp>",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Use the same optimizer-step budget as the completed Stage 3 run.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_CHECKPOINT_ROOT / f"global576_stage1_{timestamp}"

    if not args.clean_pkl.is_file():
        raise FileNotFoundError(f"Clean Stage 3 pool not found: {args.clean_pkl}")
    if not args.projection_b.is_file():
        raise FileNotFoundError(f"Projection-B weights not found: {args.projection_b}")

    config = Stage3Config(
        clean_pkl_path=args.clean_pkl,
        projection_b_path=str(args.projection_b),
        lora_output_dir=output_dir,
        training_log_path=output_dir / "training_log.json",
        max_optimizer_steps=args.steps,
        max_train_samples=args.max_train_samples,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        peak_learning_rate=args.learning_rate,
    )

    print("Matched baseline configuration")
    print(f" data: {config.clean_pkl_path}")
    print(f" Projection-B: {config.projection_b_path}")
    print(f" output: {config.lora_output_dir}")
    print(f" steps: {config.max_optimizer_steps:,}")
    print("visual input: 576 global tokens only")

    train_samples = load_stage3_clean_pool(
        config.clean_pkl_path,
        max_samples=config.max_train_samples,
    )
    frozen_vit, clip_image_processor = load_frozen_vit(config)
    projection_head_b = load_projection_b_weights(config.projection_b_path, config)
    qwen, qwen_tokenizer = load_qwen_with_lora(config)

    train_stage3_lora(
        qwen=qwen,
        qwen_tokenizer=qwen_tokenizer,
        frozen_vit=frozen_vit,
        projection_head_b=projection_head_b,
        region_extractor=None,
        train_samples=train_samples,
        clip_image_processor=clip_image_processor,
        config=config,
        include_regions=False,
    )

    print(f"Best global-only adapter: {output_dir / config.lora_best_subdir}")

if __name__ == "__main__":
    main()
