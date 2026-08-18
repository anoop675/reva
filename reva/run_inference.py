from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final ReVA (Stage 3 concat576) inference on one image/question pair."
    )
    parser.add_argument("--image", required=True, help="Local image path or URL")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument(
        "--projection-b",
        type=Path,
        required=True,
        help="Path to Projection-B weights (.pt)",
    )
    parser.add_argument(
        "--projection-a",
        type=Path,
        required=True,
        help="Path to Projection-A weights (.pt)",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="Path to the final Stage 3 LoRA adapter directory",
    )
    parser.add_argument(
        "--box-source",
        choices=("hybrid", "ram", "question"),
        default="hybrid",
        help="Automatic region proposal source when manual boxes are not provided",
    )
    parser.add_argument(
        "--manual-boxes",
        default=None,
        help="Semicolon-separated boxes as x1,y1,x2,y2;x1,y1,x2,y2 in original image pixels",
    )
    parser.add_argument(
        "--grounding-prompt",
        default=None,
        help="Optional custom prompt passed to Grounding DINO instead of auto-derived text",
    )
    parser.add_argument(
        "--format-prompt",
        default=None,
        help="Optional answer-format instruction override",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of generated answer tokens",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=20,
        help="Maximum number of regions to keep",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Visualize selected boxes during inference",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to save the full inference result as JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output to stdout instead of answer-only text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from .evaluation import load_stage3_eval_stack
    from .inference import parse_manual_boxes, run_combined_inference

    manual_boxes = parse_manual_boxes(args.manual_boxes) if args.manual_boxes else None

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_boxes <= 0:
        raise ValueError("--max-boxes must be positive")

    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    load_ram = args.box_source in {"hybrid", "ram"}

    stack = load_stage3_eval_stack(
        projection_b_path=args.projection_b,
        projection_a_path=args.projection_a,
        lora_path=args.lora,
        device=resolved_device,
        max_boxes=args.max_boxes,
        box_source=args.box_source,
        load_ram=load_ram,
    )

    result = run_combined_inference(
        args.image,
        args.question,
        frozen_vit=stack.frozen_vit,
        projection_head_b=stack.projection_head_b,
        region_extractor=stack.region_extractor,
        frozen_qwen=stack.qwen,
        qwen_tokenizer=stack.tokenizer,
        clip_image_processor=stack.image_processor,
        config=stack.config,
        manual_boxes=manual_boxes,
        grounding_dino=stack.grounding_dino,
        ram_proposer=stack.ram_proposer,
        text_prompt=args.grounding_prompt,
        format_prompt=args.format_prompt,
        show_boxes=args.show_boxes,
        max_new_tokens=args.max_new_tokens,
        box_source=args.box_source,
    )

    payload = {
        "image": args.image,
        "question": args.question,
        "answer": result["answer"],
        "grounding_prompt": result.get("grounding_prompt"),
        "box_source": args.box_source,
        "boxes_px": result.get("boxes_px"),
        "labels": result.get("labels"),
        "scores": result.get("scores"),
        "num_global_tokens": result.get("num_global_tokens"),
        "num_region_tokens": result.get("num_region_tokens"),
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(payload["answer"])


if __name__ == "__main__":
    main()
