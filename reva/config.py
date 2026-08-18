import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch


def _path_from_env(var_name: str, default: Union[str, Path]) -> Path:
    value = os.environ.get(var_name)
    return Path(value).expanduser() if value else Path(default).expanduser()


DEFAULT_DATA_ROOT = _path_from_env("REVA_DATA_ROOT", Path.home() / "reva-data")
DEFAULT_HF_CACHE_ROOT = _path_from_env(
    "REVA_HF_CACHE_ROOT",
    DEFAULT_DATA_ROOT / "hf_cache",
)
DEFAULT_CHECKPOINT_ROOT = _path_from_env(
    "REVA_CHECKPOINT_ROOT",
    DEFAULT_DATA_ROOT / "checkpoints",
)
DEFAULT_EVAL_RESULTS_ROOT = _path_from_env(
    "REVA_EVAL_RESULTS_ROOT",
    DEFAULT_DATA_ROOT / "eval_results",
)
DEFAULT_REGION_DATA_ROOT = _path_from_env(
    "REVA_REGION_DATA_ROOT",
    DEFAULT_DATA_ROOT / "region_data",
)
DEFAULT_DECONTAMINATION_ROOT = _path_from_env(
    "REVA_DECONTAMINATION_ROOT",
    DEFAULT_DATA_ROOT / "decontamination",
)
DEFAULT_GROUNDING_DINO_ROOT = _path_from_env(
    "REVA_GROUNDING_DINO_ROOT",
    DEFAULT_DATA_ROOT / "groundingdino",
)
DEFAULT_VQAV2_ROOT = _path_from_env(
    "REVA_VQAV2_ROOT",
    DEFAULT_DATA_ROOT / "vqav2",
)
DEFAULT_TEST_IMAGES_ROOT = _path_from_env(
    "REVA_TEST_IMAGES_ROOT",
    DEFAULT_DATA_ROOT / "test_images",
)


def configure_hf_cache(cache_root: Optional[Union[str, Path]] = None) -> Path:
    """Point Hugging Face downloads to a configurable cache root.

    Must run before importing ``transformers`` or ``huggingface_hub`` so hub
    constants pick up the custom paths. Safe to call multiple times.
    """
    root = Path(cache_root or DEFAULT_HF_CACHE_ROOT)
    hub_cache = root / "hub"
    datasets_cache = root / "datasets"
    for directory in (root, hub_cache, datasets_cache):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    # Legacy transformers <=4.x still reads this at import time.
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    return root


# Default project cache location (used by Stage3Config.cache_dir).
configure_hf_cache(DEFAULT_HF_CACHE_ROOT)


@dataclass
class TrainingConfig:

    # Model identifiers
    #vision_encoder_hf_id: str = "openai/clip-vit-large-patch14"  # 224x224 pixel input image resolution
    vision_encoder_hf_id : str = "openai/clip-vit-large-patch14-336"  # increasing the input image resolution from 224x224 to 336x336 pixels
    #vision_encoder_hf_id: str = "facebook/dinov2-large"
    language_model_hf_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Architecture dimensions
    vit_patch_feature_dim: int = 1024 # CLIP ViT-L hidden size
    qwen_embedding_dim: int = 3584    # Qwen 2.5 7B hidden size
    # CLIP ViT-L/14 at 224px produces 16x16 = 256 patch tokens, each 1024-dim, We exclude the CLS token, so num_image_patch_tokens = 256.
    #num_image_patch_tokens: int = 256
    # CLIP ViT-L/14 at 336px produces 24x24 = 576 patch tokens, each 1024-dim, We exclude the CLS token, so num_image_patch_tokens = 576.
    num_image_patch_tokens : int = 576

    # Dataset
    dataset_hf_id: str = "liuhaotian/LLaVA-Pretrain" # LCS 558K (Llava 558K Pretrain) dataset huggingface id
    max_caption_token_len: int = 128 # capping the caption token length to 128 tokens (set to 192 for using full caption coverage during training)

    # Training hyperparameters
    per_device_batch_size: int = 16 #8
    #gradient_accumulation_steps: int = 8   
    gradient_accumulation_steps: int = 16 #8 
    # effeective batch size = 8 x 32 = 256
    num_training_epochs: int = 1
    peak_learning_rate: float = 1e-3 #2e-3 
    weight_decay: float = 0.0
    gradient_clip_max_norm: float = 1.
    warmup_ratio: float = 0.03  # 3% of steps used for LR warm-up

    compute_dtype: torch.dtype = torch.bfloat16  # adjusting floating point precision if needed

    # Checkpointing
    checkpoint_output_dir: Path = DEFAULT_CHECKPOINT_ROOT
    save_every_n_steps: int = 500

    # Local logging
    log_every_n_steps: int = 10  # log after 10 steps
    plot_output_path: Path = DEFAULT_CHECKPOINT_ROOT / "loss_curve.png"
    log_json_path: Path = DEFAULT_CHECKPOINT_ROOT / "training_log.json"

    # Early stopping
    early_stopping_patience: int = 99999   # optimiser steps without improvement. Use 99999 to disable.
    early_stopping_min_delta: float = 0.001  # minimum drop to count as improvement
    loss_ema_smoothing_factor: float = 0.98  # higher = smoother but slower to react

    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dataloader_num_workers: int = 4


@dataclass
class EvalConfig:

    # Format prompts
    vqa_format_prompt: str = "Answer the question using a single word or phrase."
    gqa_format_prompt: str = "Answer the question using a single word or phrase."

    # No. of samples
    gqa_num_samples: int = None    # full GQA val is ~132,062 samples
    vqav2_num_samples: int = None  # full VQAv2 val is ~214,354 samples

    # Generation settings
    max_new_tokens: int = 16   # short answers only (keeps inference fast)
    do_sample: bool = False    # greedy decoding for reproducibility

    # Batch size for inference
    eval_batch_size: int = 32

    # Output paths
    results_dir: Path = DEFAULT_EVAL_RESULTS_ROOT
    gqa_results_path: Path = DEFAULT_EVAL_RESULTS_ROOT / "gqa_testdev_baseline.json"
    vqav2_results_path: Path = DEFAULT_EVAL_RESULTS_ROOT / "vqav2_baseline.json"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype: torch.dtype = torch.bfloat16


@dataclass
class ProjectionAConfig:

    # Model identifiers
    vision_encoder_hf_id: str = "openai/clip-vit-large-patch14-336"
    language_model_hf_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # ViT feature dimensions
    vit_hidden_dim: int = 1024   # CLIP ViT-L hidden size at every layer
    coordconv_dim: int = 1026    # 1024 + 2 CoordConv channels (x and y grids)

    # Feature pyramid
    num_levels: int = 4
    # Layers 14, 17, 20, 23 following GPT4RoI (Zhang et al., 2023) i.e. second-to-last, fifth-to-last, eighth-to-last, eleventh-to-last transformer blocks.
    # These span a semantic spectrum from local texture (layer 14) to rich object-level semantics (layer 23, penultimate), giving region features
    # access to both fine-grained and abstract visual information.
    # Note: layer 24 (last_hidden_state) is used by Projection-B for global tokens.
    level_indices: List[int] = field(default_factory=lambda: [14, 17, 20, 23])

    # Spatial dimensions
    # 336px / 14px patch size = 24 patches per side -> 24×24 = 576 patch tokens
    num_spatial_patches: int = 576
    spatial_grid_size: int = 24

    # RoI Align at 14×14 following GPT4RoI (Zhang et al., 2023). Each output cell maps to ~1.71×1.71 input cells (24/14), a modest downsampling that preserves
    # spatial detail within the detected region. Not using 7×7 as it would map to ~3.43×3.43 input cells (too aggressive for fine-grained attribute encoding).
    # Gives attention pooling 196 positions to select from per region.
    roi_output_size: int = 14

    # spatial_scale maps box pixel coordinates to feature map coordinates.
    # Feature map is 24×24, image is 336×336, so scale = 24/336 = 0.0714.
    # If this is wrong, RoI Align crops the wrong locations — silent failure.
    spatial_scale: float = 24 / 336

    # num_attn_heads was 4 — changed to 6 because coordconv_dim=1026 must be divisible
    # by num_heads for nn.MultiheadAttention. 1026 / 4 = 256.5 (invalid).
    # 1026 / 6 = 171 (valid). 6 is the closest valid value to the original 4.
    num_attn_heads: int = 6
    fused_dim: int = 4104        # 4 levels × 1026 = 4104 after concatenation
    qwen_embedding_dim: int = 3584   # Qwen 2.5 7B hidden size (target embedding space)
    attn_dropout: float = 0.1
    # num_region_tokens controls K in MultiTokenAttentionPooling.
    # Replaced the implicit K=1 of the old AttentionPooling single-query design.
    # K=4 quadruples region token information capacity vs the original single-token design,
    # making region tokens harder for Qwen to disregard against 576 global tokens.
    num_region_tokens: int = 16

    # ITC (Image-Text Contrastive) loss weight — added alongside captioning loss to directly
    # align region tokens to their description embeddings in Qwen's space via InfoNCE.
    # Addresses the weakness that captioning loss alone does not penalise the pipeline
    # for producing similar tokens for visually different regions.
    # Set to 0.0 to disable and fall back to captioning loss only.
    itc_loss_weight: float = 0.5
    # Temperature for InfoNCE contrastive loss — lower = sharper distribution,
    # higher = softer. 0.07 follows CLIP; slightly higher here since we are in
    # Qwen's embedding space rather than a dedicated contrastive space.
    itc_temperature: float = 0.07

    # Grounding DINO inference (Stage 3 region proposals)
    dino_max_boxes: int = 20  # max boxes kept after dedup; align with Stage 3 GT cap
    dino_per_tag: bool = True   # localize each tag separately (better recall than mega-prompt)
    dino_max_boxes_per_tag: int = 1
    # Box proposal mode at inference:
    #   hybrid / ram — spaCy question nouns ∪ RAM++ tags → DINO
    #   question     — spaCy question nouns only → DINO
    box_source: str = "hybrid"

    # RAM++ (Recognize Anything) — image tags unioned with spaCy question nouns
    ram_hf_repo: str = "xinyu1205/recognize-anything-plus-model"
    ram_checkpoint_name: str = "ram_plus_swin_large_14m.pth"
    ram_image_size: int = 384
    max_ram_tags: int = 20          # cap RAM tags before union with question nouns
    max_qwen_selected_tags: int = 3 # legacy; unused by current box proposal path

    curriculum_num_epochs: int = 3      # total epochs for curriculum training
    curriculum_batch_size: int = 16     # use Stage 2b batch size — VG descriptions are longer
    curriculum_gradient_accum: int = 8  # effective batch = 16 * 8 = 128
    #curriculum_peak_lr: float = 1e-4    # conservative LR
    #curriculum_peak_lr: float = 5e-5  # was 1e-4, halve for K=16 # still conservative
    curriculum_peak_lr: float = 7e-5 # showed fast convergance
    #curriculum_warmup_ratio: float = 0.03
    curriculum_warmup_ratio: float = 0.06

    # Projection-B path — used only for loading at inference, not during Stage 2 training
    projection_b_path: str = str(DEFAULT_CHECKPOINT_ROOT / "projection_b_best_weights.pt")

    max_description_tokens: int = 64  # max target length (covers all VG descriptions)
    weight_decay: float = 0.0
    gradient_clip_max_norm: float = 1.0
    ema_smoothing: float = 0.98       # EMA for loss monitoring (high alpha = slow decay)

    # Cap VG at 15 regions per image for dataset diversity without capping, images with hundreds of region annotations would dominate training.
    vg_max_annotations_per_image: int = 15

    # Paths
    cache_dir: Path = DEFAULT_HF_CACHE_ROOT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_ROOT / "projection_a"
    data_dir: Path = DEFAULT_REGION_DATA_ROOT

    grounding_dino_weights_path1: Path = DEFAULT_GROUNDING_DINO_ROOT / "GroundingDINO_SwinT_OGC.py"
    grounding_dino_weights_path2: Path = DEFAULT_GROUNDING_DINO_ROOT / "groundingdino_swint_ogc.pth"
        
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    # Post-DINO deduplication (applies to region + combined inference via propose())
    dino_nms_iou: float = 0.95              # remove only near-exact duplicates (IoU >= this); 0 = off
    dino_suppress_contained: bool = False   # optional: drop small boxes nested inside larger ones
    dino_containment_ratio: float = 0.85    # only used when dino_suppress_contained=True
    dino_containment_max_area_ratio: float = 0.50

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype: torch.dtype = torch.bfloat16


@dataclass
class Stage3Config(ProjectionAConfig):
    """Stage 3 — simple concat576 prefix + LoRA.

    Prefix: [576 Projection-B globals] + [K × 16 Projection-A regions] + text.
    Only LoRA is trained; Projection-A/B and ViT stay frozen.
    """

    clean_pkl_path: Path = DEFAULT_DECONTAMINATION_ROOT / "stage3_eval_clean.pkl"
    lora_output_dir: Path = DEFAULT_CHECKPOINT_ROOT / "concat576_stage3"
    lora_best_subdir: str = "best_lora"
    projection_a_weights_path: str = "projection_a_curriculum_best_weights.pt"

    tokens_per_box: int = 16
    max_boxes_per_image: int = 20
    max_optimizer_steps: int = 5000

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    peak_learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_clip_max_norm: float = 1.0
    max_answer_tokens: int = 32
    max_aokvqa_answer_tokens: int = 8

    log_every_n_steps: int = 1
    max_train_samples: Optional[int] = None
    dataloader_num_workers: int = 0
    training_log_path: Path = DEFAULT_CHECKPOINT_ROOT / "concat576_stage3" / "training_log.json"
