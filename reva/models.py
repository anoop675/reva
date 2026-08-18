import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModel
import torch.nn.functional as F
from groundingdino.util.inference import load_model
#import torchvision.transforms as T
#import datasets.transforms as T 
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict
import torchvision


def load_state_dict_file(weights_path, map_location):
    """Load a plain state dict from either .pt or .safetensors."""
    weights_path = Path(weights_path)
    if weights_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors weights requires safetensors: pip install safetensors"
            ) from exc
        return load_file(str(weights_path), device=str(map_location))
    return torch.load(weights_path, map_location=map_location)


def _box_areas_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def _intersection_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(box_a[0], box_b[0])
    y1 = torch.max(box_a[1], box_b[1])
    x2 = torch.min(box_a[2], box_b[2])
    y2 = torch.min(box_a[3], box_b[3])
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def dedupe_dino_proposals( boxes: torch.Tensor, scores: torch.Tensor, labels: list, *, nms_iou: float = 0.95, suppress_contained: bool = False, containment_ratio: float = 0.85, containment_max_area_ratio: float = 0.50, verbose: bool = False, ):
    """
    Remove redundant Grounding DINO boxes before the final top-k cap.

    Default behaviour: NMS only at high IoU (e.g. 0.95) to drop near-exact duplicate
    boxes. Normal overlap and nested boxes (cat on couch, part inside person) are kept.

    Optional containment filter (off by default) can additionally drop small nested parts.
    """
    if boxes.shape[0] == 0:
        return boxes, scores, labels

    if nms_iou > 0 and boxes.shape[0] > 1:
        n_before = boxes.shape[0]
        keep = torchvision.ops.nms(boxes.float(), scores.float(), nms_iou)
        keep_list = keep.tolist()
        keep_set = set(keep_list)
        if len(keep_list) < n_before and verbose:
            for idx in range(n_before):
                if idx in keep_set:
                    continue
                best_iou, best_kept = 0.0, None
                for kept in keep_set:
                    inter = _intersection_xyxy(boxes[idx], boxes[kept]).item()
                    union = (
                        _box_areas_xyxy(boxes[idx:idx + 1]).item()
                        + _box_areas_xyxy(boxes[kept:kept + 1]).item()
                        - inter
                    )
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou, best_kept = iou, kept
                keeper = labels[best_kept] if best_kept is not None else "?"
                print(
                    f"  NMS removed '{labels[idx]}' ({scores[idx]:.2f}) "
                    f"≈ duplicate of '{keeper}' (IoU={best_iou:.2f})"
                )
        boxes = boxes[keep_list]
        scores = scores[keep_list]
        labels = [labels[i] for i in keep_list]

    if suppress_contained and boxes.shape[0] > 1:
        areas = _box_areas_xyxy(boxes)
        order = scores.argsort(descending=True).tolist()
        kept_idx = []
        for idx in order:
            area_inner = areas[idx].item()
            if area_inner <= 0:
                continue
            nested = False
            nested_by = None
            for kept in kept_idx:
                inter = _intersection_xyxy(boxes[idx], boxes[kept]).item()
                area_outer = areas[kept].item()
                overlap_frac = inter / area_inner if area_inner > 0 else 0.0
                area_frac = area_inner / area_outer if area_outer > 0 else 1.0
                if (
                    overlap_frac >= containment_ratio
                    and area_inner < area_outer
                    and area_frac <= containment_max_area_ratio
                ):
                    nested = True
                    nested_by = labels[kept]
                    break
            if nested:
                if verbose:
                    print(
                        f"  Containment removed '{labels[idx]}' ({scores[idx]:.2f}) "
                        f"inside '{nested_by}'"
                    )
            else:
                kept_idx.append(idx)
        boxes = boxes[kept_idx]
        scores = scores[kept_idx]
        labels = [labels[i] for i in kept_idx]

    return boxes, scores, labels


class ProjectionHeadB(nn.Module):
    """
    The trainable bridge between the frozen ViT and frozen Qwen.

    Takes each image patch feature vector (1024-dim, CLIP space) and maps it to Qwen's word embedding space (3584-dim), so Qwen can process image patches
    and text tokens as a single unified sequence.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.first_linear = nn.Linear(input_dim, output_dim, bias=True)
        self.gelu = nn.GELU()
        self.second_linear = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, image_patch_features: torch.Tensor) -> torch.Tensor:
        projected = self.first_linear(image_patch_features)  # (batch, 256, 3584)
        projected = self.gelu(projected)
        projected = self.second_linear(projected) # (batch, 256, 3584)
        return projected


def load_frozen_vit(config):
    cache_dir = str(getattr(config, "cache_dir", None)) if getattr(config, "cache_dir", None) else None
    image_processor = CLIPImageProcessor.from_pretrained(
        config.vision_encoder_hf_id,
        cache_dir=cache_dir,
    )

    frozen_vit = CLIPVisionModel.from_pretrained(
        config.vision_encoder_hf_id,
        torch_dtype=config.compute_dtype,
        use_safetensors=False,
        cache_dir=cache_dir,
    ).to(config.device) # loads CLIP vision transformer model

    for param in frozen_vit.parameters():
        param.requires_grad = False # disabling gradient calculation (optimizer won't compute gradients for these params)
    frozen_vit.eval() # switching model to eval mode

    vit_param_count = sum(p.numel() for p in frozen_vit.parameters()) / 1e6
    print(f"CLIP ViT-L/14 loaded: {vit_param_count:.0f}M parameters (all frozen)")
    # TrainingConfig uses num_image_patch_tokens / vit_patch_feature_dim
    # ProjectionAConfig uses num_spatial_patches / vit_hidden_dim
    num_patches = getattr(config, 'num_image_patch_tokens', getattr(config, 'num_spatial_patches', '?'))
    feat_dim = getattr(config, 'vit_patch_feature_dim',  getattr(config, 'vit_hidden_dim', '?'))
    print(f"Patch token output shape: (batch, {num_patches}, {feat_dim})")

    return frozen_vit, image_processor

def load_frozen_dinov2(config):
    cache_dir = str(getattr(config, "cache_dir", None)) if getattr(config, "cache_dir", None) else None
    image_processor = AutoImageProcessor.from_pretrained(
        config.vision_encoder_hf_id,
        cache_dir=cache_dir,
    )
    
    frozen_vit = AutoModel.from_pretrained(
        config.vision_encoder_hf_id,
        torch_dtype=config.compute_dtype,
        use_safetensors=True,
        cache_dir=cache_dir,
    ).to(config.device) # loads DINOv2 vision transformer model
    
    for param in frozen_vit.parameters():
        param.requires_grad = False # disabling gradient calculation (optimizer won't compute gradients for these params)
    frozen_vit.eval() # switching model to eval mode
    
    vit_param_count = sum(p.numel() for p in frozen_vit.parameters()) / 1e6
    print(f"DINOv2 ViT-L/14 loaded: {vit_param_count:.0f}M parameters (all frozen)")
    print(f"Patch token output shape: (batch, {config.num_image_patch_tokens}, {config.vit_patch_feature_dim})")
    
    return frozen_vit, image_processor


def load_frozen_qwen(config):
    qwen_tokenizer = AutoTokenizer.from_pretrained( # loads text preprocessing tool for tokenizing text for Qwen model
        config.language_model_hf_id,
        use_fast=True,
        cache_dir = str(getattr(config, 'cache_dir', None)) if getattr(config, 'cache_dir', None) else None # load strictly in config.cache_dir
    )

    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token # assigns the End of Sequence (EOS) token to act as the padding token.
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id  # assigns the End of Sequence (EOS) token id as the padding token id.

    frozen_qwen = AutoModelForCausalLM.from_pretrained(
        config.language_model_hf_id,
        torch_dtype=config.compute_dtype,
        device_map="auto",
        use_safetensors=True,
        cache_dir = str(getattr(config, 'cache_dir', None)) if getattr(config, 'cache_dir', None) else None # load strictly in config.cache_dir
        #trust_remote_code=True,
    )

    for param in frozen_qwen.parameters():
        param.requires_grad = False # disabling gradient calculation (optimizer won't compute gradients for these params)

    frozen_qwen.gradient_checkpointing_enable() # enabling gradient checkpointing (for computing loss gradients that backpropagates through Qwen all the way to the projection head B while saving memory)
    frozen_qwen.train() # switching model to train mode for gradient checkpointing (evaluation mode assumes no backward pass will ever happen)

    qwen_param_count = sum(p.numel() for p in frozen_qwen.parameters()) / 1e9
    print(f"Qwen 2.5 7B loaded: {qwen_param_count:.1f}B parameters (all frozen)")
    print(f"Pad token: '{qwen_tokenizer.pad_token}' (id={qwen_tokenizer.pad_token_id})")
    print(f"Vocab size: {qwen_tokenizer.vocab_size:,}")

    return frozen_qwen, qwen_tokenizer


def build_projection_head(config):
    projection_head_b = ProjectionHeadB(
        input_dim=config.vit_patch_feature_dim,
        output_dim=config.qwen_embedding_dim,
    )
    projection_head_b = projection_head_b.to(config.device, dtype=config.compute_dtype)

    trainable_param_count = sum(p.numel() for p in projection_head_b.parameters()) / 1e6
    print(f"Projection-B MLP: {trainable_param_count:.1f}M trainable parameters")
    print(projection_head_b)

    return projection_head_b


def add_coordconv(feature_map):
    """
    Append normalised x and y coordinate channels to each spatial feature map. Addresses ViT translation invariance as there are no explicit position channels,
    two identical objects at different locations in the image produce near-identical region features. CoordConv breaks this by encoding absolute position.
    Following Liu et al. (2018) CoordConv and GPT4RoI (Zhang et al., 2023).
    """
    B, C, H, W = feature_map.shape   # e.g. (1, 1024, 24, 24)
    device = feature_map.device
    dtype = feature_map.dtype

    # Coordinate grids normalised to [-1, 1]
    x_lin = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    y_lin = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing='ij')  # (H, W) each

    x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # (B, 1, H, W)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # (B, 1, H, W)

    # Concatenate along channel dim: 1024 -> 1026
    return torch.cat([feature_map, x_grid, y_grid], dim=1)  # (B, C+2, H, W)


class PerLevelProjection(nn.Module):
    """
    Performs a linear projection on four independent linear projections parallely (one per ViT layer).
    Features from layers 14 and 23 live in different representational spaces because they have processed different numbers of global self-attention
    layers. Computing cross-level attention directly on raw features produces arbitrary dot products between incompatible representations.
    These per-level projections map each level into a compatible space before cross-level attention, ensuring attention weights are meaningful.
    Each level has independent weights, they are not shared.
    
Each linear projection also acts as the convolution operation required in CoordConv (Liu et al. (2018))(CoordConv expects 1x1 convolution on the coordinate-channel-concatenated feature maps, instead we concatenate the coordinate channels to the per level feature maps, reshape them, stack them together, and perform this PerLevelProjection - Linear Projection for each level (because mathematically a Linear layer functions the same as a 1x1 conv operation)
    """
    def __init__(self, dim=1026, num_levels=4):
        super().__init__()
        # Independent weights per level (must not share parameters)
        self.projections = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(num_levels)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_levels)])

    def forward(self, x):
        # x: (B, 576, 4, 1026) — batch, spatial positions, levels, dim
        B, S, L, D = x.shape
        outputs = []
        for i in range(L):
            lf = x[:, :, i, :].reshape(B * S, D) # (B*576, 1026) — flatten spatial into batch
            lf = self.projections[i](lf) # (B*576, 1026)
            lf = self.norms[i](lf) # (B*576, 1026)
            outputs.append(lf.reshape(B, S, D)) # (B, 576, 1026)
        return torch.stack(outputs, dim=2) # (B, 576, 4, 1026)


class SpatialCrossLevelAttention(nn.Module):
    """
    Bidirectional cross-level attention at every spatial position.

    At each of the 576 spatial positions in the 24x24 feature map, the four level vectors (sequence of length 4) attend to each other.
    This allows semantic context from layer 23 to inform the interpretation of local texture features from layer 14 at the same spatial position —
    and vice versa — before RoI Align crops the region.

    Key implementation detail: the 576 spatial positions are treated as independent batch elements (B*576, 4, 1026), so attention operates
    over sequences of length 4 only — fast and memory-efficient.

    For B=1 (single image), the effective shape is (576, 4, 1026).
    During training with batch size B: (B*576, 4, 1026).
    """
    def __init__(self, dim=1026, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # expects (batch, seq, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, 576, 4, 1026)
        B, S, L, D = x.shape

        # Merge batch and spatial dims — each position processed independently
        x_flat = x.reshape(B * S, L, D) # (B*576, 4, 1026)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat) # (B*576, 4, 1026)

        # Residual + LayerNorm — standard transformer block pattern
        x_flat = self.norm(x_flat + attn_out) # (B*576, 4, 1026)
        return x_flat.reshape(B, S, L, D) # (B, 576, 4, 1026)


class MultiLevelViTEncoder(nn.Module):
    """
    Shared backbone: ViT hidden states -> CoordConv -> PerLevelProjection ->
    SpatialCrossLevelAttention -> enriched spatial maps (one per level).

    Used by RegionFeatureExtractor (RoI path).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.coordconv_dim
        L = config.num_levels
        self.per_level_proj = PerLevelProjection(dim=D, num_levels=L)
        self.cross_level_attn = SpatialCrossLevelAttention(
            dim=D, num_heads=config.num_attn_heads, dropout=config.attn_dropout
        )

    def forward(self, hidden_states):
        """
        Returns list of L enriched maps, each (B, 1026, 24, 24).
        """
        D = self.config.coordconv_dim
        H = self.config.spatial_grid_size
        B = hidden_states[0].shape[0]

        spatial_maps = []
        for idx in self.config.level_indices:
            hs = hidden_states[idx][:, 1:, :]
            hs = hs.reshape(B, H, H, -1).permute(0, 3, 1, 2)
            hs = add_coordconv(hs)
            spatial_maps.append(hs)

        sequences = [m.reshape(B, D, -1).permute(0, 2, 1) for m in spatial_maps]
        stacked = torch.stack(sequences, dim=2)
        projected = self.per_level_proj(stacked)
        attended = self.cross_level_attn(projected)

        enriched_maps = []
        for i in range(self.config.num_levels):
            level_seq = attended[:, :, i, :]
            level_map = level_seq.permute(0, 2, 1).reshape(B, D, H, H)
            enriched_maps.append(level_map)
        return enriched_maps


# AttentionPooling was replaced by MultiTokenAttentionPooling below.
# Single-query pooling compressed each level's 196 RoI positions into 1 vector, creating an information bottleneck — 1 region token vs 576 global tokens meant
# Qwen could largely ignore the region token. K learnable queries instead of 1 produce K tokens per level per region, directly increasing information capacity
# and making region tokens harder for Qwen to disregard (after Stage 3 Instruction tuning).
#
# class AttentionPooling(nn.Module):
#     def __init__(self, dim=1026):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.query = nn.Parameter(torch.randn(1, dim) * 0.02)  # single learnable query
#         self.key = nn.Linear(dim, dim, bias=False)
#         self.value = nn.Linear(dim, dim, bias=False)
#
#     def forward(self, x):
#         # x: (N, 1026, 14, 14)
#         N, D, H, W = x.shape
#         x_seq = x.reshape(N, D, H * W).permute(0, 2, 1)  # (N, 196, 1026)
#         keys = self.key(x_seq)                             # (N, 196, 1026)
#         values = self.value(x_seq)                         # (N, 196, 1026)
#         query = self.query.unsqueeze(0).expand(N, 1, D)   # (N, 1, 1026)
#         scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale  # (N, 1, 196)
#         weights = scores.softmax(dim=-1)                   # (N, 1, 196)
#         pooled = torch.bmm(weights, values).squeeze(1)    # (N, 1026) — single vector per region
#         return pooled


class MultiTokenAttentionPooling(nn.Module):
    """
    Pools 196 RoI spatial positions (14x14) into K tokens instead of 1.

    Replaces AttentionPooling to resolve the single-token information bottleneck:
    1 region token competing against 576 global tokens gave Qwen insufficient signal to attend to region content. K independent learnable queries each
    attend over the 196 positions and specialise on different visual attributes (colour, texture, shape, state), collectively carrying far more information.

    With the named-tag prompt format <obj-X>, each <obj-X> placeholder expands to K token embeddings in inputs_embeds, mirroring how <image> expands to
    576 global tokens. K=4 is the default sweet spot — quadruples capacity over the single-token design at modest sequence length cost.

    Uses full Q/K/V attention following standard scaled dot-product attention.
    One MultiTokenAttentionPooling instance per feature level (four total).
    """
    def __init__(self, dim=1026, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.scale = dim ** -0.5
        
        # Orthogonal query initialisation
        random_matrix = torch.randn(dim, num_tokens)
        Q, _ = torch.linalg.qr(random_matrix)
        self.queries = nn.Parameter(Q.T)  # (K, dim)
        
        # Key projection — initialise with larger std to create diverse key vectors
        # This ensures different queries get different attention distributions
        self.key = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.key.weight, std=1.0)  # larger than default 1/sqrt(dim)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        N, D, H, W = x.shape
        x_seq = x.reshape(N, D, H * W).permute(0, 2, 1)  # (N, 196, 1026)
        
        keys   = self.key(x_seq)    # (N, 196, 1026)
        values = self.value(x_seq)  # (N, 196, 1026)
        
        queries = self.queries.unsqueeze(0).expand(N, -1, -1)  # (N, K, 1026)
        scores  = torch.bmm(queries, keys.transpose(1, 2)) * self.scale  # (N, K, 196)
        
        # Standard softmax — no temperature scaling
        weights = scores.softmax(dim=-1)  # (N, K, 196)
        
        pooled = torch.bmm(weights, values)  # (N, K, 1026)
        return pooled


class ProjectionHeadA(nn.Module):
    """
    Maps fused four-level region features into Qwen's embedding space.

    Uses LLaVA-1.5's empirically validated layer ordering:
        Linear -> GELU -> LayerNorm -> Linear -> LayerNorm
    GELU before LayerNorm on the first stage lets the non-linearity act on raw mixed activations before variance normalisation. This preserves
    relative magnitude differences between high-energy semantic features (layer 23) and lower-amplitude attribute features (layer 14), preventing
    attribute signal from being suppressed by normalisation before gating.
    """
    def __init__(self, input_dim=4104, output_dim=3584):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim, bias=True)  # 4104 -> 3584: maps ViT space to Qwen space
        self.gelu = nn.GELU()
        self.layer_norm_1 = nn.LayerNorm(output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim, bias=True) # 3584 -> 3584: refines within Qwen space
        self.layer_norm_2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (N, 4104) — N regions, concatenated four-level features
        x = self.linear_1(x) # (N, 3584) — project into Qwen space
        x = self.gelu(x) # soft gate — amplifies informative dims
        x = self.layer_norm_1(x) # normalise after gating
        x = self.linear_2(x) # (N, 3584) — refine within Qwen space
        x = self.layer_norm_2(x) # stabilise output distribution for Qwen
        return x


class RegionFeatureExtractor(nn.Module):
    """
    Full region path for processing ViT hidden states into region tokens.

    Pipeline:
    1. Select hidden states at layers 14, 17, 20, 23
    2. Add CoordConv (position awareness)
    3. Stack and apply per-level projections (align representational spaces)
    4. Spatial cross-level attention at every position (before RoI Align)
    5. Reshape back to spatial feature maps
    6. RoI Align at 14x14 using bounding box coordinates
    7. Multi-token attention pooling per level (196 positions -> K vectors)
    8. Concatenate four level sequences (4 x K x 1026 -> K x 4104)
    9. ProjectionHeadA -> K region tokens per region in Qwen's embedding space

    Only this module's parameters train during Stage 2.
    ViT and Qwen are both frozen throughout.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.coordconv_dim   # 1026
        L = config.num_levels      # 4

        self.encoder = MultiLevelViTEncoder(config)
        self.attention_pooling = nn.ModuleList([
            MultiTokenAttentionPooling(dim=D, num_tokens=config.num_region_tokens) for _ in range(L)
        ])
        self.projection_head_a = ProjectionHeadA(
            input_dim=config.fused_dim,
            output_dim=config.qwen_embedding_dim,
        )

    def encode_enriched_maps(self, hidden_states):
        """Shared multi-level maps after CoordConv -> PerLevelProj -> CrossLevelAttn."""
        return self.encoder(hidden_states)

    def forward(self, hidden_states, boxes_pixels):
        enriched_maps = self.encode_enriched_maps(hidden_states)
        return self.forward_from_enriched_maps(enriched_maps, boxes_pixels)

    def forward_from_enriched_maps(self, enriched_maps, boxes_pixels):
        from torchvision.ops import roi_align
        D = self.config.coordconv_dim
        B = 1
        N = boxes_pixels.shape[0]
        device = boxes_pixels.device
        dtype = self.config.compute_dtype

        batch_idx = torch.zeros(N, 1, device=device, dtype=boxes_pixels.dtype)
        boxes_roi = torch.cat([batch_idx, boxes_pixels.float()], dim=1)

        roi_outputs = []
        for feat_map in enriched_maps:
            roi = roi_align(
                input=feat_map.float(),
                boxes=boxes_roi.float(),
                output_size=self.config.roi_output_size,
                spatial_scale=self.config.spatial_scale,
                aligned=True,
            ).to(dtype)
            roi_outputs.append(roi)

        level_sequences = []
        for i, roi in enumerate(roi_outputs):
            pooled = self.attention_pooling[i](roi)
            level_sequences.append(pooled)

        fused = torch.cat(level_sequences, dim=2)
        region_tokens = self.projection_head_a(fused)
        return region_tokens


def build_region_feature_extractor(config):
    """Instantiate and move RegionFeatureExtractor to the configured device/dtype."""
    region_extractor = RegionFeatureExtractor(config)
    region_extractor = region_extractor.to(config.device, dtype=config.compute_dtype)

    total = sum(p.numel() for p in region_extractor.parameters() if p.requires_grad) / 1e6
    print(f"RegionFeatureExtractor total trainable: {total:.1f}M")
    for name, module in region_extractor.named_children():
        n = sum(p.numel() for p in module.parameters()) / 1e6
        print(f" {name}: {n:.2f}M")

    return region_extractor

def migrate_region_extractor_state_dict(state_dict):
    """Map legacy keys to encoder.* after MultiLevelViTEncoder refactor."""
    if any(k.startswith('encoder.') for k in state_dict):
        return state_dict
    migrated = {}
    for k, v in state_dict.items():
        if k.startswith(('per_level_proj.', 'cross_level_attn.')):
            migrated[f'encoder.{k}'] = v
        else:
            migrated[k] = v
    return migrated


def load_region_feature_extractor(region_extractor, region_extractor_path, config):
    region_state_dict = load_state_dict_file(region_extractor_path, config.device)
    region_state_dict = migrate_region_extractor_state_dict(region_state_dict)
    region_extractor.load_state_dict(region_state_dict, strict=False)
    region_extractor.eval()
    
    print("Region extractor loaded successfully")
    print(f"Region extractor params: {sum(p.numel() for p in region_extractor.parameters())/1e6:.1f}M")
    return region_extractor


def load_projection_b_weights(weights_path, config) -> ProjectionHeadB:
    """Load frozen Projection-B from a plain state dict or training checkpoint."""
    weights_path = Path(weights_path)
    projection_head_b = ProjectionHeadB(
        input_dim=getattr(config, "vit_patch_feature_dim", config.vit_hidden_dim),
        output_dim=config.qwen_embedding_dim,
    ).to(config.device, dtype=config.compute_dtype)
    state = load_state_dict_file(weights_path, config.device)
    if isinstance(state, dict) and "projection_head_b_state_dict" in state:
        state = state["projection_head_b_state_dict"]
    projection_head_b.load_state_dict(state)
    projection_head_b.eval()
    for param in projection_head_b.parameters():
        param.requires_grad = False
    print(f"Projection-B loaded: {weights_path}")
    return projection_head_b


def load_qwen_with_lora(config):
    """Load Qwen 2.5 7B with LoRA adapters for Stage 3 training."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Stage 3 LoRA requires peft: pip install peft") from exc

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        config.language_model_hf_id,
        use_fast=True,
        cache_dir=str(config.cache_dir) if getattr(config, "cache_dir", None) else None,
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id

    qwen = AutoModelForCausalLM.from_pretrained(
        config.language_model_hf_id,
        torch_dtype=config.compute_dtype,
        device_map={"": config.device},
        use_safetensors=True,
        cache_dir=str(config.cache_dir) if getattr(config, "cache_dir", None) else None,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    qwen = get_peft_model(qwen, lora_config)
    # Required for LoRA + gradient checkpointing (avoids CheckpointError on backward).
    if hasattr(qwen, "config"):
        qwen.config.use_cache = False
    qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    qwen.enable_input_require_grads()
    qwen.print_trainable_parameters()
    return qwen, qwen_tokenizer


def save_lora_adapter(qwen, qwen_tokenizer, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen.save_pretrained(output_dir)
    qwen_tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved -> {output_dir}")
    return output_dir


class GroundingDINOProposer:
    def __init__(self, config, device):
        # Newer transformers removed BertModel.get_head_mask and changed
        # get_extended_attention_mask(device -> dtype); GroundingDINO still uses the old API.
        _apply_transformers_compat_shim()
        self.model = load_model(
            config.grounding_dino_weights_path1,
            config.grounding_dino_weights_path2,
        )
        _rebind_grounding_dino_bert_methods(self.model)
        self.model = self.model.to(device)
        self.device = device
        self.box_threshold = config.box_threshold
        self.text_threshold = config.text_threshold
        self.nms_iou = getattr(config, 'dino_nms_iou', 0.95)
        self.suppress_contained = getattr(config, 'dino_suppress_contained', False)
        self.containment_ratio = getattr(config, 'dino_containment_ratio', 0.85)
        self.containment_max_area_ratio = getattr(config, 'dino_containment_max_area_ratio', 0.50)
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _predict_raw(self, pil_image, text_prompt):
        """Run Grounding DINO once; return xyxy boxes in 336-space, scores, phrase labels."""
        image_pil = pil_image.convert("RGB")
        image_transformed, _ = self.transform(image_pil, None)
        image_transformed = image_transformed.to(self.device)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        if boxes.shape[0] == 0:
            return torch.empty(0, 4), torch.empty(0), []

        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2) * 336
        y1 = (cy - h / 2) * 336
        x2 = (cx + w / 2) * 336
        y2 = (cy + h / 2) * 336
        boxes_336 = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes_336, logits, list(phrases)

    def _finalize_proposals(self, boxes_336, logits, phrases_list, max_boxes, *, verbose=True):
        if boxes_336.shape[0] == 0:
            print(f"Grounding DINO found no matches, falling back to whole-image box")
            whole_image_box = torch.tensor([[0.0, 0.0, 336.0, 336.0]], dtype=torch.float32)
            whole_image_score = torch.tensor([0.0])
            return whole_image_box, whole_image_score, ["[fallback: whole image]"]

        n_before = boxes_336.shape[0]
        boxes_336, logits, phrases_list = dedupe_dino_proposals(
            boxes_336,
            logits,
            phrases_list,
            nms_iou=self.nms_iou,
            suppress_contained=self.suppress_contained,
            containment_ratio=self.containment_ratio,
            containment_max_area_ratio=self.containment_max_area_ratio,
            verbose=verbose,
        )
        if boxes_336.shape[0] < n_before and verbose:
            print(
                f"DINO dedup: {n_before} -> {boxes_336.shape[0]} boxes "
                f"(nms_iou={self.nms_iou}, suppress_contained={self.suppress_contained})"
            )

        n_after_dedup = boxes_336.shape[0]
        all_order = logits.argsort(descending=True)
        order = all_order[:max_boxes]
        if n_after_dedup > max_boxes and verbose:
            dropped = [phrases_list[i] for i in all_order[max_boxes:].tolist()]
            print(f"DINO top-k: kept {max_boxes}/{n_after_dedup} (dropped: {dropped})")
        return boxes_336[order], logits[order], [phrases_list[i] for i in order.tolist()]

    @torch.no_grad()
    def propose_per_tags( self, pil_image, tags: list, *, max_boxes_per_tag: int = 1, max_total_boxes: int = 10, verbose: bool = True, ):
        """Localize each tag with its own DINO forward pass, then merge + dedupe."""
        tags = [t.strip() for t in tags if t and t.strip()]
        if not tags:
            return self._finalize_proposals(
                torch.empty(0, 4), torch.empty(0), [], max_total_boxes, verbose=verbose
            )

        all_boxes, all_scores, all_labels = [], [], []
        for tag in tags:
            boxes_336, logits, phrases = self._predict_raw(pil_image, tag)
            if boxes_336.shape[0] == 0:
                if verbose:
                    print(f"  tag '{tag}': no detection")
                continue
            order = logits.argsort(descending=True)[:max_boxes_per_tag]
            for idx in order.tolist():
                all_boxes.append(boxes_336[idx])
                all_scores.append(logits[idx])
                phrase = phrases[idx] if idx < len(phrases) and phrases[idx] else tag
                all_labels.append(phrase)
            if verbose:
                best = order[0].item()
                print(
                    f"  tag '{tag}': score={logits[best]:.3f} "
                    f"box={[round(v, 1) for v in boxes_336[best].tolist()]}"
                )

        if not all_boxes:
            return self._finalize_proposals(
                torch.empty(0, 4), torch.empty(0), [], max_total_boxes, verbose=verbose
            )

        merged_boxes = torch.stack(all_boxes)
        merged_scores = torch.stack(all_scores)
        if verbose:
            print(f"Per-tag DINO: {len(tags)} tags -> {merged_boxes.shape[0]} boxes before dedup")
        return self._finalize_proposals(
            merged_boxes, merged_scores, all_labels, max_total_boxes, verbose=verbose
        )

    @torch.no_grad()
    def propose(self, pil_image, text_prompt, max_boxes=10):
        boxes_336, logits, phrases_list = self._predict_raw(pil_image, text_prompt)
        if boxes_336.shape[0] == 0:
            print(f"Grounding DINO found no matches for '{text_prompt}', falling back to whole-image box")
            whole_image_box = torch.tensor([[0.0, 0.0, 336.0, 336.0]], dtype=torch.float32)
            whole_image_score = torch.tensor([0.0])
            return whole_image_box, whole_image_score, ["[fallback: whole image]"]
        return self._finalize_proposals(boxes_336, logits, phrases_list, max_boxes, verbose=True)

def load_grounding_dino(config):
    grounding_dino = GroundingDINOProposer(config, device=config.device)
    print(f"Grounding DINO loaded")
    return grounding_dino


_TRANSFORMERS_COMPAT_SHIM_VERSION = 3


def _rebind_grounding_dino_bert_methods(dino_model) -> None:
    """Rebind BertModelWarper helpers after patching transformers mixins."""
    import types

    import transformers.modeling_utils as _mu

    bert = getattr(dino_model, "bert", None)
    if bert is None:
        return
    bert.get_extended_attention_mask = types.MethodType(
        _mu.ModuleUtilsMixin.get_extended_attention_mask,
        bert,
    )
    if hasattr(_mu.PreTrainedModel, "get_head_mask"):
        bert.get_head_mask = types.MethodType(
            _mu.PreTrainedModel.get_head_mask,
            bert,
        )


def _grounding_dino_extended_attention_mask( self, attention_mask, input_shape, dtype=None, *args, **kwargs ):
    """
    Compatibility replacement for transformers.ModuleUtilsMixin.get_extended_attention_mask.

    GroundingDINO still calls the old signature with a ``device`` 3rd arg and often
    passes a bool 2D/3D mask. Newer transformers treat the 3rd arg as dtype and then
    do ``1.0 - mask``, which raises on bool tensors.
    """
    if isinstance(dtype, torch.device):
        dtype = None
    if dtype is None or dtype == torch.bool:
        try:
            dtype = self.dtype
        except Exception:
            dtype = torch.float32
    if dtype is None or dtype == torch.bool:
        dtype = torch.float32

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 4:
        extended_attention_mask = attention_mask
    else:
        raise ValueError(
            f"Wrong shape for attention_mask (shape {tuple(attention_mask.shape)}) "
            f"with input_shape {input_shape}"
        )

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def _apply_transformers_compat_shim():
    """Patch transformers API gaps for GroundingDINO / RAM BERT warpers."""
    import inspect

    import transformers.modeling_utils as _mu

    applied_version = getattr(
        _mu.PreTrainedModel, "_transformers_compat_shim_version", 0
    )
    if applied_version >= _TRANSFORMERS_COMPAT_SHIM_VERSION:
        return

    if not hasattr(_mu, "apply_chunking_to_forward"):
        from transformers.pytorch_utils import apply_chunking_to_forward as _achtf

        _mu.apply_chunking_to_forward = _achtf

    if not hasattr(_mu, "prune_linear_layer"):
        from transformers.pytorch_utils import prune_linear_layer as _pll

        _mu.prune_linear_layer = _pll

    if not hasattr(_mu, "find_pruneable_heads_and_indices"):
        from typing import List, Set, Tuple

        def find_pruneable_heads_and_indices( heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int], ) -> Tuple[Set[int], torch.LongTensor]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        _mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB

    if not hasattr(_PTB, "additional_special_tokens_ids"):
        _PTB.additional_special_tokens_ids = property(
            lambda self: self.extra_special_tokens_ids,
            lambda self, value: setattr(self, "extra_special_tokens_ids", value),
        )
    if not hasattr(_PTB, "additional_special_tokens"):
        _PTB.additional_special_tokens = property(
            lambda self: self.extra_special_tokens,
            lambda self, value: setattr(self, "extra_special_tokens", value),
        )

    _orig_tie_weights = _mu.PreTrainedModel.tie_weights
    _tie_params = set(inspect.signature(_orig_tie_weights).parameters)

    def _patched_tie_weights(self, missing_keys=None, recompute_mapping=True):
        if not recompute_mapping and not hasattr(self, "all_tied_weights_keys"):
            recompute_mapping = True
        kwargs = {}
        if "missing_keys" in _tie_params:
            kwargs["missing_keys"] = missing_keys
        if "recompute_mapping" in _tie_params:
            kwargs["recompute_mapping"] = recompute_mapping
        return _orig_tie_weights(self, **kwargs)

    _mu.PreTrainedModel.tie_weights = _patched_tie_weights
    _mu.ModuleUtilsMixin.get_extended_attention_mask = (
        _grounding_dino_extended_attention_mask
    )

    # Always restore get_head_mask — GroundingDINO BertModelWarper requires it.
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    _mu.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
    _mu.PreTrainedModel.get_head_mask = get_head_mask
    _mu.PreTrainedModel._transformers_compat_shim_version = _TRANSFORMERS_COMPAT_SHIM_VERSION
    _mu.PreTrainedModel._transformers_compat_shim_applied = True
    _mu.PreTrainedModel._ram_shim_applied = True  # back-compat alias


def _apply_ram_transformers_shim():
    """Back-compat alias used by older call sites."""
    _apply_transformers_compat_shim()


class RAMTagProposer:
    """Question-agnostic open-vocabulary tags via RAM++."""

    def __init__(self, config, device):
        from huggingface_hub import hf_hub_download

        _apply_transformers_compat_shim()
        from ram.models import ram_plus
        from ram import inference_ram, get_transform

        cache_dir = str(getattr(config, 'cache_dir', None)) if getattr(config, 'cache_dir', None) else None
        ram_ckpt = hf_hub_download(
            config.ram_hf_repo,
            config.ram_checkpoint_name,
            cache_dir=cache_dir,
        )
        self.model = ram_plus(
            pretrained=ram_ckpt,
            image_size=config.ram_image_size,
            vit='swin_l',
        ).eval().to(device)
        self.transform = get_transform(image_size=config.ram_image_size)
        self.device = device
        self._inference_ram = inference_ram

    @staticmethod
    def _parse_tags(en_text: str) -> list:
        return [t.strip() for t in str(en_text).split('|') if t.strip()]

    @torch.no_grad()
    def get_tags(self, pil_image, max_tags=None) -> list:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.autocast('cuda', enabled=(self.device == 'cuda')):
            en, _ = self._inference_ram(x, self.model)
        tags = self._parse_tags(en)
        if max_tags is not None:
            tags = tags[:max_tags]
        return tags


def load_ram(config):
    ram_proposer = RAMTagProposer(config, device=config.device)
    print("RAM++ loaded for hybrid box proposals")
    return ram_proposer