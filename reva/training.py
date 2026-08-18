import json as json_module
import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from .models import ProjectionHeadB, migrate_region_extractor_state_dict
from torch.utils.data import DataLoader
import random as random_module
import torch.nn.functional as F
import numpy as np
import glob

class EarlyStopper:

    def __init__(self, patience: int, min_delta: float, ema_factor: float):
        self.patience = patience
        self.min_delta = min_delta
        self.ema_factor = ema_factor
        self.best_smoothed_loss = float('inf')
        self.smoothed_loss = None   # initialised on first update
        self.steps_since_last_improvement = 0
        self.should_stop = False
        self.new_best_this_step = False  # flag for checkpoint saving

    def update(self, raw_loss: float) -> bool:
        self.new_best_this_step = False

        if self.smoothed_loss is None:
            self.smoothed_loss = raw_loss
        else:
            self.smoothed_loss = (self.ema_factor * self.smoothed_loss) + ((1 - self.ema_factor) * raw_loss)

        improvement = self.best_smoothed_loss - self.smoothed_loss
        if improvement > self.min_delta:
            self.best_smoothed_loss = self.smoothed_loss
            self.steps_since_last_improvement = 0
            self.new_best_this_step = True
        else:
            self.steps_since_last_improvement += 1

        if self.steps_since_last_improvement >= self.patience:
            self.should_stop = True

        return self.should_stop

    def status(self) -> str:
        return (
            f"ema={self.smoothed_loss:.4f} "
            f"best={self.best_smoothed_loss:.4f} "
            f"patience={self.steps_since_last_improvement}/{self.patience}"
        )

class LossTracker:
    def __init__(self, log_json_path: Path, plot_output_path: Path):
        self.log_json_path = log_json_path
        self.plot_output_path = plot_output_path

        self.steps: List[int] = []
        self.raw_losses: List[float] = []
        self.smoothed_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.best_loss_line: List[float] = []

    def record(self, global_step: int, raw_loss: float, smoothed_loss: float, best_loss: float, learning_rate: float):
        self.steps.append(global_step)
        self.raw_losses.append(raw_loss)
        self.smoothed_losses.append(smoothed_loss)
        self.learning_rates.append(learning_rate)
        self.best_loss_line.append(best_loss)

    def save_json_log(self):
        """Writing all recorded values to a JSON file for future inspection"""
        log_data = {
            'steps': self.steps,
            'raw_losses': self.raw_losses,
            'smoothed_losses': self.smoothed_losses,
            'learning_rates': self.learning_rates,
            'best_loss_values': self.best_loss_line,
        }
        with open(self.log_json_path, 'w') as f:
            json_module.dump(log_data, f, indent=2)

    def plot_and_save(self):
        if len(self.steps) < 2:
            return
        fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('Projection-B Alignment Training', fontsize=13, fontweight='bold')
        ax_loss.plot(self.steps, self.raw_losses, color='#a8c4e0', linewidth=0.8, alpha=0.6, label='Raw loss (per step)')
        ax_loss.plot(self.steps, self.smoothed_losses, color='#1f77b4', linewidth=1.8, label='Smoothed loss (EMA)')
        ax_loss.plot(self.steps, self.best_loss_line, color='#2ca02c', linewidth=1.2, linestyle='--', label='Best smoothed loss')
        ax_loss.set_ylabel('Cross-entropy loss')
        ax_loss.legend(loc='upper right', fontsize=9)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax_lr.plot(self.steps, self.learning_rates, color='#d62728', linewidth=1.4, label='Learning rate')
        ax_lr.set_ylabel('LR')
        ax_lr.set_xlabel('Optimiser step')
        ax_lr.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        ax_lr.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def run_alignment_forward_pass( processed_images: torch.Tensor, caption_token_ids: torch.Tensor, caption_attention_mask: torch.Tensor, frozen_vit, projection_head_b, frozen_qwen, config ) -> torch.Tensor:
    """Full alignment forward pass. Returns the scalar cross-entropy loss
    computed only over caption token positions."""

    batch_size = processed_images.shape[0]

    with torch.no_grad():
        vit_output = frozen_vit(pixel_values=processed_images)
        # last_hidden_state: (B, 257, 1024) — drop CLS token (index 0)
        image_patch_features = vit_output.last_hidden_state[:, 1:, :]  # (B, 256, 1024)

    projected_image_tokens = projection_head_b(image_patch_features)  # (B, 256, 3584)

    num_actual_image_tokens = projected_image_tokens.shape[1]

    with torch.no_grad():
        caption_token_embeddings = frozen_qwen.get_input_embeddings()(caption_token_ids)

    combined_token_sequence = torch.cat([projected_image_tokens, caption_token_embeddings], dim=1)

    image_token_attention_mask = torch.ones(
        batch_size, num_actual_image_tokens,
        dtype=torch.long, device=processed_images.device
    )
    full_attention_mask = torch.cat([image_token_attention_mask, caption_attention_mask], dim=1)

    image_position_labels = torch.full(
        (batch_size, num_actual_image_tokens),
        fill_value=-100,
        dtype=torch.long,
        device=caption_token_ids.device
    )

    caption_labels = caption_token_ids.clone()
    padding_positions = caption_attention_mask == 0
    caption_labels[padding_positions] = -100

    all_labels = torch.cat([image_position_labels, caption_labels], dim=1)

    # qwen_output = frozen_qwen(
    #     inputs_embeds=combined_token_sequence,
    #     attention_mask=full_attention_mask,
    #     labels=all_labels,
    # )

    # # return qwen_output.loss
    # qwen_output = frozen_qwen(
    #     inputs_embeds=combined_token_sequence,
    #     attention_mask=full_attention_mask,
    #     num_logits_to_keep=caption_token_ids.shape[1] + 1,
    #     # not passing labels, computing loss manually below
    # )
    
    # # qwen_output.logits shape: (B, 129, 32000)
    # # We need to align logits with caption labels only
    # # Shift: predict token t+1 from token t
    # # logits[:, :-1] predicts positions 576 to 703 (caption tokens)
    # # caption_labels shifted: caption_token_ids[:, 1:] with -100 for padding
    
    # logits = qwen_output.logits[:, :-1, :]          # (B, 128, 32000)
    # caption_labels = caption_token_ids.clone()       # (B, 128)
    # padding_positions = caption_attention_mask == 0
    # caption_labels[padding_positions] = -100
    
    # # Shift labels by 1 to align with next-token prediction
    # shift_labels = caption_labels[:, 1:]             # (B, 127)
    # shift_logits = logits[:, -127:, :]              # (B, 127, 32000)
    
    # loss = torch.nn.functional.cross_entropy(
    #     shift_logits.reshape(-1, shift_logits.size(-1)),
    #     shift_labels.reshape(-1),
    #     ignore_index=-100,
    # )
    # return loss
    qwen_output = frozen_qwen(
        inputs_embeds=combined_token_sequence,
        attention_mask=full_attention_mask,
        labels=all_labels,  # already built correctly above with -100 masking
    )
    return qwen_output.loss

def build_optimiser_and_scheduler(projection_head_b, train_dataloader, config):
    """Create the AdamW optimiser and cosine LR scheduler."""
    optimiser = AdamW(
        projection_head_b.parameters(),
        lr=config.peak_learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    total_optimiser_steps = (
        math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
        * config.num_training_epochs
    )
    num_warmup_steps = int(total_optimiser_steps * config.warmup_ratio)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimiser,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimiser_steps,
    )

    print(f"Total optimiser steps: {total_optimiser_steps:,}")
    print(f"Warm-up steps: {num_warmup_steps:,} ({config.warmup_ratio * 100:.0f}% of total)")
    print(f"Peak learning rate: {config.peak_learning_rate}")

    return optimiser, lr_scheduler


def save_projection_b_checkpoint(projection_head_b, optimiser, lr_scheduler, global_step: int, running_avg_loss: float, config):
    """
    Saves a checkpoint containing:
    - projection_head_b weights (the trained artefact)
    - optimiser state (to resume mid-run training if needed)
    - scheduler state
    - metadata (step, loss)
    """
    checkpoint_path = config.checkpoint_output_dir / f"projection_b_step_{global_step}.pt"

    torch.save({
        'projection_head_b_state_dict': projection_head_b.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'global_step': global_step,
        'running_avg_loss': running_avg_loss,
        'config': vars(config),
    }, checkpoint_path)

    print(f"\tCheckpoint saved -> {checkpoint_path} (step {global_step}, loss {running_avg_loss:.4f})")
    return checkpoint_path


def load_projection_b_from_checkpoint(checkpoint_path: str, config) -> ProjectionHeadB:
    """Load a previously saved projection head for inference or continued training."""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    loaded_projection_head = ProjectionHeadB(
        input_dim=config.vit_patch_feature_dim,
        output_dim=config.qwen_embedding_dim,
    ).to(config.device, dtype=config.compute_dtype)

    loaded_projection_head.load_state_dict(checkpoint['projection_head_b_state_dict'])
    step = checkpoint['global_step']
    loss = checkpoint['running_avg_loss']

    print(f"Loaded Projection-B from step {step} (loss: {loss:.4f})")
    return loaded_projection_head


def train_projection_b(frozen_vit, projection_head_b, frozen_qwen, train_dataloader, optimiser, lr_scheduler, early_stopper, loss_tracker, config):
    """
    Full training loop for Projection-B alignment with:
      - Gradient accumulation
      - Cosine LR decay
      - Early stopping on smoothed training loss
      - Automatic best-weights checkpoint on every new best loss
    Returns (global_step, stop_reason) where stop_reason is either
    'completed' or 'early_stopped'.
    """

    global_step = 0
    cumulative_loss = 0.0
    batches_accumulated = 0
    stop_reason = 'completed'

    frozen_vit.eval()
    frozen_qwen.train() #for gradient checkpointing
    projection_head_b.train()

    best_weights_path = config.checkpoint_output_dir / "projection_b_best_weights.pt"

    for epoch_num in range(1, config.num_training_epochs + 1):

        progress_bar = tqdm(train_dataloader,
                            desc=f"Epoch {epoch_num}/{config.num_training_epochs}",
                            dynamic_ncols=True)
        optimiser.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):

            processed_images = batch['processed_image'].to(config.device, dtype=config.compute_dtype)
            caption_token_ids = batch['caption_token_ids'].to(config.device)
            caption_attention_mask = batch['caption_attention_mask'].to(config.device)

            batch_loss = run_alignment_forward_pass(
                processed_images=processed_images,
                caption_token_ids=caption_token_ids,
                caption_attention_mask=caption_attention_mask,
                frozen_vit=frozen_vit,
                projection_head_b=projection_head_b,
                frozen_qwen=frozen_qwen,
                config=config,
            )

            scaled_loss = batch_loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            cumulative_loss += batch_loss.item()
            batches_accumulated += 1

            is_accumulation_complete = (
                ((batch_idx + 1) % config.gradient_accumulation_steps == 0)
                or ((batch_idx + 1) == len(train_dataloader))
            )

            if is_accumulation_complete:
                torch.nn.utils.clip_grad_norm_(
                    projection_head_b.parameters(),
                    max_norm=config.gradient_clip_max_norm
                )

                optimiser.step()
                lr_scheduler.step()
                optimiser.zero_grad()

                global_step += 1
                avg_loss_this_step = cumulative_loss / batches_accumulated
                current_lr = lr_scheduler.get_last_lr()[0]

                cumulative_loss = 0.0
                batches_accumulated = 0

                should_stop_now = early_stopper.update(avg_loss_this_step)

                if early_stopper.new_best_this_step:
                    torch.save(projection_head_b.state_dict(), best_weights_path)

                progress_bar.set_postfix({
                    'loss': f"{avg_loss_this_step:.4f}",
                    'ema': f"{early_stopper.smoothed_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'patience': f"{early_stopper.steps_since_last_improvement}/{early_stopper.patience}",
                })

                if global_step % config.log_every_n_steps == 0:
                    loss_tracker.record(
                        global_step=global_step,
                        raw_loss=avg_loss_this_step,
                        smoothed_loss=early_stopper.smoothed_loss,
                        best_loss=early_stopper.best_smoothed_loss,
                        learning_rate=current_lr,
                    )
                    loss_tracker.save_json_log()
                    loss_tracker.plot_and_save()

                if global_step % config.save_every_n_steps == 0:
                    save_projection_b_checkpoint(
                        projection_head_b=projection_head_b,
                        optimiser=optimiser,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        running_avg_loss=avg_loss_this_step,
                        config=config,
                    )

                if should_stop_now:
                    print(f"\nEarly stopping triggered at step {global_step}.")
                    print(f"{early_stopper.status()}")
                    print(f"Best weights already saved {best_weights_path}")
                    stop_reason = 'early_stopped'
                    break

        if stop_reason == 'early_stopped':
            break

    return global_step, stop_reason


def region_alignment_forward_pass(batch, frozen_vit, region_extractor, frozen_qwen, qwen_tokenizer, config):
    """
    Training forward pass for Projection-A alignment.

    For each sample: run frozen ViT, extract region tokens using GT box (teacher
    forcing), compute cross-entropy + ITC loss on description tokens.

    Processes images one at a time because each image has one box in training
    (RoI Align requires a separate feature map per image).
    """
    import torch.nn.functional as F

    pixel_values = batch['pixel_values'].to(config.device, dtype=config.compute_dtype) # (B, 3, 336, 336)
    boxes = batch['boxes'].to(config.device, dtype=torch.float32)                      # (B, 4)
    input_ids = batch['input_ids'].to(config.device)                                   # (B, max_tokens)
    attn_mask = batch['attention_mask'].to(config.device)                              # (B, max_tokens)
    B = pixel_values.shape[0]

    all_region_tokens = []

    for i in range(B):
        single_image = pixel_values[i:i+1]
        single_box  = boxes[i:i+1]          # (1, 4) — description target for region extractor
    
        with torch.no_grad():
            vit_out = frozen_vit(pixel_values=single_image, output_hidden_states=True)

        enriched_maps = region_extractor.encode_enriched_maps(vit_out.hidden_states)
        region_token = region_extractor.forward_from_enriched_maps(enriched_maps, single_box)
        all_region_tokens.append(region_token)

    region_tokens = torch.cat(all_region_tokens, dim=0)              # (B, K, 3584)

    with torch.no_grad():
        text_embeds = frozen_qwen.get_input_embeddings()(input_ids)   # (B, max_tokens, 3584)

    # --- ITC loss ---
    if config.itc_loss_weight > 0.0:
        region_repr = region_tokens.mean(dim=1)                        # (B, 3584)
        mask_expanded = attn_mask.unsqueeze(-1).float()                # (B, max_tokens, 1)
        text_repr = (text_embeds * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        region_norm = F.normalize(region_repr.float(), dim=-1)         # (B, 3584)
        text_norm   = F.normalize(text_repr.float(), dim=-1)           # (B, 3584)
        logits = region_norm @ text_norm.T / config.itc_temperature    # (B, B)
        itc_labels = torch.arange(B, device=config.device)
        itc_loss = (
            F.cross_entropy(logits, itc_labels) +
            F.cross_entropy(logits.T, itc_labels)
        ) / 2
    else:
        itc_loss = torch.tensor(0.0, device=config.device)

    # --- Captioning loss ---
    input_embeds = torch.cat([region_tokens, text_embeds], dim=1)     # (B, K+max_tokens, 3584)
    region_label_mask = torch.full(
        (B, region_tokens.shape[1]), fill_value=-100, dtype=torch.long, device=config.device
    )
    labels = torch.cat([region_label_mask, input_ids], dim=1)         # (B, K+max_tokens)
    padding_mask = (input_ids == qwen_tokenizer.pad_token_id)
    labels[:, region_tokens.shape[1]:][padding_mask] = -100
    region_attn = torch.ones(B, region_tokens.shape[1], dtype=torch.long, device=config.device)
    full_attn = torch.cat([region_attn, attn_mask], dim=1)            # (B, K+max_tokens)

    outputs = frozen_qwen(
        inputs_embeds=input_embeds,
        attention_mask=full_attn,
        labels=labels,
        #num_logits_to_keep=config.max_description_tokens + 1,
    )
    captioning_loss = outputs.loss

    total_loss = captioning_loss + config.itc_loss_weight * itc_loss
    
    return total_loss, captioning_loss, itc_loss


def save_projection_a_checkpoint(region_extractor, optimiser, lr_scheduler, global_step, running_avg_loss, stage, config):
    path = config.checkpoint_dir / f"projection_a_{stage}_step_{global_step}.pt"
    torch.save({
        'region_extractor_state_dict': region_extractor.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'global_step': global_step,
        'running_avg_loss': running_avg_loss,
        'stage': stage,
    }, path)
    #print(f"Checkpoint saved: {path}")


def save_projection_a_weights(region_extractor, stage, label, config):
    path = config.checkpoint_dir / f"projection_a_{stage}_{label}_weights.pt"
    torch.save(region_extractor.state_dict(), path)
    #print(f"Weights saved: {path}")


def load_projection_a_weights(region_extractor, weights_path, config):
    from .models import RegionFeatureExtractor, load_state_dict_file
    state_dict = load_state_dict_file(weights_path, config.device)
    if 'region_extractor_state_dict' in state_dict:
        state_dict = state_dict['region_extractor_state_dict']
    state_dict = migrate_region_extractor_state_dict(state_dict)
    region_extractor.load_state_dict(state_dict, strict=False)
    region_extractor.eval()
    print(f"Weights loaded from {weights_path}")
    return region_extractor


def train_projection_a(region_extractor, frozen_vit, frozen_qwen, coco_samples, refcoco_samples, vg_samples, clip_image_processor, qwen_tokenizer, config, save_every_steps=500, resume_from_checkpoint=None, verbose=False, balance_families=True):
    """
    Curriculum training replacing sequential Stage 2a then Stage 2b.

    Each epoch does a full combined pass over all short + VG samples, shuffled
    freshly each epoch so every sample is seen exactly once per epoch.

    Supports resuming from a checkpoint via resume_from_checkpoint argument.
    Pass the path to a projection_a_curriculum_step_N.pt checkpoint file to
    resume from that step, restoring model weights, optimiser state, scheduler
    state, global_step, and running_avg_loss.
    """
    # Lazy import so Stage 3 LoRA training does not pull dataset.py (faiss) at import time.
    from .dataset import (
        FamilyBalancedBatchSampler,
        RegionAlignmentDataset,
        collate_region_batch,
    )

    batch_size = config.curriculum_batch_size
    gradient_accum = config.curriculum_gradient_accum #Gradient accumulation for efficient GPU utlization: splitting a large batch into micro-batches, running a forward and backward pass on each micro-batch, and adding up the gradients.
    peak_lr = config.curriculum_peak_lr
    num_epochs = config.curriculum_num_epochs
    warmup_ratio = config.curriculum_warmup_ratio

    print(f"\nCurriculum training | COCO+RefCOCO: {len(coco_samples)+len(refcoco_samples):,} | VG: {len(vg_samples):,}")
    print(f"Batch: {batch_size} | Gradient Accumulation: {gradient_accum} | LR: {peak_lr}")

    coco_family_samples = coco_samples + refcoco_samples

    trainable_params = list(region_extractor.parameters())
    print(f"Trainable params: region_extractor = {sum(p.numel() for p in trainable_params)/1e6:.1f}M")

    optimiser = torch.optim.AdamW(
        region_extractor.parameters(),
        lr=peak_lr,
        weight_decay=config.weight_decay,
    )

    # Estimate total steps using full combined dataset size — each epoch sees all samples
    max_epoch_size = len(coco_family_samples) + len(vg_samples)
    estimated_steps_per_epoch = (max_epoch_size // batch_size) // gradient_accum
    total_steps = estimated_steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimiser, warmup_steps, total_steps)
    print(f"Estimated total steps: {total_steps:,} | Warmup: {warmup_steps}")

    region_extractor.train()
    frozen_vit.eval()
    # frozen_qwen stays in train() mode (required for gradient checkpointing)

    global_step = 0
    running_avg_loss = None
    best_loss = float('inf')
    training_log = []
    start_epoch = 0
    batches_to_skip = 0

    # Resume from checkpoint if provided
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=config.device)
        region_extractor.load_state_dict(
            migrate_region_extractor_state_dict(checkpoint['region_extractor_state_dict']),
            strict=False,
        )
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        global_step = checkpoint['global_step']
        running_avg_loss = checkpoint['running_avg_loss']
        best_loss = running_avg_loss

        # Calculate which epoch and batch to resume from
        full_epoch_size = len(coco_family_samples) + len(vg_samples)
        steps_per_epoch = (full_epoch_size // batch_size) // gradient_accum
        start_epoch = global_step // steps_per_epoch
        batches_to_skip = (global_step % steps_per_epoch) * gradient_accum

        print(f"Resumed: global_step={global_step} | ema={running_avg_loss:.4f} | start_epoch={start_epoch} | skipping {batches_to_skip} batches")

    # Balanced sampling: build the dataset + family-balanced sampler once over the
    # full sample list so every batch carries an equal share of each family
    # (COCO / RefCOCO / VG / GRIT) regardless of their very different raw sizes.
    all_samples = coco_family_samples + vg_samples
    balanced_sampler = None
    balanced_dataloader = None
    if balance_families:
        balanced_dataset = RegionAlignmentDataset(
            samples=all_samples,
            image_processor=clip_image_processor,
            tokenizer=qwen_tokenizer,
            max_tokens=config.max_description_tokens,
        )
        balanced_sampler = FamilyBalancedBatchSampler(
            sources=[s.get('source', 'other') for s in all_samples],
            batch_size=batch_size,
            seed=42,
        )
        balanced_dataloader = DataLoader(
            balanced_dataset,
            batch_sampler=balanced_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_region_batch,
        )

    for epoch in range(start_epoch, num_epochs):
        if balance_families:
            # Fixed dataset/loader; sampler reshuffles per epoch via set_epoch
            balanced_sampler.set_epoch(epoch)
            dataloader = balanced_dataloader
            print(f"\nEpoch {epoch+1}/{num_epochs} | family-balanced | "
                  f"samples={len(all_samples):,} | batches={len(dataloader):,}")
        else:
            # Full combined pass over all samples every epoch
            epoch_samples = coco_family_samples + vg_samples
            random_module.shuffle(epoch_samples)

            print(f"\nEpoch {epoch+1}/{num_epochs} | samples={len(epoch_samples):,} (full pass: {len(coco_family_samples):,} short + {len(vg_samples):,} VG)")

            dataset = RegionAlignmentDataset(
                samples=epoch_samples,
                image_processor=clip_image_processor,
                tokenizer=qwen_tokenizer,
                max_tokens=config.max_description_tokens,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_region_batch,
                drop_last=True,
            )

        optimiser.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch{epoch+1}/{num_epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(progress_bar):
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue

            loss, captioning_loss, itc_loss = region_alignment_forward_pass(
                batch, frozen_vit, region_extractor, frozen_qwen, qwen_tokenizer, config,
            )

            (loss / gradient_accum).backward()

            if (batch_idx + 1) % gradient_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.gradient_clip_max_norm)
                optimiser.step()
                lr_scheduler.step()
                optimiser.zero_grad()
                global_step += 1

                loss_val = loss.item()
                if running_avg_loss is None:
                    running_avg_loss = loss_val
                else:
                    running_avg_loss = config.ema_smoothing * running_avg_loss + (1 - config.ema_smoothing) * loss_val

                progress_bar.set_postfix({
                    'loss': f"{loss_val:.4f}",
                    'cap': f"{captioning_loss.item():.4f}",
                    'itc': f"{itc_loss.item():.4f}",
                    'ema': f"{running_avg_loss:.4f}",
                    'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

                training_log.append({
                    'step': global_step,
                    'epoch': epoch + 1,
                    'loss': loss_val,
                    'captioning_loss': captioning_loss.item(),
                    'itc_loss': itc_loss.item(),
                    'ema_loss': running_avg_loss,
                    'lr': lr_scheduler.get_last_lr()[0],
                })

                if global_step % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, step {global_step}:  loss={loss_val:.4f} | "
                          f"cap_loss={captioning_loss.item():.4f} | itc_loss={itc_loss.item():.4f} | "
                          f"ema_loss={running_avg_loss:.4f} | "
                          f"lr={lr_scheduler.get_last_lr()[0]:.2e}")
                    
                if global_step % save_every_steps == 0:
                    save_projection_a_checkpoint(region_extractor, optimiser, lr_scheduler, global_step, running_avg_loss, "curriculum", config)

                if running_avg_loss < best_loss and global_step > 100:
                    best_loss = running_avg_loss
                    save_projection_a_weights(region_extractor, "curriculum", "best", config)

    save_projection_a_weights(region_extractor, "curriculum", "final", config)
    log_path = config.checkpoint_dir / "training_log_curriculum.json"
    with open(log_path, 'w') as f:
        json_module.dump(training_log, f, indent=2)
    print(f"\nCurriculum complete. Best EMA loss: {best_loss:.4f} | Log: {log_path}")
    return region_extractor, training_log


def train_stage3_lora( qwen, qwen_tokenizer, frozen_vit, projection_head_b, region_extractor, train_samples, clip_image_processor, config, *, include_regions=True, ):
    """
    Train LoRA on the Stage 3 fine-tuning pool.

    Prefix is [576 globals] + [K×16 regions] + text when include_regions=True,
    or [576 globals] + text for the matched Stage 1 LoRA baseline.
    Projection-B and Projection-A stay frozen.
    """
    from .models import save_lora_adapter
    from .stage3_train import (
        Stage3VQADataset,
        build_stage3_weighted_sampler,
        collate_stage3_batch,
        stage3_forward_pass,
        write_stage3_training_manifest,
    )

    batch_size = config.micro_batch_size
    gradient_accum = config.gradient_accumulation_steps
    total_steps = config.max_optimizer_steps
    tokens_per_box = int(getattr(config, "tokens_per_box", 16))
    max_boxes = int(getattr(config, "max_boxes_per_image", 20))

    dataset = Stage3VQADataset(
        train_samples,
        clip_image_processor,
        max_boxes_per_image=max_boxes if include_regions else 0,
    )
    sampler = build_stage3_weighted_sampler(train_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_stage3_batch,
        pin_memory=True,
    )

    lora_params = [p for p in qwen.parameters() if p.requires_grad]
    optimiser = AdamW(
        lora_params,
        lr=config.peak_learning_rate,
        weight_decay=config.weight_decay,
    )

    frozen_vit.eval()
    projection_head_b.eval()
    for p in projection_head_b.parameters():
        p.requires_grad = False
    if include_regions:
        if region_extractor is None:
            raise ValueError("region_extractor is required when include_regions=True")
        region_extractor.eval()
        for p in region_extractor.parameters():
            p.requires_grad = False

    qwen.config.use_cache = False
    qwen.train()

    warmup_steps = int(total_steps * config.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimiser, warmup_steps, total_steps)

    config.lora_output_dir.mkdir(parents=True, exist_ok=True)
    write_stage3_training_manifest(
        config,
        pool_size=len(train_samples),
        include_regions=include_regions,
        extra={"total_steps": total_steps},
    )

    mode_name = "concat576 Stage 3" if include_regions else "global576 Stage 1 LoRA"
    visual_prefix = (
        f"≤{576 + max_boxes * tokens_per_box} "
        f"({576} global + {max_boxes}×{tokens_per_box} region)"
        if include_regions
        else "576 global"
    )
    print(
        f"\n{mode_name} | samples={len(train_samples):,} | "
        f"prefix={visual_prefix} | "
        f"micro_batch={batch_size} | accum={gradient_accum} | steps={total_steps:,}"
    )

    global_step = 0
    micro_step = 0
    best_ema = float("inf")
    ema = None
    training_log = []
    accum_loss = 0.0
    optimiser.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, desc=mode_name)
    data_iter = iter(dataloader)

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss = stage3_forward_pass(
            batch,
            frozen_vit=frozen_vit,
            projection_head_b=projection_head_b,
            region_extractor=region_extractor,
            qwen=qwen,
            qwen_tokenizer=qwen_tokenizer,
            config=config,
            include_regions=include_regions,
        )
        (loss / gradient_accum).backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % gradient_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, config.gradient_clip_max_norm)
            optimiser.step()
            lr_scheduler.step()
            optimiser.zero_grad(set_to_none=True)

            step_loss = accum_loss / gradient_accum
            accum_loss = 0.0
            ema = step_loss if ema is None else 0.98 * ema + 0.02 * step_loss
            training_log.append({"step": global_step, "loss": step_loss, "ema": ema})

            if ema < best_ema:
                best_ema = ema
                save_lora_adapter(
                    qwen,
                    qwen_tokenizer,
                    config.lora_output_dir / config.lora_best_subdir,
                )

            pbar.set_postfix(loss=f"{step_loss:.3f}", ema=f"{ema:.3f}", refresh=False)
            global_step += 1
            pbar.update(1)

    pbar.close()
    save_lora_adapter(qwen, qwen_tokenizer, config.lora_output_dir / "final_lora")

    log_path = Path(config.training_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json_module.dump(training_log, f, indent=2)

    print(f"\n{mode_name} complete. Best EMA={best_ema:.4f} | log: {log_path}")
    return qwen, training_log