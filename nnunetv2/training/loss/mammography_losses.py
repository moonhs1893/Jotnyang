"""
This module implements loss functions tailored for binary mammography segmentation.

The research paper “Improved Loss Function for Mass Segmentation in Mammography Images
Using Density and Mass Size” (J. Imaging 2024, 10, 20) proposes several adaptive
sample‑level prioritizing (ASP) losses.  These losses leverage sample‑specific
characteristics—such as the ratio of a mass to the overall image or the tissue
density—to dynamically reweight contributions of different loss terms.  The goal
is to combat severe class imbalance and the variable appearance of masses in
mammography data.

The implementation here focuses on the ratio‑based ASP (R‑ASP) variant.  It
computes the relative mass size for each training sample and then adjusts the
weighting between a Dice loss and a binary cross‑entropy (BCE) loss.  Samples
with very small masses (high class imbalance) receive a higher Dice weight,
whereas samples with larger masses receive a higher BCE weight.  The
reweighting follows Equation (1) from the paper:

    L_i^RASP = (I_dice + (1 − p_i) × g) × L_i^Dice + (I_bce + p_i × g) × L_i^BCE

where p_i ∈ {0,1} indicates whether sample i belongs to the “large” mass group,
g controls the strength of the reweighting, and I_dice and I_bce are the
initial weights for Dice and BCE.  In practice, we determine the large vs
small grouping by computing the median mass ratio within each mini‑batch and
setting p_i = 1 for samples with ratio ≥ median and 0 otherwise.  A median
split is simple, dataset‑agnostic and requires no precomputed thresholds.

This loss is only designed for binary segmentation (one foreground class).
Multiclass problems fall back to the standard nnU‑Net compound losses.

"""

from __future__ import annotations

from typing import List, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class RatioAdaptiveBCEDiceLoss(nn.Module):
    """Ratio‑adaptive combination of Dice and BCE loss for binary segmentation.

    This loss implements the R‑ASP formulation described in the
    aforementioned paper.  It computes the Dice loss and the BCE loss per
    sample, then combines them using weights that depend on the relative size
    of the mass (foreground) in each image.  The mass ratio is the number of
    foreground pixels divided by the total number of pixels.  A median split
    over the current mini‑batch decides which samples are treated as “large"
    (p_i = 1) and which as “small” (p_i = 0).  You may adjust the hyper
    parameter ``g`` to control the influence of the adaptive term, and
    ``initial_dice_weight``/``initial_bce_weight`` to set the baseline
    contribution of each loss term.

    Args:
        initial_dice_weight (float): Baseline weight for the Dice loss (I_dice).
        initial_bce_weight (float): Baseline weight for the BCE loss (I_bce).
        g (float): Strength of the adaptive weighting term.
        smooth (float): Smoothing constant passed to the Dice loss to prevent
            division by zero.
        ddp (bool): If true, the Dice loss will gather tensors across DDP
            processes.  Defaults to True to match nnU‑Net behaviour.

    Example:

        >>> loss = RatioAdaptiveBCEDiceLoss()
        >>> out = torch.randn(2, 1, 32, 32)  # logits
        >>> tgt = torch.randint(0, 2, (2, 32, 32))
        >>> value = loss(out, tgt)

    """

    def __init__(self,
                 initial_dice_weight: float = 1.0,
                 initial_bce_weight: float = 1.0,
                 g: float = 0.3,
                 smooth: float = 1e-5,
                 ddp: bool = True,
                 apply_nonlin: bool = False) -> None:
        super().__init__()
        self.initial_dice_weight = initial_dice_weight
        self.initial_bce_weight = initial_bce_weight
        self.g = g
        # Use MemoryEfficientSoftDiceLoss which applies sigmoid internally
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid,
                                               batch_dice=False,
                                               do_bg=False,
                                               smooth=smooth,
                                               ddp=ddp)
        # BCEWithLogitsLoss allows logits as input; we compute per‑voxel loss
        # and later reduce per sample.
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.apply_nonlin = apply_nonlin

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the ratio‑adaptive BCE + Dice loss.

        Args:
            logits: Tensor of shape (B, 1, ...) containing raw network outputs.
            target: Tensor of shape (B, ...) or (B, 1, ...).  Values must be 0
                or 1.  If a one‑hot encoded tensor with channel dimension is
                passed, only the first channel (foreground) is used.

        Returns:
            Scalar tensor representing the mean loss over the batch.
        """
        # Ensure target has shape (B, 1, ...)
        if target.ndim == logits.ndim - 1:
            # shape (B, ...)
            target_reshaped = target.unsqueeze(1)
        elif target.ndim == logits.ndim:
            # shape (B, 1, ...)
            target_reshaped = target[:, :1]
        else:
            raise ValueError(f"Unexpected target shape {target.shape} for logits shape {logits.shape}")

        b = logits.shape[0]
        # Compute Dice loss per sample.  MemoryEfficientSoftDiceLoss returns a
        # scalar averaged over the batch; to get per‑sample values we loop.
        # Looping over batch is acceptable because batch sizes are small in
        # medical imaging and Dice is relatively cheap compared to the rest of
        # nnU‑Net.  We detach intermediate computations to avoid
        # unnecessary graph building.
        dice_losses: List[torch.Tensor] = []
        for i in range(b):
            # select sample i.  We keep dims to match expected input shape (1, 1, ...)
            sample_logits = logits[i:i+1]
            sample_target = target_reshaped[i:i+1]
            dice_loss = self.dice(sample_logits, sample_target)
            # MemoryEfficientSoftDiceLoss returns negative DSC; convert to
            # positive loss by removing the sign
            dice_losses.append(dice_loss)
        dice_losses_tensor = torch.stack(dice_losses).flatten()  # shape (B,)

        # Compute BCE loss per voxel, then reduce per sample by mean.
        bce_per_voxel = self.bce(logits, target_reshaped.float())  # shape (B, 1, ...)
        # Reduce over channels and spatial dimensions
        spatial_dims = tuple(range(2, bce_per_voxel.ndim))
        bce_losses_tensor = bce_per_voxel.mean(dim=spatial_dims).squeeze(1)  # shape (B,)

        # Compute foreground ratio per sample: number of ones divided by total
        total_voxels = target_reshaped[0].numel()  # same for all samples
        # Sum over all dims except batch and channel
        fg_counts = target_reshaped.sum(dim=tuple(range(2, target_reshaped.ndim)))
        # fg_counts shape: (B, 1); convert to (B,) for ease
        fg_counts = fg_counts.squeeze(1).float()
        ratios = fg_counts / float(total_voxels)

        # Determine median ratio to split samples.  We detach to avoid backprop
        median_ratio = torch.median(ratios).item() if ratios.numel() > 0 else 0.0
        # Determine p_i: 1 if ratio >= median, else 0
        p = (ratios >= median_ratio).float()

        # Compute adaptive weights per sample
        dice_weights = self.initial_dice_weight + (1.0 - p) * self.g
        bce_weights = self.initial_bce_weight + p * self.g

        # Combine losses per sample
        total_losses = dice_weights * dice_losses_tensor + bce_weights * bce_losses_tensor
        # Return mean over batch
        return total_losses.mean()


def determine_if_binary(target: torch.Tensor) -> bool:
    """Helper to determine if the segmentation problem is binary.

    The nnUNet trainer passes the target either as a label map with shape
    (B, 1, ...) containing integer labels, or as a one‑hot encoded tensor with
    shape (B, C, ...).  If C == 1 then the task is binary.  If C > 1 it is
    multi‑class.  This helper inspects the tensor shape to detect binary
    segmentation.  It does not examine the values, so one‑hot encoded
    foreground/background (C == 2) is considered multi‑class.

    Args:
        target: A tensor provided to the loss function.

    Returns:
        True if binary, False otherwise.
    """
    if target.ndim < 2:
        return False
    # If second dimension is 1, treat as binary
    return target.shape[1] == 1