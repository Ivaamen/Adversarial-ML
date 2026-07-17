"""
Adversarial patch: init, EOT transforms, compositing onto images.

Core idea: the patch is a learnable tensor. Each training step we apply a
random affine transform (scale/rotation/location) + brightness/contrast jitter
before pasting it onto the image, so the learned patch generalizes across
viewing conditions instead of overfitting to one placement.
"""

import torch
import torch.nn.functional as F
import random
import math


def init_patch(size=300, mode="random"):
    """Initialize patch tensor, shape (3, size, size), values in [0,1]."""
    if mode == "random":
        patch = torch.rand(3, size, size)
    elif mode == "gray":
        patch = torch.full((3, size, size), 0.5)
    else:
        raise ValueError(f"unknown init mode: {mode}")
    patch.requires_grad_(True)
    return patch


def random_transform_params(
    img_h, img_w,
    scale_range=(0.15, 0.35),   # patch size as fraction of person bbox height
    rotation_range=(-20, 20),   # degrees
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
):
    scale = random.uniform(*scale_range)
    rotation = random.uniform(*rotation_range)
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    return {
        "scale": scale,
        "rotation": rotation,
        "brightness": brightness,
        "contrast": contrast,
    }


def apply_photometric(patch, brightness, contrast):
    """Jitter brightness/contrast. patch: (3,H,W) in [0,1]."""
    patch = patch * contrast + (brightness - 1.0)
    return torch.clamp(patch, 0, 1)


def rotate_patch(patch, angle_deg):
    """Rotate patch (and generate an alpha mask for the rotated corners)."""
    c, h, w = patch.shape
    angle = math.radians(angle_deg)
    theta = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
    ], dtype=patch.dtype).unsqueeze(0)

    grid = F.affine_grid(theta, [1, c, h, w], align_corners=False)
    rotated = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False, padding_mode="zeros")

    # alpha mask: 1 where patch content exists, 0 in the rotated-out corners
    ones = torch.ones(1, 1, h, w)
    alpha = F.grid_sample(ones, grid, align_corners=False, padding_mode="zeros")

    return rotated.squeeze(0), alpha.squeeze(0)


def paste_patch_on_box(image, patch, box, transform_params):
    """
    Paste a transformed patch onto `image` centered on `box`.

    image: (3, H, W) tensor in [0,1]
    patch: (3, P, P) tensor, requires_grad
    box:   (x1, y1, x2, y2) in pixel coords — the person bbox to attack
    transform_params: dict from random_transform_params

    Returns the composited image (differentiable w.r.t. patch).
    """
    _, img_h, img_w = image.shape
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    box_w = x2 - x1

    patch_size = max(8, int(box_h * transform_params["scale"]))

    patch_resized = F.interpolate(
        patch.unsqueeze(0), size=(patch_size, patch_size),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    patch_resized = apply_photometric(
        patch_resized, transform_params["brightness"], transform_params["contrast"]
    )

    patch_rot, alpha = rotate_patch(patch_resized, transform_params["rotation"])

    # place roughly on the torso: center-x of box, upper-middle of box height
    cx = x1 + box_w / 2
    cy = y1 + box_h * 0.35

    px1 = int(cx - patch_size / 2)
    py1 = int(cy - patch_size / 2)
    px2 = px1 + patch_size
    py2 = py1 + patch_size

    # clip to image bounds
    src_x1, src_y1 = max(0, -px1), max(0, -py1)
    px1c, py1c = max(0, px1), max(0, py1)
    px2c, py2c = min(img_w, px2), min(img_h, py2)

    if px2c <= px1c or py2c <= py1c:
        return image  # patch fell entirely off-frame, skip this step

    src_x2 = src_x1 + (px2c - px1c)
    src_y2 = src_y1 + (py2c - py1c)

    patch_crop = patch_rot[:, src_y1:src_y2, src_x1:src_x2]
    alpha_crop = alpha[:, src_y1:src_y2, src_x1:src_x2]

    out = image.clone()
    region = out[:, py1c:py2c, px1c:px2c]
    out[:, py1c:py2c, px1c:px2c] = region * (1 - alpha_crop) + patch_crop * alpha_crop

    return out


def eot_composite(image, patch, box, scale_range=(0.15, 0.35)):
    """Convenience wrapper: sample random transform, paste patch, return
    (composited_image, params). Params are returned (not just used
    internally) so callers can log what scale/rotation/etc was actually
    drawn each step -- needed to check whether loss behavior correlates
    with patch-to-person coverage ratio (occlusion) vs something else.

    scale_range is forwarded to random_transform_params so callers (train.py)
    can widen it via CLI to push more training mass into higher-coverage
    conditions, where the coverage-bucket analysis showed the biggest
    trained-vs-gray gap."""
    _, img_h, img_w = image.shape
    params = random_transform_params(img_h, img_w, scale_range=scale_range)
    return paste_patch_on_box(image, patch, box, params), params
