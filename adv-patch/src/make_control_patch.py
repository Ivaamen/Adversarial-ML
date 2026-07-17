"""
Generates a non-adversarial control patch (flat gray, same size as your
trained patch) so you can isolate occlusion effects from adversarial effects.

Place this file in your src/ directory (needs patch.py importable), then run:
    python make_control_patch.py --patch-size 300 --out ../outputs/patch_control_gray.pt
"""
import argparse
import torch
from patch import init_patch

parser = argparse.ArgumentParser()
parser.add_argument("--patch-size", type=int, default=300)
parser.add_argument("--out", default="patch_control_gray.pt")
args = parser.parse_args()

# mode="gray" -> flat 0.5 gray, no adversarial content, no gradient history
patch = init_patch(size=args.patch_size, mode="gray").detach()
torch.save(patch, args.out)
print(f"saved control patch to {args.out}, shape {patch.shape}")
