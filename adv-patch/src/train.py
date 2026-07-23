"""
Train an adversarial patch against a frozen YOLOv8 model.

Usage:
    python train.py --data data/INRIAPerson --epochs 20 --patch-size 300

Key point on extracting raw (pre-NMS) predictions: ultralytics wraps the
model with its own postprocessing (NMS etc). We need logits BEFORE NMS
because NMS is non-differentiable and depends on model.eval() internals
that don't expose gradients cleanly. We call model.model (the underlying
nn.Module) directly to get raw output.
"""

import argparse
import csv
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from patch import init_patch, eot_composite
from loss import total_loss
from dataset import CocoPersonDataset, collate_single


def get_raw_predictions(yolo_model, image_batch):
    """
    image_batch: (1, 3, H, W) tensor in [0,1], already resized to model's
    expected input size.

    Returns raw predictions reshaped to (N, 5+num_classes): x,y,w,h,obj,classes.
    ultralytics v8 raw output shape is (1, 4+num_classes, num_anchors) with
    objectness folded into class scores in some versions — verify against your
    installed ultralytics version's Detect head output before trusting this
    blindly; print raw.shape once and adjust the reshape/permute below if it
    doesn't match.
    """
    raw = yolo_model.model(image_batch)  # underlying nn.Module, bypasses NMS
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    # expected: (1, 4+num_classes, num_anchors) -> transpose to (num_anchors, 4+num_classes)
    raw = raw[0].transpose(0, 1)

    # ultralytics v8 detect head has no separate objectness channel (it's
    # class-scores only, already sigmoid'd in some export paths); if your
    # version differs, insert a dummy objectness=1 column so loss.py's
    # indexing (obj at col 4, classes from col 5) still lines up:
    if raw.shape[1] == 4 + 80:  # no explicit objectness column present
        obj_col = torch.ones(raw.shape[0], 1, device=raw.device)
        raw = torch.cat([raw[:, :4], obj_col, raw[:, 4:]], dim=1)

    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/coco")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=300)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--tv-weight", type=float, default=2.5e-3,
                         help="Weight on the total-variation smoothness "
                              "penalty. Lower = less pressure toward smooth/"
                              "printable patches, more room for the "
                              "optimizer to find aggressive high-frequency "
                              "patterns. Tradeoff: too low risks converging "
                              "to unprintable single-pixel noise (the exact "
                              "failure mode tv_loss exists to prevent).")
    parser.add_argument("--scale-min", type=float, default=0.15,
                         help="Min patch size as fraction of person bbox "
                              "height, sampled per EOT step.")
    parser.add_argument("--scale-max", type=float, default=0.35,
                         help="Max patch size as fraction of person bbox "
                              "height, sampled per EOT step. Coverage-bucket "
                              "analysis showed the trained-vs-gray gap peaks "
                              "around coverage_ratio 0.25-0.5; raising this "
                              "shifts more training steps into that regime. "
                              "NOTE: this changes what the patch is being "
                              "optimized for -- a patch trained at larger "
                              "scale is a different (less realistic as a "
                              "small torso patch) attack, not just a better-"
                              "trained version of the same one.")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed. Doesn't change the science, just makes "
                              "'did my code change' vs 'did the random draw change' "
                              "answerable while debugging.")
    parser.add_argument("--gray-baseline", action="store_true",
                         help="Run the same loop with a fixed flat-gray patch "
                              "(no training, no optimizer step) instead of "
                              "learning a patch. Use with the SAME --seed as a "
                              "real training run so the image order and EOT "
                              "param draws (scale/rotation/etc) line up step "
                              "for step -- this gives a matched comparison: "
                              "same coverage_ratio at the same step means same "
                              "image+box+scale, so any obj_loss difference at "
                              "matched coverage is attributable to patch "
                              "CONTENT (adversarial pattern vs gray), not to "
                              "which images/placements got sampled.")
    parser.add_argument("--profile", action="store_true",
                         help="Time each pipeline segment per step (EOT "
                              "compositing, model forward pass, loss+IoU-loop "
                              "+backward) and print a summary at the end. "
                              "Uses torch.cuda.synchronize() around GPU timing "
                              "since CUDA calls are async -- without syncing, "
                              "a naive time.time() only measures kernel launch "
                              "overhead, not actual execution time. Adds sync "
                              "overhead of its own, so don't leave this on for "
                              "real training runs, only for one diagnostic run.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.model)
    yolo.model.to(device).eval()
    for p in yolo.model.parameters():
        p.requires_grad_(False)  # freeze detector weights — only the patch trains

    dataset = CocoPersonDataset(args.data, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    # init_patch() already sets requires_grad=True on the CPU tensor. Calling
    # .to(device) on a tensor that already requires grad produces a NON-LEAF
    # tensor (the result of the device-copy op), which torch.optim.Adam
    # refuses to optimize -- "can't optimize a non-leaf Tensor". This is
    # silent/harmless on CPU (where .to() on a no-op device transfer may
    # return the same underlying leaf) but throws on GPU, where .to('cuda')
    # is always a real op. Fix: move first, then detach + re-enable grad so
    # the tensor becomes a genuine leaf at its final device.
    patch = init_patch(size=args.patch_size, mode="gray").to(device).detach()
    patch.requires_grad_(True)
    if args.gray_baseline:
        # Fixed gray, never updated -- optimizer is created but never
        # .step()'d, so the patch stays at its initial gray value throughout.
        optimizer = torch.optim.Adam([patch], lr=args.lr)  # unused, never .step()'d
    else:
        optimizer = torch.optim.Adam([patch], lr=args.lr)

    # Per-step diagnostic log (CSV, written incrementally, flushed every step).
    # This is the thing that would have caught the dead-gradient bug in ~10
    # steps instead of 5 epochs: grad_mean/grad_max show whether gradients are
    # healthy, and n_kept/empty_keep show whether objectness_loss is actually
    # connected to the current step's prediction, or silently a no-op.
    log_name = "train_log_gray_baseline.csv" if args.gray_baseline else "train_log.csv"
    log_path = os.path.join(args.out, log_name)
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "step", "obj_loss", "tv_loss", "n_kept", "empty_keep",
        "grad_mean", "grad_max", "grad_is_none",
        "box_h", "box_w", "scale", "patch_size", "coverage_ratio",
    ])

    global_step = 0
    # profiling accumulators (only used if --profile)
    prof_totals = {"eot": 0.0, "forward": 0.0, "loss_backward": 0.0}
    prof_steps = 0

    for epoch in range(args.epochs):
        epoch_obj_loss, epoch_tv_loss, n = 0.0, 0.0, 0
        epoch_empty_keep = 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            image, target_box, _all_boxes = batch[0]
            image = image.to(device)

            if args.profile and device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            composited, eot_params = eot_composite(
                image, patch, target_box,
                scale_range=(args.scale_min, args.scale_max),
            )

            if args.profile and device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            raw_preds = get_raw_predictions(yolo, composited.unsqueeze(0))

            if args.profile and device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            loss, obj_l, tv_l, n_kept = total_loss(
                raw_preds, target_box, patch, tv_weight=args.tv_weight,
            )

            optimizer.zero_grad()
            loss.backward()

            if args.profile and device == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            if args.profile:
                prof_totals["eot"] += (t1 - t0)
                prof_totals["forward"] += (t2 - t1)
                # this segment includes total_loss (with its Python-level IoU
                # loop over every raw anchor) AND backward() -- both are CPU-
                # side/autograd-graph work, lumped together deliberately since
                # separating them needs another sync point and total_loss's
                # cost (the suspected bottleneck) dominates this segment
                # anyway for a frozen, tiny (nano) detector.
                prof_totals["loss_backward"] += (t3 - t2)
                prof_steps += 1

            empty_keep = int(n_kept == 0)
            if patch.grad is None:
                grad_mean, grad_max, grad_is_none = 0.0, 0.0, 1
            else:
                grad_mean = patch.grad.abs().mean().item()
                grad_max = patch.grad.abs().max().item()
                grad_is_none = 0

            # box/coverage diagnostics -- to test whether loss is driven by
            # patch-to-person coverage ratio (occlusion) rather than the
            # adversarial pattern itself. patch_size mirrors the exact
            # computation in patch.py's paste_patch_on_box: box_h * scale.
            x1, y1, x2, y2 = target_box
            box_h = y2 - y1
            box_w = x2 - x1
            scale = eot_params["scale"]
            patch_size = max(8, int(box_h * scale))
            # fraction of the person's bbox area nominally covered by the
            # (pre-rotation-clipping) square patch footprint
            coverage_ratio = (patch_size * patch_size) / max(1e-6, (box_h * box_w))

            log_writer.writerow([
                epoch + 1, global_step, obj_l, tv_l, n_kept, empty_keep,
                grad_mean, grad_max, grad_is_none,
                box_h, box_w, scale, patch_size, coverage_ratio,
            ])
            log_file.flush()

            if not args.gray_baseline:
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)

            epoch_obj_loss += obj_l
            epoch_tv_loss += tv_l
            epoch_empty_keep += empty_keep
            n += 1
            global_step += 1
            pbar.set_postfix(
                obj=epoch_obj_loss / n,
                tv=epoch_tv_loss / n,
                empty_keep_pct=100.0 * epoch_empty_keep / n,
                grad_mean=grad_mean,
            )

        # checkpoint the patch each epoch (skip for gray-baseline: it's a
        # fixed patch that never changes, saving it every epoch is pointless)
        if not args.gray_baseline:
            torch.save(patch.detach().cpu(), os.path.join(args.out, f"patch_epoch{epoch+1}.pt"))
            from torchvision.utils import save_image
            save_image(patch.detach().cpu(), os.path.join(args.out, f"patch_epoch{epoch+1}.png"))

        print(
            f"epoch {epoch+1}: obj={epoch_obj_loss/n:.4f} tv={epoch_tv_loss/n:.4f} "
            f"empty_keep={100.0*epoch_empty_keep/n:.1f}% of steps"
        )

    log_file.close()

    if args.profile and prof_steps > 0:
        total = sum(prof_totals.values())
        print("\n=== per-step timing breakdown (--profile) ===")
        print(f"steps measured: {prof_steps}")
        for name in ["eot", "forward", "loss_backward"]:
            avg_ms = 1000 * prof_totals[name] / prof_steps
            pct = 100 * prof_totals[name] / total
            print(f"  {name:14s}: avg {avg_ms:7.2f} ms/step  ({pct:5.1f}% of measured time)")
        print(f"  {'TOTAL':14s}: avg {1000*total/prof_steps:7.2f} ms/step")
        print(
            "note: 'loss_backward' includes total_loss's Python-level IoU "
            "loop over every raw anchor (see loss.py objectness_loss) as "
            "well as the actual backward() call. If this segment dominates "
            "and 'forward' is small by comparison, the IoU loop -- not the "
            "GPU model pass -- is the bottleneck, and vectorizing it (batched "
            "tensor IoU instead of a per-anchor Python loop) is the fix, not "
            "a faster GPU."
        )

    print(f"done. patches saved to {args.out}/  step log: {log_path}")


if __name__ == "__main__":
    main()
