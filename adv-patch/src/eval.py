"""
Evaluate patch effectiveness: person-detection recall with vs. without the
patch applied, across the held-out set. This is the number that goes in
your writeup / README — not just "look, it works on this one image."

Usage:
    python eval.py --patch outputs/patch_epoch20.pt --data data/coco --split test
"""

import argparse
import random
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from patch import paste_patch_on_box, random_transform_params
from dataset import CocoPersonDataset, collate_single


def detect_person(yolo_model, image_tensor, conf_thresh=0.25, person_class_id=0):
    """Run full (postprocessed) inference. Returns (detected, max_person_conf):
      detected       - True if a person is detected above conf_thresh (same
                        as before, used for the recall metric).
      max_person_conf - the highest confidence score among ANY person-class
                        box YOLO reported, even below conf_thresh (0.0 if
                        none at all). This is a continuous signal that the
                        binary detected/not-detected recall metric throws
                        away: a patch that reliably drops confidence from
                        0.9 to 0.3 look identical to one that barely moves
                        it, if both still cross conf_thresh -- or identical
                        to "no effect" if conf_thresh itself isn't crossed
                        either way. Logging this lets you tell those cases
                        apart instead of only seeing pass/fail.

    Note: yolo_model.predict is called with conf=0.001 here (not conf_thresh)
    so YOLO's own postprocessing doesn't discard low-confidence person boxes
    before we get a chance to read their score -- conf_thresh is applied
    manually below instead, only for the `detected` boolean.
    """
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    results = yolo_model.predict(img_np, verbose=False, conf=0.001)
    max_conf = 0.0
    for r in results:
        for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(cls_id) == person_class_id:
                max_conf = max(max_conf, conf)
    detected = max_conf >= conf_thresh
    return detected, max_conf


def evaluate(yolo, dataset, patch=None, n_trials_per_image=3, device="cpu",
             scale_range=(0.15, 0.35)):
    """
    patch=None -> baseline (clean) recall.
    patch=tensor -> recall with patch applied at random EOT transforms,
                    averaged over n_trials_per_image placements per image.

    scale_range MUST match (or at least cover) whatever range the patch was
    actually trained at (train.py's --scale-min/--scale-max). Previously this
    silently used random_transform_params's hardcoded default (0.15, 0.35)
    regardless of what the patch was trained for -- so a patch trained at a
    wider range (e.g. up to 0.5) was only ever being tested in a narrower
    slice of conditions than it was optimized for, systematically
    undercounting its real effectiveness.

    Returns (recall, mean_conf, median_conf, all_confs):
      recall     - same binary detected/not metric as before.
      mean_conf  - average max-person-confidence across all trials, even
                   ones where conf_thresh wasn't crossed. This is the
                   continuous signal recall alone discards -- two patches
                   can have identical recall but very different mean_conf,
                   which tells you one is suppressing detection much more
                   aggressively even if it doesn't (yet) flip the binary
                   outcome.
      median_conf - less sensitive to a few outlier high-confidence misses
                   than the mean; useful if confidence is skewed rather
                   than roughly symmetric.
      all_confs  - raw list, in case you want to plot the distribution or
                   bucket it against coverage_ratio the way train_log.csv
                   already lets you do for training loss.
    """
    detected, total = 0, 0
    all_confs = []

    for image, target_box, _ in tqdm(dataset, desc="evaluating"):
        # dataset yields CPU tensors; move to match patch's device (which may
        # be CUDA) before compositing, or paste_patch_on_box's alpha-blend
        # will crash mixing CPU `image` with CUDA `patch_crop`/`alpha_crop`.
        image = image.to(device)
        for _ in range(n_trials_per_image if patch is not None else 1):
            if patch is not None:
                params = random_transform_params(
                    image.shape[1], image.shape[2], scale_range=scale_range,
                )
                test_img = paste_patch_on_box(image, patch, target_box, params)
            else:
                test_img = image

            is_detected, conf = detect_person(yolo, test_img)
            all_confs.append(conf)
            if is_detected:
                detected += 1
            total += 1

    recall = detected / total if total > 0 else 0.0
    mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    sorted_confs = sorted(all_confs)
    median_conf = sorted_confs[len(sorted_confs) // 2] if sorted_confs else 0.0
    return recall, mean_conf, median_conf, all_confs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True)
    parser.add_argument("--data", default="data/coco")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--adv-trained-model", default=None,
                         help="optional: path to a YOLO checkpoint fine-tuned "
                              "on patched images, for the defense comparison")
    parser.add_argument("--n-trials", type=int, default=3,
                         help="number of random EOT placements averaged per image "
                              "when evaluating with a patch")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed so repeated eval runs draw the same EOT "
                              "placements — makes runs comparable for debugging. "
                              "Does not remove real-world placement variance.")
    parser.add_argument("--scale-min", type=float, default=0.15,
                         help="Min patch size as fraction of person bbox height, "
                              "sampled per EOT trial. Should match (or cover) "
                              "whatever range the patch was trained at -- "
                              "otherwise eval tests the patch outside the "
                              "conditions it was optimized for and will "
                              "understate its real effectiveness.")
    parser.add_argument("--scale-max", type=float, default=0.35,
                         help="Max patch size as fraction of person bbox height. "
                              "MUST match train.py's --scale-max for the patch "
                              "being evaluated, or you're testing a different "
                              "coverage regime than it was trained for.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CocoPersonDataset(args.data, split=args.split)
    patch = torch.load(args.patch).to(device).detach()

    yolo = YOLO(args.model)

    print("\n=== Baseline model ===")
    scale_range = (args.scale_min, args.scale_max)
    clean_recall, clean_mean_conf, clean_med_conf, _ = evaluate(
        yolo, dataset, patch=None, n_trials_per_image=args.n_trials, device=device, scale_range=scale_range)
    patched_recall, patched_mean_conf, patched_med_conf, _ = evaluate(
        yolo, dataset, patch=patch, n_trials_per_image=args.n_trials, device=device, scale_range=scale_range)
    print(f"Recall (no patch):   {clean_recall:.3f}   mean_conf={clean_mean_conf:.3f}  median_conf={clean_med_conf:.3f}")
    print(f"Recall (with patch): {patched_recall:.3f}   mean_conf={patched_mean_conf:.3f}  median_conf={patched_med_conf:.3f}")
    print(f"Drop:                {clean_recall - patched_recall:.3f}")
    print(f"Mean confidence drop: {clean_mean_conf - patched_mean_conf:.3f}  "
          f"(continuous signal -- can differ from the binary recall drop above; "
          f"a bigger confidence drop with a similar recall drop means the patch "
          f"is suppressing detection more aggressively even where it doesn't "
          f"flip the pass/fail outcome)")

    if args.adv_trained_model:
        print("\n=== Adversarially-trained model (defense) ===")
        yolo_defended = YOLO(args.adv_trained_model)
        defended_clean_recall, defended_clean_mean_conf, _, _ = evaluate(
            yolo_defended, dataset, patch=None, n_trials_per_image=args.n_trials, device=device, scale_range=scale_range)
        defended_patched_recall, defended_patched_mean_conf, _, _ = evaluate(
            yolo_defended, dataset, patch=patch, n_trials_per_image=args.n_trials, device=device, scale_range=scale_range)
        print(f"Recall (no patch):   {defended_clean_recall:.3f}")
        print(f"Recall (with patch): {defended_patched_recall:.3f}")
        print(f"Drop:                {defended_clean_recall - defended_patched_recall:.3f}")
        print(f"\nRobustness gain from adv. training: "
              f"{(defended_patched_recall - patched_recall):.3f} recall recovered")


if __name__ == "__main__":
    main()
