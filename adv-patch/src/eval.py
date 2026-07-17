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
    """Run full (postprocessed) inference and return True if a person is detected."""
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    results = yolo_model.predict(img_np, verbose=False, conf=conf_thresh)
    for r in results:
        for cls_id in r.boxes.cls.tolist():
            if int(cls_id) == person_class_id:
                return True
    return False


def evaluate(yolo, dataset, patch=None, n_trials_per_image=3):
    """
    patch=None -> baseline (clean) recall.
    patch=tensor -> recall with patch applied at random EOT transforms,
                    averaged over n_trials_per_image placements per image.
    """
    detected, total = 0, 0

    for image, target_box, _ in tqdm(dataset, desc="evaluating"):
        for _ in range(n_trials_per_image if patch is not None else 1):
            if patch is not None:
                params = random_transform_params(image.shape[1], image.shape[2])
                test_img = paste_patch_on_box(image, patch, target_box, params)
            else:
                test_img = image

            if detect_person(yolo, test_img):
                detected += 1
            total += 1

    recall = detected / total if total > 0 else 0.0
    return recall


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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CocoPersonDataset(args.data, split=args.split)
    patch = torch.load(args.patch).to(device)

    yolo = YOLO(args.model)

    print("\n=== Baseline model ===")
    clean_recall = evaluate(yolo, dataset, patch=None, n_trials_per_image=args.n_trials)
    patched_recall = evaluate(yolo, dataset, patch=patch, n_trials_per_image=args.n_trials)
    print(f"Recall (no patch):   {clean_recall:.3f}")
    print(f"Recall (with patch): {patched_recall:.3f}")
    print(f"Drop:                {clean_recall - patched_recall:.3f}")

    if args.adv_trained_model:
        print("\n=== Adversarially-trained model (defense) ===")
        yolo_defended = YOLO(args.adv_trained_model)
        defended_clean_recall = evaluate(yolo_defended, dataset, patch=None, n_trials_per_image=args.n_trials)
        defended_patched_recall = evaluate(yolo_defended, dataset, patch=patch, n_trials_per_image=args.n_trials)
        print(f"Recall (no patch):   {defended_clean_recall:.3f}")
        print(f"Recall (with patch): {defended_patched_recall:.3f}")
        print(f"Drop:                {defended_clean_recall - defended_patched_recall:.3f}")
        print(f"\nRobustness gain from adv. training: "
              f"{(defended_patched_recall - patched_recall):.3f} recall recovered")


if __name__ == "__main__":
    main()
