"""
COCO-person (Roboflow YOLOv8-seg export) dataset loader.

Expected layout after extracting the Roboflow zip:

    data/coco/
        train/
            images/*.jpg
            labels/*.txt
        valid/
            images/*.jpg
            labels/*.txt
        test/
            images/*.jpg
            labels/*.txt
        data.yaml

Label format (verified against actual export, NOT plain YOLO bbox format):
    Each line is a polygon segmentation, not a bbox:
        class_id x1 y1 x2 y2 x3 y3 ... xn yn
    All coordinates normalized to [0,1] relative to image width/height.
    One line per object instance; an image with N people has N lines.
    nc=1, single class "person" (class_id is always 0).

We convert each polygon to its axis-aligned bounding box (min/max over the
polygon's x's and y's) since the patch-attack pipeline only needs boxes,
not masks.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def parse_yolo_seg_annotation(path):
    """Parse a YOLOv8-seg label file into a list of (x1,y1,x2,y2) boxes,
    normalized to [0,1] (still relative to original image size — caller
    rescales to pixel coords).
    """
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 (x,y) pairs
                continue
            coords = list(map(float, parts[1:]))  # drop class_id (always 0)
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                continue
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


class CocoPersonDataset(Dataset):
    def __init__(self, root, split="train", img_size=416):
        """
        root: path to the extracted Roboflow dataset (contains train/valid/test).
        split: "train", "valid", or "test" — matches the folder names in
               data.yaml (note: "valid", not "val").
        """
        self.img_dir = os.path.join(root, split, "images")
        self.ann_dir = os.path.join(root, split, "labels")
        self.img_size = img_size

        self.samples = []
        for fname in sorted(os.listdir(self.img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ann_name = os.path.splitext(fname)[0] + ".txt"
            ann_path = os.path.join(self.ann_dir, ann_name)
            if os.path.exists(ann_path):
                self.samples.append((fname, ann_path))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {self.img_dir}. "
                f"Check that the dataset was extracted with the "
                f"train/valid/test + images/labels layout."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, ann_path = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # boxes come back normalized [0,1]; convert to original pixel coords
        norm_boxes = parse_yolo_seg_annotation(ann_path)
        boxes = [
            (x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h)
            for (x1, y1, x2, y2) in norm_boxes
        ]

        if not boxes:
            raise RuntimeError(
                f"No valid person boxes parsed from {ann_path} — "
                f"check the label file isn't empty/corrupt."
            )

        img_resized = img.resize((self.img_size, self.img_size))
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h
        boxes_resized = [
            (x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (x1, y1, x2, y2) in boxes
        ]

        tensor = TF.to_tensor(img_resized)  # (3,H,W) in [0,1]

        # pick the largest box (closest/most prominent person) to attack per-sample
        target_box = max(boxes_resized, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        return tensor, target_box, boxes_resized


def collate_single(batch):
    """Keep batch size effectively 1 for the compositing step (box geometry
    differs per image); trivial collate that just returns the list."""
    return batch
