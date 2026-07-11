"""
INRIA Person dataset loader.

Download: http://pascal.inrialpes.fr/data/human/ (or the common mirror on
Kaggle: "INRIAPerson"). Expected layout after extraction:

    data/INRIAPerson/
        Train/
            pos/         # images containing people
            annotations/ # one .txt per image, INRIA format bounding boxes

We only need positive images + their person boxes — the patch is pasted onto
each labeled person and trained to suppress detection there.
"""

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def parse_inria_annotation(path):
    """INRIA annotation files list bounding boxes like:
    'Bounding box for object 1 "PASperson" (x1, y1) - (x2, y2)'
    """
    boxes = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            m = re.search(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", line)
            if m:
                x1, y1, x2, y2 = map(int, m.groups())
                boxes.append((x1, y1, x2, y2))
    return boxes


class INRIAPersonDataset(Dataset):
    def __init__(self, root, split="Train", img_size=416):
        self.img_dir = os.path.join(root, split, "pos")
        self.ann_dir = os.path.join(root, split, "annotations")
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
                "Check that INRIAPerson has been downloaded and extracted."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, ann_path = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        boxes = parse_inria_annotation(ann_path)

        img_resized = img.resize((self.img_size, self.img_size))
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h
        boxes_resized = [
            (x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (x1, y1, x2, y2) in boxes
        ]

        tensor = TF.to_tensor(img_resized)  # (3,H,W) in [0,1]

        # pick the largest box (closest/most prominent person) to attack per-sample
        target_box = max(boxes_resized, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

        return tensor, target_box, boxes_resized


def collate_single(batch):
    """Keep batch size effectively 1 for the compositing step (box geometry
    differs per image); trivial collate that just returns the list."""
    return batch
