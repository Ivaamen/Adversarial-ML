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
import os
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from patch import init_patch, eot_composite
from loss import total_loss
from dataset import INRIAPersonDataset, collate_single


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
    parser.add_argument("--data", default="data/INRIAPerson")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=300)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.model)
    yolo.model.to(device).eval()
    for p in yolo.model.parameters():
        p.requires_grad_(False)  # freeze detector weights — only the patch trains

    dataset = INRIAPersonDataset(args.data, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    patch = init_patch(size=args.patch_size, mode="gray").to(device)
    patch.requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    for epoch in range(args.epochs):
        epoch_obj_loss, epoch_tv_loss, n = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            image, target_box, _all_boxes = batch[0]
            image = image.to(device)

            composited = eot_composite(image, patch, target_box)
            raw_preds = get_raw_predictions(yolo, composited.unsqueeze(0))

            loss, obj_l, tv_l = total_loss(raw_preds, target_box, patch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0, 1)

            epoch_obj_loss += obj_l
            epoch_tv_loss += tv_l
            n += 1
            pbar.set_postfix(obj=epoch_obj_loss / n, tv=epoch_tv_loss / n)

        # checkpoint the patch each epoch
        torch.save(patch.detach().cpu(), os.path.join(args.out, f"patch_epoch{epoch+1}.pt"))
        from torchvision.utils import save_image
        save_image(patch.detach().cpu(), os.path.join(args.out, f"patch_epoch{epoch+1}.png"))

    print(f"done. patches saved to {args.out}/")


if __name__ == "__main__":
    main()
