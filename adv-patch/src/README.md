# Adversarial Patch Attack on YOLO

Trains a printable patch that suppresses person detection when composited
into a scene. Based on Thys et al. 2019 ("Fooling automated surveillance
cameras"), adapted for YOLOv8 via ultralytics.

## Setup

```bash
pip install -r requirements.txt
```

Extract the Roboflow COCO-person-only export to `data/coco/` with structure:
```
data/coco/
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
  test/images/*.jpg
  test/labels/*.txt
  data.yaml
```

Note: labels are YOLOv8-**segmentation** polygons (`class x1 y1 x2 y2 ...`),
not plain bboxes — `dataset.py` converts each polygon to its axis-aligned
bounding box. nc=1, single class "person".

## Train

```bash
cd src
python train.py --data ../data/coco --split train --epochs 20
```

Patches saved to `outputs/patch_epochN.png` each epoch — check visually that
it's converging to structured noise, not staying uniform gray (means loss
isn't flowing) or going to pure random static (TV weight too low).

## Evaluate

```bash
python eval.py --patch ../outputs/patch_epoch20.pt --data ../data/coco --split test
```

## Known gaps you'll need to close (this is a scaffold, not a finished repo)

1. **`get_raw_predictions` shape — verified.** Confirmed against a live
   `yolov8n.pt` (nc=80): raw output is `(1, 84, 3549)` at 416px input,
   matching the expected `(1, 4+num_classes, num_anchors)`. The
   objectness-padding branch in `get_raw_predictions` is dead-weight
   (v8 has no real objectness channel; it pads a constant 1, which
   makes `objectness * person_score` in loss.py reduce to just
   `person_score` — harmless but should be deleted for clarity, not
   left in as if it does something). Person class index 0 is correct
   for the frozen COCO-pretrained detector regardless of what
   person-only dataset is used for training images — the dataset's
   own class count doesn't affect this, since the detector weights
   are frozen and untouched. Sanity check still stands: after a few
   epochs, run eval.py and confirm patched recall actually drops.

2. **`objectness_loss`'s box-matching loop uses `.detach().tolist()`** to
   compute IoU for filtering, which is fine (IoU itself doesn't need
   gradients — only the score values being maximized need gradients, and
   they still carry the graph). But it's O(N) Python loop over every anchor
   per image (N=3549+ anchors at 416px input), run once per image per epoch.
   This dataset has 3721 train images — ~6x the ~600 INRIA was scaled for.
   Not vectorized yet; expect this loop to dominate epoch time. Vectorize
   the IoU computation (batch it with tensor ops instead of a Python loop)
   before running a full 20-epoch job, or budget for a much longer smoke
   test to see actual per-epoch wall time first.

3. **No test coverage.** Given code volume, actually run a 1-epoch,
   5-image smoke test before committing to a 20-epoch run overnight.

4. **Patch placement is fixed to "upper-middle of bbox"** (torso-ish). The
   paper found placement matters; you may want to sweep this as an ablation
   for your writeup rather than hardcoding one location.

## Suggested writeup structure (for the project README once it's done)

- Baseline recall vs. patched recall (the headline number)
- Patch visualization across training epochs
- Ablation: patch size vs. effectiveness
- Defense: adversarially-trained model recall with/without patch
- Failure cases: distances/angles where the patch stops working
