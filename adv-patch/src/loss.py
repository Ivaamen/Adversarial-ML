"""
Loss functions for the patch attack.

objectness_loss: drives detector confidence for the target box toward zero
                  (or toward a wrong class, if targeted).
tv_loss:         total variation — keeps the patch spatially smooth so it's
                  printable and doesn't degenerate into single-pixel noise.
"""

import torch


def iou(box_a, box_b):
    """box: (x1,y1,x2,y2). Used to match raw predictions to the attacked box."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def objectness_loss(raw_preds, target_box, person_class_id=0, iou_thresh=0.1):
    """
    raw_preds: pre-NMS model output, shape (N, 4+1+num_classes) typically —
               (x,y,w,h, objectness, class_scores...) for ultralytics YOLO,
               obtained by hooking the model *before* the NMS postprocessing
               step (see train.py for how this is extracted).
    target_box: the person bbox being attacked, in the same coord system.

    Returns (loss, n_kept):
        loss   - max (objectness * person_class_score) over predictions that
                 overlap target_box. This is what we minimize.
        n_kept - how many raw anchors passed the IoU filter. Callers should
                 log this: if it's 0 on a meaningful fraction of steps, the
                 "loss" for those steps is not connected to the patch at all
                 (see fallback below), and averaging it in with real steps
                 will mask a dead-gradient problem as a slow plateau.

    IMPORTANT: raw_preds must still require grad here (i.e. be the live
    output of get_raw_predictions on a graph rooted at `patch`, not a
    detached/copied tensor) or the fallback path below has nothing to
    attach to either.
    """
    if raw_preds.shape[0] == 0:
        # No anchors at all (shouldn't happen for YOLO, but just in case).
        # A fresh tensor here is FOR REAL disconnected from the patch -- this
        # previously used requires_grad=True on a leaf, which does NOT create
        # a path back to the patch. There is no way to get a gradient signal
        # out of literally zero predictions, so the honest thing is to make
        # that visible rather than fake a connected zero.
        return raw_preds.sum() * 0.0, 0

    boxes = raw_preds[:, :4]
    objectness = raw_preds[:, 4]
    class_scores = raw_preds[:, 5:]
    person_score = class_scores[:, person_class_id]

    scores = objectness * person_score

    # keep only predictions overlapping the attacked box (differentiable mask
    # via a hard filter is fine here since we don't need gradients through IoU)
    keep = []
    for i in range(boxes.shape[0]):
        b = boxes[i].detach().tolist()
        box_xyxy = (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)
        if iou(box_xyxy, target_box) > iou_thresh:
            keep.append(i)

    if not keep:
        # FIX: this was `torch.tensor(0.0, requires_grad=True)` — a brand new
        # leaf tensor with NO connection to raw_preds or patch whatsoever.
        # backward() on it is a no-op for patch.grad: nothing accumulates,
        # but the step still *looks* like a normal iteration in the tqdm log
        # (obj loss = 0.000 gets averaged into epoch_obj_loss like real data).
        #
        # Fix: fall back to the single highest-scoring anchor overall
        # (still connected to raw_preds -> patch through autograd), instead
        # of a disconnected constant. This keeps every step differentiable,
        # even the ones where nothing overlapped target_box tightly enough.
        # It's a softer signal ("push down whatever YOLO's most confident
        # about anywhere") rather than no signal at all.
        return scores.max(), 0

    keep_idx = torch.tensor(keep, device=raw_preds.device)
    relevant_scores = scores[keep_idx]

    return relevant_scores.max(), len(keep)


def tv_loss(patch):
    """Total variation loss — penalizes high-frequency noise, encourages
    smooth printable regions."""
    dx = torch.abs(patch[:, :, 1:] - patch[:, :, :-1]).sum()
    dy = torch.abs(patch[:, 1:, :] - patch[:, :-1, :]).sum()
    return dx + dy


def total_loss(raw_preds, target_box, patch, tv_weight=2.5e-3, person_class_id=0):
    obj_loss, n_kept = objectness_loss(raw_preds, target_box, person_class_id)
    tv = tv_loss(patch)
    return obj_loss + tv_weight * tv, obj_loss.item(), tv.item(), n_kept
