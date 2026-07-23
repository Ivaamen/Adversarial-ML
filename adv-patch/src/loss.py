"""
Loss functions for the patch attack.

objectness_loss: drives detector confidence for the target box toward zero
                  (or toward a wrong class, if targeted).
tv_loss:         total variation — keeps the patch spatially smooth so it's
                  printable and doesn't degenerate into single-pixel noise.
"""

import torch


def iou(box_a, box_b):
    """box: (x1,y1,x2,y2). Used to match raw predictions to the attacked box.
    Kept for reference/tests -- objectness_loss now uses iou_batch instead,
    since calling this per-anchor in a Python loop was the actual bottleneck
    (profiled at ~90% of total step time; see train.py --profile)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_batch(boxes_xyxy, target_box):
    """Vectorized IoU between every row of boxes_xyxy (N,4) and a single
    target_box (x1,y1,x2,y2), returning an (N,) tensor -- same math as
    iou() above, just batched instead of looped. This is the fix for the
    profiled bottleneck: one tensor op over all anchors instead of N
    Python-level calls (N ~3500+ for a 416x416 YOLOv8n).

    boxes_xyxy is expected to already be detached (caller's responsibility,
    matching the original semantics of "we don't need gradients through
    IoU") -- this function doesn't detach internally so it can also be used
    on plain tensors/tests without surprising side effects.
    """
    device = boxes_xyxy.device
    dtype = boxes_xyxy.dtype
    tb = torch.tensor(target_box, device=device, dtype=dtype)

    ax1, ay1, ax2, ay2 = boxes_xyxy.unbind(dim=1)
    bx1, by1, bx2, by2 = tb[0], tb[1], tb[2], tb[3]

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    # avoid div-by-zero the same way the scalar version's `if union > 0`
    # guard did -- degenerate/zero-area boxes get iou=0, not NaN
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def objectness_loss(raw_preds, target_box, person_class_id=0, iou_thresh=0.1, k=10):
    """
    raw_preds: pre-NMS model output, shape (N, 4+1+num_classes) typically —
               (x,y,w,h, objectness, class_scores...) for ultralytics YOLO,
               obtained by hooking the model *before* the NMS postprocessing
               step (see train.py for how this is extracted).
    target_box: the person bbox being attacked, in the same coord system.
    k: number of top-scoring anchors to average over, instead of taking a
       hard max. Fixes the whack-a-mole dynamic where only the single
       highest-scoring anchor got gradient each step -- see debugging log
       ("Root cause" #2). Every step now pushes gradient into up to k
       anchors at once.

    Returns (loss, n_kept):
        loss   - mean of the top-k (objectness * person_class_score) over
                 predictions that overlap target_box (or over all
                 predictions, in the no-overlap fallback below). This is
                 what we minimize.
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
    #
    # PERFORMANCE FIX: this used to be a Python for-loop calling iou() once
    # per anchor (profiled at ~114ms/step, ~90% of total step time, vs ~11ms
    # for the actual GPU model forward pass -- see train.py --profile output).
    # Replaced with one vectorized batched-tensor IoU call. Semantics are
    # unchanged: boxes are still detached before the IoU comparison (no
    # gradient through geometry, same as before), same xywh->xyxy conversion,
    # same threshold, same resulting index list -- only the implementation
    # is different, not the math or the returned keep set.
    boxes_detached = boxes.detach()
    bx = boxes_detached[:, 0]
    by = boxes_detached[:, 1]
    bw = boxes_detached[:, 2]
    bh = boxes_detached[:, 3]
    boxes_xyxy = torch.stack([
        bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2,
    ], dim=1)

    ious = iou_batch(boxes_xyxy, target_box)
    keep_idx = torch.nonzero(ious > iou_thresh, as_tuple=True)[0]

    if keep_idx.numel() == 0:
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
        kf = min(k, scores.numel())
        return torch.topk(scores, kf).values.mean(), 0

    relevant_scores = scores[keep_idx]

    kf = min(k, relevant_scores.numel())
    topk = torch.topk(relevant_scores, kf).values
    return topk.mean(), keep_idx.numel()


def tv_loss(patch):
    """Total variation loss — penalizes high-frequency noise, encourages
    smooth printable regions."""
    dx = torch.abs(patch[:, :, 1:] - patch[:, :, :-1]).sum()
    dy = torch.abs(patch[:, 1:, :] - patch[:, :-1, :]).sum()
    return dx + dy


def total_loss(raw_preds, target_box, patch, tv_weight=2.5e-3, person_class_id=0, k=10):
    obj_loss, n_kept = objectness_loss(raw_preds, target_box, person_class_id, k=k)
    tv = tv_loss(patch)
    return obj_loss + tv_weight * tv, obj_loss.item(), tv.item(), n_kept
