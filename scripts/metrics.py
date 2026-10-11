"""Shared detection metrics, so Faster R-CNN and the YOLO baseline are scored identically.

preds: list (one per image) of (boxes Nx4 xyxy, scores N) numpy arrays
gts:   list (one per image) of boxes Mx4 xyxy numpy arrays
"""
import numpy as np


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(preds, gts, iou_thr=0.5):
    """Greedy matching per image, highest score first. Returns (scores, is_tp, n_gt)."""
    all_s, all_tp = [], []
    for (boxes, scores), gt in zip(preds, gts):
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        iou = box_iou(boxes, gt)
        used = np.zeros(len(gt), bool)
        tp = np.zeros(len(boxes), bool)
        for i in range(len(boxes)):
            if len(gt) == 0:
                break
            cand = np.where(~used & (iou[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = tp[i] = True
        all_s.append(scores)
        all_tp.append(tp)
    return np.concatenate(all_s), np.concatenate(all_tp), sum(len(g) for g in gts)


def average_precision(preds, gts, iou_thr=0.5):
    """Area under the PR curve, COCO 101-point interpolation (Eq. 3.14 with C=1)."""
    return ap_from_matches(*match(preds, gts, iou_thr))


def ap_from_matches(scores, tp, n_gt):
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    ctp = np.cumsum(tp[order])
    cfp = np.cumsum(~tp[order])
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    out = []
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        out.append(precision[idx] if idx < len(precision) else 0.0)
    return float(np.mean(out))


def map50_95(preds, gts):
    return float(np.mean([average_precision(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)]))


def counts_at(preds, gts, conf, iou_thr=0.5):
    """Lesion-level TP/FP/FN and image-level TP/FP/TN/FN at a confidence threshold.

    Image level: an image is positive if it has any ground-truth caries, and predicted
    positive if any box scores >= conf. This gives specificity (Eq. 3.10) a defined TN.
    """
    kept = [(b[s >= conf], s[s >= conf]) for b, s in preds]
    _, tp, n_gt = match(kept, gts, iou_thr)
    les = dict(TP=int(tp.sum()), FP=int((~tp).sum()), FN=int(n_gt - tp.sum()))
    img = dict(TP=0, FP=0, TN=0, FN=0)
    for (b, _), g in zip(kept, gts):
        key = ("T" if (len(b) > 0) == (len(g) > 0) else "F") + ("P" if len(b) > 0 else "N")
        img[key] += 1
    return les, img


def summary(preds, gts, conf=0.25):
    les, img = counts_at(preds, gts, conf)
    p = les["TP"] / max(les["TP"] + les["FP"], 1)
    r = les["TP"] / max(les["TP"] + les["FN"], 1)
    return dict(
        mAP50=average_precision(preds, gts),
        mAP50_95=map50_95(preds, gts),
        precision=p,
        sensitivity=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
        img_sensitivity=img["TP"] / max(img["TP"] + img["FN"], 1),
        img_specificity=img["TN"] / max(img["TN"] + img["FP"], 1),
        **{f"les_{k}": v for k, v in les.items()},
        **{f"img_{k}": v for k, v in img.items()},
    )


def best_f1_conf(preds, gts, iou_thr=0.5):
    """Confidence threshold that maximises lesion-level F1. Pick it on validation, apply to test."""
    scores, tp, n_gt = match(preds, gts, iou_thr)
    order = np.argsort(-scores)
    ctp = np.cumsum(tp[order])
    f1 = 2 * ctp / (np.arange(1, len(ctp) + 1) + n_gt)
    return float(scores[order][np.argmax(f1)])
