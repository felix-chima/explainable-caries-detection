"""Explainability faithfulness study (Chapter 4.7): are the heatmaps faithful to the model?

Faster R-CNN (primary): Grad-CAM, HiResCAM, EigenCAM and RISE. The gradient methods and RISE
explain the same
quantity, T(x) = caries probability that the RoI head gives a fixed box (RPN bypassed), so
the methods are compared on one target. EigenCAM is class-agnostic (first principal
component of the FPN features). YOLOv8s: EigenCAM on the neck outputs (P3-P5).

Per explained true-positive detection:
  localisation  pointing game, share of heatmap energy in the box, IoU of heatmap >= 0.5
                with the box (Chapter 3's Grad-CAM IoU, against boxes because DENTEX has no
                masks), chance level = box area / image area
  faithfulness  deletion and insertion AUC (Petsiuk et al., 2018), blurred baseline,
                against a random-heatmap baseline
  sanity        model-parameter randomisation (Adebayo et al., 2018): Spearman correlation
                between heatmaps of the trained and progressively randomised model

Resumable: per-image results are cached in runs/explain/cache/.

  python scripts/explain.py                 # 20 test images, CPU
  python scripts/explain.py --report-only   # tables and figures from the cache
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from metrics import box_iou  # noqa: E402
from train_frcnn import DATA, ROOT, Dentex, build_model  # noqa: E402

OUT = ROOT / "runs/explain"
EVAL = ROOT / "runs/eval"
DS = 4  # heatmaps are kept at 1/4 of the original resolution
STAGES = {"heads": ["roi_heads.box_head", "roi_heads.box_predictor"],
          "heads+FPN": ["roi_heads.box_head", "roi_heads.box_predictor", "backbone.fpn"],
          "all layers": ["roi_heads", "backbone.fpn", "backbone.body"]}
METHOD_ORDER = ["Grad-CAM", "HiResCAM", "EigenCAM", "RISE", "Random", "Random (in box)", "YOLO EigenCAM", "YOLO Random"]
COLOR = {"Grad-CAM": "#2a78d6", "HiResCAM": "#e34948", "EigenCAM": "#1baf7a", "RISE": "#4a3aa7", "Random": "#8a8984", "Random (in box)": "#52514e",
         "YOLO EigenCAM": "#eb6834", "YOLO Random": "#c3c2b7"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def norm(a):
    a = a - a.min()
    return a / (a.max() + 1e-12)


def blurred(x):
    h, w = x.shape[-2:]
    return F.interpolate(F.interpolate(x[None], scale_factor=1 / 32, mode="area"), size=(h, w),
                         mode="bilinear", align_corners=False)[0]


# ---------- Faster R-CNN ----------

class BoxScore:
    """T(x): caries logit margin from the RoI head for fixed boxes, with the FPN features."""

    def __init__(self, model):
        self.m = model

    def __call__(self, x, boxes):
        il, _ = self.m.transform([x])
        (h, w), (H, W) = il.image_sizes[0], x.shape[-2:]
        scale = torch.tensor([w / W, h / H, w / W, h / H], dtype=x.dtype, device=x.device)
        feats = self.m.backbone(il.tensors)
        rh = self.m.roi_heads
        logits, _ = rh.box_predictor(rh.box_head(rh.box_roi_pool(feats, [boxes * scale], il.image_sizes)))
        info = dict(hw=(h, w), padded=tuple(il.tensors.shape[-2:]), out=(H // DS, W // DS))
        return logits[:, 1] - logits[:, 0], feats, info

    def prob(self, x, boxes):
        with torch.no_grad():
            return torch.sigmoid(self(x, boxes)[0]).cpu().numpy()


def to_image(maps, info):
    (ph, pw), (h, w) = info["padded"], info["hw"]
    acc = sum(F.interpolate(m.float(), size=(ph, pw), mode="bilinear", align_corners=False)[..., :h, :w]
              for m in maps)
    return norm(F.interpolate(acc, size=info["out"], mode="bilinear", align_corners=False)[0, 0].cpu().numpy())


def gradcam(T, x, box, hires=False):
    """Grad-CAM (channel weights = spatially averaged gradients) or HiResCAM (Draelos & Carin,
    2020: element-wise gradient x activation, which keeps the RoI-local gradient signal)."""
    with torch.enable_grad():  # weights are frozen, so let gradients flow from the input
        margin, feats, info = T(x.detach().clone().requires_grad_(True), box[None])
        levels = [feats[k] for k in "0123"]
        for f in levels:
            f.retain_grad()
        T.m.zero_grad()
        margin[0].backward()
    cams = [F.relu(((f.grad if hires else f.grad.mean((2, 3), keepdim=True)) * f).sum(1, keepdim=True)).detach()
            for f in levels if f.grad is not None]
    return to_image(cams, info) if cams else np.zeros(info["out"])


def eigencam_from(feats, info):
    ref = feats["1"].shape[-2:]
    a = torch.cat([F.interpolate(feats[k], size=ref, mode="bilinear", align_corners=False) for k in "0123"], 1)[0]
    A = a.reshape(a.shape[0], -1).T.double()
    A = A - A.mean(0)
    v = torch.linalg.eigh(A.T @ A).eigenvectors[:, -1]
    proj = (A @ v).reshape(1, 1, *ref)
    if torch.corrcoef(torch.stack([proj.flatten(), a.mean(0).flatten().double()]))[0, 1] < 0:
        proj = -proj  # principal component sign is arbitrary; orient it with the mean activation
    return to_image([proj], info)


def eigencam(T, x):
    with torch.no_grad():
        _, feats, info = T(x, torch.zeros(1, 4, device=x.device))
    return eigencam_from(feats, info)


def rise(T, x, boxes, n, gen, p=0.5, s=16):
    H, W = x.shape[-2:]
    h, w = H // DS, W // DS
    base = blurred(x)
    ch, cw = -(-h // s), -(-w // s)
    sal = np.zeros((len(boxes), h, w))
    for _ in range(n):
        grid = (torch.rand(1, 1, s, s, generator=gen) < p).float()
        up = F.interpolate(grid, size=((s + 1) * ch, (s + 1) * cw), mode="bilinear", align_corners=False)
        dy, dx = (int(torch.randint(0, c, (1,), generator=gen)) for c in (ch, cw))
        m = up[..., dy:dy + h, dx:dx + w]
        mf = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0].to(x.device)
        sal += T.prob(x * mf + base * (1 - mf), boxes)[:, None, None] * m[0, 0].numpy()
    return [norm(a) for a in sal / (n * p)]


def deletion_insertion(score_fn, x_base, x_full, sal, steps, blend):
    """Scores as the most salient pixels are removed (deletion) or restored (insertion)."""
    order = np.argsort(-sal.ravel())
    dels, ins = [], []
    for k in range(steps + 1):
        m = np.zeros(sal.size, np.float32)
        m[order[: int(k / steps * sal.size)]] = 1
        m = m.reshape(sal.shape)
        dels.append(score_fn(blend(x_full, x_base, m)))
        ins.append(score_fn(blend(x_base, x_full, m)))
    return np.array(dels), np.array(ins)


def blend_torch(keep, fill, m):
    """Pixels where m == 1 come from fill, others from keep."""
    mf = F.interpolate(torch.from_numpy(m)[None, None], size=keep.shape[-2:], mode="nearest")[0].to(keep.device)
    return keep * (1 - mf) + fill * mf


def randomised(model, stage):
    r = copy.deepcopy(model)
    torch.manual_seed(123)
    for name in STAGES[stage]:
        for sub in r.get_submodule(name).modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    return r.eval()


# ---------- YOLOv8s ----------

class Yolo:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.y = YOLO(weights)
        self.feats = None
        det = self.y.model.model[-1]
        det.register_forward_pre_hook(lambda mod, inp: setattr(self, "feats", [t.detach() for t in inp[0]]))
        self.y.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, verbose=False)  # build predictor

    def predict(self, img):
        r = self.y.predict(img, imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def score(self, img, box):
        b, s = self.predict(img)
        iou = box_iou(box[None], b)[0] if len(b) else np.zeros(0)
        return float(s[iou >= 0.5].max()) if (iou >= 0.5).any() else 0.0

    def eigencam(self, img):
        self.predict(img)
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)  # Ultralytics LetterBox geometry (auto=True, stride 32, centred)
        nh, nw = round(h * r), round(w * r)
        dh, dw = ((640 - nh) % 32) / 2, ((640 - nw) % 32) / 2
        top, left = round(dh - 0.1), round(dw - 0.1)
        ph, pw = nh + round(dh + 0.1) + top, nw + round(dw + 0.1) + left
        assert self.feats[0].shape[-2:] == (ph // 8, pw // 8), (self.feats[0].shape, ph, pw)
        ref = self.feats[0].shape[-2:]
        a = torch.cat([F.interpolate(f.float(), size=ref, mode="bilinear", align_corners=False)
                       for f in self.feats], 1)[0].cpu()
        A = a.reshape(a.shape[0], -1).T.double()
        A = A - A.mean(0)
        v = torch.linalg.eigh(A.T @ A).eigenvectors[:, -1]
        proj = (A @ v).reshape(1, 1, *ref)
        if torch.corrcoef(torch.stack([proj.flatten(), a.mean(0).flatten().double()]))[0, 1] < 0:
            proj = -proj
        up = F.interpolate(proj, size=(ph, pw), mode="bilinear", align_corners=False)[..., top:top + nh, left:left + nw]
        return norm(F.interpolate(up, size=(h // DS, w // DS), mode="bilinear", align_corners=False)[0, 0].numpy())


def in_box_random(box, shape, rng):
    """Control: random ranking inside the detection box, everything outside ranked below it."""
    b = (box / DS).round().astype(int)
    m = rng.random(shape) * 1e-3
    m[b[1]:b[3] + 1, b[0]:b[2] + 1] = 0.5 + 0.5 * rng.random((b[3] - b[1] + 1, b[2] - b[0] + 1))[
        : max(0, min(b[3], shape[0] - 1) - b[1] + 1), : max(0, min(b[2], shape[1] - 1) - b[0] + 1)]
    return m


def blend_np(keep, fill, m):
    mf = cv2.resize(m, (keep.shape[1], keep.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None]
    return (keep * (1 - mf) + fill * mf).astype(np.uint8)


# ---------- metrics ----------

def localisation(sal, box, all_gt):
    b = (box / DS).round().astype(int)
    inside = np.zeros_like(sal, bool)
    inside[b[1]:b[3] + 1, b[0]:b[2] + 1] = True
    any_gt = np.zeros_like(sal, bool)
    for g in (all_gt / DS).round().astype(int):
        any_gt[g[1]:g[3] + 1, g[0]:g[2] + 1] = True
    yx = np.unravel_index(np.argmax(sal), sal.shape)
    binar = sal >= 0.5
    return dict(pointing=bool(inside[yx]), pointing_any_gt=bool(any_gt[yx]),
                energy_in_box=float(sal[inside].sum() / (sal.sum() + 1e-12)),
                energy_in_any_gt=float(sal[any_gt].sum() / (sal.sum() + 1e-12)),
                iou_box=float((binar & inside).sum() / max((binar | inside).sum(), 1)),
                chance=float(inside.mean()), chance_any_gt=float(any_gt.mean()))


def auc(y):
    return float(np.trapezoid(y, dx=1 / (len(y) - 1)))


def true_positives(boxes, scores, gt, conf, k):
    """Up to k highest-scoring detections at the operating threshold, with their matched GT."""
    keep = scores >= conf
    boxes, scores = boxes[keep], scores[keep]
    order = np.argsort(-scores)
    iou = box_iou(boxes[order], gt)
    used, out = set(), []
    for i, row in zip(order, iou):
        cand = [j for j in np.argsort(-row) if row[j] >= 0.5 and j not in used]
        if cand:
            used.add(cand[0])
            out.append((boxes[i], float(scores[i]), gt[cand[0]]))
    return out[:k]


# ---------- per image ----------

def process(i, a, ctx):
    T, model, yolo, ds, fr, yo = ctx["T"], ctx["model"], ctx["yolo"], ctx["ds"], ctx["fr"], ctx["yo"]
    dev = ctx["dev"]
    x, tgt = ds[i]
    x = x.to(dev)
    gt = tgt["boxes"].numpy()
    gen = torch.Generator().manual_seed(1000 + i)
    rng = np.random.default_rng(1000 + i)
    res = dict(image=ds.imgs[i].name, rows=[], curves=[], sanity=[], maps={})

    dets = true_positives(*fr["preds"][i], gt, fr["conf"], a.per_image)
    eig = eigencam(T, x)
    dboxes = torch.tensor(np.array([d[0] for d in dets]), dtype=torch.float32, device=dev)
    rise_maps = rise(T, x, dboxes, a.masks, gen) if dets else []
    base = blurred(x)
    for d, (box, score, g) in enumerate(dets):
        tb = torch.tensor(box, dtype=torch.float32, device=dev)
        maps = {"Grad-CAM": gradcam(T, x, tb), "HiResCAM": gradcam(T, x, tb, hires=True), "EigenCAM": eig, "RISE": rise_maps[d],
                "Random": rng.random(eig.shape), "Random (in box)": in_box_random(box, eig.shape, rng)}
        for name, sal in maps.items():
            dl, il = deletion_insertion(lambda z: float(T.prob(z, tb[None])[0]), base, x, sal, a.steps, blend_torch)
            res["rows"].append(dict(image=res["image"], det=d, model="Faster R-CNN", method=name, score=score,
                                    deletion_auc=auc(dl), insertion_auc=auc(il), **localisation(sal, g, gt)))
            res["curves"].append(dict(method=name, deletion=dl, insertion=il))
        res["maps"][d] = dict(box=box, gt=g, **{k: v.astype(np.float16) for k, v in maps.items() if "Random" not in k})

    # YOLO: EigenCAM on the same image, its own true positives
    img = cv2.imread(str(ds.imgs[i]))
    yeig = yolo.eigencam(img)
    ybase = cv2.resize(cv2.resize(img, None, fx=1 / 32, fy=1 / 32, interpolation=cv2.INTER_AREA),
                       (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    for d, (box, score, g) in enumerate(true_positives(*yo["preds"][i], gt, yo["conf"], a.per_image)):
        for name, sal in {"YOLO EigenCAM": yeig, "YOLO Random": rng.random(yeig.shape)}.items():
            dl, il = deletion_insertion(lambda z: yolo.score(z, box), ybase, img, sal, a.steps, blend_np)
            res["rows"].append(dict(image=res["image"], det=d, model="YOLOv8s", method=name, score=score,
                                    deletion_auc=auc(dl), insertion_auc=auc(il), **localisation(sal, g, gt)))
            res["curves"].append(dict(method=name, deletion=dl, insertion=il))
    res["maps"]["yolo_eigencam"] = yeig.astype(np.float16)

    # sanity check on the first images only
    if i in ctx["sanity_imgs"] and dets:
        box = torch.tensor(dets[0][0], dtype=torch.float32, device=dev)
        ref = {k: res["maps"][0][k].astype(float) for k in ("Grad-CAM", "HiResCAM")} | {"EigenCAM": eig}
        if a.sanity_rise:
            ref["RISE"] = rise_maps[0]
        for stage in STAGES:
            Tr = BoxScore(randomised(model, stage).to(dev))
            new = {"Grad-CAM": gradcam(Tr, x, box), "HiResCAM": gradcam(Tr, x, box, hires=True),
                   "EigenCAM": eigencam(Tr, x)}
            if a.sanity_rise and stage == "heads":
                new["RISE"] = rise(Tr, x, box[None], a.masks, torch.Generator().manual_seed(1000 + i))[0]
            b = (dets[0][0] / DS).round().astype(int)
            for name, m in new.items():
                small = lambda z: cv2.resize(z.astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                inbox = lambda z: z[b[1]:b[3] + 1, b[0]:b[2] + 1].ravel()
                rho = spearmanr(small(ref[name]).ravel(), small(m).ravel()).statistic
                rho_box = spearmanr(inbox(ref[name]), inbox(m)).statistic
                res["sanity"].append(dict(image=res["image"], stage=stage, method=name,
                                          spearman=float(np.nan_to_num(rho)),
                                          spearman_in_box=float(np.nan_to_num(rho_box))))
    return res


# ---------- report ----------

def overlay(ax, img, sal, title, boxes=()):
    ax.imshow(img, cmap="gray")
    if sal is not None:
        ax.imshow(cv2.resize(sal.astype(np.float32), (img.shape[1], img.shape[0])), cmap="inferno",
                  alpha=0.5, vmin=0, vmax=1)
    for (x1, y1, x2, y2), c, ls in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=1.5, ls=ls))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def report():
    cache = sorted((OUT / "cache").glob("*.pkl"))
    res = [pickle.load(open(f, "rb")) for f in cache]
    rows = pd.DataFrame([r for x in res for r in x["rows"]])
    rows.to_csv(OUT / "per_detection.csv", index=False)
    cols = ["pointing", "pointing_any_gt", "energy_in_box", "energy_in_any_gt", "iou_box",
            "deletion_auc", "insertion_auc", "chance", "chance_any_gt"]
    order = [m for m in METHOD_ORDER if m in set(rows.method)]
    g = rows.groupby("method")[cols]
    summ = (g.mean().round(3).astype(str) + " ± " + g.std(ddof=1).round(3).astype(str)).loc[order]
    summ.insert(0, "n", rows.groupby("method").size().loc[order])
    summ.to_csv(OUT / "summary.csv")
    print(f"{len(res)} images, {rows[rows.method == 'Grad-CAM'].shape[0]} Faster R-CNN detections, "
          f"{rows[rows.method == 'YOLO EigenCAM'].shape[0]} YOLO detections\n")
    print(summ.to_string(), "\n")

    # paired comparison against the random baseline (Wilcoxon signed-rank)
    from scipy.stats import wilcoxon
    tests = []
    for model, rnd in (("Faster R-CNN", "Random"), ("Faster R-CNN", "Random (in box)"), ("YOLOv8s", "YOLO Random")):
        sub = rows[rows.model == model]
        r = sub[sub.method == rnd].set_index(["image", "det"])
        for m in sub.method.unique():
            if "Random" in m:
                continue
            s = sub[sub.method == m].set_index(["image", "det"]).loc[r.index]
            for k, better in (("deletion_auc", "less"), ("insertion_auc", "greater")):
                if len(s) > 5:
                    tests.append(dict(method=m, baseline=rnd, metric=k, mean=s[k].mean(), baseline_mean=r[k].mean(),
                                      p=wilcoxon(s[k], r[k], alternative=better).pvalue))
    pd.DataFrame(tests).to_csv(OUT / "vs_random.csv", index=False)
    print(pd.DataFrame(tests).round(4).to_string(index=False), "\n")

    san = pd.DataFrame([r for x in res for r in x["sanity"]])
    if len(san):
        ss = san.groupby(["method", "stage"])[["spearman", "spearman_in_box"]].agg(["mean", "std", "count"]).round(3)
        ss.to_csv(OUT / "sanity.csv")
        print(ss.to_string(), "\n")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, col, t in zip(axes, ("spearman", "spearman_in_box"), ("whole image", "inside the box")):
            for m in [m for m in METHOD_ORDER if m in set(san.method)]:
                v = san[san.method == m].groupby("stage")[col].mean().reindex(list(STAGES))
                ax.plot(list(STAGES), v.values, "-o", color=COLOR[m], lw=2, ms=7, mec="white", mew=1.5, label=m)
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set(ylim=(-0.3, 1.05), xlabel="Layers randomised", title=f"Sanity check, {t}")
        axes[0].set_ylabel("Spearman correlation with\ntrained-model heatmap")
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(OUT / "fig_sanity_check.png")
        plt.close(fig)

    curves = [c for x in res for c in x["curves"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, kind in zip(axes, ("deletion", "insertion")):
        for m in order:
            ys = np.mean([c[kind] for c in curves if c["method"] == m], 0)
            ax.plot(np.linspace(0, 1, len(ys)), ys, color=COLOR[m], lw=2 if "Random" not in m else 1.2,
                    ls="--" if "Random" in m else "-", label=m)
        ax.set(xlabel=f"Fraction of most-salient pixels {'removed' if kind == 'deletion' else 'restored'}",
               ylabel="Detection score", title=f"{kind.capitalize()} curves (mean)", ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_deletion_insertion.png")
    plt.close(fig)

    # qualitative panel: four detections, crop around the box
    picks = [(x, d) for x in res for d in x["maps"] if d != "yolo_eigencam"][:: max(1, len(res) // 4)][:4]
    cols_ = ["Grad-CAM", "HiResCAM", "EigenCAM", "RISE"]
    fig, axes = plt.subplots(len(picks), 6, figsize=(16.5, 2.6 * len(picks)))
    for row, (x, d) in zip(np.atleast_2d(axes), picks):
        img = cv2.imread(str(DATA / "images/test" / x["image"]), cv2.IMREAD_GRAYSCALE)
        mp = x["maps"][d]
        b, gbox = mp["box"], mp["gt"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        half = max(b[2] - b[0], b[3] - b[1]) * 1.6
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half * 0.7, 0))
        x1, y1 = int(min(cx + half, img.shape[1])), int(min(cy + half * 0.7, img.shape[0]))
        crop = img[y0:y1, x0:x1]
        bx = [((gbox - [x0, y0, x0, y0]), "#1baf7a", "-"), ((b - [x0, y0, x0, y0]), "#2a78d6", "--")]
        cropmap = lambda s: cv2.resize(s.astype(np.float32), (img.shape[1], img.shape[0]))[y0:y1, x0:x1]
        overlay(row[0], crop, None, f"{x['image'][-12:]}  GT solid, prediction dashed", bx)
        for ax, m in zip(row[1:5], cols_):
            overlay(ax, crop, cropmap(mp[m]), f"Faster R-CNN {m}", bx)
        overlay(row[5], crop, cropmap(x["maps"]["yolo_eigencam"]), "YOLOv8s EigenCAM", bx[:1])
    fig.savefig(OUT / "fig_heatmaps.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--per-image", type=int, default=2, help="true positives explained per image")
    ap.add_argument("--masks", type=int, default=300, help="RISE masks per image")
    ap.add_argument("--steps", type=int, default=16, help="deletion/insertion steps")
    ap.add_argument("--sanity-images", type=int, default=10)
    ap.add_argument("--sanity-rise", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    if a.report_only:
        return report()

    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    model = build_model(800, 1333)
    model.load_state_dict(torch.load(ROOT / "runs/frcnn/frcnn_s0/weights/best.pt", map_location="cpu"))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def cached(name):
        z = np.load(EVAL / f"{name}_test.npz", allow_pickle=True)
        import json
        return dict(preds=list(zip(z["boxes"], z["scores"])), gts=list(z["gts"]),
                    conf=json.loads((EVAL / f"{name}_test.json").read_text())["conf"])

    fr, yo = cached("frcnn_s0"), cached("yolov8s")
    eligible = [i for i in range(len(fr["gts"])) if true_positives(*fr["preds"][i], fr["gts"][i], fr["conf"], 1)]
    sample = sorted(np.random.default_rng(0).choice(eligible, a.n_images, replace=False).tolist())
    ctx = dict(T=BoxScore(model), model=model, yolo=Yolo(str(ROOT / "runs/detect/full/weights/best.pt")),
               ds=Dentex("test"), fr=fr, yo=yo, dev=dev, sanity_imgs=set(sample[: a.sanity_images]))
    for n, i in enumerate(sample, 1):
        f = OUT / "cache" / f"{ctx['ds'].imgs[i].stem}.pkl"
        if f.exists():
            continue
        import time
        t0 = time.time()
        pickle.dump(process(i, a, ctx), open(f, "wb"))
        print(f"[{n}/{len(sample)}] {f.stem} {time.time() - t0:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
