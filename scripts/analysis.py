"""Chapter Four statistics and figures from the cached test predictions (runs/eval/*.npz).

Per-seed and mean +- SD results, paired bootstrap CIs and test of the Faster R-CNN vs
YOLOv8s difference, FROC, detection calibration (D-ECE), image-level Wilson CIs, and the
figures. CPU only, so it can run while the GPU trains. Rerun after new seeds finish.

  python scripts/analysis.py              # -> runs/analysis/
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from metrics import ap_from_matches, match  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL, OUT = ROOT / "runs/eval", ROOT / "runs/analysis"
IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)
EXTERNAL = {"children": "Children's panoramic set (Zhang et al., 2023)"}
FPPI = [0.125, 0.25, 0.5, 1, 2, 4, 8]
B = 2000
# model -> {eval name: training dir}; seed 0 YOLO is the original runs/detect/full
MODELS = {
    "Faster R-CNN": {f"frcnn_s{s}": ROOT / f"runs/frcnn/frcnn_s{s}" for s in range(3)},
    "YOLOv8s": {"yolov8s": ROOT / "runs/detect/full",
                **{f"yolov8s_s{s}": ROOT / f"runs/detect/yolo_s{s}" for s in (1, 2)}},
}
COLOR = {"Faster R-CNN": "#2a78d6", "YOLOv8s": "#eb6834"}
GT_COLOR, INK, MUTED = "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight"})


class Run:
    """One trained model's test predictions, matched once per image and IoU threshold."""

    def __init__(self, name):
        z = np.load(EVAL / f"{name}_test.npz", allow_pickle=True)
        self.n_img = len(z["gts"])
        self.name = name
        self.preds, self.gts = list(zip(z["boxes"], z["scores"])), list(z["gts"])
        self.conf = json.loads((EVAL / f"{name}_test.json").read_text())["conf"]
        self.m = {t: [match([p], [g], t) for p, g in zip(self.preds, self.gts)] for t in (0.3, *IOUS)}

    def cat(self, idx, t=0.5):
        rows = [self.m[t][i] for i in idx]
        return (np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]),
                sum(r[2] for r in rows))

    def metrics(self, idx, full=True, iou=0.5):
        s, tp, n = self.cat(idx, iou)
        k = s >= self.conf
        TP, FP = int(tp[k].sum()), int((~tp[k]).sum())
        p, r = TP / max(TP + FP, 1), TP / max(n, 1)
        out = dict(mAP50=ap_from_matches(s, tp, n), precision=p, sensitivity=r,
                   f1=2 * p * r / (p + r) if p + r else 0.0, CPM=froc(s, tp, n, len(idx))[1])
        if full:
            out["mAP50_95"] = float(np.mean([ap_from_matches(*self.cat(idx, t)) for t in IOUS]))
        return out

    def image_counts(self):
        c = dict(TP=0, FP=0, TN=0, FN=0)
        for (b, s), g in zip(self.preds, self.gts):
            pos, pred = len(g) > 0, bool((s >= self.conf).any())
            c[("T" if pos == pred else "F") + ("P" if pred else "N")] += 1
        return c


def froc(s, tp, n_gt, n_img):
    order = np.argsort(-s)
    sens = np.cumsum(tp[order]) / max(n_gt, 1)
    fppi = np.cumsum(~tp[order]) / n_img
    pts = [float(sens[fppi <= f].max()) if (fppi <= f).any() else 0.0 for f in FPPI]
    return pts, float(np.mean(pts))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load_runs():
    runs = {m: [Run(n) for n in names if (EVAL / f"{n}_test.npz").exists()] for m, names in MODELS.items()}
    return {m: r for m, r in runs.items() if r}


def tables(runs):
    n_img = len(next(iter(runs.values()))[0].gts)
    idx = np.arange(n_img)
    rows = [dict(model=m, run=r.name, conf=r.conf, **r.metrics(idx)) for m, rs in runs.items() for r in rs]
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "results_per_seed.csv", index=False)
    cols = ["mAP50", "mAP50_95", "precision", "sensitivity", "f1", "CPM"]
    g = per_seed.groupby("model")[cols]
    summ = g.mean().round(3).astype(str) + " ± " + g.std(ddof=1).fillna(0).round(3).astype(str)
    summ.insert(0, "seeds", per_seed.groupby("model").size())
    summ.to_csv(OUT / "results_summary.csv")

    # paired bootstrap over test images; each model's metric is its mean over seeds
    rng = np.random.default_rng(0)
    keys = ["mAP50", "mAP50_95", "precision", "sensitivity", "f1", "CPM"]
    draws = {m: {k: [] for k in keys} for m in runs}
    for _ in range(B):
        bi = rng.integers(0, n_img, n_img)
        for m, rs in runs.items():
            ms = [r.metrics(bi) for r in rs]
            for k in keys:
                draws[m][k].append(np.mean([x[k] for x in ms]))
    point = per_seed.groupby("model")[keys].mean()
    brows = []
    for k in keys:
        for m in runs:
            lo, hi = np.percentile(draws[m][k], [2.5, 97.5])
            brows.append(dict(metric=k, comparison=m, estimate=point.loc[m, k], ci_low=lo, ci_high=hi, p=None))
        if len(runs) == 2:
            a, b = list(runs)
            d = np.array(draws[a][k]) - np.array(draws[b][k])
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            brows.append(dict(metric=k, comparison=f"{a} - {b}", estimate=point.loc[a, k] - point.loc[b, k],
                              ci_low=lo, ci_high=hi, p=p))
    boot = pd.DataFrame(brows)
    boot.to_csv(OUT / "bootstrap.csv", index=False)

    frows, crows, irows = [], [], []
    for m, rs in runs.items():
        for r in rs:
            pts, cpm = froc(*r.cat(idx), n_img)
            frows.append(dict(model=m, run=r.name, **{f"sens@{f}": v for f, v in zip(FPPI, pts)}, CPM=cpm))
            ece, _ = calibration(r)
            c = r.image_counts()
            sens_ci, spec_ci = wilson(c["TP"], c["TP"] + c["FN"]), wilson(c["TN"], c["TN"] + c["FP"])
            crows.append(dict(model=m, run=r.name, D_ECE=ece))
            irows.append(dict(model=m, run=r.name, **c,
                              sensitivity=c["TP"] / max(c["TP"] + c["FN"], 1),
                              sens_ci=f"{sens_ci[0]:.3f}-{sens_ci[1]:.3f}",
                              specificity=c["TN"] / max(c["TN"] + c["FP"], 1),
                              spec_ci=f"{spec_ci[0]:.3f}-{spec_ci[1]:.3f}"))
    pd.DataFrame(frows).to_csv(OUT / "froc.csv", index=False)
    pd.DataFrame(crows).to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(irows).to_csv(OUT / "image_level.csv", index=False)
    return per_seed, summ, boot


def calibration(run, bins=10, min_score=0.05):
    """Detection expected calibration error (Kuppers et al., 2020): confidence vs precision."""
    s, tp, _ = run.cat(range(len(run.gts)))
    k = s >= min_score
    s, tp = s[k], tp[k]
    edges = np.linspace(min_score, 1, bins + 1)
    b = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    rows = [(s[b == i].mean(), tp[b == i].mean(), (b == i).sum()) for i in range(bins) if (b == i).any()]
    ece = sum(n / len(s) * abs(c - a) for c, a, n in rows)
    return float(ece), rows


# ---------- figures ----------

def seed_style(i):
    return dict(lw=2 if i == 0 else 1.2, alpha=1 if i == 0 else 0.55, ls="-" if i == 0 else "--")


def fig_training(runs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in MODELS:
        for i, (name, d) in enumerate(MODELS[m].items()):
            f = d / "results.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            col = "val_mAP50" if "val_mAP50" in df else "metrics/mAP50(B)"
            a.plot(df["epoch"], df[col], color=COLOR[m], label=f"{m} seed {name[-1] if '_s' in name else 0}",
                   **seed_style(i))
            if m == "Faster R-CNN":
                b.plot(df["epoch"], df["train_loss"], color=COLOR[m], label=f"seed {name[-1]}", **seed_style(i))
    a.set(xlabel="Epoch", ylabel="Validation mAP50", title="Validation mAP50 by epoch")
    b.set(xlabel="Epoch", ylabel="Training loss", title="Faster R-CNN training loss")
    a.legend(frameon=False, fontsize=8)
    b.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_training_curves.png")
    plt.close(fig)


def fig_pr(runs):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for m, rs in runs.items():
        for i, r in enumerate(rs):
            s, tp, n = r.cat(range(len(r.gts)))
            o = np.argsort(-s)
            ctp, cfp = np.cumsum(tp[o]), np.cumsum(~tp[o])
            rec, prec = ctp / n, ctp / (ctp + cfp)
            ax.plot(rec, np.maximum.accumulate(prec[::-1])[::-1], color=COLOR[m],
                    label=f"{m} ({r.name}), AP50 {ap_from_matches(s, tp, n):.3f}", **seed_style(i))
            k = (s[o] >= r.conf).sum() - 1
            if i == 0 and k >= 0:
                ax.plot(rec[k], prec[k], "o", ms=8, color=COLOR[m], mec="white", mew=2)
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision-recall, DENTEX test (IoU 0.5)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig_pr_curves.png")
    plt.close(fig)


def fig_froc(runs):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for m, rs in runs.items():
        for i, r in enumerate(rs):
            s, tp, n = r.cat(range(len(r.gts)))
            o = np.argsort(-s)
            ax.plot(np.cumsum(~tp[o]) / len(r.gts), np.cumsum(tp[o]) / n, color=COLOR[m],
                    label=f"{m} ({r.name})", **seed_style(i))
    ax.set_xscale("log", base=2)
    ax.set(xlim=(0.125, 8), ylim=(0, 1), xlabel="False positives per image", ylabel="Lesion sensitivity",
           title="FROC, DENTEX test")
    ax.set_xticks(FPPI, [str(f) for f in FPPI])
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig_froc.png")
    plt.close(fig)


def fig_calibration(runs):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Perfect calibration")
    for m, rs in runs.items():
        ece, rows = calibration(rs[0])
        c, a, _ = zip(*rows)
        ax.plot(c, a, "-o", color=COLOR[m], ms=6, mec="white", mew=1.5, lw=2,
                label=f"{m} ({rs[0].name}), D-ECE {ece:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Precision (fraction TP)",
           title="Detection calibration")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(OUT / "fig_calibration.png")
    plt.close(fig)


def fig_confusion(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.4), gridspec_kw=dict(wspace=0.45))
    for ax, (m, rs) in zip(np.atleast_1d(axes), runs.items()):
        c = rs[0].image_counts()
        mat = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]])
        ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#1c5cab"]))
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, v, ha="center", va="center", fontsize=13, color="white" if v > mat.max() / 2 else INK)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Caries", "No caries"],
               yticklabels=["Caries", "No caries"], xlabel="Predicted", ylabel="Ground truth",
               title=f"{m} ({rs[0].name})")
        ax.grid(False)
    fig.suptitle("Image-level confusion matrices, DENTEX test (n = 114)", y=1.02)
    fig.savefig(OUT / "fig_confusion_image_level.png")
    plt.close(fig)


def per_image(run):
    rows = []
    for i, (s_tp_n, (b, s)) in enumerate(zip(run.m[0.5], run.preds)):
        s_sorted, tp, n = s_tp_n
        k = s_sorted >= run.conf
        TP = int(tp[k].sum())
        rows.append(dict(i=i, n_gt=n, TP=TP, FP=int(k.sum()) - TP, FN=n - TP))
    df = pd.DataFrame(rows)
    df["f1"] = 2 * df.TP / (2 * df.TP + df.FP + df.FN).clip(lower=1)
    return df


def draw(img, gt, boxes, scores, conf, color):
    img = img.copy()
    t = max(2, img.shape[1] // 600)
    for x1, y1, x2, y2 in gt.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(GT_COLOR[i:i + 2], 16) for i in (1, 3, 5)), t)
    for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
        if s >= conf:
            c = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            for k in range(x1, x2, 12 * t):  # dashed prediction boxes, so identity is not colour alone
                cv2.line(img, (k, y1), (min(k + 6 * t, x2), y1), c, t)
                cv2.line(img, (k, y2), (min(k + 6 * t, x2), y2), c, t)
            for k in range(y1, y2, 12 * t):
                cv2.line(img, (x1, k), (x1, min(k + 6 * t, y2)), c, t)
                cv2.line(img, (x2, k), (x2, min(k + 6 * t, y2)), c, t)
    return img


def fig_examples(runs):
    imgs = sorted((ROOT / "data/dentex_yolo/images/test").glob("*.png"))
    rs = {m: r[0] for m, r in runs.items()}
    pf = {m: per_image(r) for m, r in rs.items()}
    prim = pf["Faster R-CNN"]
    both = prim.merge(pf["YOLOv8s"], on="i", suffixes=("_frcnn", "_yolo"))
    both.assign(image=[imgs[i].name for i in both.i]).to_csv(OUT / "per_image_errors.csv", index=False)
    pos = prim[prim.n_gt > 0].sort_values("f1")
    picks = {"Good case": pos.i.iloc[-1], "Typical case": pos.i.iloc[len(pos) // 2], "Poor case": pos.i.iloc[0]}
    fig, axes = plt.subplots(len(picks), len(rs), figsize=(6 * len(rs), 3.4 * len(picks)))
    for row, (label, i) in zip(np.atleast_2d(axes), picks.items()):
        img = cv2.cvtColor(cv2.imread(str(imgs[i])), cv2.COLOR_BGR2RGB)
        # crop to the dentition: union of ground truth and shown predictions, plus a margin
        allb = np.vstack([rs["Faster R-CNN"].gts[i]] + [r.preds[i][0][r.preds[i][1] >= r.conf] for r in rs.values()])
        x0, y0 = np.maximum(allb[:, :2].min(0).astype(int) - 150, 0)
        x1, y1 = np.minimum(allb[:, 2:].max(0).astype(int) + 150, img.shape[1::-1])
        for ax, (m, r) in zip(row, rs.items()):
            b, s = r.preds[i]
            d = pf[m].iloc[i]
            ax.imshow(draw(img, r.gts[i], b, s, r.conf, COLOR[m])[y0:y1, x0:x1])
            ax.set_title(f"{label}, {m}: TP {int(d.TP)}, FP {int(d.FP)}, FN {int(d.FN)}", fontsize=9)
            ax.axis("off")
    fig.text(0.5, 0.0, "Solid green: ground truth. Dashed: model prediction at the validation-tuned threshold.",
             ha="center", fontsize=9, color=MUTED)
    fig.savefig(OUT / "fig_detection_examples.png", dpi=200)
    plt.close(fig)


def external(runs):
    """Internal (DENTEX test) vs external, seed-0 model of each family, threshold from DENTEX val.

    Independent bootstrap of each set (different images), 95% CI of internal minus external.
    IoU 0.3 is a sensitivity analysis for the different box-drawing convention.
    """
    rng = np.random.default_rng(0)
    rows = []
    for ext, label in EXTERNAL.items():
        for m, rs in runs.items():
            name = f"{rs[0].name}_{ext}"
            if not (EVAL / f"{name}_test.npz").exists():
                continue
            ri, re_ = rs[0], Run(name)
            re_.conf = ri.conf  # DENTEX-validation threshold, never tuned on external data
            for iou in (0.5, 0.3):
                a = ri.metrics(np.arange(ri.n_img), full=False, iou=iou)
                b = re_.metrics(np.arange(re_.n_img), full=False, iou=iou)
                diffs = {k: [] for k in a}
                for _ in range(B):
                    x = ri.metrics(rng.integers(0, ri.n_img, ri.n_img), full=False, iou=iou)
                    y = re_.metrics(rng.integers(0, re_.n_img, re_.n_img), full=False, iou=iou)
                    for k in a:
                        diffs[k].append(x[k] - y[k])
                for k in a:
                    lo, hi = np.percentile(diffs[k], [2.5, 97.5])
                    rows.append(dict(external=label, model=m, iou=iou, metric=k, internal=a[k], external_value=b[k],
                                     drop=a[k] - b[k], drop_ci_low=lo, drop_ci_high=hi))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "external.csv", index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    print({m: [r.name for r in rs] for m, rs in runs.items()})
    per_seed, summ, boot = tables(runs)
    for f in (fig_training, fig_pr, fig_froc, fig_calibration, fig_confusion, fig_examples):
        f(runs)
    pd.set_option("display.width", 200)
    print(per_seed.round(3).to_string(index=False), "\n")
    print(summ.to_string(), "\n")
    print(boot.round(4).to_string(index=False), "\n")
    print(external(runs).round(3).to_string(index=False), "\n")
    for f in ("froc", "calibration", "image_level"):
        print(pd.read_csv(OUT / f"{f}.csv").round(3).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
