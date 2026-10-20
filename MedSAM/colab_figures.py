"""
Presentation figures and the method table, for after Step E (all runs scored on the test set).
Reads only: scores, logs and predictions on Drive (and the test photos on the runtime's disk after
cr.copy_test_data). Writes only: PNG/CSV files into experiments/<test>/figures/ on Drive.

    import colab_figures as cf
    cf.method_table()        # all methods: trainable weights, training time, validation and test metrics
    cf.fig_overview()        # test DSC of every method (B0 -> Z -> trained)
    cf.fig_size_groups()     # large vs small photos: zoom fixes one group, training the other
    cf.fig_robustness()      # boxes as given vs 20% looser
    cf.fig_training_curves() # validation Dice per epoch, R1 vs R7, 3 seeds each
    cf.fig_examples()        # example photos: ground truth, B0, Z, R7
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import colab_runs as cr

# Fixed colour per method (validated categorical slots, light mode); B0 is a neutral reference.
COL = {"B0": "#8f8e89", "Z": "#2a78d6", "R1": "#eb6834", "R7": "#1baf7a",
       "R2": "#eda100", "R3": "#e87ba4", "R6": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"B0": "B0 zero-shot", "Z": "Z zero-shot + zoom", "R1": "R1 full fine-tuning",
         "R2": "R2 frozen encoders", "R3": "R3 LoRA decoder", "R6": "R6 LoRA enc+dec", "R7": "R7 LoRA encoder"}
PARAMS = {"B0": 0, "Z": 0, "R1": 9_791_300, "R2": 4_058_340, "R3": 47_104, "R6": 106_496, "R7": 59_392}
SEEDS = {"R1": ["R1_s0", "R1_s1", "R1_s2"], "R7": ["R7_s0", "R7_s1", "R7_s2"],
         "R2": ["R2_s0"], "R3": ["R3_s0"], "R6": ["R6_s0"]}
REF = {"B0": ("og_ckpt", "B0_p20"), "Z": ("Z", "Z_p20")}


def _style():
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def _root():
    return f"{cr.REPO_D}/experiments"


def _label(ds):
    ds = ds or cr.TEST
    return {cr.TEST: "Test set (63 photos)", "alphadent_valid": "AlphaDent validation (external)"}.get(ds, ds)


def _out(name, ds=None):
    d = f"{_root()}/{ds or cr.TEST}/figures"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{name}"


def _scores(name, ds=None):
    d = cr.load(name, ds or cr.TEST, _root())
    if d is None:
        raise FileNotFoundError(f"No scores for '{name}' on {ds or cr.TEST} on Drive: run Step E first")
    return d.set_index("case")


def _method_scores(m, suf="", ds=None):
    """Per-photo DSC of a method: reference, or mean over its seeds. Also returns the per-seed means."""
    if m in REF:
        s = _scores(REF[m][0] if suf == "" else REF[m][1], ds).dsc
        return s, [s.mean()]
    runs = [_scores(r + suf, ds).dsc for r in SEEDS[m]]
    return pd.concat(runs, axis=1).mean(axis=1), [r.mean() for r in runs]


def _save(fig, name, ds=None):
    p = _out(name, ds)
    try:
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("Saved", p)
    except OSError as e:
        print("Could not save to Drive (Drive dropped?):", e)
    plt.show()


# ------------------------------------------------------------------ table
def method_table():
    """One row per method: trainable weights, epochs and training minutes per run (from the logs on Drive),
    validation and test DSC (mean ± sd over seeds), and DSC with 20% looser boxes. Saved as method_table.csv."""
    rows = []
    for m in ["B0", "Z", "R3", "R2", "R6", "R1", "R7"]:
        row = {"method": NAMES[m], "trainable weights": PARAMS[m]}
        if m in SEEDS:
            logs = [pd.read_csv(f"{cr.RUNS_D}/{r}/log.csv") for r in SEEDS[m]]
            row.update({"seeds": len(logs),
                        "epochs run": "/".join(str(len(l)) for l in logs),
                        "best epoch": "/".join(str(int(l.epoch[l.val_dice.idxmax()])) for l in logs),
                        "s per epoch": round(np.mean([l.seconds.mean() for l in logs])),
                        "train min per run": round(np.mean([l.seconds.sum() for l in logs]) / 60, 1)})
        else:
            row.update({"seeds": "-", "epochs run": "-", "best epoch": "-", "s per epoch": 0, "train min per run": 0})
        for ds, tag in [(cr.VAL, "val"), (cr.TEST, "test")]:
            for suf, lab in [("", ""), ("_p20", " 20% looser")]:
                _, per_seed = _method_scores(m, suf, ds)
                row[f"{tag} DSC{lab}"] = round(float(np.mean(per_seed)), 4)
                if lab == "" and len(per_seed) > 1:
                    row[f"{tag} DSC sd"] = round(float(np.std(per_seed, ddof=1)), 4)
            if m in REF:
                n = REF[m][0]
            else:
                n = None
            nsd = [_scores(r, ds).nsd.mean() for r in (SEEDS[m] if n is None else [n])]
            row[f"{tag} NSD"] = round(float(np.mean(nsd)), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["method", "trainable weights", "seeds", "epochs run", "best epoch", "s per epoch", "train min per run",
            "val DSC", "val DSC sd", "val NSD", "val DSC 20% looser",
            "test DSC", "test DSC sd", "test NSD", "test DSC 20% looser"]
    df = df[[c for c in cols if c in df.columns]]
    try:
        df.to_csv(_out("method_table.csv"), index=False)
        print("Saved", _out("method_table.csv"))
    except OSError as e:
        print("Could not save to Drive:", e)
    display(df)
    return df


# ------------------------------------------------------------------ figures
def _bar_labels(ax, bars, vals, fmt="{:.3f}"):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, fmt.format(v),
                ha="center", va="bottom", fontsize=10, color=INK)


def fig_overview(ds=None):
    """DSC (boxes as given) of every method on the test set (or another scored set, e.g. alphadent_valid);
    R1 and R7 = mean of 3 seeds with the seed range."""
    _style()
    order = ["B0", "Z", "R3", "R2", "R6", "R1", "R7"]
    means, lo, hi = [], [], []
    for m in order:
        _, per = _method_scores(m, "", ds)
        means.append(np.mean(per)); lo.append(np.mean(per) - min(per)); hi.append(max(per) - np.mean(per))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(order))
    bars = ax.bar(x, means, width=0.62, color=[COL[m] for m in order], edgecolor="white", linewidth=2)
    multi = [i for i, m in enumerate(order) if len(SEEDS.get(m, [])) > 1]
    ax.errorbar([x[i] for i in multi], [means[i] for i in multi], yerr=[[lo[i] for i in multi], [hi[i] for i in multi]],
                fmt="none", ecolor=INK, elinewidth=1.4, capsize=5)
    for i, (b, v) in enumerate(zip(bars, means)):         # label above the error bar where there is one
        ax.text(b.get_x() + b.get_width() / 2, v + hi[i] + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, color=INK)
    ax.set_xticks(x, [NAMES[m].replace(" ", "\n", 1) + ("\n(3 seeds)" if m in ("R1", "R7") else
                      "\n(1 seed)" if m in SEEDS else "") for m in order], fontsize=9.5)
    lo_y = max(0.0, np.floor((min(means) - 0.08) * 10) / 10)
    ax.set_ylim(lo_y, max(0.9, max(means) + 0.05)); ax.set_ylabel("Mean DSC per photo")
    ax.set_title(f"{_label(ds)}: every method, boxes as given", loc="left", fontsize=13, color=INK)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig1_overview.png", ds)


def fig_size_groups():
    """Large vs small photos: B0, Z, R1, R7 (per-photo mean over seeds)."""
    _style()
    split = cr.split.set_index("case").size_group
    meths = ["B0", "Z", "R1", "R7"]
    groups = [("large", "Large photos (> 2,000 px)"), ("small", "Small photos")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.19
    for j, m in enumerate(meths):
        s, _ = _method_scores(m)
        vals = [s[split.reindex(s.index) == g].mean() for g, _ in groups]
        bars = ax.bar(np.arange(2) + (j - 1.5) * w, vals, width=w - 0.02, color=COL[m], label=NAMES[m])
        _bar_labels(ax, bars, vals, "{:.2f}")
    n = [int((split.reindex(_scores("Z").index) == g).sum()) for g, _ in groups]
    ax.set_xticks(np.arange(2), [f"{lab}\nn = {k}" for (_, lab), k in zip(groups, n)])
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("Mean DSC (test)")
    ax.set_title("Zoom lifts the large photos; training lifts the small ones", loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9.5)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig2_size_groups.png")


def fig_robustness(ds=None):
    """Boxes as given vs 20% looser: a slope per method, labelled at both ends."""
    _style()
    meths = ["B0", "Z", "R1", "R7"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ends = {m: (np.mean(_method_scores(m, "", ds)[1]), np.mean(_method_scores(m, "_p20", ds)[1])) for m in meths}

    def spread(vals, gap=0.014):                           # label positions at least `gap` apart, in value order
        order = sorted(vals, key=vals.get); pos, last = {}, -1.0
        for m in order:
            pos[m] = max(vals[m], last + gap); last = pos[m]
        return pos
    left, right = spread({m: e[0] for m, e in ends.items()}), spread({m: e[1] for m, e in ends.items()})
    for m, (a, b) in ends.items():
        ax.plot([0, 1], [a, b], color=COL[m], linewidth=2.2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=2)
        ax.text(-0.04, left[m], f"{NAMES[m]}  {a:.3f}", ha="right", va="center", fontsize=10, color=INK)
        ax.text(1.04, right[m], f"{b:.3f}  ({b - a:+.3f})", ha="left", va="center", fontsize=10, color=INK)
    ax.set_xticks([0, 1], ["Boxes as given", "Boxes 20% looser"])
    allv = [v for e in ends.values() for v in e]
    ax.set_xlim(-0.9, 1.45); ax.set_ylim(max(0.0, min(allv) - 0.05), max(0.9, max(allv) + 0.03))
    ax.set_ylabel("Mean DSC per photo")
    ax.set_title(f"{_label(ds)}: boxes as given vs 20% looser", loc="left", fontsize=13, color=INK)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig3_robustness.png", ds)


def fig_training_curves():
    """Validation Dice per epoch for the 3 seeds of R1 and of R7 (same y-axis); dot = kept (best) epoch."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, m in zip(axes, ["R1", "R7"]):
        for k, r in enumerate(SEEDS[m]):
            l = pd.read_csv(f"{cr.RUNS_D}/{r}/log.csv")
            ax.plot(l.epoch, l.val_dice, color=COL[m], linewidth=2, alpha=[1, 0.7, 0.45][k], label=f"seed {k}")
            b = l.val_dice.idxmax()
            ax.plot(l.epoch[b], l.val_dice[b], "o", color=COL[m], markersize=8, markeredgecolor="white",
                    markeredgewidth=2)
        ax.set_title(NAMES[m], loc="left", fontsize=12, color=INK)
        ax.set_xlabel("epoch"); ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Validation Dice (per lesion)")
    fig.suptitle("Full fine-tuning peaks early and swings; encoder LoRA improves steadily (dot = kept epoch)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    _save(fig, "fig4_training_curves.png")


def _pred(name, case, ds=None):
    ds = ds or cr.TEST
    for base in (f"/content/work/data/{ds}/MedSAM_test/{cr.sub(ds)}",
                 f"{cr.REPO_D}/data/{ds}/MedSAM_test/{cr.sub(ds)}"):
        p = f"{base}/segs_{name}/{case}"
        if os.path.exists(p):
            return np.load(p)["segs"] > 0
    return None


def _b0_preds(cases, ds=None):
    """B0 predictions for a few photos, made on the fly (whole photo, same as B0) if not stored."""
    ds = ds or cr.TEST
    missing = [c for c in cases if _pred("og_ckpt", c, ds) is None]
    out = {c: _pred("og_ckpt", c, ds) for c in cases if c not in missing}
    if missing:
        T = f"/content/work/data/{ds}/MedSAM_test/{cr.sub(ds)}"
        tmp_i, tmp_o = "/content/fig_tmp/imgs", "/content/fig_tmp/segs"
        os.makedirs(tmp_i, exist_ok=True)
        for c in missing:
            cr.sh(f'cp "{T}/imgs/{c}" {tmp_i}/')
        cr.sh(f"cd {cr.CODE} && python infer_zoom.py -i {tmp_i} -o {tmp_o} -lite_medsam_checkpoint_path {cr.OG} "
              f"-zoom_above 100000 -device {cr.device()}")
        for c in missing:
            out[c] = np.load(f"{tmp_o}/{c}")["segs"] > 0
    return out


def fig_examples(ds=None):
    """Rows chosen by rule, not by eye. Dental test set: the 2 small photos where R7 gains most over Z, the large photo
    where Z gains most over B0, and the photo where R7 loses most to Z. Other sets (e.g. AlphaDent, all photos large):
    the same rule without the size condition. R7 = its median seed on that set (no cherry-picking)."""
    _style()
    ds = ds or cr.TEST
    by_size = ds == cr.TEST
    T = f"/content/work/data/{ds}/MedSAM_test/{cr.sub(ds)}"
    if not os.path.isdir(f"{T}/imgs"):
        raise FileNotFoundError(f"Photos of {ds} not on this runtime: run cr.copy_eval_data('{ds}') first")
    seeds = sorted(SEEDS["R7"], key=lambda r: _scores(r, ds).dsc.mean())
    r7 = seeds[1]
    split = cr.split.set_index("case").size_group
    z, b0, r = _scores("Z", ds).dsc, _scores("og_ckpt", ds).dsc, _scores(r7, ds).dsc
    g7 = (r - z).sort_values()
    if by_size:
        top = [c for c in g7.index[::-1] if split.get(c) == "small"][:2]
        zb = [(z - b0)[split.reindex(z.index) == "large"].idxmax()]
        labs = ["Small photo: R7 gains most over Z", "Small photo: 2nd largest R7 gain",
                "Large photo: zoom gains most over B0"]
    else:
        top = list(g7.index[::-1][:2])
        zb = [(z - b0).drop(top).idxmax()]
        labs = ["Photo where R7 gains most over Z", "2nd largest R7 gain over Z", "Photo where zoom gains most over B0"]
    worst = [c for c in g7.index if c not in top + zb][:1]
    rows = list(zip(top + zb + worst, labs + ["Photo where R7 loses most to Z"]))
    b0p = _b0_preds([c for c, _ in rows], ds)
    fig, ax = plt.subplots(len(rows), 4, figsize=(15, 3.9 * len(rows)))
    for i, (c, label) in enumerate(rows):
        with np.load(f"{T}/imgs/{c}") as f:
            img, boxes = f["imgs"], np.asarray(f["boxes"]).reshape(-1, 4)
        gt = np.load(f"{T}/gts/{c}")["gts"] > 0
        H, W = gt.shape[:2]
        pad = int(0.08 * max(H, W))
        x0, y0 = max(int(boxes[:, 0].min()) - pad, 0), max(int(boxes[:, 1].min()) - pad, 0)
        x1, y1 = min(int(boxes[:, 2].max()) + pad, W), min(int(boxes[:, 3].max()) + pad, H)
        panels = [("Ground truth", gt, INK, None), ("B0", b0p[c], COL["B0"], b0.get(c)),
                  ("Z", _pred("Z", c, ds), COL["Z"], z.get(c)), (f"R7 ({r7})", _pred(r7, c, ds), COL["R7"], r.get(c))]
        for k, (t, m, colr, d) in enumerate(panels):
            a = ax[i, k]
            a.imshow(img[y0:y1, x0:x1])
            if m is not None:
                ov = np.zeros((y1 - y0, x1 - x0, 4))
                rgb = tuple(int(colr[j:j + 2], 16) / 255 for j in (1, 3, 5))
                ov[m[y0:y1, x0:x1]] = (*rgb, 0.5)
                a.imshow(ov)
            for bx in boxes:
                a.add_patch(plt.Rectangle((bx[0] - x0, bx[1] - y0), bx[2] - bx[0], bx[3] - bx[1], fill=False,
                                          edgecolor="white", linewidth=1.2, linestyle="--"))
            a.set_title(t + ("" if d is None else f"  DSC {d:.2f}"), fontsize=11, color=INK)
            a.axis("off")
        gain = (f"R7 - Z: {r[c] - z[c]:+.2f} DSC" if i != 2 else f"Z - B0: {z[c] - b0[c]:+.2f} DSC")
        ax[i, 0].text(-0.04, 0.5, f"{chr(65 + i)}. {label}\n{gain}  |  {W}x{H} px\n{c[:22]}",
                      transform=ax[i, 0].transAxes, rotation=90, ha="right", va="center", fontsize=10, color=INK)
    rule = ("A, B = the 2 small photos where R7 gains most over Z;  C = the large photo where zoom (Z) gains most "
            "over B0;  D = the photo where R7 loses most to Z.\n") if by_size else (
            "A, B = the 2 photos where R7 gains most over Z;  C = the photo where zoom (Z) gains most over B0;  "
            "D = the photo where R7 loses most to Z.\n")
    fig.suptitle(f"{_label(ds)}: photos chosen by a fixed rule, not by eye:\n" + rule +
                 f"R7 shown = its median seed on this set ({r7}).  Coloured fill = predicted mask; dashed white = given boxes.",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "fig5_examples.png", ds)
