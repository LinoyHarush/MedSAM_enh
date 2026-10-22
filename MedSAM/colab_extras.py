"""
Extra analyses after Step E (agreed 2026-10-20; each one fixed before its results are seen).

    import colab_extras as cx
    cx.zoom_ablation()     # G1: is zoom still needed after training? R1/R7 on test with and without zoom
    cx.inference_cost()    # G2: seconds per photo for B0, Z, R1, R7, measured in one session on one device
    cx.error_analysis()    # G3: which test photos R7 still gets wrong, and what they have in common
    cx.forgetting_score(ds); cx.forgetting_table(ds)   # K: forgetting check on kvasir_seg / kvasir_instruments

All of them only read existing results/checkpoints and write NEW names (…_nozoom on test, B0/R… on kvasir_*), so the
overwrite guards of colab_runs apply. Tables and plots go to experiments/<dataset>/figures/ on Drive.
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import colab_runs as cr
from compare_paired import summarize

R1 = ["R1_s0", "R1_s1", "R1_s2"]
R7 = ["R7_s0", "R7_s1", "R7_s2"]
KVASIR = "kvasir_seg"


def _root():
    return f"{cr.REPO_D}/experiments"


def _save_csv(df, ds, name):
    d = f"{_root()}/{ds}/figures"
    try:
        os.makedirs(d, exist_ok=True)
        df.to_csv(f"{d}/{name}", index=False)
        print("Saved", f"{d}/{name}")
    except OSError as e:
        print("Could not save to Drive:", e)


def _dsc(name, ds):
    d = cr.load(name, ds, _root())
    if d is None:
        raise FileNotFoundError(f"No scores for {name} on {ds} on Drive")
    return d.set_index("case")


def _seed_mean(runs, ds, col="dsc", suf=""):
    return pd.concat([_dsc(r + suf, ds)[col] for r in runs], axis=1).mean(axis=1)


def _line(label, k):
    return (f"{label}: wins/losses {k['wins']}/{k['losses']} | mean {k['mean_diff']:+.3f} | "
            f"sign p {k['sign_p_one_sided']:.4f} | Wilcoxon p {k['wilcoxon_p']:.4f}")


# ---------------------------------------------------------------- G1: zoom ablation
def zoom_ablation(runs=R1 + R7):
    """Score each trained run on the test set WITHOUT zoom (whole photo, as B0; names <run>_nozoom), then compare with
    zoom per size group. Primary question (fixed 2026-10-20): on the 27 large test photos, does R7 (mean of 3 seeds)
    score higher with zoom than without? Two-sided Wilcoxon. Small photos are a check: zoom never acts there, so
    with and without must be identical."""
    ds = cr.TEST
    cr.copy_test_data(runs)
    for r in runs:
        n = f"{r}_nozoom"
        if cr.load(n, ds, _root()) is None:
            cr.score(n, cr.infer_ckpt(r), ds=ds, zoom_above=100000, variants=("given",))
            cr.save(n, ds, trained=False, variants=("given",))
        else:
            print(f"{n} already scored on Drive; reused")
    split = cr.split.set_index("case").size_group
    rows = []
    for m, rr in [("R1", R1), ("R7", R7)]:
        z = _seed_mean([r for r in rr if r in runs], ds)
        nz = _seed_mean([r for r in rr if r in runs], ds, suf="_nozoom")
        for g in ["large", "small", "all"]:
            idx = z.index if g == "all" else z.index[split.reindex(z.index) == g]
            k = summarize((z[idx] - nz[idx]).to_numpy(), 0.03, 0.05)
            rows.append({"method": m, "photos": g, "n": len(idx), "DSC with zoom": z[idx].mean(),
                         "DSC without zoom": nz[idx].mean(), "gain from zoom": k["mean_diff"],
                         "wins/losses": f"{k['wins']}/{k['losses']}", "Wilcoxon p": k["wilcoxon_p"]})
    zr, b0 = _dsc("Z", ds).dsc, _dsc("og_ckpt", ds).dsc
    for g in ["large", "small", "all"]:
        idx = zr.index if g == "all" else zr.index[split.reindex(zr.index) == g]
        k = summarize((zr[idx] - b0[idx]).to_numpy(), 0.03, 0.05)
        rows.append({"method": "untrained (Z vs B0)", "photos": g, "n": len(idx), "DSC with zoom": zr[idx].mean(),
                     "DSC without zoom": b0[idx].mean(), "gain from zoom": k["mean_diff"],
                     "wins/losses": f"{k['wins']}/{k['losses']}", "Wilcoxon p": k["wilcoxon_p"]})
    df = pd.DataFrame(rows).round(4)
    _save_csv(df, ds, "zoom_ablation.csv")
    display(df)
    print("\nPrimary (fixed in advance): R7, large photos ->",
          df[(df.method == "R7") & (df.photos == "large")][["gain from zoom", "Wilcoxon p"]].to_dict("records")[0])
    return df


# ---------------------------------------------------------------- G2: inference cost
def inference_cost(models=None, n_photos=63):
    """Time infer_zoom per photo for each model, in this session on this device (times from different sessions are
    not comparable). Writes nothing to Drive except the table; predictions go to /content/timing/ and are discarded."""
    ds = cr.TEST
    T = f"/content/work/data/{ds}/MedSAM_test/RGB_Teeth"
    r7_med = sorted(R7, key=lambda r: _dsc(r, ds).dsc.mean())[1]
    models = models or [("B0 (whole photo)", cr.OG, 100000), ("Z (zoom above 2,000 px)", cr.OG, 2000),
                        (f"R7 ({r7_med}, zoom)", cr.infer_ckpt(r7_med), 2000), ("R1 (R1_s0, zoom)", cr.infer_ckpt("R1_s0"), 2000)]
    cr.copy_test_data([r7_med, "R1_s0"])
    dev = cr.device()
    print(f"Device: {dev}. Warm-up, then {len(models)} models x {n_photos} photos; "
          f"about 2-5 min in total on GPU, 15-40 min on CPU.", flush=True)
    # warm-up so the first model doesn't pay for loading CUDA
    cr.sh(f"mkdir -p /content/timing/warm_in && cp $(ls {T}/imgs/*.npz | head -3) /content/timing/warm_in/")
    cr.sh(f"cd {cr.CODE} && python infer_zoom.py -i /content/timing/warm_in -o /content/timing/warm_out "
          f"-lite_medsam_checkpoint_path {cr.OG} -zoom_above 2000 -device {dev} > /dev/null 2>&1")
    split = cr.split.set_index("case").size_group
    rows = []
    for label, ck, za in models:
        out = f"/content/timing/{label.split()[0]}_{za}"
        print(f"timing {label} on {dev} ({len(rows) + 1} of {len(models)}; output hidden so it doesn't affect the time) ...",
              flush=True)
        t0 = time.time()
        cr.sh(f'cd {cr.CODE} && python infer_zoom.py -i "{T}/imgs" -o {out} -lite_medsam_checkpoint_path {ck} '
              f"-zoom_above {za} -device {dev} > /dev/null 2>&1")
        wall = time.time() - t0
        print(f"   done in {wall / 60:.1f} min", flush=True)
        e = pd.read_csv(f"{out}/efficiency.csv").set_index("case").time
        rows.append({"model": label, "device": dev, "photos": len(e), "s/photo mean": e.mean(),
                     "s/photo median": e.median(), "large photos mean": e[split.reindex(e.index) == "large"].mean(),
                     "small photos mean": e[split.reindex(e.index) == "small"].mean(), "wall time incl. loading (s)": wall})
    df = pd.DataFrame(rows).round(3)
    _save_csv(df, ds, f"inference_cost_{dev}.csv")
    display(df)
    print("Same session, same device for all rows. LoRA adds no cost at inference (adapters are merged into the weights).")
    return df


# ---------------------------------------------------------------- G3: error analysis
def error_analysis(k=10):
    """Per test photo: R7 (mean of 3 seeds) DSC, with photo size, number of boxes, lesion size and gain over Z.
    Spearman correlations of DSC with each feature, the k worst photos, and a scatter of DSC vs lesion size."""
    from scipy.stats import spearmanr
    ds = cr.TEST
    T = f"/content/work/data/{ds}/MedSAM_test/RGB_Teeth"
    if not os.path.isdir(f"{T}/gts"):
        cr.copy_test_data([])
    r7 = _seed_mean(R7, ds)
    z = _dsc("Z", ds).dsc
    split = cr.split.set_index("case")
    rows = []
    for c in r7.index:
        gt = np.load(f"{T}/gts/{c}")["gts"] > 0
        with np.load(f"{T}/imgs/{c}") as f:
            boxes = np.asarray(f["boxes"]).reshape(-1, 4)
        H, W = gt.shape[:2]
        bw, bh = boxes[:, 2] - boxes[:, 0] + 1, boxes[:, 3] - boxes[:, 1] + 1
        # lesion size as the model sees it: box side after the photo (or its crop) is resized to 256 px
        rows.append({"case": c, "R7 DSC": r7[c], "Z DSC": z[c], "R7 - Z": r7[c] - z[c],
                     "size group": split.size_group.get(c), "long side px": max(H, W), "boxes": len(boxes),
                     "lesion % of photo": 100 * gt.mean(), "median box side px": float(np.median(np.maximum(bw, bh))),
                     "box side at 256 px (no zoom)": float(np.median(np.maximum(bw, bh))) * 256 / max(H, W)})
    df = pd.DataFrame(rows)
    feats = ["long side px", "boxes", "lesion % of photo", "median box side px", "box side at 256 px (no zoom)"]
    corr = pd.DataFrame([{"feature": f, "Spearman rho with R7 DSC": spearmanr(df[f], df["R7 DSC"])[0],
                          "p": spearmanr(df[f], df["R7 DSC"])[1]} for f in feats]).round(4)
    print("Which photo features go with lower R7 DSC (Spearman, 63 test photos):"); display(corr)
    worst = df.sort_values("R7 DSC").head(k).round(3)
    print(f"\nThe {k} test photos where R7 scores lowest:"); display(worst)
    _save_csv(df.round(4), ds, "error_analysis_per_photo.csv")
    _save_csv(corr, ds, "error_analysis_correlations.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for g, col in [("large", "#2a78d6"), ("small", "#eb6834")]:
        s = df[df["size group"] == g]
        ax.scatter(s["median box side px"], s["R7 DSC"], s=40, color=col, edgecolor="white", linewidth=1.2,
                   label=f"{g} photos (n = {len(s)})")
    ax.set_xscale("log"); ax.set_xlabel("Median lesion box side (px, original photo, log scale)")
    ax.set_ylabel("R7 DSC (mean of 3 seeds)"); ax.legend(frameon=False)
    ax.set_title("Where R7 still fails: DSC vs lesion size, test photos", loc="left", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#e4e3df")
    try:
        fig.savefig(f"{_root()}/{ds}/figures/fig6_error_vs_lesion_size.png", dpi=200, bbox_inches="tight")
        print("Saved", f"{_root()}/{ds}/figures/fig6_error_vs_lesion_size.png")
    except OSError as e:
        print("Could not save to Drive:", e)
    plt.show()
    return df, corr


# ---------------------------------------------------------------- K: forgetting check on Kvasir (already prepared)
# Uses the Kvasir sets prepared earlier in LiteMedSAM_Baseline_production.ipynb with pre_rgb_bbox_datasets.py
# (official test splits, the datasets' own boxes, binary masks):
#   kvasir_seg:          data/kvasir_seg/MedSAM_test/RGB_KvasirSeg/          120 test images (polyps)
#   kvasir_instruments:  data/kvasir_instruments/MedSAM_test/RGB_KvasirInstruments/   118 test images (instruments)
KV_SETS = ["kvasir_seg", "kvasir_instruments"]
KV_MODELS = [("B0", None)] + [(r, r) for r in ["R2_s0", "R3_s0", "R6_s0"] + R1 + R7]
PARAMS = {"R1": 9_791_300, "R2": 4_058_340, "R3": 47_104, "R6": 106_496, "R7": 59_392}


def score_set(ds, plan):
    """Score a list of (name, run or None for the original model, zoom_above, variants) on a dataset that is already
    on the runtime's disk. Checkpoints are fetched from Drive if needed; names already scored on Drive are skipped,
    so the cell can be re-run after a reset."""
    for name, run, za, variants in plan:
        names = [name if v == "given" else f"{name}_p20" for v in variants]
        if all(cr.load(n, ds, _root()) is not None for n in names):
            print(f"{name}: already scored on {ds}; reused"); continue
        if run is not None and not os.path.exists(cr.infer_ckpt(run)):
            cr.sync(f"{cr.RUNS_D}/{run}_infer.pth", "/content/runs/")
        todo = tuple(v for v, n in zip(variants, names) if cr.load(n, ds, _root()) is None)
        cr.score(name, cr.OG if run is None else cr.infer_ckpt(run), ds=ds, zoom_above=za, variants=todo)
        cr.save(name, ds, trained=False, variants=todo)


def forgetting_score(ds, models=KV_MODELS):
    """Score every model on one Kvasir test set (boxes as given; images < 2,000 px, so zoom never acts: Z = B0).
    Names on Drive: experiments/<ds>/<model> (B0 is a new name; the old og_ckpt folder there is left untouched)."""
    cr.copy_eval_data(ds, [r for _, r in models if r])
    score_set(ds, [(name, run, 100000, ("given",)) for name, run in models])


def forgetting_table(ds):
    """Forgetting check (fixed 2026-10-20): every model vs B0 on one Kvasir test set, per image, two-sided Wilcoxon,
    Holm-corrected. A drop vs B0 = the dental training made the model worse there."""
    b0 = _dsc("B0", ds)
    rows = [{"model": "B0 (original LiteMedSAM)", "trainable weights": 0, "DSC": b0.dsc.mean(), "NSD": b0.nsd.mean()}]
    for name, _ in KV_MODELS[1:]:
        d = _dsc(name, ds)
        k = summarize((d.dsc - b0.dsc.reindex(d.index)).to_numpy(), 0.03, 0.05)
        rows.append({"model": name, "trainable weights": PARAMS[name[:2]], "DSC": d.dsc.mean(), "NSD": d.nsd.mean(),
                     "change vs B0": k["mean_diff"], "better/worse images": f"{k['wins']}/{k['losses']}",
                     "Wilcoxon p": k["wilcoxon_p"]})
    df = pd.DataFrame(rows)
    m = df["Wilcoxon p"].notna()
    df.loc[m, "Wilcoxon p (Holm)"] = cr.holm(df.loc[m, "Wilcoxon p"].values)
    extra = []
    for mth, runs in [("R1 mean of 3 seeds", R1), ("R7 mean of 3 seeds", R7)]:
        s = df[df.model.isin(runs)]
        extra.append({"model": mth, "trainable weights": PARAMS[runs[0][:2]], "DSC": s.DSC.mean(), "NSD": s.NSD.mean(),
                      "change vs B0": s["change vs B0"].mean()})
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True).round(4)
    _save_csv(df, ds, "forgetting_vs_B0.csv")
    display(df)
    print(f"{ds}: {len(b0)} test images. LoRA (R3/R6/R7) can be switched off for non-dental images, which restores "
          "B0 exactly; full fine-tuning (R1, R2) cannot.")
    return df


def _bases(ds):
    return (f"/content/work/data/{ds}/MedSAM_test/{cr.sub(ds)}", f"{cr.REPO_D}/data/{ds}/MedSAM_test/{cr.sub(ds)}")


def _segs_dir(name, ds):
    """Folder holding segs_<name> (runtime first, then Drive)."""
    for base in _bases(ds):
        if os.path.isdir(f"{base}/segs_{name}"):
            return base
    raise FileNotFoundError(f"No predictions segs_{name} for {ds} on the runtime or on Drive")


def _gts_dir(ds):
    """Folder holding the ground-truth masks (runtime first, then Drive). AlphaDent's exist only on the runtime,
    after AD4 (prepare_alphadent.py) has run in this session."""
    for base in _bases(ds):
        if os.path.isdir(f"{base}/gts"):
            return f"{base}/gts"
    raise FileNotFoundError(f"No ground-truth masks for {ds} on the runtime or on Drive (AlphaDent: run AD0-AD4 first)")


def b0_reproduce(ds):
    """Added 2026-10-22: the new B0 on a Kvasir set vs the B0 scored earlier in LiteMedSAM_Baseline_production.ipynb
    (og_ckpt). The new B0 goes through infer_zoom.py with zoom off, which should repeat the original inference exactly.
    Prints per-image DSC of both (if the old scores are on Drive) and pixel agreement of the two predictions (if the
    old predictions are). Reads saved files only."""
    new, old = cr.load("B0", ds, _root()), cr.load("og_ckpt", ds, _root())
    if old is None:
        print(f"No old og_ckpt scores in experiments/{ds}/og_ckpt/ on Drive")
    else:
        m = new.merge(old, on="case", suffixes=("_new", "_old"))
        d = m.dsc_new - m.dsc_old
        print(f"{ds}: B0 new {new.dsc.mean():.4f} ({len(new)} images) vs old og_ckpt {old.dsc.mean():.4f} ({len(old)} images); "
              f"{len(m)} in both: mean change {d.mean():+.4f}, identical DSC (to 1e-4) on {int((d.abs() < 1e-4).sum())}")
        if len(m) < max(len(new), len(old)):
            print("   the two runs cover different images: compare the image lists before anything else")
        print(m.assign(change=d).sort_values("change").iloc[[0, 1, 2, -3, -2, -1]][["case", "dsc_old", "dsc_new", "change"]]
              .round(4).to_string(index=False))
    T = f"{cr.REPO_D}/data/{ds}/MedSAM_test/{cr.sub(ds)}"
    a, b = f"{T}/segs_B0", f"{T}/segs_og_ckpt"
    if not (os.path.isdir(a) and os.path.isdir(b)):
        print("Old or new predictions not on Drive: pixel comparison skipped")
        return
    same, ious = 0, []
    common = sorted(set(x for x in os.listdir(a) if x.endswith(".npz")) & set(os.listdir(b)))
    for f in common:
        pa, pb = np.load(f"{a}/{f}")["segs"] > 0, np.load(f"{b}/{f}")["segs"] > 0
        if pa.shape != pb.shape:
            ious.append(np.nan); continue
        same += int((pa == pb).all())
        u = (pa | pb).sum(); ious.append((pa & pb).sum() / u if u else 1.0)
    print(f"Predictions: {len(common)} images in both, identical on {same}; IoU new vs old: median {np.nanmedian(ious):.4f}, "
          f"min {np.nanmin(ious):.4f}; shape mismatches {int(np.isnan(ious).sum())}")


def mask_bias(ds, models=None):
    """Added 2026-10-22, after the K results (exploratory, not fixed in advance): does a model draw masks that are too
    small or too large? Per image: predicted area / true area, precision (share of the predicted pixels that are right)
    and recall (share of the true pixels found). Reads saved predictions only, runs no model (CPU, a few minutes).
    Default models: the Kvasir set (B0 and every trained run). For the dental test set pass e.g.
    ["og_ckpt", "Z", "R1_s0", "R1_s1", "R1_s2", "R7_s0", "R7_s1", "R7_s2"]."""
    models = models or [m for m, _ in KV_MODELS]
    rows = []
    for m in models:
        base, G = _segs_dir(m, ds), _gts_dir(ds)
        ratio, prec, rec = [], [], []
        for f in sorted(x for x in os.listdir(f"{base}/segs_{m}") if x.endswith(".npz")):
            pr = np.load(f"{base}/segs_{m}/{f}")["segs"] > 0
            gt = np.load(f"{G}/{f}")["gts"] > 0
            tp = (pr & gt).sum()
            ratio.append(pr.sum() / max(gt.sum(), 1)); prec.append(tp / max(pr.sum(), 1)); rec.append(tp / max(gt.sum(), 1))
        rows.append({"model": m, "images": len(ratio), "predicted / true area (median)": np.median(ratio),
                     "images with predicted < true area": f"{int((np.array(ratio) < 1).sum())}/{len(ratio)}",
                     "precision (mean)": np.mean(prec), "recall (mean)": np.mean(rec)})
    df = pd.DataFrame(rows).round(3)
    _save_csv(df, ds, "mask_bias.csv")
    display(df)
    print("Area ratio < 1 with high precision and lower recall = masks too small (under-segmentation); "
          "> 1 with lower precision = masks spill outside the object.")
    return df


def error_by_seen_size():
    """Added 2026-10-22 (exploratory): G3 used 'box side at 256 px without zoom', which is wrong for the 27 large photos
    (zoom crops them). Here the lesion size is measured as R7 actually saw it: box side x 256 / crop side, from R7's own
    zoom_boxes.csv (median seed). Spearman with R7 DSC (mean of 3 seeds), for all photos, small and large separately."""
    from scipy.stats import spearmanr
    ds = cr.TEST
    r7_med = sorted(R7, key=lambda r: _dsc(r, ds).dsc.mean())[1]
    zb = pd.read_csv(f"{_segs_dir(r7_med, ds)}/segs_{r7_med}/zoom_boxes.csv")
    zb["seen side px"] = (np.maximum(zb.box_x2 - zb.box_x1, zb.box_y2 - zb.box_y1) + 1) * 256 / np.maximum(zb.crop_w, zb.crop_h)
    per = zb.groupby("case")["seen side px"].median().rename("median box side as seen (px of 256)")
    r7 = _seed_mean(R7, ds).rename("R7 DSC")
    df = pd.concat([r7, per], axis=1).join(cr.split.set_index("case").size_group)
    rows = []
    for g, d in [("all", df), ("small", df[df.size_group == "small"]), ("large", df[df.size_group == "large"])]:
        rho, p = spearmanr(d.iloc[:, 1], d["R7 DSC"])
        rows.append({"photos": g, "n": len(d), "median seen side": d.iloc[:, 1].median(), "Spearman rho": rho, "p": p})
    out = pd.DataFrame(rows).round(4)
    _save_csv(df.reset_index().rename(columns={"index": "case"}), ds, "error_by_seen_size_per_photo.csv")
    _save_csv(out, ds, "error_by_seen_size.csv")
    display(out)
    return out


# ---------------------------------------------------------------- E3: AlphaDent checks (after prepare_alphadent.py)
def check_alphadent(src, ds="alphadent_valid", split="valid", n_png=10, n_show=4, seed=0):
    """Checks that our converted masks mean the same as AlphaDent's own labels, before any model is run.
    (a) counts from photos.csv; (b) our caries mask vs the union of AlphaDent's own caries PNG masks, per photo
    (IoU should be ~1; a size mismatch means a rotation problem); if the archive has no PNG masks (the Zenodo
    archive doesn't): (b1) no photo carries an EXIF rotation and (b2) all label coordinates lie in [0, 1];
    (c) every box contains caries pixels;
    (d) an overlay figure of n_show photos (photo, our caries mask, boxes) to look at by eye."""
    import glob
    import re as _re
    import cv2
    from pathlib import Path
    T = f"/content/work/data/{ds}/MedSAM_test/{cr.sub(ds)}"
    pc = pd.read_csv(f"/content/work/data/{ds}/photos.csv")
    kept = pc[pc.skipped.isna() | (pc.skipped == "")]
    print(f"(a) {len(pc)} photos read, {len(kept)} with caries kept; skipped: {pc.skipped.value_counts().to_dict()}")
    print(f"    caries boxes {int(kept.caries_boxes.sum())}, per photo median {kept.caries_boxes.median():.0f}, "
          f"max {int(kept.caries_boxes.max())}; long side median {int(kept.long_side.median())} px")
    lab_root = [p for p in Path(src).rglob(split) if p.is_dir() and p.parent.name == "labels"][0]
    rng = np.random.default_rng(seed)
    cases = list(rng.choice(kept.case.values, size=min(n_png, len(kept)), replace=False))
    ious, notes = [], []
    for c in cases:
        stem = c[:-4]
        ours = np.load(f"{T}/gts/{c}")["gts"] > 0
        pngs = [f for f in glob.glob(f"{lab_root}/{stem}/*.png")
                if int(_re.search(r"class_(\d+)", Path(f).stem).group(1)) in (3, 4, 5, 6, 7, 8)]
        if not pngs:
            notes.append(f"{stem}: no PNG masks found"); continue
        theirs = np.zeros_like(ours)
        for f in pngs:
            m = cv2.imread(f, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if m.shape != ours.shape:
                notes.append(f"{stem}: PNG {m.shape} vs ours {ours.shape} -> ROTATION/SIZE MISMATCH"); break
            theirs |= m > 0
        else:
            inter, union = (ours & theirs).sum(), (ours | theirs).sum()
            ious.append(inter / union if union else 1.0)
    if ious:
        print(f"(b) our caries mask vs AlphaDent's caries PNGs on {len(ious)} photos: IoU min {min(ious):.3f}, "
              f"median {np.median(ious):.3f}")
        for n in notes:
            print("    ", n)
    else:
        print(f"(b) AlphaDent's PNG masks are not in this archive ({len(notes)} of {len(cases)} photos checked have none),"
              " so instead: (b1) orientation and (b2) label coordinates")
    # (b1) The YOLO polygons refer to the photo as YOLO tools load it, i.e. after EXIF rotation; we read the pixels as
    # stored. The two agree only if no photo carries an EXIF rotation (orientation tag missing or 1).
    from PIL import Image
    img_root = [p for p in Path(src).rglob(split) if p.is_dir() and p.parent.name == "images"][0]
    orient = {}
    for c, name in zip(kept.case, kept.image):
        with Image.open(img_root / name) as im:
            orient[c] = int(im.getexif().get(274, 1) or 1)
    rotated = [c for c, o in orient.items() if o != 1]
    print(f"(b1) EXIF orientation: {len(orient) - len(rotated)} of {len(orient)} photos unrotated (tag missing or 1)"
          + ("" if not rotated else f"; ROTATED: {len(rotated)} -> "
             + ", ".join(f"{c[:-4]} (tag {orient[c]})" for c in rotated[:8])
             + ". STOP and tell Claude: labels and pixels may disagree on these"))
    # (b2) Normalised YOLO coordinates must lie in [0, 1]; values outside mean the labels use another convention.
    out_of_range, n_coords = 0, 0
    for name in kept.image:
        for line in (lab_root / f"{Path(name).stem}.txt").read_text().split("\n"):
            v = line.split()
            if len(v) >= 7:
                xy = np.array(v[1:], dtype=float)
                n_coords += len(xy); out_of_range += int(((xy < -1e-6) | (xy > 1 + 1e-6)).sum())
    print(f"(b2) label coordinates outside [0, 1]: {out_of_range} of {n_coords} (expect 0)")
    bad = 0
    for c in kept.case:
        with np.load(f"{T}/imgs/{c}") as f:
            boxes = np.asarray(f["boxes"]).reshape(-1, 4)
        gt = np.load(f"{T}/gts/{c}")["gts"] > 0
        bad += sum(not gt[b[1]:b[3] + 1, b[0]:b[2] + 1].any() for b in boxes)
    print(f"(c) boxes with no caries pixels inside: {bad} (expect 0)")
    show = rotated[:2] + [c for c in rng.choice(kept.case.values, size=min(n_show, len(kept)), replace=False)
                          if c not in rotated][:n_show - len(rotated[:2])]   # rotated photos first, if any
    fig, ax = plt.subplots(1, len(show), figsize=(5 * len(show), 4.5))
    for a, c in zip(np.atleast_1d(ax), show):
        with np.load(f"{T}/imgs/{c}") as f:
            img, boxes = f["imgs"], np.asarray(f["boxes"]).reshape(-1, 4)
        gt = np.load(f"{T}/gts/{c}")["gts"] > 0
        s = 1600 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = cv2.resize(gt.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ov = np.zeros((*g.shape, 4)); ov[g] = (0.106, 0.686, 0.478, 0.55)
        a.imshow(small); a.imshow(ov)
        for b in boxes * s:
            a.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor="white",
                                      linewidth=1, linestyle="--"))
        a.set_title(f"{c[:24]}  ({len(boxes)} caries boxes)", fontsize=10); a.axis("off")
    fig.suptitle(f"{ds}: our binary caries mask (green) and boxes (dashed), random photos", x=0.01, ha="left")
    fig.tight_layout()
    try:
        os.makedirs(f"{_root()}/{ds}/figures", exist_ok=True)
        fig.savefig(f"{_root()}/{ds}/figures/check_conversion.png", dpi=150, bbox_inches="tight")
        print("Saved", f"{_root()}/{ds}/figures/check_conversion.png")
    except OSError as e:
        print("Could not save to Drive:", e)
    plt.show()
    return ious


# ---------------------------------------------------------------- E3: AlphaDent scoring plan (inference only)
def alphadent_plan():
    """B0 (whole photo) as og_ckpt + B0_p20, Z, and every trained run, each with boxes as given and 20% looser
    (same names as on our test set, so cr.test_table / cr.final_table / colab_figures work unchanged)."""
    return ([("og_ckpt", None, 100000, ("given",)), ("B0", None, 100000, ("p20",)), ("Z", None, 2000, ("given", "p20"))]
            + [(r, r, 2000, ("given", "p20")) for r in R1 + R7 + ["R2_s0", "R3_s0", "R6_s0"]])


def r7_vs_r1(ds):
    """Per photo, each method's DSC averaged over its 3 seeds; two-sided Wilcoxon R7 vs R1 (as given, then looser)."""
    for suf, label in [("", "as given (main)"), ("_p20", "20% looser")]:
        a, b = _seed_mean(R7, ds, suf=suf), _seed_mean(R1, ds, suf=suf)
        k = summarize((a - b.loc[a.index]).to_numpy(), 0.03, 0.05)
        print(f"R7 vs R1 on {ds}, {label}: DSC {a.mean():.3f} vs {b.mean():.3f} | " + _line("per photo", k))
