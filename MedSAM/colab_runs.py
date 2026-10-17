"""
Colab helpers for the training runs of run list B (MedSAM_enh_repo_review.md, section 10b).
Used by both of us from the notebook; the cells are in Shelly_Training.md and Linoy_Training.md.

    import sys; sys.path.insert(0, "/content/code/MedSAM")
    import colab_runs as cr
    cr.init(REPO_D)                 # paths; call once per session, after the code is copied to /content/code/MedSAM
    cr.copy_data()                  # training + validation data to the runtime's disk (retries if Drive drops)
    run = cr.train("R2", 0)         # train on the runtime's disk + export the inference checkpoint
    cr.score(run, cr.infer_ckpt(run))   # zoom inference + eval_run, boxes as given and 20% looser
    cr.save(run)                    # copy ONLY this run's folders back to Drive
    cr.compare(run)                 # per-photo comparison with Z
    cr.board()                      # all runs on Drive (both of us), saved as experiments/<ds>/board.csv

Why save() never copies runs_summary.csv: eval_run.py appends rows to the copy on the runtime's disk, which only
holds what was on Drive at session start plus this session's runs; copying it back would delete the other person's
rows. board() rebuilds the shared table from the per-run folders instead.
"""
import os
import re
import subprocess
import time

import pandas as pd
from IPython import get_ipython
from IPython.display import display

from compare_paired import summarize

CODE = "/content/code/MedSAM"
TR = "/content/train_ready_npz"
VA = "/content/work/data/dental_caries_val42/train_ready_npz"
OG = "/content/lite_medsam.pth"
LOCAL_EXP = "/content/work/experiments"
VAL = "dental_caries_val42"
TEST = "dental_caries_test63"

FLAGS = {                                                   # the runs of list B
    "R1": "",                                               # full fine-tuning
    "R2": "--freeze_image_encoder --freeze_prompt_encoder", # frozen encoders, full decoder
    "R3": "--use_lora --lora_scope mask_decoder",           # LoRA decoder q,v, r=8
    "R6": "--use_lora --lora_scope both",                   # LoRA encoder qkv + decoder q,v, r=8
    "R7": "--use_lora --lora_scope image_encoder",          # LoRA encoder qkv only, r=8
}

REPO_D = RUNS_D = None
split = None


def sh(cmd):
    """Run a shell command with its output shown live in the notebook cell."""
    get_ipython().system(cmd)


def init(repo_d):
    """Set the Drive paths. REPO_D = the MedSAM_enh folder on Drive."""
    global REPO_D, RUNS_D, split
    REPO_D = repo_d
    RUNS_D = f"{repo_d}/MedSAM/work_dir/runs"
    split = pd.read_csv(f"{repo_d}/data/split_val42_test63.csv").rename(columns={"file": "case"})[["case", "size_group"]]
    os.chdir(CODE)
    print(f"colab_runs ready. Drive repo: {REPO_D}")


def sync(src, dst, extra=(), tries=6):
    """rsync src -> dst; on failure remount Drive and retry. Already-copied files are skipped."""
    from google.colab import drive
    for t in range(1, tries + 1):
        os.chdir("/content")                                # never sit inside the Drive mount
        os.makedirs(dst, exist_ok=True)
        r = subprocess.run(["rsync", "-a", *extra, src, dst])
        if r.returncode == 0:
            print(f"OK: {src}")
            os.chdir(CODE)
            return
        print(f"rsync code {r.returncode} on try {t}; remounting Drive and retrying")
        try:
            drive.flush_and_unmount()
        except Exception as e:
            print("unmount:", e)
        time.sleep(5)
        drive.mount("/content/drive", force_remount=True)
        time.sleep(5)
    raise RuntimeError(f"Gave up after {tries} tries: {src}")


def copy_data():
    """Training data, validation data (without old predictions) and validation experiments, to the runtime's disk."""
    train_d = f"{REPO_D}/data/dental_caries_segmentation_processed/MedSAM_train/RGB_Teeth/train_ready_npz"
    sync(f"{train_d}/", f"{TR}/")
    sync(f"{REPO_D}/data/{VAL}", "/content/work/data/", extra=["--exclude=segs_*", "--exclude=metrics_*"])
    sync(f"{REPO_D}/experiments/{VAL}", f"{LOCAL_EXP}/")
    print("train npz:", len(os.listdir(TR)), "(expect 414) | val npz:", len(os.listdir(VA)), "(expect 42)")


def copy_test_data(final_runs):
    """For E2: test data, test experiments (B0 = og_ckpt), and the inference checkpoints of the final runs."""
    sync(f"{REPO_D}/data/{TEST}", "/content/work/data/", extra=["--exclude=segs_*", "--exclude=metrics_*"])
    sync(f"{REPO_D}/experiments/{TEST}", f"{LOCAL_EXP}/")
    for r in final_runs:
        sync(f"{RUNS_D}/{r}_infer.pth", "/content/runs/")
    print("test photos:", len(os.listdir(f"/content/work/data/{TEST}/MedSAM_test/RGB_Teeth/imgs")), "(expect 63)")
    print("B0 on test already scored (og_ckpt):", os.path.exists(f"{LOCAL_EXP}/{TEST}/og_ckpt/metrics_per_case.csv"))


def infer_ckpt(run):
    return f"/content/runs/{run}_infer.pth"


def train(method, seed):
    """Train one run on the runtime's disk and export its inference checkpoint. Epochs/patience/lr: trainer defaults."""
    run = f"{method}_s{seed}"
    sh(f"cd {CODE} && python train_one_gpu_multibox.py -train_npz_dir {TR} -val_npz_dir {VA} "
       f"-pretrained_checkpoint {OG} -work_dir /content/runs/{run} -seed {seed} {FLAGS[method]}")
    ck = f"/content/runs/{run}/medsam_lite_multibox_best_val.pth"
    if not os.path.exists(ck):
        raise RuntimeError(f"No {ck}: training failed, read the output above")
    sh(f"cd {CODE} && python export_infer_ckpt.py -ckpt {ck} -out {infer_ckpt(run)}")
    return run


def score(name, inf, ds=VAL, zoom_above=2000, variants=("given", "p20")):
    """Zoom inference, then eval_run, with the boxes as given (<name>) and/or 20% looser (<name>_p20, seed 0 as in E1)."""
    T = f"/content/work/data/{ds}/MedSAM_test/RGB_Teeth"
    n_imgs = len(os.listdir(f"{T}/imgs"))
    for v in variants:
        n = name if v == "given" else f"{name}_p20"
        extra = "" if v == "given" else "-box_perturb 0.2 -perturb_seed 0"
        sh(f'cd {CODE} && python infer_zoom.py -i "{T}/imgs" -o "{T}/segs_{n}" -lite_medsam_checkpoint_path {inf} '
           f"-zoom_above {zoom_above} -device cuda {extra}")
        n_segs = len(os.listdir(f"{T}/segs_{n}")) if os.path.isdir(f"{T}/segs_{n}") else 0
        if n_segs != n_imgs:   # eval_run would otherwise fall back to the old whole-photo inference
            raise RuntimeError(f"infer_zoom wrote {n_segs}/{n_imgs} predictions for {n}; read the output above")
        sh(f"cd {CODE} && python eval_run.py --run_name {n} --ckpt {inf} --repo /content/work --dataset {ds}")


def save(name, ds=VAL, trained=True, variants=("given", "p20")):
    """Copy ONE run back to Drive: checkpoints (if trained), its experiments folders, its predictions."""
    T = f"/content/work/data/{ds}/MedSAM_test/RGB_Teeth"
    if trained:
        sync(f"/content/runs/{name}", f"{RUNS_D}/", extra=["--exclude=medsam_lite_multibox_latest.pth"])
        sync(infer_ckpt(name), f"{RUNS_D}/")
    for v in variants:
        n = name if v == "given" else f"{name}_p20"
        sync(f"{LOCAL_EXP}/{ds}/{n}", f"{REPO_D}/experiments/{ds}/")
        sync(f"{T}/segs_{n}", f"{REPO_D}/data/{ds}/MedSAM_test/RGB_Teeth/")


def load(name, ds=VAL, root=LOCAL_EXP):
    """Per-photo DSC and NSD of one run, or None if it hasn't been scored."""
    p = f"{root}/{ds}/{name}/metrics_per_case.csv"
    return pd.read_csv(p)[["case", "dsc", "nsd"]] if os.path.exists(p) else None


def compare(run, ref="Z", ref_p20="Z_p20", ds=VAL):
    """Per-photo DSC comparison, boxes as given and 20% looser. Rule: sign p <= 0.05 and mean gain >= +0.03."""
    for a, b in [(run, ref), (f"{run}_p20", ref_p20)]:
        m = load(a, ds).merge(load(b, ds), on="case", suffixes=("_c", "_r")).merge(split, on="case")
        print(f"\n{a} vs {b}:  DSC {m.dsc_c.mean():.3f} vs {m.dsc_r.mean():.3f} | NSD {m.nsd_c.mean():.3f} vs {m.nsd_r.mean():.3f}")
        for g in ["all", "large", "small"]:
            mm = m if g == "all" else m[m.size_group == g]
            k = summarize((mm.dsc_c - mm.dsc_r).to_numpy(), 0.03, 0.05)
            print(f"  {g:<5} n={len(mm):>2}  wins/losses {k['wins']:>2}/{k['losses']:<2}  mean {k['mean_diff']:+.3f}  "
                  f"sign p {k['sign_p_one_sided']:.4f}  Wilcoxon p {k['wilcoxon_p']:.4f}  {'PASSES' if k['passes'] else '-'}")


def board(ds=VAL):
    """All trained runs on Drive (both of us), against Z. Used for every decision. Saved as experiments/<ds>/board.csv."""
    root = f"{REPO_D}/experiments"
    z, zp = load("Z", ds, root), load("Z_p20", ds, root)
    rows = [{"run": "Z", "DSC": z.dsc.mean(), "NSD": z.nsd.mean(), "DSC_p20": zp.dsc.mean()}]
    for d in sorted(os.listdir(f"{root}/{ds}")):
        if not re.fullmatch(r"R\d+_s\d+", d) or load(d, ds, root) is None:
            continue
        a, ap = load(d, ds, root), load(f"{d}_p20", ds, root)
        m = a.merge(z, on="case", suffixes=("", "_z"))
        k = summarize((m.dsc - m.dsc_z).to_numpy(), 0.03, 0.05)
        rows.append({"run": d, "DSC": a.dsc.mean(), "NSD": a.nsd.mean(),
                     "DSC_p20": ap.dsc.mean() if ap is not None else float("nan"),
                     "vs Z wins/losses": f"{k['wins']}/{k['losses']}", "vs Z mean": k["mean_diff"],
                     "sign p": k["sign_p_one_sided"], "passes vs Z": k["passes"]})
    df = pd.DataFrame(rows).sort_values("DSC", ascending=False).round(4).reset_index(drop=True)
    try:                                                    # rebuilt complete every time, so overwriting is safe
        df.to_csv(f"{root}/{ds}/board.csv", index=False)
        print(f"Saved {root}/{ds}/board.csv")
    except OSError as e:
        print("Could not write board.csv (Drive dropped?); the table below is still complete:", e)
    display(df)
    return df


def final_table(final, ds=TEST):
    """E2: references and the mean over seeds of each final method. final = {"R6": ["R6_s0", "R6_s1", "R6_s2"], ...}"""
    import numpy as np
    rows = [{"model": n, "DSC": load(n, ds).dsc.mean(), "NSD": load(n, ds).nsd.mean()}
            for n in ["og_ckpt", "B0_p20", "Z", "Z_p20"]]
    for mth, runs in final.items():
        for suf, label in [("", "as given"), ("_p20", "20% looser")]:
            d = [load(r + suf, ds).dsc.mean() for r in runs]
            n = [load(r + suf, ds).nsd.mean() for r in runs]
            rows.append({"model": f"{mth} ({label}), mean of {len(runs)} seeds", "DSC": np.mean(d),
                         "DSC sd": np.std(d, ddof=1), "NSD": np.mean(n), "NSD sd": np.std(n, ddof=1)})
    df = pd.DataFrame(rows).round(4)
    display(df)
    return df
