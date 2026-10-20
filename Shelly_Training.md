# Shelly's Training

Round 2 of the training runs (plan agreed 2026-10-17, after all seed-0 runs). Linoy's part is in `Linoy_Training.md`. The seed-0 results and the reasoning are in the review file, section 10b.

## Where we are

Seed 0 is done for R1, R2, R3, R6 and R7 (validation DSC 0.862 / 0.846 / 0.824 / 0.853 / 0.859; Z 0.820). Adapting the image encoder is what helps: R1 (full fine-tuning) and R7 (LoRA on the encoder only) lead.

## The plan from here (fixed before anyone looks at test results)

1. **Seeds 1 and 2 for R7 and R1.** Shelly trains seed 1 of each, Linoy seed 2 of each.
2. **R7 extension, only if R7 caps out on both new seeds,** i.e. R7_s1 and R7_s2 are both cut off by the 20-epoch cap while still improving, like R7_s0 (best epoch 17, early stopping still had patience left). A run that reaches epoch 19 because early stopping ends it right there does *not* count. Then the best of R7_s0 / s1 / s2 by validation DSC (boxes as given, from the board) continues from its best checkpoint for up to 20 more epochs, once, as a new run `R7_s<k>_ext`. Shelly runs it. If only one of the two caps out, send Claude the board and decide together.
3. **Test set:** the five seed-0 models are scored now (Step A). Every run of this round is scored on test too, whatever it shows (Step E). No choice from here on uses test numbers.

| Order | Shelly (this file) | Linoy (`Linoy_Training.md`) |
|---|---|---|
| 0 | One-time backup of the results folders | – |
| A | Test set: B0, Z and the five seed-0 models | – |
| 1 | **R7 seed 1** (~70 min) | **R7 seed 2** (~70 min) |
| 2 | **R1 seed 1** (~25 min) | **R1 seed 2** (~25 min) |
| 3 | R7 extension, if both new R7 seeds cap out (~70 min at most) | – |
| E | Test set: the new runs, the final tables, and R7 vs R1 | – |
| F | Figures and the method table for the presentation | – |

Times include about 5 minutes of data preparation and scoring per run.

## Safety: how earlier results are protected

Everything already on Drive (R1_s0, R2_s0, R3_s0, R6_s0, R7_s0, Z, B0, the E1 runs, `board.csv`) must stay exactly as it is.
1. **Every new run has a new name.** Shelly trains seed **1**, Linoy seed **2**; nobody trains seed 0 again. The extension gets its own name, `R7_s<k>_ext`.
2. **The script refuses to overwrite.** `cr.train`, `cr.extend` and `cr.save` stop with an error if the name already exists on Drive. If you see that error, the seed or name is wrong: fix it, don't force it. `overwrite=True` is only for finishing an interrupted save of *your own* run in the *same* session (see "If the runtime resets").
3. **Only `cr.save` writes to Drive**, and only the run's own folders. Never `rsync` or copy whole folders (`experiments`, `runs`, `data`) back to Drive, and never delete or rename anything on Drive by hand. The one file that is rewritten on purpose is `board.csv`, which is rebuilt from all runs each time.
4. **Don't re-run old cells** (the trial run, the first R2 cells, or cells from earlier messages): they don't have these checks.
5. **The extension never touches the original run.** It copies the chosen seed's folder from Drive to the runtime, continues training there, and saves as `R7_s<k>_ext`.
6. **Git:** pull before you push, and only Shelly edits the md files and `colab_runs.py`, so the two of you never push conflicting versions.
7. **One-time backup (Shelly, Step 0).** Before anything else, a copy of the results folders is made on Drive.

## The board, and why the two of you never overwrite each other

**What gets saved where.** After each run, `save()` copies to Drive *only that run's own folders*, all named after the run (e.g. `R3_s0`):
- `MedSAM/work_dir/runs/R3_s0/` (log, curves, best checkpoint) and `R3_s0_infer.pth`;
- `experiments/dental_caries_val42/R3_s0/` and `R3_s0_p20/` (per-photo scores, `summary.json`);
- `data/dental_caries_val42/MedSAM_test/RGB_Teeth/segs_R3_s0/` and `segs_R3_s0_p20/` (predictions).

Run names never repeat between the two of you, so two sessions saving at the same time write to different folders and can't clash.

**The one file that would clash** is `experiments/dental_caries_val42/runs_summary.csv`: `eval_run.py` appends a row to the copy *on the runtime's disk*, and that copy only holds what was on Drive when the session started plus your own runs. Copying it back would replace the Drive copy and delete the other person's rows. So `save()` never copies it, and the Drive copy simply stops being updated. Ignore it from now on and use the board, or its saved copy `board.csv`.

**The board** (`board()`) is the shared results table. It reads every run's `metrics_per_case.csv` straight from Drive, so it shows both of your runs as soon as each `save()` has finished, and it rebuilds the table from scratch every time it's run. It also saves the table to Drive as `experiments/dental_caries_val42/board.csv`, overwriting the previous one. That's safe: each version is rebuilt from all runs on Drive, so whoever runs `board()` last writes the most complete table, and no row can be lost. Open `board.csv` in Drive to see the results without running anything. One row per trained run on validation (42 photos), plus Z, sorted by DSC:

| Column | Meaning |
|---|---|
| `run` | run name, `<method>_s<seed>` |
| `DSC`, `NSD` | mean over the 42 photos, boxes as given. **`DSC` is what every decision uses.** |
| `DSC_p20` | mean DSC with the same seeded 20% looser boxes. Reported, never decides |
| `vs Z wins/losses` | on how many photos the run scored higher / lower than Z |
| `vs Z mean` | mean per-photo DSC gain over Z |
| `sign p` | one-sided sign test: chance of at least this many wins if the run were no better than Z |
| `passes vs Z` | `True` if `sign p` ≤ 0.05 **and** mean gain ≥ +0.03, the same rule as the zoom decision |

`compare(run)` gives the same comparison for one run in more detail: large and small photos separately, NSD, the Wilcoxon test, and the 20% looser boxes against `Z_p20`.

## Session setup (run at the start of every Colab session, in this order)

**S0 — GPU runtime.** Runtime → Change runtime type → **T4 GPU**. Training on CPU is not an option.

**Before S1, run the notebook's two usual setup cells**, as at the start of every session:
1. the **daily setup cell** (mounts Drive, pulls the repo with your own `GITHUB_PAT`);
2. the **`pip install -e .` cell**. A fresh runtime doesn't have the trainer's dependencies (e.g. `monai`), so this is needed every time the runtime is new.

The git identity cell (your name for commits) is only needed if you commit from Colab.

**S1 — copy the code and the model to the runtime's disk, check the GPU**
```python
%cd /content
REPO_D = "/content/drive/MyDrive/Deep_Learning_Course/MedSAM/MedSAM_enh"   # the only line to edit if the shared folder sits elsewhere in your Drive
!mkdir -p /content/code/MedSAM /content/work/data /content/work/experiments /content/train_ready_npz /content/runs
!rsync -a --exclude='work_dir' --exclude='__pycache__' "{REPO_D}/MedSAM/" /content/code/MedSAM/
!grep -c "def board" /content/code/MedSAM/colab_runs.py
!cp "{REPO_D}/MedSAM/work_dir/LiteMedSAM/lite_medsam.pth" /content/lite_medsam.pth
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to GPU")
```
Expect `1` from `grep` (the daily setup cell pulled the new `colab_runs.py`; a `0` or "No such file" means it didn't) and `GPU: Tesla T4`.

**S2 — load the helpers and copy the data to the runtime's disk**
```python
import sys; sys.path.insert(0, "/content/code/MedSAM")
import colab_runs as cr          # MedSAM/colab_runs.py: train, score, save, compare, board, ...
cr.init(REPO_D)
cr.copy_data()                   # retries and reconnects Drive by itself if the mount drops
```
Expect `train npz: 414 (expect 414) | val npz: 42 (expect 42)`.

**What the helpers in `MedSAM/colab_runs.py` do** (read the file for details; both of you run the same version, pulled from git by the daily setup cell):

| Function | What it does |
|---|---|
| `cr.train(method, seed)` | trains run `<method>_s<seed>` on the runtime's disk with the trainer's defaults for that run type, then exports its inference checkpoint |
| `cr.score(run, cr.infer_ckpt(run))` | zoom inference (above 2,000 px), then `eval_run`, with the boxes as given (`<run>`) and with the seeded 20% looser boxes (`<run>_p20`). Stops if inference is incomplete, so `eval_run` never falls back to the old whole-photo inference |
| `cr.save(run)` | copies only this run's folders back to Drive (see "The board" above) |
| `cr.compare(run)` | per-photo comparison with Z, all / large / small photos, as given and 20% looser |
| `cr.board()` | the shared results table from Drive, also saved as `board.csv` |
| `cr.sync(src, dst)` | `rsync` that reconnects Drive and retries if the mount drops |
| `cr.guard(name)` | stops with an error if anything named `name` already exists on Drive. `train`, `extend` and `save` call it for you |
| `cr.capped(run)` | `True` if the cap cut the run off while it was still improving (early stopping had patience left); `False` if early stopping ended it; `None` if the run isn't on Drive yet |
| `cr.extend(run)` | continues `run` from its best checkpoint for up to 20 more epochs as a **new** run `<run>_ext`; the original is only read |
| `cr.copy_test_data(runs)` / `cr.test_table(runs)` | for the test set: copy the test photos and checkpoints; the comparison table vs Z and B0 |

**If the runtime resets** (disconnect, idle timeout, usage limit): everything on the runtime's disk is gone. Run the daily setup and pip cells and S0–S2 again, then restart the experiment you were on from its **Train** cell. A run is safe only after its **Save** cell has finished.
- Reset **during training or scoring**: nothing reached Drive, so just start that experiment again from its Train cell.
- A save **stopped with an error but the runtime is still alive** (e.g. Drive gave up): run the same save again with `overwrite=True`, e.g. `cr.save(run, overwrite=True)`. That's the only case for `overwrite=True`.
- Reset **in the middle of a save**: part of the run is on Drive and the Train cell will now refuse to start, by design. Don't delete anything yourself; tell Shelly, who checks with Claude.

**Test-set sessions** (Shelly's Step A and Step E) don't need the training data: in S2 run only the first three lines, without `cr.copy_data()`.

## Step 0 — one-time backup (run once, before anything else)

Copies the current scores and checkpoints to a folder **outside** the git repo, so nothing can be lost even by mistake.
```python
import os
BK = "/content/drive/MyDrive/Deep_Learning_Course/MedSAM/results_backup_2026-10-17"   # outside the repo
assert not os.path.exists(BK), "The backup already exists: don't run this cell twice"
cr.sync(f"{REPO_D}/experiments", f"{BK}/")
cr.sync(f"{REPO_D}/MedSAM/work_dir/runs", f"{BK}/")
print(sorted(os.listdir(BK)), "\n", sorted(os.listdir(f"{BK}/runs")))
```
Expect `experiments` and `runs` in the backup, and the five `R*_s0` folders plus their `_infer.pth` files in `runs`. Never write into this folder afterwards.

## Step A — test set: B0, Z and the five seed-0 models

A test-set session: in S2 skip `cr.copy_data()`.

**A1 · Copy the test photos and the five checkpoints**
```python
TS = cr.TEST                                            # "dental_caries_test63"
RUNS0 = ["R1_s0", "R2_s0", "R3_s0", "R6_s0", "R7_s0"]
cr.copy_test_data(RUNS0)
```
Expect `test photos: 63` and `B0 on test already scored (og_ckpt): True` (B0 with boxes as given was scored on 2026-10-14 and is reused).

**A2 · The references: Z (as given and 20% looser) and B0 with 20% looser boxes**
```python
cr.score("Z", cr.OG, ds=TS)
cr.score("B0", cr.OG, ds=TS, zoom_above=100000, variants=("p20",))   # whole photo, as B0
cr.save("Z", TS, trained=False)
cr.save("B0", TS, trained=False, variants=("p20",))
```

**A3 · The five models** (about 1 min each)
```python
for r in RUNS0:
    cr.score(r, cr.infer_ckpt(r), ds=TS)
    cr.save(r, TS, trained=False)
```

**A4 · The comparison table** (every model vs Z and vs B0; saved to Drive)
```python
t0 = cr.test_table(RUNS0, out_name="test_comparison_seed0.csv")
```
Columns: per-photo comparison of each model with Z and with B0, with the boxes as given and 20% looser. **Wilcoxon p (Holm)** is the paired test corrected for comparing five models with the same reference: report that one. Send Claude the table.

Optional, large vs small photos per model:
```python
for r in RUNS0:
    cr.compare(r, "Z", "Z_p20", ds=TS)
```

## Step 1 — R7 seed 1 (LoRA on the encoder only)

A training session: run S2 in full, including `cr.copy_data()`.

**R7_s1 · Train** (~65 min)
```python
run = cr.train("R7", 1)
```
Check: `Run type: LoRA -> num_epochs 20, patience 5, lr 0.0001`, `LoRA layers: decoder 0, encoder 10`, `trainable: 59,392`, `encoder BN change 0.00e+00`, and after training `Merged 10 LoRA modules`.

**R7_s1 · Score, save, compare with Z**
```python
run = "R7_s1"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()
```

## Step 2 — R1 seed 1 (full fine-tuning)

Same session is fine if the runtime is still alive (no need to re-run S2).

**R1_s1 · Train** (~25 min)
```python
run = cr.train("R1", 1)
```
Check: `Run type: full model -> num_epochs 10, patience 3, lr 5e-05`, `trainable: 9,791,300`, and `No LoRA weights found; exporting model weights as-is`. Here `encoder BN change` above 0 is expected.

**R1_s1 · Score, save, compare with Z**
```python
run = "R1_s1"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()
```

## Step 3 — R7 extension (only if both new R7 seeds cap out)

Wait until Linoy's R7_s2 is on the board.

**3a · Check the rule**
```python
b = cr.board()
caps = {r: cr.capped(r) for r in ["R7_s1", "R7_s2"]}
r7 = b[b.run.isin(["R7_s0", "R7_s1", "R7_s2"])].sort_values("DSC", ascending=False)
print("\nCut off by the cap:", caps)
if None in caps.values():
    print("WAIT: not every R7 seed is on Drive yet")
else:
    print("Best R7 seed by validation DSC:", r7.run.iloc[0])
    print("RUN THE EXTENSION" if all(caps.values()) else
          "Don't extend" if not any(caps.values()) else "Only one was cut off: send Claude the board first")
```

**3b · The extension** (only if 3a says RUN THE EXTENSION; up to ~70 min)
```python
BEST = "R7_s?"                  # the "Best R7 seed" printed by 3a
ext = cr.extend(BEST)           # new run BEST_ext; the original folder is only read
```
Check: `Resumed from ...: continuing at epoch N`, where N is the best epoch + 1; the run stops at epoch N + 19 at the latest, or earlier if validation Dice doesn't improve for 5 epochs.

**3c · Score, save, compare**
```python
cr.score(ext, cr.infer_ckpt(ext))
cr.save(ext)
cr.compare(ext)
cr.board()
```

## Step E — test set: the new runs, and the final tables (when all runs of this round are on the board)

A test-set session (skip `cr.copy_data()` in S2).

**E1 · Score the new runs on test**
```python
TS = cr.TEST
NEW = ["R7_s1", "R7_s2", "R1_s1", "R1_s2"]          # no extension ran (Step 3a: "Don't extend")
cr.copy_test_data(NEW)
for r in NEW:
    cr.score(r, cr.infer_ckpt(r), ds=TS)
    cr.save(r, TS, trained=False)
```
(Z, B0 and the seed-0 models are already scored from Step A: `cr.copy_test_data` brings their scores from Drive, so they're reused, not redone.)

**E2 · The final tables**
```python
cr.final_table({"R1": ["R1_s0", "R1_s1", "R1_s2"], "R7": ["R7_s0", "R7_s1", "R7_s2"]})
t = cr.test_table(["R1_s0", "R1_s1", "R1_s2", "R7_s0", "R7_s1", "R7_s2"] + [r for r in NEW if r.endswith("_ext")],
                  out_name="test_comparison_final.csv")
```
`final_table` gives the mean ± sd over the 3 seeds of R1 and R7 (the main result); `test_table` gives every seed against Z and B0. If the extension ran, report it as its own row ("best R7 seed, trained longer"), not as R7's average.

**E3 · R7 vs R1, directly** (fixed on 2026-10-19 before any round-2 test result was seen; run once, report as is)

For each test photo, each method's DSC is averaged over its 3 seeds; then a two-sided Wilcoxon test compares R7 with R1. Boxes as given is the main comparison, 20% looser is secondary.
```python
from compare_paired import summarize
def seed_mean(runs, suf=""):
    m = None
    for r in runs:
        d = cr.load(r + suf, TS)[["case", "dsc"]].rename(columns={"dsc": r})
        m = d if m is None else m.merge(d, on="case")
    return m.set_index("case").mean(axis=1)
R7, R1 = ["R7_s0", "R7_s1", "R7_s2"], ["R1_s0", "R1_s1", "R1_s2"]
for suf, label in [("", "as given (main)"), ("_p20", "20% looser")]:
    a, b = seed_mean(R7, suf), seed_mean(R1, suf)
    k = summarize((a - b.loc[a.index]).to_numpy(), 0.03, 0.05)
    print(f"R7 vs R1, {label}: DSC {a.mean():.3f} vs {b.mean():.3f} | wins/losses {k['wins']}/{k['losses']} | "
          f"mean {k['mean_diff']:+.3f} | Wilcoxon p {k['wilcoxon_p']:.4f}")
```

## Step F — figures and the method table (after Step E, same session)

Everything here only **reads** scores, logs and predictions from Drive, and only **writes** new files into `experiments/dental_caries_test63/figures/` on Drive. Run in the same session as Step E (the test photos must be on the runtime's disk for F6).

**F1 · Load the figures module**
```python
import importlib, colab_figures as cf
importlib.reload(cf)
```

**F2 · The method table** (trainable weights, epochs, training time per run from the logs, validation and test DSC/NSD with sd over seeds, DSC with 20% looser boxes)
```python
t = cf.method_table()          # also saved as figures/method_table.csv
```

**F3 · Overview: every method on test (boxes as given)**: B0 → Z → the trained models; R1 and R7 are 3-seed means with the seed range
```python
cf.fig_overview()
```

**F4 · Large vs small photos**: zoom lifts the large photos, training the small ones (B0, Z, R1, R7)
```python
cf.fig_size_groups()
```

**F5 · Robustness**: each method with the boxes as given vs 20% looser
```python
cf.fig_robustness()
```

**F6 · Training curves**: validation Dice per epoch, 3 seeds of R1 vs 3 seeds of R7 (dot = the epoch that was kept)
```python
cf.fig_training_curves()
```

**F7 · Example photos**: ground truth, B0, Z and R7, four rows chosen by a fixed rule (not by eye): the 2 small photos where R7 gains most over Z, the large photo where zoom gains most over B0, and the photo where R7 loses most to Z. R7 = its median seed on test. B0's masks are recomputed for these 4 photos if they aren't stored (a few seconds).
```python
cf.fig_examples()
```
All PNGs are saved at 200 dpi in `experiments/dental_caries_test63/figures/` on Drive, ready for the slides. Send Claude the E2 tables, the E3 line, the method table and the figures.

