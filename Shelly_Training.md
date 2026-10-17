# Shelly's Training

Shelly's half of the training experiments of **run list B** (review file `MedSAM_enh_repo_review.md`, section 10b), in the order to run them, with every Colab cell needed. Linoy's half is in `Linoy_Training.md`. Written 2026-10-17, after the trial run passed (list B step 1).

## Who runs what

| Order | Shelly (this file) | Linoy (`Linoy_Training.md`) |
|---|---|---|
| 1 | **R2 seed 0**: frozen encoders, full decoder (~30 min) | **R3 seed 0**: LoRA decoder q,v (~40 min) |
| 2 | **R6 seed 0**: LoRA encoder + decoder (~45 min) | **R1 seed 0**: full fine-tuning (~40 min) |
| 3 | ⏸ **Decision 1**: which method(s) get more seeds (needs R2, R3, R6 seed 0 on the board) | – |
| 4 | **Seeds 1 and 2** of the chosen method(s): 2 runs, or 4 if two methods qualify | **R7 seed 0**: LoRA encoder only, *only if R6_s0 > R3_s0* |
| 5 | ⏸ **Decision 2**: which models go to the final test | – |
| 6 | **E2 final test** (63 photos) | E3 AlphaDent, if time allows |

Shelly has the larger share (4–6 training runs plus E2; Linoy 2–3 runs), since she has more time.

Times are rough upper bounds on a T4 (early stopping usually ends runs sooner), plus about 2 min of data preparation and about 3 min of scoring per run.

## Rules for both of us

- **Trainer defaults decide the training settings.** Epoch cap, patience and learning rate come from the run type (full model 10/3/5e-5; frozen encoders 15/4/5e-5; LoRA 20/5/1e-4); box jitter is a random 0–20% of the box size per side. Don't pass extra flags.
- **Every run is scored the same way:** zoom inference above 2,000 px with the boxes as given (`<run>`, decides) and with the same seeded 20% looser boxes (`<run>_p20`, reported, never decides). The `score()` helper does both, and always runs `infer_zoom` before `eval_run`.
- **Run names:** `<method>_s<seed>`, e.g. `R2_s0`. Same names on Drive for checkpoints (`MedSAM/work_dir/runs/`), experiments and predictions, so the two of you never overwrite each other.
- **Save only your own run** with `save()`. Never rsync the whole `experiments` folder back: its `runs_summary.csv` would overwrite the other person's rows. `board()` rebuilds the table from the per-run folders on Drive.
- **The 63 test photos are used once**, in E2, after every decision is made.
- After each experiment, send Claude (through Shelly) the "Done. Best validation Dice … at epoch …" line and the output of `compare()`.


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

**If the runtime resets** (disconnect, idle timeout, usage limit): everything on the runtime's disk is gone. Run the daily setup and pip cells and S0–S2 again, then restart the experiment you were on from its **Train** cell. A run is safe only after its **Save** cell has finished.

## Experiment 1 — R2 seed 0 (frozen encoders, full decoder)

**R2 · Train**
```python
run = cr.train("R2", 0)
```
Check at the top of the output: `Run type: frozen encoders -> num_epochs 15, patience 4, lr 5e-05`, `trainable: 4,058,340`, the two jitter lines, `train: 414 photos -> 1278 lesion samples`, `validation: 42 photos -> 149 lesion samples`, and `encoder BN change 0.00e+00` on every epoch.

**R2 · Score, save, compare with Z**
```python
run = "R2_s0"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
```

## Experiment 2 — R6 seed 0 (LoRA encoder qkv + decoder q,v)

**R6 · Train**
```python
run = cr.train("R6", 0)
```
Check: `Run type: LoRA -> num_epochs 20, patience 5, lr 0.0001`, `LoRA layers: decoder 14, encoder 10`, `trainable: 106,496`, `encoder BN change 0.00e+00`, and after training `Merged 24 LoRA modules`.

**R6 · Score, save, compare with Z**
```python
run = "R6_s0"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()          # shows Linoy's runs too, once she has saved them
```

## Decision 1: which method gets seeds 1 and 2 (when R2_s0, R3_s0 and R6_s0 are all on the board)
```python
cr.board()
```
- **Seeds go to the best of R2_s0 / R3_s0 / R6_s0** by validation DSC with the boxes as given (the `DSC` column), **and to any other of the three within 0.03 of it**. Differences below about 0.03 are noise on 42 photos, so a close runner-up needs its own seeds before we can call it worse.
- Shelly runs seeds 1 and 2 of every method chosen.
- **R7 gate** (for Linoy): R7 runs only if R6_s0's DSC is higher than R3_s0's. Tell Linoy which way it went.
- Shelly sends the board to Claude and we confirm the choice together before any seed run starts. R1 isn't part of this choice (list B keeps it as the upper-bound check); if R1 comes out clearly best, we discuss it then.

## Experiment 3 — seeds 1 and 2 of the chosen method(s)

**Seeds · Train, score, save** (2 runs per method, about 1–1.5 h per method. If the runtime resets, remove the runs already on the board from `TODO` and run the cell again)
```python
SEED_METHODS = ["R?"]          # from Decision 1, e.g. ["R6"] or ["R6", "R2"]
TODO = [(m, s) for m in SEED_METHODS for s in (1, 2)]
for mth, seed in TODO:
    run = cr.train(mth, seed)
    cr.score(run, cr.infer_ckpt(run))
    cr.save(run)
    cr.compare(run)
cr.board()
```

## Decision 2: which models go to the final test (when all seed runs, and Linoy's R1 and R7, are on the board)
```python
cr.board()
```
Decision 1 picked which methods deserve 3 seeds; Decision 2 picks, from those, which ones we report on the 63 test photos. It's the last choice made on validation, because the test set may only be used once.
- **Final method** = the method with the highest validation DSC averaged over its 3 seeds (boxes as given). One seed can be lucky; the 3-seed mean is the fair comparison.
- **A runner-up** whose 3-seed mean is within 0.03 goes into E2 as well, so the test can show whether the two really differ.
- E2 always includes **B0 and Z** as references, so `FINAL` in E2a only lists the trained runs.
- Send the board to Claude and we fix the list together **before** E2 runs. After E2, nothing is changed or re-chosen.

## Experiment 4 — E2 final test (63 test photos, used once)

**E2a · Copy the test data and the final checkpoints**
```python
TS = cr.TEST                                       # "dental_caries_test63"
FINAL = {"R?": ["R?_s0", "R?_s1", "R?_s2"]}        # from Decision 2; add a second method if a runner-up qualifies
cr.copy_test_data([r for runs in FINAL.values() for r in runs])
```
Expect `test photos: 63` and `B0 on test already scored (og_ckpt): True`.

**E2b · The two references on test: Z, and B0 with 20% looser boxes**
```python
cr.score("Z", cr.OG, ds=TS)                                            # Z and Z_p20
cr.score("B0", cr.OG, ds=TS, zoom_above=100000, variants=("p20",))     # B0_p20; B0 with boxes as given = og_ckpt (scored 2026-10-14)
cr.save("Z", TS, trained=False)
cr.save("B0", TS, trained=False, variants=("p20",))
```

**E2c · The final models on test**
```python
for runs in FINAL.values():
    for r in runs:
        cr.score(r, cr.infer_ckpt(r), ds=TS)
        cr.save(r, TS, trained=False)
```

**E2d · Results: per seed against Z and B0, then the mean over seeds**
```python
for runs in FINAL.values():
    for r in runs:
        cr.compare(r, "Z", "Z_p20", ds=TS)
        cr.compare(r, "og_ckpt", "B0_p20", ds=TS)
cr.final_table(FINAL)
```
Send Claude the full output of E2d. This is the reported result.
