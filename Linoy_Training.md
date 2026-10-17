# Linoy's Training

Linoy's half of the training experiments of **run list B** (review file `MedSAM_enh_repo_review.md`, section 10b), in the order to run them, with every Colab cell needed. Shelly's half is in `Shelly_Training.md`. Written 2026-10-17, after the trial run passed (list B step 1).

## Who runs what

| Order | Linoy (this file) | Shelly (`Shelly_Training.md`) |
|---|---|---|
| 1 | **R3 seed 0**: LoRA decoder q,v (~40 min) | **R2 seed 0**: frozen encoders, full decoder (~30 min) |
| 2 | **R1 seed 0**: full fine-tuning (~40 min) | **R6 seed 0**: LoRA encoder + decoder (~45 min) |
| 3 | – | ⏸ **Decision 1**: which method(s) get more seeds |
| 4 | **R7 seed 0**: LoRA encoder only, *only if R6_s0 > R3_s0* | **Seeds 1 and 2** of the chosen method(s) |
| 5 | – | ⏸ **Decision 2**: which models go to the final test |
| 6 | E3 AlphaDent, if time allows | **E2 final test** (63 photos) |

Shelly has the larger share (4–6 training runs plus E2; Linoy 2–3 runs), since she has more time.

Times are rough upper bounds on a T4 (early stopping usually ends runs sooner), plus about 2 min of data preparation and about 3 min of scoring per run. None of your runs waits for a decision except R7, which needs Shelly's R6_s0 on the board: by the time R3 and R1 are done, it should be.

## Rules for both of us

- **Trainer defaults decide the training settings.** Epoch cap, patience and learning rate come from the run type (full model 10/3/5e-5; frozen encoders 15/4/5e-5; LoRA 20/5/1e-4); box jitter is a random 0–20% of the box size per side. Don't pass extra flags.
- **Every run is scored the same way:** zoom inference above 2,000 px with the boxes as given (`<run>`, decides) and with the same seeded 20% looser boxes (`<run>_p20`, reported, never decides). The `score()` helper does both, and always runs `infer_zoom` before `eval_run`.
- **Run names:** `<method>_s<seed>`, e.g. `R3_s0`. Same names on Drive for checkpoints (`MedSAM/work_dir/runs/`), experiments and predictions, so the two of you never overwrite each other.
- **Save only your own run** with `save()`. Never rsync the whole `experiments` folder back: its `runs_summary.csv` would overwrite the other person's rows. `board()` rebuilds the table from the per-run folders on Drive.
- **The 63 test photos are used once**, in E2 (Shelly), after every decision is made.
- After each experiment, send Shelly the "Done. Best validation Dice … at epoch …" line and the output of `compare()`, so she can pass them on to Claude.
- **Your Drive path:** S1 assumes the project folder is at `MyDrive/Deep_Learning_Course/MedSAM/MedSAM_enh` in *your* Drive too (for a shared folder, via "Add shortcut to Drive"). If it isn't, change `REPO_D` in S1; nothing else needs editing.


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

## Experiment 1 — R3 seed 0 (LoRA decoder q,v)

**R3 · Train**
```python
run = cr.train("R3", 0)
```
Check at the top of the output: `Run type: LoRA -> num_epochs 20, patience 5, lr 0.0001`, `LoRA layers: decoder 14, encoder 0`, `trainable: 47,104`, the two jitter lines, `train: 414 photos -> 1278 lesion samples`, `validation: 42 photos -> 149 lesion samples`, `encoder BN change 0.00e+00` on every epoch, and after training `Merged 14 LoRA modules`.

**R3 · Score, save, compare with Z**
```python
run = "R3_s0"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()          # shows Shelly's runs too, once she has saved them
```

## Experiment 2 — R1 seed 0 (full fine-tuning)

**R1 · Train**
```python
run = cr.train("R1", 0)
```
Check: `Run type: full model -> num_epochs 10, patience 3, lr 5e-05`, `trainable: 9,791,300` (everything), an empty "Frozen (eval mode)" list, and after training `No LoRA weights found; exporting model weights as-is`. Here `encoder BN change` **should** be above 0: the encoder is trained, so its BatchNorm statistics move. That's expected for R1 only.

**R1 · Score, save, compare with Z**
```python
run = "R1_s0"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()
```

## Experiment 3 — R7 seed 0 (LoRA encoder qkv only), only if R6_s0 > R3_s0

**Gate:** run `board()` (or ask Shelly). Skip this experiment if R6_s0's `DSC` is at or below R3_s0's, or if R6_s0 isn't on the board yet (then wait, or do E3 preparation first). R7 answers "where does R6's gain come from: the encoder or the decoder?", which only matters if R6 beat R3.

**R7 · Train**
```python
run = cr.train("R7", 0)
```
Check: `Run type: LoRA -> num_epochs 20, patience 5, lr 0.0001`, `LoRA layers: decoder 0, encoder 10`, `trainable: 59,392`, `encoder BN change 0.00e+00`, and after training `Merged 10 LoRA modules`.

**R7 · Score, save, compare with Z**
```python
run = "R7_s0"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()
```

## Experiment 4 — E3 AlphaDent (only if time allows, after Shelly's E2)

External test on a second dataset (another clinic and camera): B0, Z and the final model(s) of E2, scored exactly as in E2. AlphaDent first has to be converted to our npz layout (images, boxes, binary caries masks), so the cells for this step will be written together with Claude when we get there.
