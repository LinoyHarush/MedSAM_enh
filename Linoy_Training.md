# Linoy's Training

Round 2 of the training runs (plan agreed 2026-10-17, after all seed-0 runs). Shelly's part is in `Shelly_Training.md`. The seed-0 results and the reasoning are in the review file, section 10b.

## Where we are

Seed 0 is done for R1, R2, R3, R6 and R7 (validation DSC 0.862 / 0.846 / 0.824 / 0.853 / 0.859; Z 0.820). Adapting the image encoder is what helps: R1 (full fine-tuning) and R7 (LoRA on the encoder only) lead.

## The plan from here (fixed before anyone looks at test results)

1. **Seeds 1 and 2 for R7 and R1.** Shelly trains seed 1 of each, Linoy seed 2 of each.
2. **R7 extension** (Shelly), only if R7_s1 and R7_s2 are both cut off by the 20-epoch cap while still improving (not merely ending at epoch 19 through early stopping).
3. **Test set:** Shelly scores everything on test; no choice from here on uses test numbers.

| Order | Linoy (this file) | Shelly (`Shelly_Training.md`) |
|---|---|---|
| 0 | – | One-time backup of the results folders |
| A | – | Test set: B0, Z and the five seed-0 models |
| 1 | **R7 seed 2** (~70 min) | **R7 seed 1** (~70 min) |
| 2 | **R1 seed 2** (~25 min) | **R1 seed 1** (~25 min) |
| 3 | – | R7 extension, if both new R7 seeds cap out |
| E | – | Test set: the new runs, and the final tables |

Your two runs don't depend on anything else, so you can start as soon as Shelly has pushed this file and `colab_runs.py`. Times include about 5 minutes of data preparation and scoring per run.

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

## Step 1 — R7 seed 2 (LoRA on the encoder only)

Run S2 in full, including `cr.copy_data()`.

**R7_s2 · Train** (~65 min)
```python
run = cr.train("R7", 2)
```
Check: `Run type: LoRA -> num_epochs 20, patience 5, lr 0.0001`, `LoRA layers: decoder 0, encoder 10`, `trainable: 59,392`, `encoder BN change 0.00e+00`, and after training `Merged 10 LoRA modules`.

**R7_s2 · Score, save, compare with Z**
```python
run = "R7_s2"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()
```
Tell Shelly when R7_s2 is saved: her Step 3 (the extension check) waits for it.

## Step 2 — R1 seed 2 (full fine-tuning)

Same session is fine if the runtime is still alive (no need to re-run S2).

**R1_s2 · Train** (~25 min)
```python
run = cr.train("R1", 2)
```
Check: `Run type: full model -> num_epochs 10, patience 3, lr 5e-05`, `trainable: 9,791,300`, and `No LoRA weights found; exporting model weights as-is`. Here `encoder BN change` above 0 is expected.

**R1_s2 · Score, save, compare with Z**
```python
run = "R1_s2"
cr.score(run, cr.infer_ckpt(run))
cr.save(run)
cr.compare(run)
cr.board()
```
Send Shelly the "Done. Best validation Dice … at epoch …" line, the `compare` output and the `log.csv` of each run.
