# MedSAM_enh repo review against the presentation's plan

**Date:** 2026-10-10, updated 2026-10-14 (sections 4, 6 and 7)
**Repo state reviewed:** `main` at `97febca` (2026-10-10) and `43d015f` (2026-10-14), plus the Colab notebook (versions of 2026-10-10 and 2026-10-14)
**Plan reviewed against:** slides "First, fix the training signal" (steps 1–5) and "Next runs on the corrected pipeline" (B0, R1–R5)

## Summary

- **Update 2026-10-14:** `diagnose_bbox_survival.py` is fixed and has been run on all 414 training photos (sections 4 and 6). The main finding: 59 of 1,278 lesions vanish at 256 px, all in photos above 2,000 px, and 38% of lesions cover fewer than 20 pixels at 256 px, because large photos are shrunk 10–30×. The agreed order of next steps is in section 7.
- `train_one_gpu_multibox.py` is not called anywhere in the notebook yet; the last training cells still use `train_one_gpu.py`. None of the trainer fixes below have been made yet.
- `train_one_gpu_multibox.py` is a good base: it trains on one stored box per lesion, `-bbox_shift` works, it has no default resume path, and it has a validation loop that saves the best checkpoint. Before the next runs it still needs fixes for plan steps 2, 4 and 5. It has no LoRA, so R3–R5 can't run on it.
- `preprocess_tooth_ds.py`, which creates the binary masks and `bboxes.json`, is now in git (`MedSAM/`, commit `97febca`). Its box filter still needs a fix (section 2).
- Inference handles several boxes per image correctly. The mismatch is on the training side, which the new trainer mostly fixes.

## How we checked

- Read every data, training and evaluation script in the repo, plus the notebook's cells and outputs.
- Dry-ran the dataset code of `train_one_gpu_multibox.py` and the whole of `diagnose_bbox_survival.py` on synthetic 3000×2000 images. The test lesions were a large lesion, a 6×6 px lesion, an L-shaped lesion whose box contains a second lesion, and a 1-px-wide crack.
- The PyTorch training and validation loops were **not** run, because PyTorch couldn't be installed in the review environment. Findings about them come from reading the code.

## 1. Repo hygiene

| Item | Finding |
|---|---|
| `preprocess_tooth_ds.py` | **Fixed: committed as `MedSAM/preprocess_tooth_ds.py` (commit `97febca`), same content as the notebook version.** Notebook cells 51–53 still write and run it from the repo root (`%%writefile preprocess_tooth_ds.py`, `python preprocess_tooth_ds.py`). Delete cell 51 and run `MedSAM/preprocess_tooth_ds.py` instead, so nobody overwrites the committed file with an old copy. |
| `train_one_gpu.py` `-resume` default | Still `workdir/medsam_lite_latest.pth`. If `-work_dir` is left at its default, a fresh run silently resumes an old run. The new trainer fixed this, but the old script is still used for LoRA. |

## 2. Data pipeline (as the notebook runs it)

```
COCO export
  └─ MedSAM/preprocess_tooth_ds.py  binary mask PNG + bboxes.json (one box per external contour, contourArea > 5)
      ├─ pre_rgb_bbox_datasets.py   test: imgs/*.npz (imgs, boxes) + gts/*.npz (gts)  --binary_mask      → inference + metrics
      └─ pre_rgb_bbox_train.py      train: one npz per photo (imgs, gts, boxes)        --binary_mask      → train_ready_npz
            ├─ npz_to_npy.py        old trainer input (boxes dropped)
            └─ train_one_gpu_multibox.py reads train_ready_npz directly (boxes kept)
```

**Issues in `preprocess_tooth_ds.py`:**

1. **Thin lesions get no box.** The `> 5` filter uses `cv2.contourArea`, which measures the area enclosed by the outline, not the pixel count. A 1-px-wide crack 181 px long has `contourArea == 0`, so it gets no box. Its pixels stay in the ground-truth mask, so at test time they always count as missed. **Fix:** filter by pixel count, e.g. `(component pixels) > 5` from `connectedComponentsWithStats`. **Checked 2026-10-14 with `count_unboxed_lesions.py`: only 3 lesions are affected (2 in train, 21 px together; 1 in test, 11 px), no photo was dropped, and they make up 0.00% of lesion pixels. Decision: keep the data as it is and mention it as a limitation.**
2. **Boxes can contain other lesions.** One box per external contour is correct, but a lesion's box can overlap a neighbouring lesion (L-shapes, lesions close together). Because the mask is 0/1, anything that later takes "the mask inside the box" also picks up the neighbour (see 3.2).
3. Minor: `xmax = x + w` is one pixel past the lesion's edge. This is harmless.

## 3. `train_one_gpu_multibox.py`

### 3.1 Status per plan step

| Plan step | Status |
|---|---|
| 1 · One box per lesion | **Mostly done.** One sample per stored box. The target is the 0/1 mask inside that box, so overlapping neighbours leak in (3.2). |
| 2 · Drop vanished lesions | **Not met.** Lesions are filtered at full resolution. One that vanishes at 256 px gets a **filled rectangle** as its target (lines 180–182). |
| 3 · Validation split, best val DSC, early stop | **Partly done.** `-val_npz_dir` works and `medsam_lite_multibox_best_val.pth` is saved. Still missing: early stopping, the script that splits off the 42 validation photos, and a 63-photo test folder for `eval_run.py`. |
| 4 · Trainer fixes | **2 of 5 done.** `-bbox_shift` is applied and there is no default resume path. Still open: seed, epoch 0, frozen encoder in eval mode (3.3). |
| 5 · 3 seeds + Wilcoxon | **Not possible yet.** There is no `-seed` flag and no statistics script. |
| R1 full fine-tune, R2 frozen encoders | Supported by the flags. |
| R3–R5 LoRA | **Blocked.** LoRA exists only in `train_one_gpu.py`, which still uses one box drawn around all lesions. |

### 3.2 Correctness issues (with evidence)

| # | Issue | Where | Evidence | Suggested fix |
|---|---|---|---|---|
| A | Lesion that vanishes at 256 px → target becomes a filled rectangle | lines 180–182 | 6×6 px lesion on a 3000 px photo: 0 real pixels at 256 px, target = 4-px filled square | In `__init__`, check that each box still contains lesion pixels at 256 px and skip it if not. Remove the rectangle fallback. |
| B | Box overlapping a neighbouring lesion → both lesions are the target | lines 173–178 | L-shaped lesion whose box contains a second lesion: target had 2 connected components | Per-lesion target (see section 5) |
| C | No seed | whole file | — | `-seed` flag; seed `random`, `numpy`, `torch` (+ CUDA) and the DataLoader generator |
| D | Fresh run starts at epoch 1 and runs `num_epochs − 1` epochs | line 435 `range(start_epoch + 1, num_epochs)` | Same as the old trainer ("ep 1–19" on the slides) | `range(start_epoch, num_epochs)`, and save `epoch + 1` as the resume point |
| E | Frozen TinyViT BatchNorm keeps updating its statistics | lines 316, 434 (`model.train()`) | TinyViT uses `BatchNorm2d` (`tiny_vit_sam.py:29`) | After every `model.train()`, call `model.image_encoder.eval()` when `--freeze_image_encoder` is set |
| F | No early stopping | — | — | `-patience N` on validation DSC |
| G | LR scheduler steps on training loss; its state isn't saved on resume | lines 408, 475 | — | Step on validation loss when a validation set exists; save `lr_scheduler.state_dict()` |

### 3.3 Performance risk (compute budget)

- `__getitem__` decompresses the full-size npz (up to 37.5 MP) from Drive for **every lesion sample**, then resizes it (line 143).
- `__init__` also reads `data["imgs"]` for every file even though it doesn't need the image (line 112), so startup decompresses every photo.
- There are now more samples per epoch than photos, so an epoch could take far longer than the 4–5 h per run assumed on the compute-budget slide.
- **Fix:** resize every photo and mask to 256 px once in `__init__` and keep them in memory (414 × 256×256×3 ≈ 80 MB). All later samples are then fast.

### 3.4 Smaller points

- The validation Dice is per lesion, at 256 px, averaged per batch. That's fine for picking a checkpoint, but it isn't the per-photo, full-resolution DSC from `compute_metrics_binary.py`. Final numbers must still come from `eval_run.py`.
- Training rounds box coordinates (`int(x*s + 0.5)`), while inference truncates them (`int(x*s)` in `CVPR24_LiteMedSAM_infer.py::resize_box_to_256`). The 1-px difference matters only for tiny lesions. Use one shared function for both.
- The epoch log doesn't print the learning rate.

## 4. `diagnose_bbox_survival.py`

**Updated 2026-10-14: fixed and pushed as `43d015f`.** Linoy's rewrite (three "fix diagnose_bbox_survival.py" commits) added an npy mode and skipped unreadable files, but had three bugs, all reproduced on synthetic data:
- It crashed (`UnboundLocalError`) when the first box of a file collapses at 256 px.
- For a collapsed box after a valid one, the CSV pixel counts were copied from the previous box (a 9-px lesion was reported as 120,000 px).
- A Google Drive disconnect was counted as a "corrupt file" and skipped, so a run could finish "successfully" having read almost nothing.

The fixed version also checks **each lesion's own pixels** (via connected components), so it now catches a lesion that vanishes while a neighbour sits inside its box (`lesion_vanished_neighbor_in_box`). It only loads the photo when `--save_failed_vis` is set, and prints the lesion size distribution at 256 px. Results are in section 6.

## 5. Per-lesion target: do we need to drop `--binary_mask`?

**No.** There is a simpler route that keeps every 0/1 mask as it is.

- **Inference doesn't read masks at all.** `CVPR24_LiteMedSAM_infer.py` reads only `imgs` and `boxes`. For each box it runs the decoder once and pastes that mask into `segs` with id `1, 2, 3…`. So the model already handles several boxes per photo: one box per decoder call. The zero-shot numbers back this up: photos with several boxes score *higher* (DSC 0.719) than single-box photos (0.647).
- **Metrics binarize both masks** (`compute_metrics_binary.py`, which `eval_run.py` uses). A ground-truth mask with one id per lesion would score the same as the 0/1 mask. Only the old `compute_metrics.py` would break.
- **Recommended:** keep `--binary_mask` everywhere and derive each lesion's own pixels inside the training dataset:
  1. Run `cv2.connectedComponentsWithStats(gt, connectivity=8)` on the full-resolution mask once per photo. 8-connectivity matches the outlines that `findContours` used to make the boxes.
  2. For each stored box, take the component whose bounding box matches it.
  3. Resize that single lesion to 256 px with nearest-neighbour. Skip it if it's empty there, which also handles step 2.

  This changes no preprocessing, no test data and no evaluation. It also lets `diagnose_bbox_survival.py` report vanished lesions exactly.
- Alternative: `preprocess_tooth_ds.py` saves an extra mask with one id per lesion for the training split only, and the test split stays binary. This works, but it changes more files.

## 6. Per-lesion survival results (2026-10-14)

Run on all 414 training photos (`train_ready_npz`), from Colab's local disk. Output on Drive: `MedSAM/work_dir/bbox_survival_per_lesion/` (CSV + 59 failure pictures).

| | Lesion boxes | Share |
|---|---|---|
| Total | 1,278 (about 3 per photo) | |
| Fine at 256 px | 1,219 | 95.4% |
| Box collapses to zero width or height | 25 | 2.0% |
| Box survives, lesion inside it vanishes | 33 | 2.6% |
| Lesion vanishes, a neighbour's pixels are in its box | 1 | 0.1% |
| Box without a matching lesion | 0 | |

By the photo's longest side:

| Photo longest side | Photos | Lesions | Problem lesions | Under 20 px at 256 | Median lesion at 256 px | Median lesion, full size |
|---|---|---|---|---|---|---|
| up to 640 px | 148 | 333 | 0 | 61 (18%) | 215 px | 387 px |
| 641–1024 | 71 | 240 | 0 | 77 (32%) | 75 px | 789 px |
| 1025–2048 | 19 | 71 | 0 | 37 (52%) | 18 px | 464 px |
| 2049–4000 | 70 | 272 | 27 | 133 (49%) | 22 px | 3,081 px |
| 4001+ | 106 | 362 | 32 | 177 (49%) | 20 px | 7,797 px |

**What it means:**
- All 59 problem lesions come from photos whose long side is above 2,000 px. They are not small in the original photo (median 96 px); they vanish because the photo is shrunk 10–30× to 256 px.
- 485 of 1,278 lesions (38%) cover fewer than 20 pixels at 256 px, mostly in the large photos, where lesions are actually the *largest* at full size. One wrong pixel on a 3-px lesion moves its Dice by 0.3–0.5.
- So the main problem is **downscaling, not tiny lesions**. A rough estimate: cropping a margin around each box before resizing (no upscaling) would cut the under-20-px lesions from 485 to about 20.
- Inference shrinks the test photos (up to 26 MP) the same way, so this likely affects the test scores too, not only training.
- Colab note: reading many files straight from the Drive mount kept disconnecting it ("Transport endpoint is not connected"). Copy the data to `/content` with `rsync` first, run `%cd /content` before copying, run scripts from `/content`, and copy results back to Drive at the end.

## 7. Next steps, in order (2026-10-14)

The order follows what each step depends on: fix the data before measuring anything on it, split the data before deciding anything, decide on zoom before finishing the trainer, and check the trainer before spending GPU hours.

| # | Step | Owner | Why at this point |
|---|---|---|---|
| 1 | ✅ **Done 2026-10-14. Count thin lesions without a box.** A small script over the masks + `bboxes.json`: lesions with more than 5 pixels but no box (the `contourArea` filter in `preprocess_tooth_ds.py`). | Shelly | Fixing the filter changes the test boxes, so every score after it would change. If the count is ~0, keep the data as it is and note it as a limitation. If it's large, fix the filter and regenerate the data now, before anything is measured on it. **Result:** 3 of 1,611 lesions (2 train, 1 test; 32 px in total), no photo dropped → data kept as it is. |
| 2 | ✅ **Done 2026-10-14. Split script: 42 validation / 63 test photos** (fixed seed). Validation as combined npz (`prepare_training_npz.py`), test as its own `MedSAM_test` folder. | Shelly | Every decision from here on (zoom or not, best epoch, learning rate) must be made on validation, so the 63 test photos stay untouched until the final comparison. **Result:** `split_val_test.py`, seed 2024, list in `MedSAM/splits/split_val42_test63.csv`. Val 42 photos / 149 boxes, test 63 photos / 181 boxes; each size × lesion-count group split 40/60. Folders on Drive: `data/dental_caries_val42/` (incl. `train_ready_npz/` for `-val_npz_dir`) and `data/dental_caries_test63/`. |
| 3 | ✅ **Done 2026-10-14. Re-score B0 on validation and on test.** No new inference needed: take the existing `segs_og_ckpt` predictions and subset the per-photo metrics. | Shelly | The reference every run is compared against. Minutes of work. **Result:** B0 validation DSC 0.708 / NSD 0.765; B0 test DSC 0.686 / NSD 0.740. Old fine-tuned runs: all below B0 on validation (−0.035 to −0.070 DSC), frz_20e +0.030 above B0 on test, the rest within ±0.003. The ranking flips between the two parts, so differences of ~0.03 on 42–63 photos are within noise (see section 6 note). |
| 4 | ✅ **Done 2026-10-15. Zero-shot zoom test (Z0).** A crop-around-the-box version of the inference script; run it on the 42 validation photos and compare with B0, overall and for photos above 2,000 px. | Shelly | Needs no training, so it's the cheapest experiment left, and section 6 suggests it's where the biggest gain is. Its answer decides whether training should also use crops (step 6). **Plan fixed in section 9** (one version, never enlarge; decision rule set before running). **Result:** large photos +0.26 DSC (17 of 18 better), small photos no change → crop only photos above 2,000 px (section 9). |
| 5 | **Trainer fixes that don't depend on zoom** (review sections 3.2–3.3): skip the 59 problem boxes, per-lesion target, seed, start at epoch 0, frozen encoder in eval mode, early stopping on validation DSC, caching at 256 px, LoRA in a shared module. | Linoy | Can run in parallel with steps 1–4, since none of them touch the trainer. |
| 6 | ✅ **Decided 2026-10-15: crops for photos above 2,000 px, whole photo otherwise.** If Z0 clearly helps, add a crop option to the trainer (same crop rule as inference); otherwise keep whole photos. | Both | Training and inference must see lesions the same way, or the earlier box mismatch comes back in a new form. |
| 7 | **Short trial run:** 1 epoch on ~20 photos. Check the sample count, that epochs start at 0, that frozen BatchNorm statistics don't change, and that `best_val.pth` is written. | Linoy | Catches bugs in minutes instead of after a 4–5 h run. |
| 8 | **Runs R1–R5** from the "Next runs" slide (plus their zoom versions if step 6 says so), checkpoint chosen on validation, scored once on the 63 test photos with `eval_run.py`. | Both, split across the two Colab accounts | The main experiments, only worth running once steps 1–7 are done. |
| 9 | **Statistics script:** paired Wilcoxon signed-rank test on per-photo DSC and NSD, each run against B0. | Shelly | Can be written during step 8; it only needs the per-photo CSVs. |
| 10 | **External test on AlphaDent** with the best model and B0. | Both | Last: it tests the final choice on unseen data and must not influence any decision before it. |

Also still open from earlier: point notebook cells 52–53 at `MedSAM/preprocess_tooth_ds.py` and delete cell 51 (section 1).

## 8. What the re-scoring (step 3) means for how we work (2026-10-14)

B0 and the four old fine-tuned runs, re-scored on the two parts of the split (no new inference; the existing predictions were split along with the photos):

| Run | DSC val (42) | DSC test (63) | DSC vs B0, val | DSC vs B0, test |
|---|---|---|---|---|
| B0 zero-shot (`og_ckpt`) | **0.708** | **0.686** | – | – |
| frz_20e | 0.674 | 0.715 | −0.035 | +0.030 |
| frz_40e | 0.638 | 0.703 | −0.070 | +0.018 |
| LoRA r8 | 0.668 | 0.688 | −0.040 | +0.002 |
| LoRA r4 | 0.656 | 0.683 | −0.052 | −0.003 |

NSD: B0 0.765 (val) / 0.740 (test). Check: weighting the two parts 42:63 gives back exactly the 105-photo numbers from the presentation.

The ranking flips between the two parts. With a per-photo DSC spread of about 0.23, an average over 42 photos is uncertain by about ±0.035 and one over 63 photos by about ±0.03, so differences of about 0.03 between runs cannot be told apart from chance on these sets. This leads to four working rules:

1. **The old conclusion still holds.** Fine-tuning so far is roughly equal to zero-shot; the presentation's "+0.004" was never a real gain.
2. **Decisions need large effects or paired comparisons.** Compare the same photos run against run (the paired Wilcoxon test of step 9), not two averages, and don't call a winner on validation from a gap of 0.02–0.03.
3. **This is why the 3 seeds matter.** They show how much a run moves by chance, compared with how much the method changes the result.
4. **The test table above is now "used".** That's fine for these old runs, which are already in the presentation, but from now on no decision may look at the 63 test photos. B0 on validation (0.708 / 0.765) is the reference for every decision; B0 on test (0.686 / 0.740) only for the final comparison.

## 9. Step 4 plan: zero-shot zoom test, fixed before running (2026-10-14)

**One version only (Z1, "never enlarge"), agreed 2026-10-14.** For every box, cut a square crop centred on the box: side = 2 × the box's longer side (50% margin on each side), but **never smaller than 256 px of the original photo**, shifted to stay inside the photo and clipped to its size. Each crop goes through exactly the current preprocessing (resize longest side to 256, normalise, pad), and its mask is pasted back into the full-size photo. Script: `MedSAM/infer_zoom.py`.

- Nothing is enlarged beyond today's pipeline. Photos whose long side is 256 px or less get the whole photo as their crop, which makes them identical to B0; the script checks this against `segs_og_ckpt`.
- Expected zoom compared with today (training lesions): about 1.5× in photos up to 640 px, 3× at 641–1,024 px, 5× at 1,025–2,000 px and about 12× above 2,000 px. Lesions under 20 px at the model's input would drop from 485 to about 21.
- Why not enlarge small crops: decided by Shelly. Enlarging adds no image detail, and the zoom from "never smaller than 256 px" already reaches small lesions in all but the smallest photos. A single version also means one comparison instead of two on the same 42 photos.
- The margin (50%) is a common starting value, not tuned. For large photos it gives crops of about 260 px, close to the model's native 256 input. It will not be tuned on validation.

**Decision rule (fixed now, before any zoom result is seen).** Compare Z1 with B0 photo by photo on the 42 validation photos, separately for the 18 large (long side > 2,000 px) and 24 small photos (`MedSAM/compare_paired.py`):

- A group **passes** if Z1 wins on enough photos that the one-sided sign test gives **p ≤ 0.05** (ties, i.e. unchanged photos, are left out; with no ties that is at least 13 of 18 or 17 of 24), **and** the mean per-photo DSC gain is **at least +0.03** (the noise level seen in step 3; smaller gains aren't worth the extra complexity).
- Large passes, small passes → crop every photo, at inference and in training.
- Only large passes → crop only photos whose long side is above 2,000 px.
- Neither passes → no cropping; keep the current pipeline.
- NSD is reported alongside but doesn't change the decision.
- The 63 test photos are not touched in this step.


**Result (2026-10-15).** Z1 on the 42 validation photos, compared with B0 photo by photo (`compare_paired.py`):

| Group | Photos | Z1 better / worse / same | Mean DSC gain | Mean NSD gain | Sign test (one-sided) | Wilcoxon | Rule |
|---|---|---|---|---|---|---|---|
| Large photos (> 2,000 px) | 18 | 17 / 1 / 0 | **+0.260** | **+0.290** | p = 0.0001 | p < 0.0001 | **passes** |
| Small photos | 24 | 11 / 10 / 3 | +0.013 | +0.032 | p = 0.50 | p = 0.81 | fails |
| All | 42 | 28 / 11 / 3 | +0.119 | +0.142 | p = 0.005 | p = 0.001 | (not part of the rule) |

- Self-check passed: the 3 photos whose crops all equal the whole photo were identical to B0.
- 149 boxes, median zoom 3.7× compared with today (max 21×); 8 boxes used the whole photo. CPU time about 15 s per photo.
- **Decision, by the rule fixed above: crop only photos whose long side is above 2,000 px** (`infer_zoom.py -zoom_above 2000`). On validation this gives about **DSC 0.820 / NSD 0.889**, compared with B0's 0.708 / 0.765. That is far more than any fine-tuning run so far.
- For training (step 6) this means: the trainer needs the same crop rule for photos above 2,000 px, so training and inference see lesions at the same scale.
- Not yet done: a visual check of a few large-photo predictions (B0 vs Z1) and of the single photo where Z1 lost, to rule out artefacts before the trainer is built around this.
- The 63 test photos were not used. Zoom (B0 + zoom) becomes one of the methods in the final comparison.

## 10. Training runs: plan (agreed 2026-10-15)

**Decisions (2026-10-15):** we follow the **tight-schedule list (10b)** unless Shelly decides otherwise; the best-case list (10a) is the reference for what we would add with more time. LoRA runs use **learning rate 1e-4** (agreed).

All training runs use the rewritten `train_one_gpu_multibox.py`: one sample per lesion, crops for photos above 2,000 px (same rule as inference), validation on the 42 validation photos for early stopping. Every finished run is scored per photo with `infer_zoom.py -zoom_above 2000` on its exported checkpoint, then `eval_run.py`, and compared photo by photo with **Z** (zero-shot + zoom) using `compare_paired.py`. The 63 test photos are used once, at the end.

**Reference methods (no training):** B0 = zero-shot whole photo (done). **Z = zero-shot + zoom above 2,000 px**: run `infer_zoom.py -zoom_above 2000` once on validation to produce its official predictions (the earlier run cropped every photo).

**Epoch caps** (early stopping usually ends runs sooner; one epoch is now about 305 steps at batch 4, three times the old 104):

| Trainable part | Parameters | Cap (`-num_epochs`) | Patience | Learning rate |
|---|---|---|---|---|
| Full model | 9.79 M | 10 | 3 | 5e-5 |
| Full decoder (frozen encoders) | 4.06 M | 15 | 4 | 5e-5 |
| LoRA, decoder and/or encoder | 47 K – 107 K | 20 | 5 | 1e-4 |

### 10a. Run list A: best case (no time limit), for reference

| ID | Run | Question it answers | Seeds |
|---|---|---|---|
| Z | Zero-shot + zoom above 2,000 px | Reference for everything below | – |
| R2 | Frozen encoders, full decoder | Does a correct training signal help on top of zoom? | 3 |
| R3 | LoRA decoder q,v, r=8 | Does a parameter-efficient decoder update help? | 3 |
| R6 | LoRA encoder (qkv) + decoder (q,v), r=8 | Does adapting the image features help? | 3 |
| R7 | LoRA encoder only (qkv), r=8 | Where does the gain come from: features or decoder? | 3 |
| R1 | Full fine-tuning | Upper bound, or overfitting on 414 photos? | 3 |
| R4, R5 | Best LoRA scope at lr 5e-4 and 3e-5 | Is LoRA sensitive to the learning rate? | 1 each |
| R8 | Best LoRA scope at r=4 and r=16 | Is rank 8 enough? | 1 each |
| R9 | R2 trained on whole photos only (`-zoom_above 100000`), scored with zoom | Does training on the same crops as inference matter? | 1 |
| E1 | Box robustness: Z and the best run with boxes enlarged/shifted by 10–20% | Do results hold with less exact boxes, as a clinician would draw? | – |
| E2 | Final test (63 photos): B0, Z and the best 1–2 runs, paired Wilcoxon | The reported result | – |
| E3 | External test on AlphaDent: B0, Z, best run | Does it generalise to another clinic and camera? | – |

### 10b. Run list B: tight schedule, in order (ACTIVE: this is the list we follow)

| # | Run | Why at this point | Gate |
|---|---|---|---|
| 0 | Z on validation (inference only) | The reference all runs are compared with | – |
| 0b | **E1 box robustness** on validation (inference only): B0 and Z with boxes loosened by 10% and 20% per side (`infer_zoom.py -box_perturb`, seed 0) | Can change two training settings before any GPU time is spent: whether to train on crops, and how much box jitter to train with (added 2026-10-15) | Rules below |
| 1 | Trial: 2 epochs, 20 photos, frozen encoders | Catches bugs in minutes; measures seconds per epoch for planning | All checks in the brief pass |
| 2 | R2 seed 0 | Best earlier setup; first answer to "does training add anything on top of zoom?" | – |
| 3 | R3 seed 0 | The project's parameter-efficient method | – |
| 4 | R6 seed 0 | Encoder adaptation: the most promising untested idea | – |
| 5 | Seeds 1 and 2 for the best of R2 / R3 / R6, and for any run within ~0.03 of it | A result worth reporting needs 3 seeds | After 2–4 are scored |
| 6 | R1 seed 0 | Upper bound / overfitting check; slowest and most likely to overfit | If time allows |
| 7 | R7 seed 0 | Only if R6 beats R3: shows the gain comes from the encoder | If R6 > R3 |
| 8 | E2 final test + Wilcoxon | The reported result; uses the test set once | Always, last on our data |
| 9 | E3 AlphaDent | External validity | If time allows |

Skipped under a tight schedule: the learning-rate and rank sweeps (R4, R5, R8) and the training-on-whole-photos ablation (R9). E1 was moved into this list (step 0b) on 2026-10-15.

**E1 decision rules (fixed 2026-10-15, before running).** B0 under loosened boxes is run through `infer_zoom.py -zoom_above 100000` (whole photo, identical to B0), so both methods see exactly the same boxes.
- **Keep training on crops** if, on the 18 large validation photos, Z still beats B0 with boxes loosened by 20%: one-sided sign test p ≤ 0.05 and mean DSC gain ≥ +0.03 (the rule of section 9). Otherwise stop and rethink the crop rule before any training.
- **Raise box jitter in training** (`-bbox_shift`, e.g. from 5 to 15 px) if either method loses more than 0.05 mean DSC (all 42 photos) going from the given boxes to 20% looser boxes.
- 10% is reported for the shape of the curve; the decisions use 20%.


## 11. Why zoom helps some photos and not others (validation, zoom on every photo, 2026-10-15)

Per-photo DSC of B0 and of zoom on every photo (`zoom_z1`), from `paired_dsc.csv` and `zoom_boxes.csv`:

| Photos | Count | B0 | Zoom | Gain | Zoom better / worse |
|---|---|---|---|---|---|
| Large, one lesion | 6 | 0.548 | **0.895** | +0.347 | 6 / 0 |
| Large, several lesions | 12 | 0.632 | **0.848** | +0.217 | 11 / 1 |
| Small, one lesion | 8 | 0.737 | 0.810 | +0.073 | 4 / 2 |
| Small, several lesions | 16 | 0.811 | 0.794 | −0.017 | 7 / 8 |

- **Several lesions score lower than one lesion once zoom is used** (0.817 vs 0.846 overall; 0.848 vs 0.895 in large photos). With B0 it looked the other way round (0.734 vs 0.656) only because two of B0's three complete misses (DSC 0) were one-lesion photos; without those three photos B0 scores the same for both (0.762 vs 0.765). Multi-lesion photos remain the harder case and are worth a per-lesion look later (smaller lesions, neighbours inside each other's crops, lower model confidence).
- **Small photos are not limited by resolution, so zoom only perturbs them.** Their gain does not depend on how much they were zoomed (correlation 0.05) or on their number of lesions (0.06), but strongly on how good B0 already was (−0.86): where B0 was good, the re-cropped input (less context, different contrast normalisation) moves the score down; where B0 was poor, it moves it up. Median zoom for small photos that lost was only 1.5×. This supports the decision to zoom only photos above 2,000 px.
- The analysis explains the decision of section 9; it was not used to change it.
