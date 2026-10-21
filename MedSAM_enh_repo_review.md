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
| 0 | ✅ Z on validation (inference only), done 2026-10-17: DSC 0.820 / NSD 0.889 | The reference all runs are compared with | – |
| 0b | ✅ **Done 2026-10-17 (keep crops; train with `-bbox_shift_frac 0.2`).** **E1 box robustness** on validation (inference only): B0 and Z with boxes loosened by 10% and 20% per side (`infer_zoom.py -box_perturb`, seed 0) | Can change two training settings before any GPU time is spent: whether to train on crops, and how much box jitter to train with (added 2026-10-15) | Rules below |
| 1 | ✅ **Done 2026-10-17, all checks passed.** Trial: 2 epochs, 20 photos, frozen encoders (+ resume to epoch 2, + 1 epoch LoRA both, + export/infer_zoom/eval_run chain). Full data: train 1,278 lesion samples (638 from crops), val 149 (69 from crops), **0 skipped**. Trainable: frozen 4,058,340; LoRA both 106,496 (24 LoRA layers merged on export). Encoder BN change 0 throughout. Speed (T4): ~1.5–2 min/epoch frozen, ~15% more for LoRA both. Trial model on validation: DSC 0.800 / NSD 0.888 (plumbing check only). | Catches bugs in minutes; measures seconds per epoch for planning | All checks in the brief pass |
| 2 | R2 seed 0 | Best earlier setup; first answer to "does training add anything on top of zoom?" | – |
| 3 | R3 seed 0 | The project's parameter-efficient method | – |
| 4 | R6 seed 0 | Encoder adaptation: the most promising untested idea | – |
| 5 | Seeds 1 and 2 for the best of R2 / R3 / R6, and for any run within ~0.03 of it | A result worth reporting needs 3 seeds | After 2–4 are scored |
| 6 | R1 seed 0 | Upper bound / overfitting check; slowest and most likely to overfit | If time allows |
| 7 | R7 seed 0 | Only if R6 beats R3: shows the gain comes from the encoder | If R6 > R3 |
| 8 | E2 final test + Wilcoxon, boxes as given (main) and 20% looser (robustness), every model | The reported result; uses the test set once | Always, last on our data |
| 9 | E3 AlphaDent | External validity | If time allows |

**Split between us (2026-10-17):** steps 2–9 are divided between Shelly and Linoy, with all Colab cells, in `Shelly_Training.md` and `Linoy_Training.md` (round 1: Shelly R2, R6; Linoy R3, R1, R7. Round 2: see the agreed plan below). Colab helpers (train, score, save, compare, board) live in `MedSAM/colab_runs.py`. Results are shared through `board()`, which rebuilds the table from the per-run folders on Drive; `runs_summary.csv` on Drive is no longer updated; `board()` saves the shared table as `experiments/dental_caries_val42/board.csv`.

**Seed-0 results (2026-10-17, validation, 42 photos).**

| Run | DSC | NSD | DSC 20% looser | vs Z (wins/losses, mean, sign p) | Best epoch (per-lesion val Dice) | Stopped after | s / epoch |
|---|---|---|---|---|---|---|---|
| R1 full fine-tuning | **0.862** | 0.932 | 0.831 | 28/14, +0.042, p 0.022, **passes** | 1 (0.797) | 5 epochs | ~275 |
| R7 LoRA encoder only | 0.859 | 0.928 | **0.842** | 26/16, +0.040, p 0.082 | 17 (0.786) | **20 (hit the cap, still improving)** | ~197 |
| R6 LoRA encoder + decoder | 0.853 | 0.926 | 0.828 | 23/19, +0.033, p 0.32 | 2 (0.788) | 8 | ~160 |
| R2 frozen encoders | 0.846 | 0.927 | 0.798 | 25/17, +0.026, p 0.14 | 1 (0.767) | 6 | ~90 |
| R3 LoRA decoder | 0.824 | 0.909 | 0.794 | 17/25, +0.004, p 0.92 | 2 (0.748) | 8 | ~105 |
| Z (no training) | 0.820 | 0.889 | 0.728 | – | – | – | – |

- Adapting the image encoder (R1, R6) helps most; decoder-only LoRA (R3) adds nothing over Z. Gaps among R1 / R6 / R2 (≤ 0.016) are below the ~0.03 noise level, so seeds decide.
- With 20% looser boxes every trained model gains +0.07 to +0.10 over Z: training with the 0–20% box jitter made the models more robust to imprecise boxes (to be confirmed on test in E2).
- Training curves: R1 and R2 (millions of trainable weights) overfit after epoch 1: training loss keeps falling while validation loss rises. The LoRA runs (R3, R6) don't overfit: validation loss is still flat or falling when early stopping ends them, so they may be slightly under-trained (R3's epoch-7 Dice equals its best but missed the 0.001 threshold). The learning rate never changed in any run: the inherited scheduler (patience 5, factor 0.9) never acts before early stopping. Kept as is so all runs share one protocol; reported as a limitation. Validation Dice swings by 0.01–0.03 between epochs, the same size as the gaps between runs.
- **R7 (added after it finished):** encoder adapters alone (59 K weights, 0.6% of R1) match full fine-tuning (0.859 vs 0.862) and give the best score with looser boxes (0.842). Decoder adapters alone (R3) add nothing. So the gain comes from adapting the image features. R7 never early-stopped: it hit the 20-epoch cap with validation loss still falling, so it is under-trained under the shared protocol (limitation). R6 vs R7 (0.853 vs 0.859) is within noise; R6 stopped after 8 epochs on a noisy plateau.
- **Plan agreed 2026-10-17 (round 2, fixed before any test results were seen):** seeds 1 and 2 for **R1 and R7** (Shelly seed 1, Linoy seed 2). If R7_s1 and R7_s2 both hit the 20-epoch cap, the best R7 seed by validation DSC continues from its best checkpoint for up to 20 more epochs, once, as a new run `R7_s<k>_ext` (Shelly). The five seed-0 models are scored on test now; every round-2 run is scored on test afterwards, whatever it shows; no choice uses test numbers. Cells and safety rules: `Shelly_Training.md`, `Linoy_Training.md`; `colab_runs.py` refuses to overwrite any existing run name on Drive.
- **Decision 1 (proposed, before R7; superseded by the agreed plan above):** seeds 1 and 2 for R1, R6 and R2 (the same 0.03 rule applied to all four runs, since R1 came out best); R3 dropped (0.038 behind R1, no better than Z). Six runs, about 2–2.5 h of GPU time in total.

**Round 2, validation (2026-10-19): 3 seeds of R1 and R7.**

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± sd DSC | Mean DSC 20% looser | Best epochs |
|---|---|---|---|---|---|---|
| R7 LoRA encoder (59 K weights) | 0.859 | 0.864 | 0.864 | **0.862 ± 0.003** | **0.835 ± 0.006** | 17, 14, 14 (of 20) |
| R1 full fine-tuning (9.8 M) | 0.862 | 0.838 | 0.856 | 0.852 ± 0.012 | 0.823 ± 0.012 | 1, 0, 3 |
| Z | | | | 0.820 | 0.728 | |

- R7 is higher on average and about 5x more stable across seeds; 2 of its 3 seeds pass the rule vs Z (R7_s0: p = 0.08), only 1 of R1's 3 does. The R7 vs R1 gap (0.010) is small; the stability difference is the clearer result.
- R1 peaks after 0-3 epochs and swings between epochs; which epoch is kept is close to chance (and the epoch is chosen by per-lesion Dice, not the reported per-photo DSC). R7 improves smoothly and peaks at epochs 14-17 in every seed.
- **Extension: not run.** Neither R7_s1 nor R7_s2 was cut off by the cap (both peaked at epoch 14 and early stopping ended them at epoch 19), so the agreed rule says don't extend. 20 epochs is enough for R7.
- Large vs small photos on validation, from the `compare()` outputs in Shelly's and Linoy's notebooks (8 of the 9 runs: R1_s0, R1_s1, R1_s2, R2_s0, R3_s0, R6_s0, R7_s1, R7_s2; R7_s0's `compare()` output is empty in Linoy's notebook). On the 18 large photos the trained runs are within -0.024 to +0.020 of Z and none passes the rule. On the 24 small photos they gain +0.025 to +0.080, and 6 of the 8 pass; the two that don't are R3_s0 (+0.025) and R1_s1 (+0.041, p = 0.076). Same pattern as on test.

**Final test result, 3 seeds of R1 and R7 (Step E, 2026-10-20, 63 test photos; plan fixed before these were seen).** `experiments/dental_caries_test63/test_comparison_final.csv`.

| Model | DSC as given (mean ± sd over seeds) | NSD | DSC 20% looser |
|---|---|---|---|
| B0 (og_ckpt) | 0.686 | 0.740 | 0.644 |
| Z | 0.796 | 0.862 | 0.714 |
| R1 full fine-tuning, 3 seeds | 0.851 ± 0.008 | 0.917 | 0.825 ± 0.009 |
| **R7 LoRA encoder, 3 seeds** | **0.853 ± 0.003** | **0.918** | **0.847 ± 0.005** |

- Every one of the six runs beats Z (Wilcoxon, Holm-corrected p ≤ 0.007; +0.047 to +0.062 DSC) and B0 (+0.157 to +0.172).
- **R7 vs R1, per-photo means of 3 seeds (fixed in advance):** boxes as given +0.002 DSC, 36/27 photos, Wilcoxon p = 0.69: **no difference in accuracy**. With 20% looser boxes +0.022, 50/13 photos, p < 0.0001: **R7 is clearly more robust to imprecise boxes.**
- Conclusion: with the boxes as given, encoder LoRA (59 K trained weights) is as accurate as full fine-tuning (9.8 M), with seeds that agree about 2.5x more closely (sd 0.003 vs 0.008), and it holds up much better when the boxes are loose (R7 loses 0.006 DSC, R1 0.026).
- Z was re-scored in this session with the same result as before (0.796), a reproducibility check.

**Cost and accuracy of every method (2026-10-19).** Trainable weights; epochs, seconds per epoch and minutes of training per run, from each run's `log.csv` (Colab T4; excludes about 1 min of data preparation and 1-3 min of scoring; speed varies between sessions and accounts, e.g. R7 ran at 158, 164 and 196 s/epoch in three sessions). Validation: mean over seeds (± sd for 3 seeds). Test: seed 0 only until Step E; the exact, complete table is produced by `colab_figures.method_table()` after Step E.

| Method | Trainable weights | Seeds | Epochs run | s / epoch | Train min / run | Val DSC | Val DSC 20% looser | Test DSC (seed 0) | Test DSC 20% looser (seed 0) |
|---|---|---|---|---|---|---|---|---|---|
| B0 zero-shot | 0 | – | – | – | 0 | 0.708 | 0.645 | 0.686 | 0.644 |
| Z zero-shot + zoom | 0 | – | – | – | 0 | 0.820 | 0.728 | 0.796 | 0.714 |
| R3 LoRA decoder | 47,104 | 1 | 8 | 106 | 14 | 0.824 | 0.794 | 0.812 | 0.794 |
| R2 frozen encoders | 4,058,340 | 1 | 6 | 90 | 9 | 0.846 | 0.798 | 0.833 | 0.799 |
| R6 LoRA enc + dec | 106,496 | 1 | 8 | 160 | 21 | 0.853 | 0.828 | 0.840 | 0.830 |
| R1 full fine-tuning | 9,791,300 | 3 | 5 / 4 / 7 | 237-273 | 23 / 16 / 30 (mean 23) | 0.852 ± 0.012 | 0.823 | 0.858 | 0.831 |
| R7 LoRA encoder | 59,392 | 3 | 20 / 20 / 20 | 158-196 | 65 / ~55 / ~53 | **0.862 ± 0.003** | **0.835** | 0.850 | **0.852** |

R7 trains about 2.5x longer per run than R1 (about 53-65 vs 23 min; it never overfits, so early stopping lets it run all 20 epochs) but trains 0.6% of R1's weights and its seeds agree; R1 is fast because it overfits after 1-3 epochs.

**Test set, seed-0 models (2026-10-17, 63 photos; plan for round 2 fixed before these were seen).** `experiments/dental_caries_test63/test_comparison_seed0.csv`.

| Model | DSC as given | NSD | vs Z (wins/losses, gain, Wilcoxon p Holm) | DSC 20% looser | vs Z_p20 gain |
|---|---|---|---|---|---|
| B0 (whole photo) | 0.686 | 0.740 | – | 0.644 | – |
| Z (zoom, no training) | 0.796 | 0.862 | – | 0.714 | – |
| R1 full fine-tuning | **0.858** | 0.922 | 44/19, +0.062, 0.0005 | 0.831 | +0.116 |
| R7 LoRA encoder | 0.850 | 0.915 | 39/23, +0.054, 0.006 | **0.852** | **+0.138** |
| R6 LoRA enc + dec | 0.840 | 0.907 | 39/24, +0.044, 0.022 | 0.830 | +0.116 |
| R2 frozen encoders | 0.833 | 0.902 | 40/23, +0.037, 0.022 | 0.799 | +0.084 |
| R3 LoRA decoder | 0.812 | 0.882 | 31/32, +0.017, 0.31 (n.s.) | 0.794 | +0.080 |

- How to read the vs-Z column: wins/losses only counts on how many photos a model scored higher or lower than Z; photos with exactly the same DSC are ties and count in neither (R7: 39 + 23 = 62, so 1 tie; R6: 39 + 24 = 63). The mean gain and the Wilcoxon test also use how large each difference is, which is why R7 and R6 can have the same 39 wins but different gains (+0.054 vs +0.044) and p-values.
- Same ranking as validation (R1 > R7 > R6 > R2 > R3), and gains over Z at least as large as on validation: no sign that selecting on validation overfitted it.
- Every model except R3 beats Z significantly after Holm correction; all beat B0 (+0.13 to +0.17 DSC).
- **Zoom and training are complementary:** on test, the trained models' gain over Z comes from the 36 small photos (+0.06 to +0.10, significant); on the 27 large photos, where zoom already works, gains are +0.01 to +0.02 and not significant (R3 is even worse there, 8/19).
- **Robustness:** R7 loses nothing with 20% looser boxes (0.850 -> 0.852), while Z drops 0.796 -> 0.714 and B0 0.686 -> 0.644. R7 with loose boxes beats Z with exact boxes.
- Caveats: one seed per method; R1 vs R7 (0.008) was not tested and is within seed variation (R1_s1 on validation: 0.838 vs R1_s0 0.862). The 3-seed table (Step E) is the result to report.

Skipped under a tight schedule: the learning-rate and rank sweeps (R4, R5, R8) and the training-on-whole-photos ablation (R9). E1 was moved into this list (step 0b) on 2026-10-15.

**E1 decision rules (fixed 2026-10-15, before running).** B0 under loosened boxes is run through `infer_zoom.py -zoom_above 100000` (whole photo, identical to B0), so both methods see exactly the same boxes.
- **Keep training on crops** if, on the 18 large validation photos, Z still beats B0 with boxes loosened by 20%: one-sided sign test p ≤ 0.05 and mean DSC gain ≥ +0.03 (the rule of section 9). Otherwise stop and rethink the crop rule before any training.
- **Raise box jitter in training** (`-bbox_shift`, e.g. from 5 to 15 px; implemented as `-bbox_shift_frac`, see Decision below) if either method loses more than 0.05 mean DSC (all 42 photos) going from the given boxes to 20% looser boxes.
- 10% is reported for the shape of the curve; the decisions use 20%.

**Results (2026-10-17).** Step 0 (Z) and step 0b (E1) done, validation, 42 photos.

| Run | Boxes | Mean DSC |
|---|---|---|
| B0 | as given | 0.708 |
| **Z (official reference)** | as given | **0.820** (NSD 0.889) |
| B0 | 10% looser | 0.694 |
| Z | 10% looser | 0.793 |
| B0 | 20% looser | 0.645 |
| Z | 20% looser | 0.728 |

- Z self-check: all 24 small photos identical to B0 (OK).
- **Rule 1: KEEP training on crops.** With 20% looser boxes, Z still beats B0 on 15 of 18 large photos, mean +0.192 DSC, sign test p = 0.004.
- **Rule 2: RAISE box jitter.** Going to 20% looser boxes costs B0 −0.063 and Z −0.092 mean DSC, both beyond −0.05. Zoom is the more sensitive one, because its crop size follows the box size.
- **Decision:** the trainer gets a proportional box loosening, `-bbox_shift_frac 0.2` (each side moved out by a random 0–20% of the box size), **instead of** the old 0–5 px jitter (changed 2026-10-17 at Shelly's request; the 0–5 px mode is used only with `-bbox_shift_frac 0`). It is the trainer's default from now on, so every run in list B uses it. Proportional rather than a fixed pixel count, because E1's looser boxes are proportional and a fixed 15 px would mean very different looseness for small and large lesions.
- **Jitter is training-only, on purpose.** Validation inside the trainer uses the boxes as given, because that is exactly what inference gets: `infer_zoom.py` receives the given boxes for every model (B0, Z, every trained run, the final test). Validation is what picks each run's best epoch, so it must measure the same task the test scores; random jitter there would also make validation Dice change from epoch to epoch for reasons unrelated to the model. Robustness to loose boxes is measured separately and identically for all models, with E1's fixed, seeded 10%/20% loosening.
- Later, re-running E1 on the trained models shows whether this made them more robust than Z.

**Scoring rule for list B and E2 (agreed 2026-10-17).** Every model is scored on identical inputs; the boxes as given decide, the looser boxes are always reported but never used to choose.

| Stage | Boxes as given | 20% looser (`-box_perturb 0.2 -perturb_seed 0`) | Why |
|---|---|---|---|
| Inside the trainer (best epoch, early stopping, learning rate) | ✅ only | ❌ | A run needs one criterion, and it should be the main task. |
| Scoring a finished run on validation (choosing between runs, e.g. R6 vs R3) | ✅ decides | ✅ reported, does not decide | Decisions on the main task; the loose-box score shows early whether training jitter helps. |
| Final test E2 (63 photos), every model incl. B0 and Z | ✅ main result | ✅ robustness result | Same seeded boxes for all models, so both comparisons are fair. Nothing is tuned on the loose-box score. |

**Scoring order matters:** run `infer_zoom.py` into `segs_<run_name>` first, then `eval_run.py --run_name <run_name>`. eval_run skips its own inference when all 42 predictions exist; if they don't, it runs the old whole-photo inference instead of zoom. For the loose-box score, write `infer_zoom.py -box_perturb 0.2 -perturb_seed 0` output to `segs_<run_name>_p20` and score it with `--run_name <run_name>_p20` (same naming as E1's `Z_p20` / `B0_p20`).

Training jitter is part of each method, like the flips; it is never mirrored on the test boxes for one model only. If R2 falls clearly below Z on the boxes as given, compare it with the same run at `-bbox_shift_frac 0` to see whether the jitter cost something.


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


## 12. Extra analyses after the main result (plan fixed 2026-10-20, before any of them was run)

Cells: `Shelly_Training.md` (Steps G and K), `Linoy_Training.md` (Step R4) and `AlphaDent_Plan.md` (E3); code: `MedSAM/colab_extras.py`, `MedSAM/prepare_alphadent.py`, run types `R4d` / `R4e` in `MedSAM/colab_runs.py`. (`MedSAM/prepare_kvasir.py` is not needed: the Kvasir sets were already prepared.) Each analysis writes new names only; nothing already on Drive is touched.

| ID | Question | What is run | Main comparison (fixed now) | Cost |
|---|---|---|---|---|
| G1 | Is zoom still needed after training? | R1 and R7 (3 seeds each) on test **without** zoom (`<run>_nozoom`) | R7 (mean of 3 seeds), 27 large test photos: with vs without zoom, two-sided Wilcoxon. Small photos must come out identical (zoom never acts there): a built-in check | Inference only, ~10 min on GPU |
| G2 | What does zoom cost at inference? | B0, Z, R7 (median seed), R1_s0 timed in one session on one device | Seconds per photo, large vs small photos. LoRA adds nothing at inference (merged weights) | ~5 min |
| G3 | Where does R7 still fail? | Per-photo R7 DSC (mean of 3 seeds) vs photo size, number of boxes, lesion size | Spearman correlations; the 10 worst photos; DSC vs lesion size plot | Seconds |
| K | Does dental training hurt LiteMedSAM elsewhere (forgetting)? | The Kvasir test sets already prepared in `LiteMedSAM_Baseline_production.ipynb` (official splits, the datasets' own boxes): Kvasir-SEG (120 polyp images) and Kvasir-Instrument (118 instrument images). B0, R2, R3, R6 (seed 0), R1 and R7 (3 seeds) | Every model vs B0, per image, two-sided Wilcoxon, Holm-corrected, per dataset. Images are < 2,000 px, so zoom never acts (Z = B0) | ~30–60 min CPU or ~10 min GPU |
| R4 | Was R3 only slow to learn? Does R7 gain from a higher learning rate? | `R4d` = R3 at lr 5e-4, `R4e` = R7 at lr 5e-4, seed 0 each, validation | R4d vs R3_s0 (0.824): a gain ≥ 0.03 means R3's decoder adapters were held back by the learning rate. R4e vs R7's seed range (0.859–0.864). Single seed each: a sensitivity check, not a new final method | ~15 + ~55 min GPU |

**G1 caveat.** The trained models saw large photos only as crops during training, so running them without zoom also shows them large photos in a form they never trained on. G1 therefore answers "does our pipeline still need zoom at inference?", not "could training alone replace zoom?". The clean answer to the second question would be R9 from run list A (train on whole photos, score without zoom), which we skipped.

**How to report them.** G1–G3 and K are analyses of the final models, decided before running, so their test/Kvasir numbers can be reported as they come. R4 runs after the test set was seen, so it is reported as a sensitivity check only; its test numbers, if computed, are labelled post hoc and never replace R7 as the method we report.

**Kvasir caveat.** Kvasir-SEG may have been part of the large public collection LiteMedSAM was trained on (not verified). B0 is then on familiar data, which makes this a fair test of *forgetting*, not of generalisation. A drop for R1/R2 and none for R3/R6/R7 would show a practical advantage of LoRA: its adapters can be switched off for non-dental images, which restores B0 exactly.

**AlphaDent (E3)**: the full plan, with cells, checks and time per step, is in `AlphaDent_Plan.md` (repo root). Inference only; estimated effort about half a day to a day.


## 12a. Step F: method table and figures 1–5, what they show (2026-10-21)

Source: `experiments/dental_caries_test63/figures/` (method_table.csv, fig1–fig5). Test = 63 photos, validation = per-photo DSC on the validation split.

| Method | Trainable weights | Train min / run | Val DSC | Test DSC (sd over seeds) | Test NSD | Drop with 20% looser boxes: val / test |
|---|---|---|---|---|---|---|
| B0 zero-shot | 0 | – | 0.708 | 0.686 | 0.740 | −0.063 / −0.041 |
| Z zero-shot + zoom | 0 | – | 0.820 | 0.796 | 0.862 | −0.092 / −0.081 |
| R3 LoRA decoder | 47 k | 14 | 0.824 | 0.812 | 0.882 | −0.030 / −0.018 |
| R2 frozen encoders | 4.06 M | 9 | 0.846 | 0.833 | 0.902 | −0.048 / −0.034 |
| R6 LoRA enc+dec | 106 k | 21 | 0.853 | 0.839 | 0.907 | −0.024 / −0.009 |
| R1 full fine-tuning | 9.79 M | 23 | 0.852 | 0.851 (0.008) | 0.917 | −0.029 / −0.027 |
| R7 LoRA encoder | 59 k | 58 | 0.862 | 0.853 (0.003) | 0.918 | −0.028 / −0.007 |

What holds up:
1. **Nearly the same order on validation and test** (only R6 and R1 swap; they are 0.001 apart on validation). Every method is 0.00–0.02 lower on test, the zero-shot ones most. Nothing was tuned on test, so this is the expected small gap.
2. **Zoom and training fix different photos** (fig2): large photos B0 0.57 → Z 0.83 → R1 0.84 / R7 0.85; small photos B0 = Z = 0.77 → R1 / R7 0.86. Zoom gives almost all the gain on large photos; training gives almost all of it on small ones.
3. **R7 ≈ R1 in accuracy at 1/165 of the trainable weights**, and R7's seeds agree more closely (test sd 0.003 vs 0.008; validation 0.003 vs 0.012).
4. **Training curves** (fig4): R1's kept epoch is 0/1/3 and its validation Dice swings by ±0.03 between epochs; R7 climbs steadily and keeps epoch 14/14/17. R7 costs more training time (20 epochs × 173 s vs ~5 × 255 s), not more per epoch. Note: fig4's y-axis is the trainer's per-lesion Dice, not the per-photo DSC of the tables, so its values are lower; say so on the slide.

What needs a caveat:
5. **The robustness gap between R7 and R1 is a test-set result only.** On test R7 loses 0.007 with looser boxes and R1 0.027 (R7 vs R1 with looser boxes: +0.022, p < 0.0001). On validation both lose the same (−0.028 vs −0.029); R7 stays higher there only because it starts higher. Report it as "on the test set R7 was more robust; on validation the two lost equally", not as a general property of LoRA. The same goes for any "LoRA models are more robust" reading: R3 and R6 (one seed each) are not consistently better than R1 on validation.
6. **Zoom makes the zero-shot model more sensitive to loose boxes** (Z −0.081 vs B0 −0.041 on test, same on validation): with a looser box the crop grows, and the untrained model segments more of the surrounding tooth. Training removes most of this (all trained models lose ≤ 0.048).
7. **Examples** (fig5, rows chosen by rule): A and B are small photos where B0 = Z (zoom doesn't act) and R7 captures the whole dark cavity (0.28 → 0.94, 0.43 → 0.71); in B, R7's mask is close to the box itself. C is a 4000 × 6000 photo where B0 gets 0.00 (the lesion is about one pixel at 256 px) and zoom alone fixes it (0.82; R7 0.80). D is R7's worst loss to Z (0.72 vs 0.88), on a photo with two tiny lesions. In C and D the lesions are too small to judge at the figure's scale: use `cf.fig_examples(closeup=True)` (Step F7b), which crops each row to one box and saves `fig5b_examples_closeup.png`.

Code fixes after Step F (colab_figures.py, 2026-10-21):
- The "best epoch" column used the plain argmax of validation Dice. It now replays the trainer's rule (a gain counts only if it is above the best by 0.001), the same as `cr.capped()`. Only R3 changes: 7 → **2** (its later peak was within 0.001 of epoch 2, so the kept checkpoint is from epoch 2). R1, R2, R6 and R7 are unchanged. Fig4 dots use the same rule (no visible change); fig4's x-axis now shows whole epochs.
- Fig1: the x labels "Z zero-shot + zoom" and "R3 LoRA decoder" overlapped; names are now wrapped onto two lines.

Re-run `cf.method_table()`, `cf.fig_overview()` and `cf.fig_training_curves()` after a pull; this overwrites only those three files in `figures/`, which are derived from results that don't change.


## 13. Limitations

1. **Small evaluation sets.** 42 validation and 63 test photos. A single run's difference below about 0.03 DSC is within noise; that is why decisions use seeds and paired per-photo tests.
2. **One dental dataset.** Roboflow "Dental Caries Segmentation" (CC BY 4.0, no published description of acquisition or labelling). Generalisation to other clinics, cameras and labelling styles is untested until AlphaDent (E3).
3. **Boxes are given.** Ground-truth lesion boxes are the prompts, so the results measure how well a lesion is outlined once its location is known, not detection. Results are not directly comparable with fully automatic methods (related work). Robustness to imprecise boxes is tested only with synthetic loosening (20%, seeded), not with boxes drawn by people or a detector.
4. **Merged and noisy labels.** Caries, cavity and crack are merged into one class; tooth is background. Caries boundaries are gradual, which limits the achievable Dice. 3 of 1,611 lesions have no box and are not evaluated.
5. **Training protocol not tuned per method.** One learning rate per run type, constant in practice (the inherited scheduler never acts before early stopping), fixed epoch caps, LoRA rank 8, box jitter 0–20%, zoom threshold 2,000 px and margin 0.5, each chosen once. R7_s0 reached its 20-epoch cap while still improving. R2, R3 and R6 have one seed only.
6. **Epoch selection uses a stand-in metric.** The best epoch is chosen by per-lesion Dice at 256 px, not by the reported per-photo DSC at full resolution; two checkpoints equal on the first can differ by about 0.02 on the second (seen with R1).
7. **Test set used in two rounds.** Seed-0 models were scored on test before the round-2 runs; the round-2 plan (seeds, the R7 extension rule, and the R7 vs R1 test) was fixed before those results were seen, and every round-2 run is scored on test regardless of outcome. R4 is post hoc (section 12).
8. **Multiple comparisons.** Holm correction is applied within each table of model-vs-reference tests, not across all tables in the project.
9. **Timing.** Training and inference times come from free-tier Colab T4 sessions and vary by about ±20% between sessions and accounts (e.g. R7: 158–196 s per epoch); inference times are only compared within one session (G2).
10. **The 256-px input of LiteMedSAM** is a fixed model constraint; zoom works around it for large photos, but small photos are still downscaled, and lesions that occupy a few pixels at 256 px remain hard (G3).
