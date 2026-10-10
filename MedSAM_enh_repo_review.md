# MedSAM_enh repo review against the presentation's plan

**Date:** 2026-10-10
**Repo state reviewed:** `main` at `97febca` (pulled 2026-10-10), plus the Colab notebook `LiteMedSAM_Baseline_production (2).ipynb`
**Plan reviewed against:** slides "First, fix the training signal" (steps 1–5) and "Next runs on the corrected pipeline" (B0, R1–R5)

## Summary

- The two newest scripts, `train_one_gpu_multibox.py` and `diagnose_bbox_survival.py`, are not called anywhere in the notebook yet. The last training cells still use `train_one_gpu.py`.
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

1. **Thin lesions get no box.** The `> 5` filter uses `cv2.contourArea`, which measures the area enclosed by the outline, not the pixel count. A 1-px-wide crack 181 px long has `contourArea == 0`, so it gets no box. Its pixels stay in the ground-truth mask, so at test time they always count as missed. **Fix:** filter by pixel count, e.g. `(component pixels) > 5` from `connectedComponentsWithStats`.
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

- Runs correctly. On the synthetic test it flagged the 6×6 px lesion as `zero_fg_in_box256` and kept the others.
- **Run it first** on `train_ready_npz` to get the real count of vanished lesions (the slides name only photos 49 and 1302).
- Limitation: it counts lesion pixels inside the box at 256 px, using the 0/1 mask. A lesion that vanished but whose box overlaps a surviving neighbour is counted as valid. The per-lesion target in section 5 fixes this too.
- Minor: it resizes and pads the image but never uses the result.

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

## 6. Still to build for the next runs

1. ~~Commit `preprocess_tooth_ds.py`~~ (done, `97febca`). Still to do: the pixel-count filter, and pointing the notebook cells at `MedSAM/preprocess_tooth_ds.py`.
2. Fixes A–G above, plus caching at 256 px.
3. Move LoRA into a shared module (e.g. `lora.py`) that both trainers import, and add `--use_lora` to the multibox trainer. Save a `medsam_lite_lora_*.pth` with its `config` next to the checkpoint, because `export_infer_ckpt.py` reads `lora_alpha` from it.
4. A split script: the 42 validation photos from the 105 test photos, with a fixed seed. Write the validation set as combined npz (`prepare_training_npz.py` does this) and the remaining 63 as their own `MedSAM_test` folder so `eval_run.py` scores only those.
5. Re-score B0 on the 63 photos.
6. A statistics script: paired Wilcoxon signed-rank test on per-photo DSC and NSD against B0, over the 3 seeds.
7. Notebook cells that run `diagnose_bbox_survival.py` and the multibox trainer. A quick check before the long runs: 1 epoch on ~20 photos, confirming the number of samples and epochs, that BatchNorm statistics stay unchanged when the encoder is frozen, and that the validation checkpoint gets written.
