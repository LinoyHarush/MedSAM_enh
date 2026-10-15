# Trainer fixes for `train_one_gpu_multibox.py`: brief for Linoy

**Date:** 2026-10-14 · **Context:** step 5 of the plan in `MedSAM_enh_repo_review.md` (section 7). The full reasoning is in review sections 3 and 6.
**Scope:** only `MedSAM/train_one_gpu_multibox.py`, plus a new `MedSAM/lora.py`. Shelly is working on the data, the split, B0 and the zoom test in parallel (steps 1–4), so please leave `preprocess_tooth_ds.py`, `pre_rgb_*`, `eval_run.py`, `CVPR24_LiteMedSAM_infer.py` and `evaluation/` untouched to avoid conflicts.
**Git:** please work on a branch (e.g. `trainer-fixes`) and merge when the trial run in section 10 passes. Please don't force-push to `main`.

Line numbers below refer to the current `train_one_gpu_multibox.py` (unchanged since 2026-10-10).

---

## Why these fixes

The per-lesion diagnostic (review section 6) found 1,278 lesion boxes in the 414 training photos:
- **59** lesions don't survive the resize to 256 px. Today the trainer gives each of them a filled rectangle as its target, every epoch.
- Lesion boxes can also contain part of a neighbouring lesion, and today that neighbour becomes part of the target.
- Several trainer problems from the presentation are still open: no seed, starting at epoch 1, BatchNorm updating in the frozen encoder, and no early stopping.
- R3–R5 need LoRA, which only exists in the old trainer.

---

## 1. Per-lesion target + skip lesions that vanish (plan steps 1–2)

**Where:** `MultiBoxNpzDataset.__init__` (lines 94–136) and `__getitem__` (lines 141–216), including the rectangle fallback at lines 180–182.

**What:**
- The target for a box should be **only the lesion that box was drawn around**, not every lesion pixel inside the box.
- Boxes where that lesion vanishes at 256 px, or where the box collapses to zero width or height, should be **dropped when the dataset is built**.
- **Delete the rectangle fallback entirely.**

**How:** `diagnose_bbox_survival.py` (commit `43d015f`) already does exactly this matching and survival check, and it was run on all 414 photos. Reuse its logic:
- `lesion_component(labels, stats, box)`: finds the connected component a stored box was drawn around (same top-left corner and size, ±1 px).
- `to_256(mask, image_size)`: nearest-neighbour resize of the longest side, then pad, exactly as the trainer does.

```python
n, labels, stats, _ = cv2.connectedComponentsWithStats(gt_binary, connectivity=8)
for bi, box in enumerate(boxes):
    box_orig = clip_box_xyxy(box.tolist(), w, h)
    cid = lesion_component(labels, stats, box_orig)
    if cid == 0:
        skipped["no_lesion"] += 1; continue
    own_256 = to_256((labels == cid).astype(np.uint8), image_size)
    x1, y1, x2, y2 = box_to_256(box_orig, scale)          # see 1b
    if x2 <= x1 or y2 <= y1:
        skipped["collapsed"] += 1; continue
    if own_256.sum() == 0:
        skipped["vanished"] += 1; continue
    self.samples.append((img_idx, own_256.astype(bool), (x1, y1, x2, y2)))
print(f"lesion samples: {len(self.samples)}, skipped: {dict(skipped)}")
```

On the full training set this should print about **1,219 samples, 59 skipped** (25 collapsed + 33 vanished + 1 vanished with a neighbour in its box; that last one counts as "vanished" here). The diagnostic rounded boxes to nearest, so after the rounding change in 1b the collapsed count may differ by a few.

**1b. Use the same box rounding as inference.** `CVPR24_LiteMedSAM_infer.py::resize_box_to_256` uses `int(v * 256 / max(H, W))` (rounding down). The trainer's `transform_box_to_resized` rounds to nearest. Use the inference rule in the trainer so a lesion gets the same box in both:
```python
def box_to_256(box, scale):
    return [int(v * scale) for v in box]
```

**1c. Keep the sample-building in one function** (update 2026-10-15: cropping is now decided, see section 11), e.g. `make_sample(img_256, own_256, box_256)` holding the flip, jitter and tensor conversion. If Shelly's zoom test (step 4) shows that crops help, a crop option goes in there later. **Don't implement cropping now.** It waits for that result.

## 2. Resize once and keep everything in memory (speed)

**Where:** `__init__` (line 112 loads every image without using it) and `__getitem__` (line 143 decompresses the full-size npz for every sample, up to 37 MP).

**How:** in `__init__`, load each npz **once**:
- Resize the image's longest side to 256 (`INTER_AREA`, as now), keep it as `uint8`, and store it with its `(newh, neww)`.
- Store each lesion's `own_256` mask from section 1 as a bool array.
- Memory: about 80 MB for images plus about 80 MB for masks.

`__getitem__` then only does: min-max normalisation on the unpadded image, then pad, flip, jitter and convert to tensors, exactly as now. No file access in `__getitem__`. Expect epochs to drop from minutes to seconds.

**Data location:** read from Colab's local disk, not Drive. Reading many files from the Drive mount kept disconnecting it today. These are the cells that worked:
```python
%cd /content
SRC = "/content/drive/MyDrive/Deep_Learning_Course/MedSAM/MedSAM_enh/data/dental_caries_segmentation_processed/MedSAM_train/RGB_Teeth/train_ready_npz"
!rsync -a --info=progress2 "{SRC}/" /content/train_ready_npz/
!ls /content/train_ready_npz | wc -l     # expect 414
```
If Drive drops: `drive.mount('/content/drive', force_remount=True)`, then `%cd /content` again, then re-run the `rsync` line (it resumes).

## 3. Seed (plan step 4)

**Where:** `parse_args` and the start of `main()`, plus both `DataLoader`s (lines 376–398).

```python
p.add_argument("-seed", type=int, default=0)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(args.seed)
g = torch.Generator(); g.manual_seed(args.seed)
train_loader = DataLoader(train_ds, ..., shuffle=True, generator=g)   # or sampler=WeightedRandomSampler(..., generator=g)
```
PyTorch already seeds `random`, NumPy and torch inside each worker from the generator, so no `worker_init_fn` is needed. GPU runs won't be bit-identical (cuDNN), which is fine: the 3 seeds exist to measure run-to-run variation. Save `args.seed` in the checkpoint (it's already inside `vars(args)`) and use it in the work-dir name, e.g. `work_dir/R2_frozen_s0`.

## 4. Start at epoch 0 (plan step 4)

**Where:** line 435 `for epoch in range(start_epoch + 1, args.num_epochs)`, and the resume block at lines 420–429.

```python
for epoch in range(start_epoch, args.num_epochs):
    ...
    ckpt["epoch"] = epoch + 1          # = the next epoch to run
# resume:  start_epoch = int(r["epoch"])
```
`-num_epochs 20` then runs 20 epochs, numbered 0–19. Don't resume the old checkpoints with this version, because their `epoch` field means something different.

## 5. Keep frozen parts in eval mode (plan step 4)

**Where:** line 434 (`model.train()` before the loop) and line 316 (`model.train()` at the end of `evaluate`).

`model.train()` switches every BatchNorm in TinyViT back to updating its statistics, even when the encoder is "frozen". Replace both calls with:
```python
def set_train_mode(model, args):
    model.train()
    if args.freeze_image_encoder:
        model.image_encoder.eval()
    if args.freeze_prompt_encoder:
        model.prompt_encoder.eval()
```
For LoRA runs the encoder is frozen, so it stays in eval mode too. The decoder stays in train mode, so LoRA dropout still works.

## 6. Early stopping + scheduler on validation (plan step 3)

**Where:** argument list, the end of the epoch loop (after line 531), and the scheduler at lines 408 and 475.

```python
p.add_argument("-patience", type=int, default=5, help="epochs without val Dice improvement before stopping")
p.add_argument("-min_delta", type=float, default=1e-3)
...
if val_stats is not None:
    if val_stats["dice_metric"] > best_val_dice + args.min_delta:
        ...  # save best_val as now
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= args.patience:
        print(f"Early stop at epoch {epoch}: no val Dice gain for {args.patience} epochs"); break
```
- Step `lr_scheduler` on `val_stats["loss"]` when there is a validation set, and on the training loss otherwise.
- Save `lr_scheduler.state_dict()` and `epochs_no_improve` in the checkpoint, and restore both on resume.
- The validation dataset must use the same skip rules as training (section 1), with `data_aug=False` and `bbox_shift=0`, as now.

**Validation data:** Shelly's split script (step 2) will produce a folder of combined npz files (`imgs`, `gts`, `boxes`) for the 42 validation photos and pass it as `-val_npz_dir`. Until then, test the plumbing with a temporary folder of ~10 training files. Use that only to see that the code runs, not to make decisions.

## 7. LoRA in the multibox trainer (needed for R3–R5)

**Where:** new file `MedSAM/lora.py`; `train_one_gpu.py` lines 160–216 (move code out); `train_one_gpu_multibox.py` arguments and model setup (lines 332–358).

1. **Move** `LoRALinear`, `maybe_wrap_with_lora` and `inject_lora_mask_decoder` from `train_one_gpu.py` into `MedSAM/lora.py` unchanged. Replace them in `train_one_gpu.py` with `from lora import ...`, so old runs stay reproducible.
2. **Add the same flags** as `train_one_gpu.py`: `--use_lora`, `--lora_r` (8), `--lora_alpha` (16), `--lora_dropout` (0.05), `--lora_targets` ("q,v"), `--lora_scope` ("mask_decoder").
3. **Order in `main()`:** build the model, load pretrained weights, inject LoRA, freeze everything, set `requires_grad=True` only for parameters whose name contains `lora_`, then build the optimizer from the trainable parameters.
4. **Save the LoRA config** where `export_infer_ckpt.py` can find it. It reads `lora_alpha` from `medsam_lite_lora_best.pth` / `medsam_lite_lora_latest.pth` in the same folder as the checkpoint. Whenever you save `best_val` (and `latest`), also save:
   ```python
   torch.save({"lora": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
               "epoch": epoch, "config": {"lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
               "lora_dropout": args.lora_dropout, "lora_targets": args.lora_targets,
               "lora_scope": args.lora_scope}},
              os.path.join(args.work_dir, "medsam_lite_lora_best.pth"))
   ```
5. The printed trainable-parameter count for `--lora_r 8 --lora_targets q,v` should be **47,104**, the same as on the slide.

R4 and R5 only change `-lr` (5e-4 and 1e-4); no extra code is needed.

## 8. Logging (for the presentation)

- Print the learning rate each epoch.
- Write one row per epoch to `work_dir/<run>/log.csv`: `epoch, train_loss, train_seg, train_ce, train_iou, val_loss, val_dice, val_iou, lr, seconds`. The loss-curve and validation-curve slides then come straight from this file.

## 9. Not in scope here

- **Cropping / zoom:** waits for Shelly's step 4 result (section 1c).
- **Preprocessing changes:** waits for Shelly's step 1 result.
- **The old `train_one_gpu.py`:** only the LoRA move (section 7.1). It stays as it is for reproducing the old runs.

## 10. Trial run before the real runs (plan step 7)

Add `-max_images N` (use only the first N photos, for both datasets) so the trial is quick:
```bash
python train_one_gpu_multibox.py -train_npz_dir /content/train_ready_npz -val_npz_dir <val folder> \
  -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -work_dir work_dir/trial \
  -num_epochs 2 -max_images 20 -seed 0 --freeze_image_encoder --freeze_prompt_encoder
```
Check each of these and paste the output into the notebook:

| Check | Expected |
|---|---|
| Sample count on the full training set (without `-max_images`; building the dataset is enough) | about 1,219 samples, about 59 skipped |
| First epoch printed | `Epoch 0` |
| Number of epochs with `-num_epochs 2` | 2 |
| One frozen BatchNorm's `running_mean`, before vs after an epoch (e.g. the first `BatchNorm2d` in `model.image_encoder`) | identical |
| Files in `work_dir/trial/` | `medsam_lite_multibox_best_val.pth`, `..._latest.pth`, `log.csv` |
| Same command with `-seed 0` twice: first 5 training losses | very close |
| `--use_lora --lora_r 8` trainable params | 47,104 |
| `python export_infer_ckpt.py -ckpt work_dir/trial_lora/medsam_lite_multibox_best_val.pth -out /content/x.pth` | "Merged 14 LoRA modules" |
| Resume from `latest` | continues at the next epoch, doesn't repeat one |

After that, R1–R5 can run (step 8).


---

## 11. Update 2026-10-15: add the zoom crop for large photos (decided)

Shelly's zero-shot zoom test (review section 9) was decided by a rule fixed in advance. For photos whose long side is **above 2,000 px**, cropping around each box raised validation DSC by **+0.26** on average (17 of 18 photos better). Small photos showed no gain. Inference will therefore use `infer_zoom.py -zoom_above 2000`, and **the trainer must see lesions the same way**:

- For a lesion in a photo with long side > 2,000 px, build the sample from a **crop**, using the same function as inference: `crop_window(box, H, W, margin=0.5, min_crop=256)` in `MedSAM/infer_zoom.py` (import it; don't copy it, so the two can't drift apart). The crop is then resized to 256 exactly like a whole photo is today.
- The box inside the crop: `box_in_crop_256(box, x0, y0, cw, ch)` from the same file.
- The target is the lesion's own mask (section 1), cropped with the same window and then resized and padded with nearest-neighbour.
- Photos with a long side of 2,000 px or less: unchanged, whole photo as in section 1.
- Add a flag `-zoom_above` (default 2000; 0 = crop everything, a very large number = never crop) and save it in the checkpoint, so every run records how it was trained.
- Caching (section 2) then stores one 256-px crop per lesion for large photos instead of one image per photo. That's still small (about 640 crops × 196 KB ≈ 125 MB).
- Re-run the vanished-lesion count afterwards. With crops, far fewer large-photo lesions should vanish, so the skipped count should drop well below 59.
- Validation inside the trainer uses the same rule. Final scoring of every run uses `infer_zoom.py -zoom_above 2000` on the exported checkpoint, then `eval_run.py` (it skips inference when predictions exist).
