"""
Zero-shot LiteMedSAM inference with a crop around each box ("zoom", step 4 of the plan).

Today's inference (CVPR24_LiteMedSAM_infer.py) shrinks the whole photo to 256 px once and decodes every box
from that one encoding, so lesions in large photos end up only a few pixels big. Here every box gets its own crop:

  crop side = max(box side * (1 + 2 * margin), min_crop)       margin 0.5 -> 2x the box; min_crop 256 px
  the crop is square, centred on the box, shifted to stay inside the photo, and clipped to the photo size.

A crop is never smaller than 256 px of the original photo, so nothing is enlarged beyond what the current
pipeline already does (photos whose long side is under 256 px are still resized to 256 exactly as today,
which makes their result identical to B0). Each crop is then preprocessed exactly like the current inference
(resize longest side to 256, min-max normalise, pad), encoded, decoded with the box, and the mask is pasted
back into the full-size photo with the box's id (1, 2, 3, ...), in the same output format as today.

Outputs in -o:
  <case>.npz             segs, same format as CVPR24_LiteMedSAM_infer.py (eval_run.py / compute_metrics_binary.py read it)
  zoom_boxes.csv         one row per box: crop size, zoom factor vs today, predicted IoU
  efficiency.csv         seconds per photo

Self-check: with -reference_dir pointing at B0's predictions (segs_og_ckpt), every photo whose crops all equal
the whole photo must give exactly B0's mask. The script reports how many such photos match.

Step 4 decision (2026-10-15): zoom passed the pre-set rule for photos with a long side above 2,000 px only,
so use -zoom_above 2000 for every run from now on.

Usage (from MedSAM/):
  python infer_zoom.py -i <test_dir>/imgs -o <test_dir>/segs_zoom_z1 \
      -lite_medsam_checkpoint_path work_dir/LiteMedSAM/lite_medsam.pth -reference_dir <test_dir>/segs_og_ckpt -zoom_above 2000
"""
import argparse
import csv
from glob import glob
from os import makedirs
from os.path import basename, join, isfile
from time import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT

IMAGE_SIZE = 256


# ---------------------------------------------------------------- geometry (no torch; tested separately)
def crop_window(box, H, W, margin=0.5, min_crop=256):
    """Square window around a box: (x0, y0, w, h) in original pixels, always containing the box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    side = max(x2 - x1, y2 - y1, 1)
    c = max(int(round(side * (1 + 2 * margin))), min_crop)
    cw, ch = min(c, W), min(c, H)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = int(round(cx - cw / 2.0)); y0 = int(round(cy - ch / 2.0))
    x0 = min(max(x0, 0), W - cw); y0 = min(max(y0, 0), H - ch)
    # make sure the whole box is inside (rounding at the edges)
    x0 = min(x0, max(0, x1)); y0 = min(y0, max(0, y1))
    x0 = max(x0, min(W - cw, x2 - cw)); y0 = max(y0, min(H - ch, y2 - ch))
    return x0, y0, cw, ch


def box_in_crop_256(box, x0, y0, cw, ch):
    """Box in the 256-px input of the crop, with the same rounding as resize_box_to_256 (floor)."""
    ratio = IMAGE_SIZE / max(cw, ch)
    x1, y1, x2, y2 = box
    return np.array([int((x1 - x0) * ratio), int((y1 - y0) * ratio),
                     int((x2 - x0) * ratio), int((y2 - y0) * ratio)])


# ---------------------------------------------------------------- preprocessing (identical to CVPR24_LiteMedSAM_infer.py)
def resize_longest_side(image, target_length=IMAGE_SIZE):
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    neww, newh = int(oldw * scale + 0.5), int(oldh * scale + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)


def pad_image(image, target_size=IMAGE_SIZE):
    h, w = image.shape[0], image.shape[1]
    if image.ndim == 3:
        return np.pad(image, ((0, target_size - h), (0, target_size - w), (0, 0)))
    return np.pad(image, ((0, target_size - h), (0, target_size - w)))


# ---------------------------------------------------------------- model (same architecture as the inference script)
class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[..., :new_size[0], :new_size[1]]
        return F.interpolate(masks, size=(original_size[0], original_size[1]), mode="bilinear", align_corners=False)


def build_model(ckpt_path, device):
    image_encoder = TinyViT(img_size=256, in_chans=3, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
                            num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], mlp_ratio=4., drop_rate=0.,
                            drop_path_rate=0.0, use_checkpoint=False, mbconv_expand_ratio=4.0,
                            local_conv_size=3, layer_lr_decay=0.8)
    prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                                   input_image_size=(256, 256), mask_in_chans=16)
    mask_decoder = MaskDecoder(num_multimask_outputs=3,
                               transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                               transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
    model = MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def segment_crop(model, crop, box_256, device):
    """Same steps as MedSAM_infer_npz_2D + medsam_inference, applied to one crop and one box."""
    img_256 = resize_longest_side(crop)
    newh, neww = img_256.shape[:2]
    img_norm = (img_256 - img_256.min()) / np.clip(img_256.max() - img_256.min(), a_min=1e-8, a_max=None)
    x = torch.tensor(pad_image(img_norm)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    emb = model.image_encoder(x)
    box_t = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=device)
    sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    logits, iou = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                     sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                     multimask_output=False)
    prob = torch.sigmoid(model.postprocess_masks(logits, (newh, neww), crop.shape[:2]))
    return (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8), float(iou.squeeze().cpu())


def main():
    ap = argparse.ArgumentParser(description="Zero-shot LiteMedSAM inference with one crop per box.")
    ap.add_argument("-i", "--input_dir", required=True, help="Folder with imgs npz files (imgs, boxes).")
    ap.add_argument("-o", "--output_dir", required=True, help="Folder for the predictions (e.g. <test_dir>/segs_zoom_z1).")
    ap.add_argument("-lite_medsam_checkpoint_path", default="work_dir/LiteMedSAM/lite_medsam.pth")
    ap.add_argument("-device", default="cpu")
    ap.add_argument("-margin", type=float, default=0.5, help="Context on each side of the box, as a fraction of the box side.")
    ap.add_argument("-min_crop", type=int, default=256, help="Smallest crop side in original pixels (256 = never enlarge).")
    ap.add_argument("-zoom_above", type=int, default=0,
                    help="Only crop photos whose long side is above this many px; others are segmented exactly like B0 "
                         "(whole photo). 0 = crop every photo. Step 4 decision (2026-10-15): 2000.")
    ap.add_argument("-reference_dir", default=None, help="B0 predictions (segs_og_ckpt) for the self-check.")
    args = ap.parse_args()

    makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args.lite_medsam_checkpoint_path, device)
    files = sorted(glob(join(args.input_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No npz files in {args.input_dir}")

    box_rows, eff_rows = [], []
    whole_photo_cases, matches = 0, 0
    for f in tqdm(files, desc="Zoom inference"):
        t0 = time()
        name = basename(f)
        d = np.load(f, allow_pickle=True)
        img = d["imgs"]
        boxes = np.asarray(d["boxes"]).reshape(-1, 4)
        assert img.max() < 256, f"{name}: image should be in [0, 255]"
        H, W = img.shape[:2]
        segs = np.zeros((H, W), dtype=np.uint8)
        all_whole = True
        zoom_this = max(H, W) > args.zoom_above
        for idx, box in enumerate(boxes, start=1):
            if zoom_this:
                x0, y0, cw, ch = crop_window(box, H, W, args.margin, args.min_crop)
            else:
                x0, y0, cw, ch = 0, 0, W, H          # whole photo = exactly the B0 pipeline
            whole = (x0, y0, cw, ch) == (0, 0, W, H)
            all_whole &= whole
            crop = img[y0:y0 + ch, x0:x0 + cw]
            mask, iou = segment_crop(model, crop, box_in_crop_256(box, x0, y0, cw, ch), device)
            segs[y0:y0 + ch, x0:x0 + cw][mask > 0] = idx
            box_rows.append({"case": name, "box_idx": idx, "img_h": H, "img_w": W,
                             "crop_x0": x0, "crop_y0": y0, "crop_w": cw, "crop_h": ch,
                             "zoom_vs_today": round(max(H, W) / max(cw, ch), 3),
                             "whole_photo": whole, "pred_iou": round(iou, 4)})
        np.savez_compressed(join(args.output_dir, name), segs=segs)
        eff_rows.append({"case": name, "time": round(time() - t0, 4)})

        if args.reference_dir and all_whole:
            ref = join(args.reference_dir, name)
            if isfile(ref):
                whole_photo_cases += 1
                matches += int(np.array_equal(np.load(ref)["segs"] > 0, segs > 0))

    for fname, rows in (("zoom_boxes.csv", box_rows), ("efficiency.csv", eff_rows)):
        with open(join(args.output_dir, fname), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    z = np.array([r["zoom_vs_today"] for r in box_rows])
    print(f"Photos: {len(files)}, boxes: {len(box_rows)}, boxes whose crop is the whole photo: {int((z == 1).sum())}")
    print(f"Zoom vs today: median {np.median(z):.1f}x, max {z.max():.1f}x")
    if args.reference_dir:
        print(f"Self-check vs {args.reference_dir}: {matches} of {whole_photo_cases} whole-photo cases identical to B0"
              + ("  OK" if matches == whole_photo_cases else "  <-- MISMATCH, tell Claude"))


if __name__ == "__main__":
    main()
