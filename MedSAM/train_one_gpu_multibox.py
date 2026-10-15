"""
LiteMedSAM training, one sample per lesion, with validation, for the dental caries project.

Rewritten 2026-10-15 from the version of 2026-10-10 (see trainer_fixes_brief.md and MedSAM_enh_repo_review.md).
What a training sample is:
  * one stored box (bboxes.json, from preprocess_tooth_ds.py) = one lesion;
  * the target is that lesion's own pixels (its connected component), not every lesion pixel inside the box;
  * photos whose long side is above -zoom_above (2000 px) are cut to a crop around the box first, with exactly
    the crop rule of infer_zoom.py (imported, so training and inference cannot drift apart); smaller photos are
    used whole, as in B0;
  * the crop (or whole photo) is resized to 256 px like the inference script does, the box with the same
    rounding as inference;
  * lesions that vanish at 256 px, or whose box collapses, are skipped (no rectangle fallback).
Everything is prepared once at start-up and kept in memory, so epochs never read files.

Also: -seed, epochs counted from 0, frozen parts kept in eval mode (BatchNorm statistics don't move),
early stopping and learning-rate schedule on validation, LoRA on the mask decoder and/or the image encoder
(lora.py), per-epoch log.csv, and -max_images for a quick trial run.

Example (frozen encoders, full decoder, seed 0):
  python train_one_gpu_multibox.py -train_npz_dir /content/train_ready_npz \
      -val_npz_dir /content/work/data/dental_caries_val42/train_ready_npz \
      -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -work_dir work_dir/R2_frozen_s0 \
      -seed 0 --freeze_image_encoder --freeze_prompt_encoder
"""
import argparse
import csv
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import time

import cv2
import matplotlib
matplotlib.use("Agg")
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from infer_zoom import crop_window, box_in_crop_256          # same crop + box rule as inference
from diagnose_bbox_survival import lesion_component          # same box -> lesion matching as the diagnostic
from lora import LoRALinear, inject_lora_mask_decoder, inject_lora_image_encoder

IMAGE_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description="LiteMedSAM per-lesion training with validation (dental caries).")
    # data
    p.add_argument("-train_npz_dir", required=True, help="Training-ready npz files (imgs, gts, boxes).")
    p.add_argument("-val_npz_dir", default=None, help="Validation npz files (imgs, gts, boxes), e.g. dental_caries_val42/train_ready_npz.")
    p.add_argument("-max_images", type=int, default=0, help="Use only the first N photos of each set (quick trial). 0 = all.")
    p.add_argument("-zoom_above", type=int, default=2000, help="Crop around each box for photos whose long side is above this (px). 0 = crop all photos; a huge value = never crop.")
    p.add_argument("-margin", type=float, default=0.5, help="Crop context on each side of the box (fraction of the box side). Must match inference.")
    p.add_argument("-min_crop", type=int, default=256, help="Smallest crop side in original px (256 = never enlarge). Must match inference.")
    # model / run
    p.add_argument("-pretrained_checkpoint", default="work_dir/LiteMedSAM/lite_medsam.pth")
    p.add_argument("-resume", default=None, help="Continue a run from its medsam_lite_multibox_latest.pth.")
    p.add_argument("-work_dir", default="./work_dir/multibox_run")
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("-num_epochs", type=int, default=None,
                   help="Upper limit; early stopping usually ends the run sooner. Default by run type (plan, section 10): "
                        "full model 10, frozen encoders 15, LoRA 20.")
    p.add_argument("-batch_size", type=int, default=4)
    p.add_argument("-num_workers", type=int, default=2)
    p.add_argument("-device", default="cuda:0")
    p.add_argument("-bbox_shift", type=int, default=5, help="Random box enlargement during training, in px at 256.")
    p.add_argument("-lr", type=float, default=None, help="Default by run type: 5e-5 (full model, decoder), 1e-4 (LoRA).")
    p.add_argument("-weight_decay", type=float, default=1e-2)
    p.add_argument("-iou_loss_weight", type=float, default=1.0)
    p.add_argument("-seg_loss_weight", type=float, default=1.0)
    p.add_argument("-ce_loss_weight", type=float, default=1.0)
    p.add_argument("-patience", type=int, default=None,
                   help="Stop after this many epochs without a validation Dice gain. Default by run type: 3 / 4 / 5.")
    p.add_argument("-min_delta", type=float, default=1e-3, help="Smallest validation Dice gain that counts.")
    p.add_argument("--freeze_image_encoder", action="store_true")
    p.add_argument("--freeze_prompt_encoder", action="store_true")
    p.add_argument("--freeze_mask_decoder", action="store_true")
    p.add_argument("--use_weighted_sampler", action="store_true", help="Oversample small lesions.")
    p.add_argument("--small_obj_thr", type=float, default=0.0025, help="Lesion area / (256*256) below which a lesion counts as small.")
    p.add_argument("--small_obj_weight", type=float, default=3.0)
    # LoRA
    p.add_argument("--use_lora", action="store_true", help="Train only LoRA adapters; all base weights frozen.")
    p.add_argument("--lora_scope", default="mask_decoder", choices=["mask_decoder", "image_encoder", "both"])
    p.add_argument("--lora_targets", default="q,v", help="Decoder targets among q,k,v,o.")
    p.add_argument("--lora_encoder_targets", default="qkv", help="Encoder targets among qkv,proj.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


# ------------------------------------------------------------------ data
RUN_DEFAULTS = {  # run type: (num_epochs, patience, lr), agreed 2026-10-15 (review file, section 10)
    "full model": (10, 3, 5e-5),
    "frozen encoders": (15, 4, 5e-5),
    "LoRA": (20, 5, 1e-4),
}


def apply_run_defaults(args):
    kind = "LoRA" if args.use_lora else ("frozen encoders" if args.freeze_image_encoder else "full model")
    epochs, patience, lr = RUN_DEFAULTS[kind]
    if args.num_epochs is None:
        args.num_epochs = epochs
    if args.patience is None:
        args.patience = patience
    if args.lr is None:
        args.lr = lr
    print(f"Run type: {kind} -> num_epochs {args.num_epochs}, patience {args.patience}, lr {args.lr:g}")
    return kind


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resized_shape(h, w, target=IMAGE_SIZE):
    """(newh, neww) exactly as resize_longest_side in the inference scripts."""
    s = target * 1.0 / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box]
    return [max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(0, min(x2, w - 1)), max(0, min(y2, h - 1))]


def pad_to(a, size=IMAGE_SIZE):
    h, w = a.shape[:2]
    pad = ((0, size - h), (0, size - w)) + (((0, 0),) if a.ndim == 3 else ())
    return np.pad(a, pad)


def build_samples(npz_dir, args, desc):
    """Read every photo once; return (images, samples, skipped). images: uint8 256-px crops/photos (unpadded)."""
    files = sorted(Path(npz_dir).glob("*.npz"))
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        raise FileNotFoundError(f"No npz files in {npz_dir}")
    images, samples, skipped = [], [], Counter()
    for f in tqdm(files, desc=f"Preparing {desc}"):
        with np.load(f, allow_pickle=True) as d:
            img, gt, boxes = d["imgs"], d["gts"], np.asarray(d["boxes"]).reshape(-1, 4)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        img = img[:, :, :3].astype(np.uint8)
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.uint8)
        H, W = gt.shape
        zoom = max(H, W) > args.zoom_above
        _, labels, stats, _ = cv2.connectedComponentsWithStats(gt, connectivity=8)
        whole_idx = None
        for box in boxes:
            box = clip_box_xyxy(box.tolist(), W, H)
            cid = lesion_component(labels, stats, box)
            if cid == 0:
                skipped["no_matching_lesion"] += 1
                continue
            x0, y0, cw, ch = crop_window(box, H, W, args.margin, args.min_crop) if zoom else (0, 0, W, H)
            b = box_in_crop_256(box, x0, y0, cw, ch)
            if b[2] <= b[0] or b[3] <= b[1]:
                skipped["collapsed_box"] += 1
                continue
            newh, neww = resized_shape(ch, cw)
            own = (labels[y0:y0 + ch, x0:x0 + cw] == cid).astype(np.uint8)
            own_256 = cv2.resize(own, (neww, newh), interpolation=cv2.INTER_NEAREST)
            if own_256.sum() == 0:
                skipped["vanished_at_256"] += 1
                continue
            if zoom:
                images.append(cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (neww, newh), interpolation=cv2.INTER_AREA))
                idx = len(images) - 1
            else:
                if whole_idx is None:
                    images.append(cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA))
                    whole_idx = len(images) - 1
                idx = whole_idx
            samples.append({"img": idx, "mask": own_256.astype(bool), "box": [int(v) for v in b],
                            "file": f.name, "zoomed": zoom})
    n_zoom = sum(s["zoomed"] for s in samples)
    print(f"{desc}: {len(files)} photos -> {len(samples)} lesion samples ({n_zoom} from crops), "
          f"skipped {sum(skipped.values())}: {dict(skipped)}")
    return images, samples, skipped


class LesionDataset(Dataset):
    def __init__(self, images, samples, bbox_shift, data_aug):
        self.images, self.samples = images, samples
        self.bbox_shift, self.data_aug = bbox_shift, data_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = self.images[s["img"]].astype(np.float32)
        img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)   # as in inference
        img = pad_to(img)
        mask = pad_to(s["mask"].astype(np.uint8))
        x1, y1, x2, y2 = s["box"]
        S = IMAGE_SIZE
        if self.data_aug:
            if random.random() > 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
                x1, x2 = S - 1 - x2, S - 1 - x1
            if random.random() > 0.5:
                img, mask = img[::-1], mask[::-1]
                y1, y2 = S - 1 - y2, S - 1 - y1
        if self.bbox_shift > 0:
            x1 = max(0, x1 - random.randint(0, self.bbox_shift))
            y1 = max(0, y1 - random.randint(0, self.bbox_shift))
            x2 = min(S - 1, x2 + random.randint(0, self.bbox_shift))
            y2 = min(S - 1, y2 + random.randint(0, self.bbox_shift))
        return {
            "image": torch.tensor(np.ascontiguousarray(img).transpose(2, 0, 1)).float(),
            "gt2D": torch.tensor(np.ascontiguousarray(mask)[None]).float(),
            "bboxes": torch.tensor([[[x1, y1, x2, y2]]], dtype=torch.float32),
        }


# ------------------------------------------------------------------ model
class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)
        sparse, dense = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        return low_res_masks, iou_predictions


def build_model():
    image_encoder = TinyViT(img_size=256, in_chans=3, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
                            num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], mlp_ratio=4.0, drop_rate=0.0,
                            drop_path_rate=0.0, use_checkpoint=False, mbconv_expand_ratio=4.0,
                            local_conv_size=3, layer_lr_decay=0.8)
    prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                                   input_image_size=(256, 256), mask_in_chans=16)
    mask_decoder = MaskDecoder(num_multimask_outputs=3,
                               transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                               transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
    return MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)


def base_frozen(module):
    """True if none of the module's own (non-LoRA) weights are trained."""
    return not any(p.requires_grad for n, p in module.named_parameters() if "lora_" not in n)


def set_train_mode(model):
    """Train mode for what is trained; eval mode for frozen parts, so their BatchNorm statistics stay fixed.
    LoRA adapters inside a frozen part stay in train mode (their dropout keeps working)."""
    model.train()
    for part in (model.image_encoder, model.prompt_encoder, model.mask_decoder):
        if base_frozen(part):
            part.eval()
            for m in part.modules():
                if isinstance(m, LoRALinear):
                    m.train()


def first_batchnorm(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            return m
    return None


def cal_iou(result, reference):
    inter = torch.count_nonzero(torch.logical_and(result, reference), dim=[1, 2, 3]).float()
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[1, 2, 3]).float()
    return (inter / union.clamp_min(1.0)).unsqueeze(1)


def compute_losses(model, batch, device, losses, w):
    image, gt, boxes = batch["image"].to(device), batch["gt2D"].to(device), batch["bboxes"].to(device)
    logits, iou_pred = model(image, boxes)
    l_seg = losses["seg"](logits, gt)
    l_ce = losses["ce"](logits, gt)
    l_iou = losses["iou"](iou_pred, cal_iou(torch.sigmoid(logits) > 0.5, gt.bool()))
    total = w["seg"] * l_seg + w["ce"] * l_ce + w["iou"] * l_iou
    return total, l_seg, l_ce, l_iou, logits, gt


@torch.no_grad()
def evaluate(model, loader, device, losses, w):
    """Per-lesion Dice / IoU at 256 px, averaged over lesions (not batches). Used to pick the best epoch;
    the per-photo DSC/NSD for the report comes from infer_zoom.py + eval_run.py."""
    model.eval()
    tot = Counter()
    n = 0
    for batch in loader:
        total, l_seg, l_ce, l_iou, logits, gt = compute_losses(model, batch, device, losses, w)
        b = gt.shape[0]
        pred = torch.sigmoid(logits) > 0.5
        g = gt.bool()
        inter = torch.count_nonzero(pred & g, dim=[1, 2, 3]).float()
        psum = torch.count_nonzero(pred, dim=[1, 2, 3]).float()
        gsum = torch.count_nonzero(g, dim=[1, 2, 3]).float()
        tot["dice"] += (2 * inter / (psum + gsum).clamp_min(1)).sum().item()
        tot["iou"] += (inter / (psum + gsum - inter).clamp_min(1)).sum().item()
        tot["loss"] += total.item() * b
        n += b
    set_train_mode(model)
    return {k: v / max(n, 1) for k, v in tot.items()}


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    apply_run_defaults(args)
    os.makedirs(args.work_dir, exist_ok=True)
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # ---- model
    model = build_model()
    if args.pretrained_checkpoint and Path(args.pretrained_checkpoint).is_file():
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
        print(f"Loaded pretrained weights: {args.pretrained_checkpoint}")
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")

    if args.use_lora:
        n_dec = n_enc = 0
        if args.lora_scope in ("mask_decoder", "both"):
            t = {s.strip().lower() for s in args.lora_targets.split(",") if s.strip()}
            n_dec = inject_lora_mask_decoder(model.mask_decoder, t, args.lora_r, args.lora_alpha, args.lora_dropout)
        if args.lora_scope in ("image_encoder", "both"):
            t = {s.strip().lower() for s in args.lora_encoder_targets.split(",") if s.strip()}
            n_enc = inject_lora_image_encoder(model.image_encoder, t, args.lora_r, args.lora_alpha, args.lora_dropout)
        if n_dec + n_enc == 0:
            raise ValueError("No LoRA layers were injected; check --lora_scope and the targets.")
        print(f"LoRA layers: decoder {n_dec}, encoder {n_enc}")
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n
    else:
        for flag, part in ((args.freeze_image_encoder, model.image_encoder),
                           (args.freeze_prompt_encoder, model.prompt_encoder),
                           (args.freeze_mask_decoder, model.mask_decoder)):
            if flag:
                for p in part.parameters():
                    p.requires_grad = False
    model = model.to(device)
    set_train_mode(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("All parameters are frozen.")
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}  trainable: {sum(p.numel() for p in trainable):,} "
          f"(encoder {count(model.image_encoder):,}, prompt {count(model.prompt_encoder):,}, decoder {count(model.mask_decoder):,})")
    print("Frozen (eval mode): " + ", ".join(n for n, m in (("image encoder", model.image_encoder),
          ("prompt encoder", model.prompt_encoder), ("mask decoder", model.mask_decoder)) if base_frozen(m)))

    # ---- data
    tr_images, tr_samples, _ = build_samples(args.train_npz_dir, args, "train")
    train_ds = LesionDataset(tr_images, tr_samples, args.bbox_shift, data_aug=True)
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.use_weighted_sampler:
        thr = args.small_obj_thr * IMAGE_SIZE * IMAGE_SIZE
        weights = [args.small_obj_weight if s["mask"].sum() < thr else 1.0 for s in tr_samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                                  num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if args.val_npz_dir:
        va_images, va_samples, _ = build_samples(args.val_npz_dir, args, "validation")
        val_loader = DataLoader(LesionDataset(va_images, va_samples, 0, data_aug=False), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        print("WARNING: no -val_npz_dir; the best epoch will be chosen by training loss and there is no early stopping.")

    # ---- optimisation
    optimizer = optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, cooldown=0)
    losses = {"seg": monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean"),
              "ce": nn.BCEWithLogitsLoss(reduction="mean"), "iou": nn.MSELoss(reduction="mean")}
    w = {"seg": args.seg_loss_weight, "ce": args.ce_loss_weight, "iou": args.iou_loss_weight}

    start_epoch, best_val, best_val_epoch, best_train, no_improve = 0, -1.0, -1, 1e10, 0
    if args.resume:
        r = torch.load(args.resume, map_location="cpu", weights_only=False)   # our own checkpoint (has args, scheduler)
        model.load_state_dict(r["model"], strict=True)
        optimizer.load_state_dict(r["optimizer"])
        if "scheduler" in r:
            scheduler.load_state_dict(r["scheduler"])
        start_epoch = int(r["epoch"])                       # = next epoch to run
        best_val, best_val_epoch = r.get("best_val_dice", -1.0), r.get("best_val_epoch", -1)
        best_train, no_improve = r.get("best_train_loss", 1e10), r.get("epochs_no_improve", 0)
        print(f"Resumed from {args.resume}: continuing at epoch {start_epoch}")

    bn = first_batchnorm(model.image_encoder)
    bn_ref = bn.running_mean.detach().clone() if bn is not None else None
    log_path = Path(args.work_dir) / "log.csv"
    log_fields = ["epoch", "train_loss", "train_seg", "train_ce", "train_iou", "val_loss", "val_dice", "val_iou",
                  "lr", "seconds", "encoder_bn_max_change"]
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=log_fields).writeheader()

    history = []
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time()
        set_train_mode(model)
        sums, n = Counter(), 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            total, l_seg, l_ce, l_iou, _, gt = compute_losses(model, batch, device, losses, w)
            total.backward()
            optimizer.step()
            b = gt.shape[0]
            for k, v in (("loss", total), ("seg", l_seg), ("ce", l_ce), ("iou", l_iou)):
                sums[k] += v.item() * b
            n += b
            pbar.set_postfix(loss=f"{total.item():.4f}")
        tr = {k: v / n for k, v in sums.items()}
        va = evaluate(model, val_loader, device, losses, w) if val_loader else None
        scheduler.step(va["loss"] if va else tr["loss"])
        lr = optimizer.param_groups[0]["lr"]
        bn_change = float((bn.running_mean - bn_ref).abs().max()) if bn is not None else float("nan")

        row = {"epoch": epoch, "train_loss": tr["loss"], "train_seg": tr["seg"], "train_ce": tr["ce"],
               "train_iou": tr["iou"], "val_loss": va["loss"] if va else "", "val_dice": va["dice"] if va else "",
               "val_iou": va["iou"] if va else "", "lr": lr, "seconds": round(time() - t0, 1),
               "encoder_bn_max_change": bn_change}
        with open(log_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=log_fields).writerow(row)
        history.append(row)
        print(f"[{datetime.now():%H:%M}] epoch {epoch}: train {tr['loss']:.4f}"
              + (f" | val loss {va['loss']:.4f} dice {va['dice']:.4f} iou {va['iou']:.4f}" if va else "")
              + f" | lr {lr:.2e} | encoder BN change {bn_change:.2e} | {row['seconds']} s")

        improved = False
        if va is not None:
            if va["dice"] > best_val + args.min_delta:
                best_val, best_val_epoch, no_improve, improved = va["dice"], epoch, 0, True
            else:
                no_improve += 1
        elif tr["loss"] < best_train:
            improved = True
        best_train = min(best_train, tr["loss"])

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1, "best_val_dice": best_val, "best_val_epoch": best_val_epoch,
                "best_train_loss": best_train, "epochs_no_improve": no_improve, "args": vars(args),
                "train_loss": tr["loss"], "val_dice": va["dice"] if va else None}
        lora_cfg = {"lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                    "lora_targets": args.lora_targets, "lora_encoder_targets": args.lora_encoder_targets,
                    "lora_scope": args.lora_scope}
        torch.save(ckpt, Path(args.work_dir) / "medsam_lite_multibox_latest.pth")
        if args.use_lora:
            torch.save({"lora": lora_state(model), "epoch": epoch, "config": lora_cfg},
                       Path(args.work_dir) / "medsam_lite_lora_latest.pth")
        if improved:
            name = "medsam_lite_multibox_best_val.pth" if va is not None else "medsam_lite_multibox_best_train.pth"
            torch.save(ckpt, Path(args.work_dir) / name)
            if args.use_lora:
                torch.save({"lora": lora_state(model), "epoch": epoch, "config": lora_cfg},
                           Path(args.work_dir) / "medsam_lite_lora_best.pth")
            print(f"  saved {name} (epoch {epoch})")

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([h["train_loss"] for h in history], label="train loss")
        if va is not None:
            ax[0].plot([h["val_loss"] for h in history], label="val loss")
            ax[1].plot([h["val_dice"] for h in history], label="val Dice (per lesion, 256 px)")
            ax[1].legend()
        ax[0].legend(); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch")
        plt.tight_layout(); plt.savefig(Path(args.work_dir) / "curves.png"); plt.close(fig)

        if va is not None and no_improve >= args.patience:
            print(f"Early stop: no validation Dice gain for {args.patience} epochs. Best epoch {best_val_epoch} "
                  f"(val Dice {best_val:.4f}).")
            break

    print("Done." + (f" Best validation Dice {best_val:.4f} at epoch {best_val_epoch}." if val_loader else ""))


if __name__ == "__main__":
    main()
