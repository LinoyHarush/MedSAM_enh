import os
import random
from pathlib import Path
from glob import glob
from datetime import datetime
from time import time
from collections import defaultdict

import cv2
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT


def parse_args():
    p = argparse.ArgumentParser(description="LiteMedSAM multi-box training (object-level samples) with validation.")
    p.add_argument("-train_npz_dir", type=str, required=True, help="Directory of train-ready npz files (imgs,gts,boxes).")
    p.add_argument("-val_npz_dir", type=str, default=None, help="Directory of val npz files (imgs,gts,boxes).")
    p.add_argument("-pretrained_checkpoint", type=str, default="lite_medsam.pth")
    p.add_argument("-resume", type=str, default=None)
    p.add_argument("-work_dir", type=str, default="./workdir_multibox")
    p.add_argument("-num_epochs", type=int, default=20)
    p.add_argument("-batch_size", type=int, default=4)
    p.add_argument("-num_workers", type=int, default=2)
    p.add_argument("-device", type=str, default="cuda:0")
    p.add_argument("-image_size", type=int, default=256)
    p.add_argument("-bbox_shift", type=int, default=5)
    p.add_argument("-lr", type=float, default=5e-5)
    p.add_argument("-weight_decay", type=float, default=1e-2)
    p.add_argument("-iou_loss_weight", type=float, default=1.0)
    p.add_argument("-seg_loss_weight", type=float, default=1.0)
    p.add_argument("-ce_loss_weight", type=float, default=1.0)
    p.add_argument("--freeze_image_encoder", action="store_true")
    p.add_argument("--freeze_prompt_encoder", action="store_true")
    p.add_argument("--freeze_mask_decoder", action="store_true")
    p.add_argument("--use_weighted_sampler", action="store_true", help="Oversample small objects.")
    p.add_argument("--small_obj_thr", type=float, default=0.0025, help="Area ratio threshold in 256x256.")
    p.add_argument("--small_obj_weight", type=float, default=3.0, help="Weight for small object samples.")
    return p.parse_args()


def resize_longest_side(image, target_length=256, interpolation=cv2.INTER_AREA):
    oldh, oldw = image.shape[:2]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=interpolation), scale


def pad_image(image, target_size=256):
    h, w = image.shape[:2]
    padh = target_size - h
    padw = target_size - w
    if image.ndim == 3:
        return np.pad(image, ((0, padh), (0, padw), (0, 0)))
    return np.pad(image, ((0, padh), (0, padw)))


def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2]


def transform_box_to_resized(box, scale):
    x1, y1, x2, y2 = box
    return [int(x1 * scale + 0.5), int(y1 * scale + 0.5), int(x2 * scale + 0.5), int(y2 * scale + 0.5)]


def cal_iou(result, reference):
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    iou = intersection.float() / torch.clamp(union.float(), min=1.0)
    return iou.unsqueeze(1)


class MultiBoxNpzDataset(Dataset):
    """
    Object-level dataset:
    one sample = (image_id, one bbox from that image).
    """
    def __init__(self, npz_dir, image_size=256, bbox_shift=5, data_aug=True):
        self.npz_dir = Path(npz_dir)
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

        self.samples = []  # list of (npz_path, box_idx)
        self.meta_area_ratio = []

        npz_files = sorted(glob(str(self.npz_dir / "*.npz")))
        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            if "boxes" not in data or "gts" not in data or "imgs" not in data:
                continue
            boxes = data["boxes"]
            if len(boxes) == 0:
                continue

            img = data["imgs"]
            gt = data["gts"]
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt = (gt > 0).astype(np.uint8)
            h, w = gt.shape[:2]

            for bi, b in enumerate(boxes):
                box = clip_box_xyxy(b.tolist(), w, h)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                x1, y1, x2, y2 = box
                gt_box = gt[y1:y2 + 1, x1:x2 + 1]
                if np.count_nonzero(gt_box) == 0:
                    continue

                # approximate area ratio after resize to track small objects
                # scale by longest side
                scale = image_size / max(h, w)
                bw = max(1, int((x2 - x1) * scale + 0.5))
                bh = max(1, int((y2 - y1) * scale + 0.5))
                area_ratio = (bw * bh) / float(image_size * image_size)

                self.samples.append((npz_path, bi))
                self.meta_area_ratio.append(area_ratio)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, bi = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        img = data["imgs"]
        gt = data["gts"]
        boxes = data["boxes"]

        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.uint8)

        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        if img.ndim == 3 and img.shape[-1] > 3:
            img = img[:, :, :3]
        img = img.astype(np.uint8)

        h0, w0 = gt.shape[:2]
        box_orig = clip_box_xyxy(boxes[bi].tolist(), w0, h0)

        # resize image/mask
        img_rs, scale = resize_longest_side(img, self.image_size, interpolation=cv2.INTER_AREA)
        gt_rs, _ = resize_longest_side(gt, self.image_size, interpolation=cv2.INTER_NEAREST)
        newh, neww = gt_rs.shape[:2]

        img_rs = (img_rs - img_rs.min()) / np.clip(img_rs.max() - img_rs.min(), a_min=1e-8, a_max=None)
        img_pd = pad_image(img_rs, self.image_size)
        gt_pd = pad_image(gt_rs, self.image_size)

        box_256 = transform_box_to_resized(box_orig, scale)
        box_256 = clip_box_xyxy(box_256, self.image_size, self.image_size)

        # object target from bbox-local positive region
        x1, y1, x2, y2 = box_256
        obj_mask = np.zeros_like(gt_pd, dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            local = gt_pd[y1:y2 + 1, x1:x2 + 1]
            obj_mask[y1:y2 + 1, x1:x2 + 1] = (local > 0).astype(np.uint8)

        if np.count_nonzero(obj_mask) == 0:
            # fallback: use box rectangle as weak target to keep sample trainable
            obj_mask[y1:y2 + 1, x1:x2 + 1] = 1

        if self.data_aug:
            # horizontal flip
            if random.random() > 0.5:
                img_pd = np.ascontiguousarray(np.flip(img_pd, axis=1))
                obj_mask = np.ascontiguousarray(np.flip(obj_mask, axis=1))
                x1f = self.image_size - 1 - x2
                x2f = self.image_size - 1 - x1
                x1, x2 = min(x1f, x2f), max(x1f, x2f)
            # vertical flip
            if random.random() > 0.5:
                img_pd = np.ascontiguousarray(np.flip(img_pd, axis=0))
                obj_mask = np.ascontiguousarray(np.flip(obj_mask, axis=0))
                y1f = self.image_size - 1 - y2
                y2f = self.image_size - 1 - y1
                y1, y2 = min(y1f, y2f), max(y1f, y2f)

        # bbox jitter
        H, W = obj_mask.shape
        x1 = max(0, x1 - random.randint(0, self.bbox_shift))
        y1 = max(0, y1 - random.randint(0, self.bbox_shift))
        x2 = min(W - 1, x2 + random.randint(0, self.bbox_shift))
        y2 = min(H - 1, y2 + random.randint(0, self.bbox_shift))
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        img_chw = np.transpose(img_pd.astype(np.float32), (2, 0, 1))
        return {
            "image": torch.tensor(img_chw).float(),
            "gt2D": torch.tensor(obj_mask[None, :, :]).float(),
            "bboxes": torch.tensor(bbox[None, None, :]).float(),
            "image_name": Path(npz_path).name,
            "new_size": torch.tensor(np.array([newh, neww])).long(),
            "original_size": torch.tensor(np.array([h0, w0])).long(),
        }


class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[:, :, :new_size[0], :new_size[1]]
        masks = F.interpolate(masks, size=(original_size[0], original_size[1]), mode="bilinear", align_corners=False)
        return masks


def build_model(device):
    image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    model = MedSAM_Lite(image_encoder=image_encoder, mask_decoder=mask_decoder, prompt_encoder=prompt_encoder)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device, seg_loss, ce_loss, iou_loss, seg_w, ce_w, iou_w):
    model.eval()
    total = defaultdict(float)
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        gt2D = batch["gt2D"].to(device)
        boxes = batch["bboxes"].to(device)

        logits_pred, iou_pred = model(image, boxes)
        l_seg = seg_loss(logits_pred, gt2D)
        l_ce = ce_loss(logits_pred, gt2D)
        mask_loss = seg_w * l_seg + ce_w * l_ce
        iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
        l_iou = iou_loss(iou_pred, iou_gt)
        loss = mask_loss + iou_w * l_iou

        pred_bin = (torch.sigmoid(logits_pred) > 0.5)
        inter = torch.count_nonzero(torch.logical_and(pred_bin, gt2D.bool()), dim=[1, 2, 3]).float()
        union = torch.count_nonzero(torch.logical_or(pred_bin, gt2D.bool()), dim=[1, 2, 3]).float().clamp_min(1.0)
        iou_batch = (inter / union).mean().item()

        dice_num = (2.0 * inter)
        dice_den = (torch.count_nonzero(pred_bin, dim=[1, 2, 3]).float() +
                    torch.count_nonzero(gt2D.bool(), dim=[1, 2, 3]).float()).clamp_min(1.0)
        dice_batch = (dice_num / dice_den).mean().item()

        total["loss"] += loss.item()
        total["seg"] += l_seg.item()
        total["ce"] += l_ce.item()
        total["iou_loss"] += l_iou.item()
        total["iou_metric"] += iou_batch
        total["dice_metric"] += dice_batch
        n += 1

    for k in total:
        total[k] /= max(n, 1)
    model.train()
    return total


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    torch.cuda.empty_cache()
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"

    device = args.device
    model = build_model(device)

    if args.pretrained_checkpoint and Path(args.pretrained_checkpoint).is_file():
        print(f"Finetuning with pretrained weights {args.pretrained_checkpoint}")
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
        # supports either pure state_dict or checkpoint dict with "model"
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
    else:
        print("Pretrained checkpoint not found, training from scratch.")

    if args.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if args.freeze_prompt_encoder:
        for p in model.prompt_encoder.parameters():
            p.requires_grad = False
    if args.freeze_mask_decoder:
        for p in model.mask_decoder.parameters():
            p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("All model parameters are frozen.")

    print(f"MedSAM Lite size: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_ds = MultiBoxNpzDataset(
        npz_dir=args.train_npz_dir,
        image_size=args.image_size,
        bbox_shift=args.bbox_shift,
        data_aug=True,
    )
    print(f"Train object-level samples: {len(train_ds)}")

    if args.use_weighted_sampler:
        weights = []
        for ar in train_ds.meta_area_ratio:
            if ar < args.small_obj_thr:
                weights.append(args.small_obj_weight)
            else:
                weights.append(1.0)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True
        )
        print("Using WeightedRandomSampler for small-object oversampling.")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )

    val_loader = None
    if args.val_npz_dir:
        val_ds = MultiBoxNpzDataset(
            npz_dir=args.val_npz_dir,
            image_size=args.image_size,
            bbox_shift=0,
            data_aug=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        print(f"Val object-level samples: {len(val_ds)}")

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, cooldown=0)

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    iou_loss = nn.MSELoss(reduction="mean")

    start_epoch = 0
    best_train_loss = 1e10
    best_val_dice = -1.0
    best_train_epoch = -1
    best_val_epoch = -1

    if args.resume and Path(args.resume).is_file():
        print(f"Resuming from checkpoint {args.resume}")
        r = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(r["model"], strict=True)
        optimizer.load_state_dict(r["optimizer"])
        start_epoch = int(r.get("epoch", 0))
        best_train_loss = float(r.get("best_train_loss", r.get("loss", 1e10)))
        best_val_dice = float(r.get("best_val_dice", -1.0))
        best_train_epoch = int(r.get("best_train_epoch", -1))
        best_val_epoch = int(r.get("best_val_epoch", -1))

    train_curve = []
    val_curve = []

    model.train()
    for epoch in range(start_epoch + 1, args.num_epochs):
        epoch_loss = []
        epoch_seg = []
        epoch_ce = []
        epoch_iou_l = []
        pbar = tqdm(train_loader)

        for batch in pbar:
            image = batch["image"].to(device)
            gt2D = batch["gt2D"].to(device)
            boxes = batch["bboxes"].to(device)

            optimizer.zero_grad()
            logits_pred, iou_pred = model(image, boxes)

            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D)
            mask_loss = args.seg_loss_weight * l_seg + args.ce_loss_weight * l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            loss = mask_loss + args.iou_loss_weight * l_iou

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_seg.append(l_seg.item())
            epoch_ce.append(l_ce.item())
            epoch_iou_l.append(l_iou.item())

            pbar.set_description(
                f"Epoch {epoch} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"loss={loss.item():.4f}"
            )

        train_loss = float(np.mean(epoch_loss))
        train_seg = float(np.mean(epoch_seg))
        train_ce = float(np.mean(epoch_ce))
        train_iou_l = float(np.mean(epoch_iou_l))
        train_curve.append(train_loss)
        lr_scheduler.step(train_loss)

        val_stats = None
        if val_loader is not None:
            val_stats = evaluate(
                model=model, loader=val_loader, device=device,
                seg_loss=seg_loss, ce_loss=ce_loss, iou_loss=iou_loss,
                seg_w=args.seg_loss_weight, ce_w=args.ce_loss_weight, iou_w=args.iou_loss_weight
            )
            val_curve.append(val_stats["dice_metric"])
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} seg={train_seg:.4f} ce={train_ce:.4f} iou_l={train_iou_l:.4f} | "
                f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice_metric']:.4f} val_iou={val_stats['iou_metric']:.4f}"
            )
        else:
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} seg={train_seg:.4f} ce={train_ce:.4f} iou_l={train_iou_l:.4f}"
            )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seg_loss": train_seg,
            "train_ce_loss": train_ce,
            "train_iou_loss": train_iou_l,
            "best_train_loss": best_train_loss,
            "best_train_epoch": best_train_epoch,
            "best_val_dice": best_val_dice,
            "best_val_epoch": best_val_epoch,
            "args": vars(args),
        }
        if val_stats is not None:
            ckpt["val_loss"] = val_stats["loss"]
            ckpt["val_dice"] = val_stats["dice_metric"]
            ckpt["val_iou"] = val_stats["iou_metric"]

        torch.save(ckpt, os.path.join(args.work_dir, "medsam_lite_multibox_latest.pth"))

        # best by train loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch
            ckpt["best_train_loss"] = best_train_loss
            ckpt["best_train_epoch"] = best_train_epoch
            torch.save(ckpt, os.path.join(args.work_dir, "medsam_lite_multibox_best_train.pth"))
            print(f"New best train loss: {best_train_loss:.4f} at epoch {best_train_epoch}")

        # best by val dice
        if val_stats is not None and val_stats["dice_metric"] > best_val_dice:
            best_val_dice = val_stats["dice_metric"]
            best_val_epoch = epoch
            ckpt["best_val_dice"] = best_val_dice
            ckpt["best_val_epoch"] = best_val_epoch
            torch.save(ckpt, os.path.join(args.work_dir, "medsam_lite_multibox_best_val.pth"))
            print(f"New best val dice: {best_val_dice:.4f} at epoch {best_val_epoch}")

        # curves
        plt.figure()
        plt.plot(train_curve, label="train_loss")
        if len(val_curve) > 0:
            plt.plot(val_curve, label="val_dice")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Training curves (multibox)")
        plt.savefig(os.path.join(args.work_dir, "train_val_curves_multibox.png"))
        plt.close()

    print("Done.")
    print(f"Best train loss: {best_train_loss:.4f} (epoch {best_train_epoch})")
    if val_loader is not None:
        print(f"Best val dice: {best_val_dice:.4f} (epoch {best_val_epoch})")


if __name__ == "__main__":
    main()