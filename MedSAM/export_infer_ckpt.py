"""
Convert any LiteMedSAM checkpoint into the plain state_dict that
CVPR24_LiteMedSAM_infer.py expects (it calls model.load_state_dict(torch.load(path))).

Handles three cases:
  1. Original lite_medsam.pth (already a plain state_dict)  -> copied as-is
  2. Fine-tuned checkpoint from train_one_gpu.py ({"model": state_dict, "epoch": ...})
                                                            -> the "model" entry is extracted
  3. LoRA fine-tuned checkpoint (keys like "...q_proj.base.weight", "...q_proj.lora_A")
                                                            -> LoRA is merged into the base weights:
                                                               W = W_base + (alpha / r) * B @ A

For LoRA, alpha is read from the medsam_lite_lora_{best,latest}.pth file saved next to the
checkpoint (its "config"), or from --lora_alpha if given.

Usage:
  python export_infer_ckpt.py -ckpt work_dir/<run>/medsam_lite_best.pth -out /content/<run>_infer.pth
"""
import argparse
import os
from collections import OrderedDict


def unwrap(ckpt):
    """Return the model state_dict from either a plain state_dict or a training checkpoint."""
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def merge_lora(state, alpha):
    """Fold LoRA adapters into their base Linear layers and return a plain state_dict.

    LoRALinear (train_one_gpu.py) computes base(x) + (alpha/r) * x @ A.T @ B.T,
    which equals a single Linear with weight W_base + (alpha/r) * B @ A.
    """
    prefixes = sorted({k[: -len(".lora_A")] for k in state if k.endswith(".lora_A")})
    merged = OrderedDict()
    for k, v in state.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        merged[k.replace(".base.", ".")] = v
    for p in prefixes:
        A, B = state[p + ".lora_A"], state[p + ".lora_B"]
        r = A.shape[0]
        merged[p + ".weight"] = state[p + ".base.weight"] + (alpha / r) * (B @ A)
    return merged, len(prefixes)


def find_lora_alpha(ckpt_path):
    """Look for the LoRA config saved by train_one_gpu.py next to the checkpoint."""
    import torch
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for name in ("medsam_lite_lora_best.pth", "medsam_lite_lora_latest.pth"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            cfg = torch.load(p, map_location="cpu").get("config", {})
            if "lora_alpha" in cfg:
                return float(cfg["lora_alpha"]), p
    return None, None


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("-ckpt", required=True, help="Checkpoint to convert")
    ap.add_argument("-out", required=True, help="Where to write the plain state_dict")
    ap.add_argument("--lora_alpha", type=float, default=None,
                    help="LoRA alpha; only needed if no medsam_lite_lora_*.pth sits next to -ckpt")
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu")
    state = unwrap(raw)
    if isinstance(raw, dict) and "epoch" in raw:
        print(f"Checkpoint epoch: {raw['epoch']}  loss: {raw.get('loss', 'n/a')}")

    if any(k.endswith(".lora_A") for k in state):
        alpha, src = args.lora_alpha, "--lora_alpha"
        if alpha is None:
            alpha, src = find_lora_alpha(args.ckpt)
        if alpha is None:
            raise ValueError("LoRA checkpoint found but alpha is unknown: pass --lora_alpha")
        state, n = merge_lora(state, alpha)
        print(f"Merged {n} LoRA modules into base weights (alpha={alpha} from {src})")
    else:
        print("No LoRA weights found; exporting model weights as-is")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)
    print(f"Saved inference checkpoint to {args.out}")


if __name__ == "__main__":
    main()
