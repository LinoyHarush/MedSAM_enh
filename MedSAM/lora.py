"""
LoRA adapters for LiteMedSAM, shared by train_one_gpu.py and train_one_gpu_multibox.py.

LoRALinear wraps an nn.Linear:  out = W x + (alpha / r) * B A x,  with W frozen, A (r x in) and B (out x r)
trained, B starting at 0 so training starts exactly from the pretrained model. export_infer_ckpt.py merges
B A back into W for inference, for any wrapped layer (decoder or encoder).

Where adapters can go:
  mask decoder  (inject_lora_mask_decoder): q/k/v/out projections of the 7 attention layers of the
                two-way transformer; targets "q,v" -> 14 layers, 47,104 parameters at r = 8.
  image encoder (inject_lora_image_encoder): TinyViT's windowed self-attention blocks (stages 1-3, 10 blocks).
                Each block has one fused qkv layer and one output projection (proj); targets "qkv" -> 10 layers,
                59,392 parameters at r = 8. Stage 0 is convolutional (MBConv) and gets no adapters.
"""
import numpy as np
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base_linear
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros((r, base_linear.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base_linear.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def maybe_wrap_with_lora(parent_module, attr_name, target_flag, enabled_targets, r, alpha, dropout):
    if target_flag not in enabled_targets:
        return False
    if not hasattr(parent_module, attr_name):
        return False
    layer = getattr(parent_module, attr_name)
    if isinstance(layer, LoRALinear):
        return True
    if isinstance(layer, nn.Linear):
        setattr(parent_module, attr_name, LoRALinear(layer, r=r, alpha=alpha, dropout=dropout))
        return True
    return False


def inject_lora_mask_decoder(mask_decoder, enabled_targets, r, alpha, dropout):
    """Unchanged from train_one_gpu.py (moved here). Targets among q, k, v, o."""
    injected = 0
    t = mask_decoder.transformer
    attns = []
    for blk in t.layers:
        attns += [blk.self_attn, blk.cross_attn_token_to_image, blk.cross_attn_image_to_token]
    attns.append(t.final_attn_token_to_image)
    for attn in attns:
        injected += maybe_wrap_with_lora(attn, "q_proj", "q", enabled_targets, r, alpha, dropout)
        injected += maybe_wrap_with_lora(attn, "k_proj", "k", enabled_targets, r, alpha, dropout)
        injected += maybe_wrap_with_lora(attn, "v_proj", "v", enabled_targets, r, alpha, dropout)
        injected += maybe_wrap_with_lora(attn, "out_proj", "o", enabled_targets, r, alpha, dropout)
    return injected


def inject_lora_image_encoder(image_encoder, enabled_targets, r, alpha, dropout):
    """TinyViT attention blocks. Targets among qkv (fused query/key/value) and proj (output projection)."""
    injected = 0
    for layer in image_encoder.layers:
        for blk in getattr(layer, "blocks", []):
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            injected += maybe_wrap_with_lora(attn, "qkv", "qkv", enabled_targets, r, alpha, dropout)
            injected += maybe_wrap_with_lora(attn, "proj", "proj", enabled_targets, r, alpha, dropout)
    return injected
