#!/usr/bin/env python3
"""LTX-Video 13B on POWER8 - High Resolution"""
import sys
import os
import gc
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

for i in range(1, 8):
    if not hasattr(torch, f"uint{i}"):
        setattr(torch, f"uint{i}", torch.uint8)

import numpy as np
from PIL import Image
from safetensors import safe_open
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer

print("=" * 60)
print("LTX-Video 13B POWER8 - High Resolution")
print("=" * 60)

MODEL_13B = os.path.expanduser("~/models/ltx-video-13b/ltxv-13b-0.9.8-distilled.safetensors")
FULL_MODEL = os.path.expanduser("~/models/ltx-video-full")

# Higher quality settings
RESOLUTION = 512
FRAMES = 17  # More frames
STEPS = 8     # More steps
PATCH_SIZE = 1
PATCH_SIZE_T = 1

def pack_latents(latents, patch_size=1, patch_size_t=1):
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    
    latents = latents.reshape(
        batch_size, num_channels,
        post_patch_num_frames, patch_size_t,
        post_patch_height, patch_size,
        post_patch_width, patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    latents = latents.reshape(
        batch_size,
        post_patch_num_frames * post_patch_height * post_patch_width,
        num_channels * patch_size_t * patch_size * patch_size,
    )
    return latents

def unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1, out_channels=128):
    batch_size = latents.shape[0]
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    
    latents = latents.reshape(
        batch_size,
        post_patch_num_frames, post_patch_height, post_patch_width,
        out_channels, patch_size_t, patch_size, patch_size,
    )
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = latents.reshape(batch_size, out_channels, num_frames, height, width)
    return latents

def map_key(key):
    k = key.replace("model.diffusion_model.", "")
    mappings = [
        ("patchify_proj.", "proj_in."),
        (".q_norm.", ".norm_q."),
        (".k_norm.", ".norm_k."),
        ("adaln_single.emb.timestep_embedder.", "time_embed.emb.timestep_embedder."),
        ("adaln_single.linear.", "time_embed.linear."),
    ]
    for old, new in mappings:
        k = k.replace(old, new)
    return k

def load_and_map_weights(model, checkpoint_path):
    with safe_open(checkpoint_path, framework="pt") as f:
        checkpoint_keys = list(f.keys())
        model_keys = set(model.state_dict().keys())
        mapped_state = {}
        for ck in checkpoint_keys:
            if ".vae." in ck:
                continue
            mk = map_key(ck)
            if mk in model_keys:
                mapped_state[mk] = f.get_tensor(ck)
        model.load_state_dict(mapped_state, strict=False)
        return len(mapped_state)

def show_mem():
    import subprocess
    r = subprocess.run(["free", "-g"], capture_output=True, text=True)
    for line in r.stdout.strip().split("\n")[1:2]:
        parts = line.split()
        print(f"  RAM: {parts[3]}G used, {parts[6]}G available")

start_time = time.time()

# 1. Tokenize
print("\n[1/5] Tokenizing...")
tokenizer = T5Tokenizer.from_pretrained(FULL_MODEL, subfolder="tokenizer")
prompt = "A majestic phoenix rising from flames, golden feathers glowing, magical sparks swirling around"
inputs = tokenizer(prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
print(f"  Prompt: {prompt}")

# 2. Text encoder
print("\n[2/5] Loading T5 encoder...")
show_mem()
text_encoder = T5EncoderModel.from_pretrained(FULL_MODEL, subfolder="text_encoder", torch_dtype=torch.float32, low_cpu_mem_usage=True)
with torch.no_grad():
    prompt_embeds = text_encoder(**inputs).last_hidden_state.clone()
    encoder_attention_mask = inputs.attention_mask.clone()
del text_encoder
gc.collect()

# 3. 13B Transformer
print("\n[3/5] Loading 13B transformer...")
show_mem()

config = {
    "in_channels": 128, "out_channels": 128,
    "patch_size": PATCH_SIZE, "patch_size_t": PATCH_SIZE_T,
    "num_attention_heads": 32, "attention_head_dim": 128,
    "cross_attention_dim": 4096, "num_layers": 48,
    "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False, "norm_eps": 1e-6,
    "caption_channels": 4096, "attention_bias": True,
}

transformer = LTXVideoTransformer3DModel(**config)
matched = load_and_map_weights(transformer, MODEL_13B)
gc.collect()
print(f"  Matched: {matched} weights")
print(f"  Params: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")
show_mem()

# 4. Denoise
print(f"\n[4/5] Denoising ({STEPS} steps)...")

scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False,
)

lf = (FRAMES - 1) // 8 + 1
lh = RESOLUTION // 32
lw = RESOLUTION // 32
print(f"  Resolution: {RESOLUTION}x{RESOLUTION}, Frames: {FRAMES}")
print(f"  Latent: [{lf}, {lh}, {lw}] = {lf*lh*lw} tokens")

latents = torch.randn(1, 128, lf, lh, lw, dtype=torch.float32)
scheduler.set_timesteps(STEPS)
packed_latents = pack_latents(latents, PATCH_SIZE, PATCH_SIZE_T)

for i, t in enumerate(scheduler.timesteps):
    step_start = time.time()
    print(f"  Step {i+1}/{STEPS}: t={t.item():.0f}", end="", flush=True)
    with torch.no_grad():
        pred = transformer(
            hidden_states=packed_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t.unsqueeze(0),
            num_frames=lf,
            height=lh,
            width=lw,
        ).sample
    
    unpacked_pred = unpack_latents(pred, lf, lh, lw, PATCH_SIZE, PATCH_SIZE_T)
    unpacked_latents = unpack_latents(packed_latents, lf, lh, lw, PATCH_SIZE, PATCH_SIZE_T)
    unpacked_latents = scheduler.step(unpacked_pred, t, unpacked_latents).prev_sample
    packed_latents = pack_latents(unpacked_latents, PATCH_SIZE, PATCH_SIZE_T)
    print(f" ({time.time() - step_start:.1f}s)")

latents = unpack_latents(packed_latents, lf, lh, lw, PATCH_SIZE, PATCH_SIZE_T)
del transformer
gc.collect()
print("  Denoising complete!")

# 5. VAE decode
print("\n[5/5] VAE decode...")
show_mem()
vae = AutoencoderKLLTXVideo.from_pretrained(FULL_MODEL, subfolder="vae", torch_dtype=torch.float32)
with torch.no_grad():
    decoded = vae.decode(latents).sample

# Save
frames = decoded.squeeze(0).permute(1, 2, 3, 0).numpy()
frames = (frames * 0.5 + 0.5).clip(0, 1) * 255
frames = frames.astype(np.uint8)
pil_frames = [Image.fromarray(f) for f in frames]
outfile = os.path.expanduser("~/ltx_13b_hires.gif")
pil_frames[0].save(outfile, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)

total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"SUCCESS! {len(pil_frames)} frames @ {RESOLUTION}x{RESOLUTION}")
print(f"Time: {total_time:.1f}s ({total_time/STEPS:.1f}s/step)")
print(f"Output: {outfile}")
print("=" * 60)
