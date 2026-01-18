#!/usr/bin/env python3
"""
LTX-Video 13B on POWER8 - HYBRID Threading
- Single-thread for T5 encoder and VAE (required on POWER8)
- Multi-thread for 13B transformer (the bottleneck)
"""
import sys
import os
import gc
import time

# Start SINGLE-THREADED for T5 loading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
#torch.set_num_interop_threads(1)  # Removed: causes error after parallel work

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
print("LTX-Video 13B POWER8 - HYBRID Threading")
print("=" * 60)

MODEL_13B = os.path.expanduser("~/models/ltx-video-13b/ltxv-13b-0.9.8-distilled.safetensors")
FULL_MODEL = os.path.expanduser("~/models/ltx-video-full")

RESOLUTION = 256
FRAMES = 9
STEPS = 20
PATCH_SIZE = 1
PATCH_SIZE_T = 1

TRANSFORMER_THREADS = 32  # Reduced for OpenBLAS stability

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
        matched = 0
        for ck in checkpoint_keys:
            if ".vae." in ck:
                continue
            mk = map_key(ck)
            if mk in model_keys:
                mapped_state[mk] = f.get_tensor(ck)
                matched += 1
        model.load_state_dict(mapped_state, strict=False)
        return matched

print(f"\n[Config]")
print(f"  T5/VAE threads: 1 (required on POWER8)")
print(f"  Transformer threads: {TRANSFORMER_THREADS}")
print(f"  Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"  Frames: {FRAMES}, Steps: {STEPS}")

import subprocess
result = subprocess.run(["free", "-g"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "Mem" in line:
        parts = line.split()
        print(f"  RAM: {parts[2]}G used, {parts[6]}G available")

# Step 1: Tokenize (single-thread)
print(f"\n[1/5] Tokenizing (1 thread)...")
prompt = "A serene forest with sunlight filtering through tall pine trees, gentle breeze moving branches, morning mist"
print(f"  Prompt: {prompt}")

tokenizer = T5Tokenizer.from_pretrained(FULL_MODEL, subfolder="tokenizer")
text_inputs = tokenizer(
    prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
)

# Step 2: Encode text (single-thread - REQUIRED ON POWER8!)
print(f"\n[2/5] Loading T5 encoder (1 thread)...")
text_encoder = T5EncoderModel.from_pretrained(FULL_MODEL, subfolder="text_encoder", torch_dtype=torch.float32)

with torch.no_grad():
    prompt_embeds = text_encoder(**text_inputs).last_hidden_state

del text_encoder
gc.collect()
# Keep text_inputs.attention_mask for transformer forward
print(f"  Text encoding complete")

# Step 3: Load transformer - SWITCH TO MULTI-THREAD
print(f"\n[3/5] Loading 13B transformer ({TRANSFORMER_THREADS} threads)...")

# SWITCH TO MULTI-THREAD FOR TRANSFORMER
os.environ["OMP_NUM_THREADS"] = str(TRANSFORMER_THREADS)
os.environ["OMP_PROC_BIND"] = "spread"
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # Max for precompiled OpenBLAS
torch.set_num_threads(TRANSFORMER_THREADS)
# torch.set_num_interop_threads(16)  # Inter-op parallelism

result = subprocess.run(["free", "-g"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "Mem" in line:
        parts = line.split()
        print(f"  RAM: {parts[2]}G used, {parts[6]}G available")

latent_height = RESOLUTION // 32
latent_width = RESOLUTION // 32
num_latent_frames = (FRAMES - 1) // 8 + 1

transformer = LTXVideoTransformer3DModel(
    in_channels=128, out_channels=128,
    patch_size=PATCH_SIZE, patch_size_t=PATCH_SIZE_T,
    num_attention_heads=32, attention_head_dim=128,
    cross_attention_dim=4096, num_layers=48,
    activation_fn="gelu-approximate", qk_norm="rms_norm_across_heads", norm_elementwise_affine=False, norm_eps=1e-6, caption_channels=4096,
)

matched = load_and_map_weights(transformer, MODEL_13B)
transformer.eval()

print(f"  Matched: {matched} weights")
print(f"  Params: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B")

# Step 4: Denoise (MULTI-THREAD)
print(f"\n[4/5] Denoising ({STEPS} steps, {TRANSFORMER_THREADS} threads)...")
print(f"  Latent: [{num_latent_frames}, {latent_height}, {latent_width}] = {num_latent_frames * latent_height * latent_width} tokens")

scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
scheduler.set_timesteps(STEPS)

torch.manual_seed(42)
latents = torch.randn(1, 128, num_latent_frames, latent_height, latent_width, dtype=torch.float32)
latents = pack_latents(latents, PATCH_SIZE, PATCH_SIZE_T)
# latents = latents * scheduler.init_noise_sigma  # Not needed for Flow Matching

start = time.time()
with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        step_start = time.time()
        noise_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=text_inputs.attention_mask,
            timestep=t.unsqueeze(0),
            num_frames=num_latent_frames,
            height=latent_height,
            width=latent_width,
            return_dict=False
        )[0]
        # Unpack for scheduler step (CRITICAL FIX)
        unpacked_pred = unpack_latents(noise_pred, num_latent_frames, latent_height, latent_width, PATCH_SIZE, PATCH_SIZE_T)
        unpacked_latents = unpack_latents(latents, num_latent_frames, latent_height, latent_width, PATCH_SIZE, PATCH_SIZE_T)
        unpacked_latents = scheduler.step(unpacked_pred, t, unpacked_latents).prev_sample
        # Repack for next iteration
        latents = pack_latents(unpacked_latents, PATCH_SIZE, PATCH_SIZE_T)
        print(f"  Step {i+1}/{STEPS}: t={t.item():.0f} ({time.time()-step_start:.1f}s)")
        del noise_pred
        gc.collect()

denoise_time = time.time() - start
print(f"  Denoise total: {denoise_time:.1f}s ({denoise_time/STEPS:.1f}s/step)")

latents = unpack_latents(latents, num_latent_frames, latent_height, latent_width, PATCH_SIZE, PATCH_SIZE_T)
torch.save({"latents": latents, "frames": FRAMES, "height": RESOLUTION, "width": RESOLUTION}, "/tmp/ltx_13b_latents.pt")

del transformer
gc.collect()

# Step 5: VAE decode - SWITCH BACK TO SINGLE-THREAD
print(f"\n[5/5] VAE decode (1 thread)...")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
#torch.set_num_interop_threads(1)  # Removed: causes error after parallel work

vae = AutoencoderKLLTXVideo.from_pretrained(FULL_MODEL, subfolder="vae", torch_dtype=torch.float32)
vae.eval()

start = time.time()
with torch.no_grad():
    decoded = vae.decode(latents).sample

decode_time = time.time() - start
print(f"  VAE decode: {decode_time:.1f}s")

frames = decoded.squeeze(0).permute(1, 2, 3, 0).numpy()
frames = (frames * 0.5 + 0.5).clip(0, 1) * 255
frames = frames.astype(np.uint8)

outfile = os.path.expanduser("~/ltx_13b_nature.gif")
pil_frames = [Image.fromarray(f) for f in frames]
pil_frames[0].save(outfile, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)

print(f"\n" + "=" * 60)
print(f"SUCCESS! Output: {outfile}")
print(f"  Frames: {len(pil_frames)}")
print(f"  Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"  Denoise time: {denoise_time:.1f}s ({denoise_time/STEPS:.1f}s/step)")
print(f"  VAE time: {decode_time:.1f}s")
print(f"  Total: {denoise_time + decode_time:.1f}s")
print("=" * 60)
