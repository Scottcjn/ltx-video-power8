# LTX-Video 13B on IBM POWER8

First successful deployment of the 13B parameter LTX-Video diffusion model on IBM POWER8 architecture.

## Overview

This repository contains scripts to run the LTX-Video 13B model on IBM POWER8 S824 systems with 320GB RAM. The implementation includes:

- **Key mapping** for the distilled model architecture (QK-norm → diffusers format)
- **Latent packing/unpacking** for transformer input
- **Single-threaded workaround** for POWER8 stack corruption issues
- **Hybrid multi-threading** option for 5.7× speedup

## Hardware Requirements

- IBM POWER8 or later (tested on S824 with dual 8-core CPUs)
- 64GB+ RAM minimum (320GB recommended for high-res)
- Python 3.11+ with PyTorch 2.1+

## Model Architecture

The 13B distilled LTX-Video model has architectural differences from the standard diffusers format:

| Checkpoint Key | Diffusers Key |
|---------------|---------------|
| `patchify_proj.*` | `proj_in.*` |
| `attn*.q_norm.*` | `attn*.norm_q.*` |
| `attn*.k_norm.*` | `attn*.norm_k.*` |
| `adaln_single.*` | `time_embed.*` |

The scripts handle this mapping automatically.

## Scripts

### `ltx_13b_full.py`
Complete single-threaded pipeline. Safe but slow.
- Resolution: 256×256
- Frames: 9
- Steps: 4
- Time: ~30 seconds

### `ltx_13b_hybrid.py`
Hybrid multi-threaded pipeline with 5.7× speedup.
- Uses multi-threading for transformer (safe)
- Falls back to single-thread for VAE decode (required)
- Achieves ~65 seconds vs 370 seconds single-threaded

### `ltx_13b_hires.py`
High-resolution generation.
- Resolution: 512×512
- Frames: 17
- Steps: 8
- Time: ~54 minutes (single-threaded)

## Usage

```bash
# Download the 13B model
# Place in ~/models/ltx-video-13b/ltxv-13b-0.9.8-distilled.safetensors

# Download the full LTX-Video model (for VAE/text encoder)
# Place in ~/models/ltx-video-full/

# Run inference
cd scripts
python3 ltx_13b_hybrid.py
```

## Results

| Configuration | Resolution | Frames | Time | Output Size |
|--------------|------------|--------|------|-------------|
| Single-thread | 256×256 | 9 | 30s | 511KB |
| Hybrid | 256×256 | 9 | 65s | 531KB |
| High-res | 512×512 | 17 | 54min | 3.9MB |

## Example Outputs

### 256×256 Preview
![Crystal Rotation](examples/ltx_13b_output.gif)

*Prompt: "A glowing crystal rotating slowly in darkness"*

### 512×512 High Resolution
![Phoenix Rising](examples/ltx_13b_hires.gif)

*Prompt: "A majestic phoenix rising from flames, golden feathers glowing"*

## Technical Details

### POWER8 Stack Corruption Workaround

The POWER8 architecture exhibits stack smashing errors with multi-threaded PyTorch operations in certain code paths (particularly VAE decode). The workaround:

```python
# Force single-threaded for VAE
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
```

The hybrid script dynamically switches between multi-threaded (transformer) and single-threaded (VAE) modes.

### Latent Packing

The diffusers transformer expects pre-patchified input:

```python
def pack_latents(latents, patch_size=1, patch_size_t=1):
    # [B, C, F, H, W] → [B, F*H*W, C]
    ...
```

### RoPE Dimensions

The rotary position embeddings expect latent-space dimensions, not video dimensions:

```python
# Correct: pass latent dimensions
num_frames = (FRAMES - 1) // 8 + 1  # latent frames
height = RESOLUTION // 32           # latent height
width = RESOLUTION // 32            # latent width
```

## Dependencies

```
torch>=2.1.0
diffusers>=0.32.0
transformers>=4.39.0
safetensors
pillow
numpy
```

## License

MIT License

## Acknowledgments

- [Lightricks](https://github.com/Lightricks/LTX-Video) for the LTX-Video model
- [Hugging Face](https://huggingface.co/) for diffusers library
- IBM for POWER8 architecture documentation
