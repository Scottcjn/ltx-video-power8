# Contributing to LTX-Video 13B on IBM POWER8

Thank you for contributing to LTX-Video 13B on IBM POWER8, a port of LTX-Video for video generation inference on IBM POWER8/POWER9 systems.

## Project Overview

This project runs the LTX-Video 13B video generation model on IBM POWER8 and POWER9 systems using the Hugging Face Diffusers library, leveraging VSX vector extensions for optimized inference.

## Development Setup

### Prerequisites

- IBM POWER8 or POWER9 system (ppc64le)
- Python 3.11+
- CUDA (for GPU acceleration)
- 16GB+ VRAM recommended

### Environment Setup

```bash
git clone https://github.com/Scottcjn/ltx-video-power8.git
cd ltx-video-power8
pip install -r requirements.txt

# Install PyTorch for POWER
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install diffusers
pip install diffusers>=0.32.0 transformers accelerate
```

## Code Style

- Python PEP 8 compliant
- Use `black` for formatting
- Type hints for function signatures
- Docstrings for all public functions

## Testing

```bash
# Run inference test
python scripts/test_inference.py --model <model_path>

# Benchmark performance
python scripts/benchmark.py --batch-size 1 --num-frames 32
```

## Submitting Changes

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature`
3. Test on POWER8/POWER9 hardware
4. Submit a pull request

## Ideas for Contributions

- VSX-optimized kernels for CPU inference
- Additional video generation parameters
- Performance improvements for longer videos
- Integration with video generation pipelines
