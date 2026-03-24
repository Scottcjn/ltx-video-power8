# Contributing to LTX-Video POWER8

Thank you for your interest in contributing to LTX-Video POWER8! This project brings the LTX-Video 13B video diffusion model to IBM POWER8 systems, marking the first time a 13B parameter video generation model runs on PowerPC architecture.

## Table of Contents

- [Code of Conduct](#code-ofduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [How to Contribute](#how-to-contribute)
- [Build Instructions](#build-instructions)
- [Testing](#testing)
- [Style Guidelines](#style-guidelines)
- [Submitting Changes](#submitting-changes)
- [Performance Optimization](#performance-optimization)
- [Community](#community)

## Code of Conduct

This project expects contributors to:

- Be respectful and collaborative
- Help others learn about PowerPC and AI inference
- Share knowledge about POWER8 optimization techniques
- Respect the challenge of running modern AI on vintage hardware

## Getting Started

### Prerequisites

To contribute, you'll need:

1. **IBM POWER8 System** (S822, S824, or E880) with:
   - Minimum 128GB RAM (256GB+ recommended)
   - RHEL 7.x or Ubuntu 16.04/18.04 for ppc64le
   - NVIDIA GPU support (optional but recommended)
2. **Python 3.8+** with pip
3. **PyTorch 2.0+** compiled for ppc64le
4. **CUDA 11.8+** (if using NVIDIA GPUs)
5. **Understanding of diffusion models** and transformer architectures
6. **Knowledge of PowerPC/POWER architecture** and VSX/VMX instructions

### Understanding the Project

LTX-Video POWER8 is a port of the LTX-Video model:

- **Base Model**: LTX-Video 13B (video diffusion transformer)
- **Target Architecture**: IBM POWER8 (POWER8E/POWER8NVL)
- **ISA**: Power ISA v2.07 (ppc64le)
- **Optimization Focus**: VSX/VMX SIMD, SMT threading, memory bandwidth
- **Input**: Text prompts → Output: Video clips (up to 121 frames)

Key technical challenges:
- 13B parameters require ~52GB memory (FP16)
- Attention mechanisms need optimization for POWER8's cache hierarchy
- Matrix operations benefit from VSX vectorization

## Development Environment

### Hardware Requirements

**Minimum POWER8 Configuration:**
```
- IBM Power System S822 (or better)
- 2x POWER8 processors (10 cores each, 8 threads/core)
- 128GB DDR4 memory
- 1TB storage (SSD recommended)
- Optional: NVIDIA Tesla K80/V100 for GPU acceleration
```

**Recommended Configuration:**
```
- IBM Power System S824 or E880
- 2x POWER8+ processors (12+ cores each)
- 256GB+ DDR4 memory
- NVMe storage
- NVIDIA Tesla V100 (16GB or 32GB)
```

### Setting Up POWER8 Development Environment

```bash
# 1. Verify POWER8 system
cat /proc/cpuinfo | grep -E "(cpu|revision|platform)"

# 2. Install system dependencies (RHEL 7)
sudo yum groupinstall "Development Tools"
sudo yum install python38 python38-pip
sudo yum install cmake ninja-build

# 3. Install optimized BLAS libraries
sudo yum install openblas-openmp
sudo yum install lapack

# 4. Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Installing PyTorch for POWER8

```bash
# Install PyTorch with POWER8 optimizations
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA support (if using NVIDIA GPUs)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'POWER8: {torch.backends.cpu.get_cpu_capability()}')"
```

### Installing LTX-Video Dependencies

```bash
# Clone the repository
git clone https://github.com/Scottcjn/ltx-video-power8.git
cd ltx-video-power8

# Install Python dependencies
pip3 install -r requirements.txt

# Install POWER8-specific optimizations
pip3 install powerai-vision  # IBM's optimized vision libraries
```

## How to Contribute

### Types of Contributions

We welcome:

1. **Performance Optimizations**: VSX/VMX kernels, memory optimizations
2. **Bug Fixes**: Numerical stability, compatibility issues
3. **Documentation**: Setup guides, optimization tips
4. **Benchmarks**: Performance comparisons, profiling data
5. **Model Variants**: Smaller models, quantized versions
6. **Integration**: WebUI, API servers, batch processing

### Finding Issues

Look for issues labeled:

- `performance`: Speed/memory optimization opportunities
- `help-wanted`: Specific assistance needed
- `good-first-issue`: Beginner-friendly tasks
- `documentation`: Documentation improvements
- `power8-specific`: POWER8 architecture issues

## Build Instructions

### Building from Source

```bash
# 1. Clone repository
git clone https://github.com/Scottcjn/ltx-video-power8.git
cd ltx-video-power8

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights
python scripts/download_weights.py --model ltx-video-13b

# 5. Run inference test
python inference.py \
  --prompt "A serene lake at sunset with mountains in the background" \
  --num_frames 49 \
  --fps 24 \
  --output output.mp4
```

### Building with VSX Optimizations

```bash
# Compile custom VSX kernels
python setup.py build_ext --inplace

# Verify VSX support
python -c "from ltx_video import vsx_ops; print('VSX ops available:', vsx_ops.is_available())"
```

### Docker Build (Recommended)

```bash
# Build POWER8-optimized container
docker build -f Dockerfile.power8 -t ltx-video-power8:latest .

# Run with GPU support
docker run --gpus all --rm -it \
  -v $(pwd)/outputs:/outputs \
  ltx-video-power8:latest \
  python inference.py --prompt "Your prompt here"
```

## Testing

### Unit Tests

```bash
# Run Python tests
python -m pytest tests/ -v

# Run POWER8-specific tests
python -m pytest tests/test_power8_ops.py -v

# Run with coverage
python -m pytest tests/ --cov=ltx_video --cov-report=html
```

### Inference Tests

```bash
# Quick smoke test
python inference.py \
  --prompt "Test video" \
  --num_frames 9 \
  --height 256 \
  --width 256 \
  --output test.mp4

# Full quality test
python inference.py \
  --prompt "A beautiful mountain landscape with flowing rivers" \
  --num_frames 121 \
  --height 480 \
  --width 720 \
  --fps 30 \
  --output full_test.mp4
```

### Performance Benchmarks

```bash
# Run benchmark suite
python benchmarks/benchmark_inference.py \
  --models ltx-video-13b \
  --batch_sizes 1,2,4 \
  --output benchmark_results.json

# Profile with perf (Linux)
perf record -g python inference.py --prompt "Test"
perf report
```

### Memory Testing

```bash
# Monitor memory usage
python -m memory_profiler inference.py --prompt "Test"

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python inference.py
```

## Style Guidelines

### Python Code Style

Follow PEP 8 with POWER8-specific conventions:

```python
"""Module docstring with POWER8-specific notes."""

import torch
import torch.nn as nn


class LTXVideoTransformer(nn.Module):
    """LTX-Video transformer optimized for POWER8.
    
    Uses VSX instructions for attention computation when available.
    """
    
    def __init__(self, config: dict) -> None:
        """Initialize transformer with POWER8 optimizations.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.use_vsx = self._check_vsx_support()
        
    def _check_vsx_support(self) -> bool:
        """Check if VSX instructions are available."""
        # Implementation here
        return True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with POWER8 optimizations.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        if self.use_vsx:
            return self._vsx_forward(x)
        return self._standard_forward(x)
```

### VSX Kernel Style

For custom VSX kernels:

```c
// power8_vsx_kernels.c
#include <altivec.h>

// Use vector types for VSX operations
vector float vsx_matmul_vec(vector float a, vector float b) {
    return vec_madd(a, b, (vector float){0.0, 0.0, 0.0, 0.0});
}

// Align data for VSX (16-byte alignment)
__attribute__((aligned(16))) float input[4];
```

### Configuration Files

YAML/JSON configuration:

```yaml
# config/power8_optimized.yaml
model:
  name: "ltx-video-13b"
  checkpoint: "models/ltx-video-13b.pt"
  
optimization:
  use_vsx: true
  use_smt: true        # Simultaneous Multi-Threading
  num_threads: 160     # 10 cores * 8 threads/core * 2 sockets
  memory_pool: "256gb"
  
attention:
  flash_attention: true
  scale_factor: 0.08838834764831843  # 1/sqrt(128)
  
inference:
  batch_size: 1
  num_frames: 121
  height: 480
  width: 720
  fps: 30
```

## Submitting Changes

### Pull Request Process

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch**: `git checkout -b feature/power8-optimization`
4. **Make changes** with focused commits
5. **Test on POWER8 hardware** (or provide emulation evidence)
6. **Update documentation** if needed
7. **Submit PR** with detailed description

### PR Requirements

- [ ] Code passes unit tests
- [ ] Performance benchmarks included (if optimization)
- [ ] Documentation updated
- [ ] VSX code tested on real POWER8
- [ ] Commit messages follow format

### Commit Message Format

```
type(scope): Brief description (50 chars)

Detailed explanation. Include:
- What changed
- Why it changed
- Performance impact (if applicable)

Refs: #issue-number
```

Types:
- `feat`: New feature
- `perf`: Performance improvement
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructuring

Examples:
```
perf(attention): Add VSX-optimized attention kernel

Implements vec_madd-based attention computation for POWER8.
Reduces inference time by 15% for 121-frame videos.

Benchmarks:
- Before: 45s per video
- After: 38s per video
- Hardware: IBM S824, 256GB RAM

Refs: #23
```

## Performance Optimization

### VSX Optimization Guidelines

1. **Align data to 16-byte boundaries**
2. **Use vec_madd for multiply-accumulate**
3. **Minimize vector loads/stores**
4. **Unroll loops for SMT efficiency**

### Profiling Tools

```bash
# Linux perf
perf stat -e cycles,instructions,cache-misses python inference.py

# IBM Performance Tools
perfpmr -o profile.out

# Python profiling
python -m cProfile -o profile.stats inference.py
```

### Memory Optimization

```python
# Use memory-mapped files for large models
torch.load("model.pt", mmap=True)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache between batches
torch.cuda.empty_cache()  # If using GPU
```

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Architecture questions, optimization tips
- **IBM PowerAI Slack**: Community support

### Resources

- [IBM POWER8 Documentation](https://www.ibm.com/docs/en/power8)
- [Power ISA Specification](https://openpowerfoundation.org/specifications/isa/)
- [PyTorch on POWER](https://pytorch.org/blog/)
- [LTX-Video Paper](https://arxiv.org/abs/2401.xxxxx)

### Acknowledgments

- LTX-Video authors for the original model
- IBM PowerAI team for POWER8 optimizations
- OpenPOWER Foundation for architecture documentation
- Contributors keeping PowerPC AI alive

---

**Happy video generation on POWER8!** 🎬⚡️