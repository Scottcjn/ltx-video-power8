# Contributing to ltx-video-power8

Thanks for helping improve LTX-Video on POWER8. This project documents large
model video diffusion on ppc64le hardware, so contributions should be explicit
about model files, memory requirements, threading, and runtime environment.

## Useful Contributions

- Improve model download, placement, and checksum instructions.
- Add validation notes for different POWER8 memory sizes or thread counts.
- Clarify QK-Norm mapping, latent packing, or hybrid threading behavior.
- Improve scripts with clearer errors for missing models or dependencies.
- Add reproducible benchmark notes for prompt, frame count, and runtime.

## Development Workflow

1. Fork the repository and create a focused branch.
2. Keep script, documentation, and benchmark changes separate when practical.
3. Include Python version, dependency versions, hardware, and model path layout.
4. Do not commit model weights, generated videos, caches, or large artifacts.

## Validation

- Documentation-only changes: run `git diff --check`.
- Python changes: run the touched script or the smallest relevant smoke test.
- Inference changes: include prompt, model checkpoint names, memory size, thread
  settings, and whether generation completed.

## Pull Request Checklist

- The affected script or documentation section is named.
- Validation commands and POWER8 hardware details are included.
- Model artifact requirements are documented but not committed.
- Benchmark claims include prompt, settings, and sample size.
- Generated outputs are excluded unless intentionally documented as fixtures.

## Reporting Issues

Include POWER system model, RAM size, Linux distribution, Python version,
dependency versions, model paths, command run, and the full traceback or runtime
log excerpt.
