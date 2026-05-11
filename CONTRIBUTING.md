# Contributing to LTX-Video POWER8

Thanks for helping improve LTX-Video POWER8. This repository documents and
packages scripts for running LTX-Video model workflows on IBM POWER8 hardware.

## Good First Contributions

- Clarify README setup steps, model paths, or hardware requirements.
- Fix typos, stale links, or unclear command examples.
- Improve script comments without changing runtime behavior.
- Add notes from tested POWER8, ppc64le, or Linux environments.
- Document common failure modes and how to diagnose them.

## Before You Open a Pull Request

1. Check existing issues and pull requests to avoid duplicate work.
2. Keep each pull request focused on one topic.
3. Use clear commit messages, such as `docs: clarify model download paths`.
4. Do not commit model weights, generated videos, cache directories, logs, or
   local credentials.
5. If you change a script, describe the hardware and Python environment used
   for testing.

## Documentation Guidelines

- Prefer copy-pasteable shell commands.
- Mention whether a command is run from the repository root or `scripts/`.
- Keep model download paths consistent with the README.
- Use relative links for files in this repository.
- Separate measured results from expected or planned results.

## Script Guidelines

- Keep POWER8-specific workarounds documented near the relevant code.
- Avoid broad formatting changes in the same pull request as a behavior change.
- Preserve existing script names and command-line assumptions unless the PR is
  specifically about migration.
- Include fallback notes when a workflow requires large model files or hardware
  that reviewers may not have locally.

## Pull Request Checklist

- [ ] The change is scoped to LTX-Video POWER8.
- [ ] Documentation links, file paths, and commands were checked.
- [ ] Script changes were run, syntax-checked, or clearly marked as not run.
- [ ] No generated artifacts, model weights, cache files, or secrets were added.

## Review Process

Maintainers may ask for smaller diffs, clearer test notes, or more precise
hardware details before merging. That keeps the project easier to reproduce on
POWER8 systems and easier to review without access to every model artifact.
