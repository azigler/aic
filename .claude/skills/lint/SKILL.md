---
name: lint
description: Code quality policy for Python policy code and ROS 2 packages
---

# /lint - Code Quality

## What's Automatic

A post-write hook runs on every Python file you write or edit. It silently auto-fixes
formatting, import sorting, and safe lint issues:

| Extension | Hook runs | What it fixes |
|-----------|----------|---------------|
| `.py` | `ruff check --fix` + `ruff format` | Formatting, imports, safe lint fixes |

If the hook encounters an unfixable error, it feeds the error back to you.

## What's Automatic at Commit Time

A PreToolUse hook intercepts `git commit` and runs ruff on staged Python files:

| Staged files | Hook runs | What it catches |
|-------------|----------|-----------------|
| `.py` | `ruff check` | Final lint verification |

If the hook fails, it **blocks the commit**. Fix, re-stage, retry.

## Manual Checks

```bash
ruff check .                            # Check all Python
ruff check --fix . && ruff format .     # Fix all
ruff check path/to/file.py             # Check specific file
```

## Type Checking

This project uses pyright (see `pyrightconfig.json` at repo root):

```bash
pixi run pyright                        # Type check
```

## ROS 2 Build Verification

```bash
pixi install                            # Install/rebuild all packages
pixi reinstall <package_name>           # Reinstall specific package after changes
```

## Rules

- **Do not ignore lint errors** to unblock yourself. Fix them or ask for help.
- **Do not disable rules inline** (`# noqa`) unless the rule is genuinely wrong for
  that line, and leave a comment explaining why.
- **Do not run lint on files you did not modify.** Scope to changed files.

## Known Gaps

- **Ruff does not do type checking** -- use pyright separately for type errors
- **ROS message types** -- pyright may not resolve generated ROS message types;
  add appropriate type stubs or ignores where needed
- **Unsafe fixes** -- auto-fix only applies safe fixes. For unsafe fixes, run
  `ruff check --fix --unsafe-fixes` and review the diff

## Project-Specific Conventions

- Policy classes extend `aic_model.Policy`
- ROS 2 message imports follow `from geometry_msgs.msg import Pose` pattern
- NumPy arrays for batch operations, torch tensors for ML
- f-strings preferred over % formatting or .format()
