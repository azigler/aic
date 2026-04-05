# Known Runtime Issues and Fixes

## Issue 1: pixi doesn't pick up code changes

**Symptom:** `ModuleNotFoundError: No module named 'aic_example_policies.ros.NewPolicy'`

**Cause:** pixi installs Python packages into its environment. When you add/modify a .py
file, pixi doesn't know about it until you reinstall the package.

**Fix:** Always run `pixi reinstall ros-kilted-aic-example-policies` after changing policy files.

**Where it's fixed:**
- `scripts/remote-eval.sh` runs pixi reinstall after rsync (Step 1b)
- `~/run-eval.sh` on the GPU runs pixi reinstall at startup (Step 0)
- Both are automatic -- no manual step needed when using these scripts

## Issue 2: Zenoh connectivity between distrobox and pixi

**Symptom:** `No node with name 'aic_model' found. Retrying...` in the engine,
or `Unable to connect to a Zenoh router` in the model.

**Cause:** The Zenoh router runs inside the distrobox container. The pixi-based
policy process runs on the host. For Zenoh discovery to work, the distrobox must
be using the host network namespace (which distrobox does by default), AND the
policy process must be able to reach the router.

**When it works:** When distrobox and pixi are run from the same terminal session
or from the `~/run-eval.sh` script.

**When it fails:** When launched from a non-interactive SSH command that splits
the distrobox and pixi into separate SSH sessions.

**Fix:** Always use `~/run-eval.sh` on the GPU instance. This script runs both
distrobox and pixi in the same shell session, ensuring Zenoh connectivity.

**If Zenoh fails anyway:**
1. Kill everything: `killall -9 gz aic_engine aic_model rmw_zenohd; docker kill $(docker ps -q)`
2. Wait 5s for sockets to clear
3. Try again with `~/run-eval.sh`

## Issue 3: Docker pixi build cache

**Symptom:** Docker model container has stale code despite Dockerfile COPY changes.

**Cause:** The Dockerfile uses `--mount=type=cache,target=/root/.cache/rattler/cache`
for pixi's package cache. When source files change (COPY layer invalidated), pixi
may still use cached built packages.

**Fix:** Removed the `.pixi/build` cache mount from the Dockerfile. The rattler
cache mount is kept (download cache, not build cache).

**For submission builds:** If Docker scores differ from pixi scores, try
`docker compose build --no-cache model` to force a clean rebuild.

## Best Practice: The Reliable Eval Flow

```bash
# From this machine:
scripts/remote-eval.sh aic_example_policies.ros.MyPolicy

# This does:
# 1. rsync code to GPU
# 2. pixi reinstall on GPU (picks up code changes)
# 3. ~/run-eval.sh on GPU (distrobox + pixi in same session)
# 4. rsync scoring.yaml back
```

If remote-eval.sh doesn't work, SSH to the GPU manually:

```bash
ssh gpu
cd ~/ws_aic/src/aic
git pull  # or rsync from dev host
pixi reinstall ros-kilted-aic-example-policies
POLICY=aic_example_policies.ros.MyPolicy ~/run-eval.sh
```
