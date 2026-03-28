---
name: release
description: Build and submit the policy container for cloud evaluation
argument-hint: "[build|submit|verify]"
---

# /release - Submission

Package and submit the policy for cloud evaluation.

## Pre-Submission Checklist

- [ ] Policy passes Tier 1 locally
- [ ] All 3 trials complete without timeout
- [ ] No force or contact penalties
- [ ] Docker container builds and runs
- [ ] Scores match between pixi and Docker runs

## Build

```bash
docker compose -f docker/docker-compose.yaml build model
```

## Local Verification

```bash
docker compose -f docker/docker-compose.yaml up
```

## Submit

```bash
# Authenticate (12hr token)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

# Tag (increment version each time -- tags are immutable)
docker tag localhost/my-solution:v1 \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:vN

# Push
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:vN
```

Then register the image URI in the submission portal.

## Limit: 1 submission per day

Make it count. Always run full local eval before submitting.

## Cloud GPU Submission Flow

Docker build and push happen **on the GPU instance** for speed (NVMe storage, 22
cores, GPU available for testing). No need for Docker locally.

```bash
# SSH into the GPU instance
ssh gpu

# Build on the GPU instance
cd ~/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml build model

# Run full eval on GPU instance to verify
docker compose -f docker/docker-compose.yaml up

# Authenticate and push from GPU instance
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
docker tag localhost/my-solution:v1 \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:vN
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:vN
```

**Final submission must always be a Docker container.** The GPU instance matches
official eval hardware (NVIDIA L4), so Docker eval results there are highly
representative of cloud eval scores.
