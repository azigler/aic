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

## Remote Runner vs Docker

The Mac Studio remote runner is for **development iteration only**. It provides
faster eval cycles (~10-12 min vs ~27 min) but is not part of the submission
pipeline.

**Final submission must always be a Docker container.** Before submitting:
1. Build the Docker image (`docker compose build model`)
2. Run full eval in Docker (`docker compose up`)
3. Verify Docker scores match remote runner scores
4. Only then tag and push to ECR

Do not rely solely on remote runner scores -- always validate in Docker.
