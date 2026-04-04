---
name: submit
description: Build, verify, and submit policy container to cloud evaluation (1/day limit)
argument-hint: "[build|verify|push|status]"
---

# /submit - Cloud Submission

Submit the current best policy for official cloud evaluation. **1 submission per day.**
All steps run on the GPU instance via SSH.

## Pre-Submission Checklist

Before invoking this skill, confirm:

- [ ] Current policy is our personal best (check experiment-log.md)
- [ ] Score is consistent across 2+ local eval runs
- [ ] No force penalties (>20N for >1s)
- [ ] No off-limit contact penalties
- [ ] All 3 trials complete without timeout
- [ ] Human has approved the submission

## Step 1: Build Docker Image on GPU

```bash
ssh gpu "cd ~/ws_aic/src/aic && docker compose -f docker/docker-compose.yaml build model"
```

**Important:** The docker-compose.yaml CMD must point to the correct policy class.
Update it before building:

```bash
# On GPU instance, or edit locally and rsync
ssh gpu "cd ~/ws_aic/src/aic && \
  sed -i 's|command:.*|command: --ros-args -p policy:=<pkg>.<PolicyClass>|' \
  docker/docker-compose.yaml"
```

## Step 2: Verify in Docker (on GPU)

Run the full eval in Docker to confirm scores match pixi-based eval:

```bash
ssh gpu "cd ~/ws_aic/src/aic && docker compose -f docker/docker-compose.yaml up 2>&1 | tail -20"
```

Check scoring.yaml matches expectations. If scores differ significantly from
pixi eval, investigate before submitting.

## Step 3: Authenticate with ECR

```bash
ssh gpu "aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  973918476471.dkr.ecr.us-east-1.amazonaws.com"
```

**Requires:** AWS CLI configured on GPU instance with team credentials.
Team credentials come from the onboarding email.

One-time AWS CLI setup:
```bash
ssh gpu "aws configure --profile aic"
# Enter Access Key ID, Secret Access Key, region: us-east-1
ssh gpu "export AWS_PROFILE=aic"
```

## Step 4: Tag and Push

**Tags are immutable** -- increment version each time.

```bash
VERSION="v$(date +%Y%m%d-%H%M)"  # e.g., v20260404-1530
REPO="973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>"

ssh gpu "docker tag my-solution:v1 ${REPO}:${VERSION} && \
  docker push ${REPO}:${VERSION}"
```

## Step 5: Register on Portal

1. Log in to submission portal (credentials from onboarding email)
2. Click "AI for Industry Challenge" → "Submit"
3. Select "Qualification" phase
4. Paste the full Image URI: `${REPO}:${VERSION}`
5. Click "Submit"

## Step 6: Monitor

Check "My Submissions" in the portal:
- **Submitted** → **Queued** → **Running** → **Finished** (5-15 min)
- **Failed** = container crashed (check logs, fix, try again tomorrow)

## Step 7: Log the Submission

Create a bead to track:

```bash
br create -p 1 "submit: <date> <policy_name> <local_score>"
br update <id> --description "$(cat <<'EOF'
## Submission
- Date: YYYY-MM-DD
- Policy: <pkg>.<Class>
- Version tag: <VERSION>
- Local score: <total>
- Cloud score: <pending -- check portal>

## Local Results (from GPU pixi eval)
| Trial | T1 | T2 | T3 | Total |
|-------|----|----|----| ------|
| 1 | X | X | X | X |
| 2 | X | X | X | X |
| 3 | X | X | X | X |

## Cloud Results (from portal)
<fill in after eval completes>
EOF
)"
```

## Submission Script

For convenience, `scripts/submit.sh` automates steps 1-4:

```bash
scripts/submit.sh <policy_class> <version_tag>
```

## Rules

- **1 submission per day** -- make it count
- **Never submit untested code** -- always verify in Docker first
- **Tags are immutable** -- can't overwrite, must increment
- **Human approval required** -- never auto-submit
