# Experiment Log

## Session 1 Summary (Previous Agent)

Explored 3 branches: Classical (A), Camera Perception (B), ACT Training (C).
Branches A and B plateaued. Branch C (ACT) became primary approach.

Best claimed score: 136/300 (position-mode ACT, 24 demos, 8 configs, 50 epochs).
However: this was through pixi eval with a custom SimpleACT model that had
argument order bugs. The true score is uncertain.

## Session 2 Summary (This Agent)

Ran 10 experiments, mostly regressions due to:
1. Using Docker compose for eval instead of distrobox+pixi
2. Custom SimpleACT model had bugs (argument order, import path)
3. Docker-collected training data had different distribution

**Key finding:** We should use the competition's LeRobot ACTPolicy (RunACT.py)
instead of our custom SimpleACT. This is the fundamental reset for Session 3.

## Session 3: Fresh Start

Starting from clean upstream with LeRobot-based approach.
Goal: Fine-tune the pretrained ACTPolicy on our CheatCode demonstration data.
