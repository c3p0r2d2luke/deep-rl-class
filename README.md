---
tags:
- deep-reinforcement-learning
- reinforcement-learning
- ppo
- minesweeper
- custom-environment
library_name: stable-baselines3
---

# PPO Agent playing MineSweeper Mania

This is a trained **PPO** agent playing a custom web-based **Minesweeper** environment built using Playwright and Gymnasium.

## Video Replay

![Minesweeper Gameplay](minesweeper_gameplay.mp4)

## Model Details
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** MlpPolicy (9x9 numerical grid state)
- **Observation Space:** 9x9 Box (-1 = covered, 0-8 = revealed counts)
- **Action Space:** Discrete(81)
