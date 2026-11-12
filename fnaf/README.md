---
library_name: stable-baselines3
tags:
- reinforcement-learning
- deep-reinforcement-learning
- ppo
- fnaf
- custom-environment
---

# PPO Agent Playing Five Nights at Freddy's (Night 1)

This is a trained **PPO (`CnnPolicy`)** agent playing a custom web-based **Five Nights at Freddy's** Gymnasium environment built with Playwright and OpenCV.

## Video Replay
<video src="replay.mp4" controls width="100%"></video>

## Model Details
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** `CnnPolicy` (`84x84x3` RGB visual frames)
- **Action Space:** `Discrete(9)` (Wait/Conserve Power, Left Door, Left Light, Right Door, Right Light, Camera Flip, CAM 2B, CAM 4B, CAM 1A)
- **Evaluation Result:** Survived Night 1 to **6:00 AM** (`Total Reward: 174.0`, `741 steps`)
