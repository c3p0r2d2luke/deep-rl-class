import os
import subprocess
from huggingface_hub import HfApi

REPO_ID = "Frenchwizz/FNAF-Night1-PPO"  # Change repo name if desired
MODEL_FILE = "ppo-fnaf-night1.zip"
RAW_VIDEO = "fnaf_trained_replay.mp4"
# RAW_VIDEO1 = "fnaf_trained_replay.mp4"
# RAW_VIDEO2 = "fnaf_trained.mp4"
HF_VIDEO = "replay.mp4"

# 1. Convert video to H.264 yuv420p so Hugging Face's Video Preview widget plays it natively
print(f"Converting {RAW_VIDEO} to H.264 {HF_VIDEO}...")
subprocess.run(
    [
        "ffmpeg", "-y",
        "-i", RAW_VIDEO,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        HF_VIDEO,
    ],
    check=True,
)

# print(f"Converting {RAW_VIDEO1} to H.264 {RAW_VIDEO2}...")
# subprocess.run(
#     [
#         "ffmpeg", "-y",
#         "-i", RAW_VIDEO1,
#         "-c:v", "libx264",
#         "-pix_fmt", "yuv420p",
#         RAW_VIDEO2,
#     ],
#     check=True,
# )

# 2. Create repository if it doesn't exist yet
api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

# 3. Upload trained PPO model and replay video
print("Uploading model checkpoint...")
api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload trained PPO CnnPolicy agent for FNAF Night 1",
)

print("Uploading replay.mp4 video preview...")
api.upload_file(
    path_or_fileobj=HF_VIDEO,
    path_in_repo="replay.mp4",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add 6 AM Night 1 gameplay replay video",
)

# 4. Upload a clean Model Card (README.md) with RL tags
readme_content = f"""---
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

## Video Replays
<video src="fnaf_trained.mp4" controls width="100%"></video>
<video src="replay.mp4" controls width="100%"></video>

## Model Details
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** `CnnPolicy` (`84x84x3` RGB visual frames)
- **Action Space:** `Discrete(9)` (Wait/Conserve Power, Left Door, Left Light, Right Door, Right Light, Camera Flip, CAM 2B, CAM 4B, CAM 1A)
- **Evaluation Result:** Survived Night 1 to **6:00 AM** (`Total Reward: 174.0`, `741 steps`)
"""

with open("README.md", "w") as f:
    f.write(readme_content)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add FNAF Night 1 model card",
)

print(f"\nAll done! View your model and video preview at: https://huggingface.co{REPO_ID}")
