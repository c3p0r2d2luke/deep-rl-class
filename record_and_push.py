import io
import os
import time
import numpy as np
import imageio
from PIL import Image
from stable_baselines3 import PPO
from huggingface_hub import HfApi, create_repo
from minesweeper_env import FastWebMinesweeperEnv

def record_game_and_push():
    model_filename = "ppo-web-minesweeper.zip"
    
    # Check if model exists
    if not os.path.exists(model_filename) and os.path.exists("ppo-web-minesweeper"):
        model_filename = "ppo-web-minesweeper"
    
    print(f"Loading trained model from {model_filename}...")
    model = PPO.load(model_filename)

    print("Initializing environment for recording...")
    env = FastWebMinesweeperEnv()
    
    frames = []
    obs, _ = env.reset()
    done = False
    step_count = 0

    print("Playing game and capturing frames...")
    while not done and step_count < 200:
        # Capture current page screenshot for video frame
        screenshot_bytes = env.page.screenshot()
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        frames.append(np.array(img))

        # Model predicts action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # Short pause for visual clarity in video
        time.sleep(0.1)

    # Capture final state
    screenshot_bytes = env.page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    frames.append(np.array(img))

    env.close()

    # Save video
    video_filename = "minesweeper_gameplay.mp4"
    print(f"Saving video to {video_filename} ({len(frames)} frames)...")
    imageio.mimsave(video_filename, frames, fps=5)

    # Push to Hugging Face
    repo_id = "Frenchwizz/MineSweeper"  # Change to your HF username/repo if different
    print(f"Uploading model and video to Hugging Face Hub ({repo_id})...")

    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note on repo creation: {e}")

    # Generate Model Card README
    readme_content = f"""---
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
"""

    with open("README.md", "w") as f:
        f.write(readme_content)

    # Upload files
    files_to_upload = [
        ("ppo-web-minesweeper.zip" if os.path.exists("ppo-web-minesweeper.zip") else "ppo-web-minesweeper", "ppo-web-minesweeper.zip"),
        (video_filename, video_filename),
        ("README.md", "README.md")
    ]

    for local_file, path_in_repo in files_to_upload:
        if os.path.exists(local_file):
            print(f"Uploading {local_file} -> {path_in_repo}...")
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model"
            )

    print("Successfully pushed MineSweeper model and gameplay video to Hugging Face!")

if __name__ == "__main__":
    record_game_and_push()