import os
import glob
from stable_baselines3 import PPO
from minesweeper_env import FastWebMinesweeperEnv

# 1. Create environment
env = FastWebMinesweeperEnv(board_size=(9, 9))

# 2. Replace the browser context with one that records video natively
os.makedirs("videos", exist_ok=True)
env.page.close()
env.context.close()

env.context = env.browser.new_context(
    viewport={"width": 1280, "height": 720},
    record_video_dir="videos/",
    record_video_size={"width": 1280, "height": 720}
)
env.page = env.context.new_page()
env.page.goto("http://localhost:8000/MinesweeperMania.html")

# Re-apply the rotation lock CSS from __init__
env.page.add_style_tag(
    content="#wrongRotation { display: none !important; } #game { display: block !important; }"
)
env.page.wait_for_timeout(300)

# 3. Initialize model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    ent_coef=0.05,
    verbose=1
)

# 4. Run for 300 steps (~15 seconds total including the 5s wait in reset())
print("Starting test run...")
model.learn(total_timesteps=300)

# 5. Closing the env finalizes and saves the .webm video file
env.close()

# Rename the randomly generated Playwright video filename to agent_test.webm
saved_videos = sorted(glob.glob("videos/*.webm"), key=os.path.getmtime)
if saved_videos:
    os.replace(saved_videos[-1], "agent_test.webm")
    print("Video saved successfully to: agent_test.webm")
else:
    print("No video found in videos/ directory.")
