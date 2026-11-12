import os
import glob
from stable_baselines3 import PPO
from minesweeper_env import FastWebMinesweeperEnv

# 1. Load environment and attach native Playwright video recording
env = FastWebMinesweeperEnv(board_size=(9, 9))
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
env.page.add_style_tag(
    content="#wrongRotation { display: none !important; } #game { display: block !important; }"
)
env.page.wait_for_timeout(3500)
env.game_started = False

# 2. Load your trained PPO model
model = PPO.load("ppo-web-minesweeper")

# 3. Play 20 test games
num_games = 20
wins = 0
total_rewards = []

for game in range(1, num_games + 1):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0

    while not done and steps < 100:
        # Use deterministic=False so the agent never gets stuck clicking one open tile
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        ep_reward += reward
        steps += 1
        done = terminated or truncated

        # Brief pause on the first 3 games so the video is easy to watch
        if game <= 3:
            env.page.wait_for_timeout(150)

    total_rewards.append(ep_reward)
    if ep_reward >= 80:
        wins += 1
    print(f"Game {game:02d} | Steps: {steps:2d} | Reward: {ep_reward:6.1f} | {'WIN!' if ep_reward >= 80 else 'Loss'}")

env.close()

# Save the evaluation video
saved_videos = sorted(glob.glob("videos/*.webm"), key=os.path.getmtime)
if saved_videos:
    os.replace(saved_videos[-1], "trained_agent_eval.webm")

print(f"\nResults over {num_games} games:")
print(f"Win Rate: {wins}/{num_games} ({(wins / num_games) * 100:.1f}%)")
print(f"Average Reward: {sum(total_rewards) / num_games:.1f}")
print("Video saved to: trained_agent_eval.webm")
