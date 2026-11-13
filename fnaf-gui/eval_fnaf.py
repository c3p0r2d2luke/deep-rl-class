import cv2
import numpy as np
from stable_baselines3 import PPO
from fnaf_env import WebFNAFEnv

# 1. Initialize standard environment (no Playwright video encoder overhead)
env = WebFNAFEnv(url="http://localhost:8000/FNAF.html")

# 2. Load your trained Night 1 CNN model
model = PPO.load("ppo-fnaf-night1")

# 3. Set up direct OpenCV MP4 writer (640x360 @ 4 FPS keeps file size small & smooth)
output_path = "fnaf_trained_replay.mp4"
video_size = (640, 360)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, 4.0, video_size)

# Hook into _capture_raw_bgr so every frame captured during Night 1 is written to the MP4
original_capture = env._capture_raw_bgr

def capture_and_record():
    bgr_img = original_capture()
    if bgr_img is not None:
        resized = cv2.resize(bgr_img, video_size, interpolation=cv2.INTER_AREA)
        writer.write(resized)
    return bgr_img

env._capture_raw_bgr = capture_and_record

# 4. Play 1 full Night 1 evaluation (~8.5 minutes)
obs, _ = env.reset()
done = False
ep_reward = 0.0
step_count = 0

print("\n--- Playing & Recording Evaluation Night 1 ---", flush=True)
while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    ep_reward += reward
    step_count += 1
    done = terminated or truncated

    if step_count % 100 == 0:
        print(f"[Eval] Step {step_count} | Cumulative Reward: {ep_reward:.1f}", flush=True)

# 5. Finalize video and close browser cleanly
writer.release()
env.close()

won = reward >= 90.0
print(f"\nEvaluation Result: {'6 AM WIN!' if won else 'Jumpscare'} | Steps: {step_count} | Total Reward: {ep_reward:.1f}", flush=True)
print(f"Video saved cleanly to: {output_path}", flush=True)
