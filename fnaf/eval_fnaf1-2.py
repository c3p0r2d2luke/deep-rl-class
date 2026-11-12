import cv2
import numpy as np
from stable_baselines3 import PPO
from fnaf_env import WebFNAFEnv

# 1. Initialize multi-night environment
env = WebFNAFEnv(url="http://localhost:8000/FNAF.html")

# 2. Load your trained Nights 1 & 2 CNN model
model = PPO.load("ppo-fnaf-nights1-2")

# 3. Set up direct OpenCV MP4 writer (640x360 @ 4 FPS)
output_path = "fnaf_nights1_2_replay.mp4"
video_size = (640, 360)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, 4.0, video_size)

# Hook into _capture_raw_bgr so every frame across Night 1 & Night 2 is recorded
original_capture = env._capture_raw_bgr

def capture_and_record():
    bgr_img = original_capture()
    if bgr_img is not None:
        resized = cv2.resize(bgr_img, video_size, interpolation=cv2.INTER_AREA)
        writer.write(resized)
    return bgr_img

env._capture_raw_bgr = capture_and_record

# 4. Play Night 1, and if won, automatically continue into Night 2
results = {}

for night in range(1, 3):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    step_count = 0

    print(f"\n--- Playing & Recording Evaluation Night {night} ---", flush=True)
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, step_info = env.step(int(action))
        ep_reward += reward
        step_count += 1
        done = terminated or truncated

        if step_count % 100 == 0:
            print(f"[Night {night} Eval] Step {step_count} | Cumulative Reward: {ep_reward:.1f}", flush=True)

    won = reward >= 90.0
    results[night] = (won, step_count, ep_reward)
    print(
        f"Night {night} Result: {'6 AM WIN!' if won else 'Jumpscare'} | "
        f"Steps: {step_count} | Total Reward: {ep_reward:.1f}",
        flush=True,
    )

    # Stop evaluation if Night 1 was lost (since Night 2 won't unlock automatically)
    if not won:
        print("Stopping evaluation because Night 1 was not cleared.", flush=True)
        break

# 5. Finalize video and close browser cleanly
writer.release()
env.close()

print("\n" + "=" * 55)
print("FINAL NIGHTS 1 & 2 EVALUATION SUMMARY")
print("=" * 55)
for night, (won, steps, rew) in results.items():
    print(f"Night {night}: {'6 AM WIN!' if won else 'Jumpscare'} | Steps: {steps} | Reward: {rew:.1f}")
print(f"Full replay video saved to: {output_path}")
print("=" * 55)
