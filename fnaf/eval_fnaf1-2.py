import os
import subprocess
import cv2
import numpy as np
from stable_baselines3 import PPO
from fnaf_env import WebFNAFEnv

RAW_VIDEO = "fnaf_trained_replay.mp4"
HF_VIDEO = "replay.mp4"
VIDEO_SIZE = (640, 360)
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# 1. Initialize environment and load trained Nights 1 & 2 model
env = WebFNAFEnv(url="http://localhost:8000/FNAF.html")
model = PPO.load("ppo-fnaf-nights1-2")

writer = None
original_capture = env._capture_raw_bgr

def capture_and_record():
    bgr_img = original_capture()
    if bgr_img is not None and writer is not None:
        resized = cv2.resize(bgr_img, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
        writer.write(resized)
    return bgr_img

env._capture_raw_bgr = capture_and_record

def run_night(obs, expected_night, deterministic=True):
    done = False
    ep_reward = 0.0
    step_count = 0
    reward = 0.0
    info = {}

    print(f"\n--- Playing & Recording Evaluation Night {expected_night} ---", flush=True)
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        step_count += 1
        done = terminated or truncated

        if step_count % 100 == 0:
            print(
                f"[Eval Night {expected_night}] Step {step_count} | Cumulative Reward: {ep_reward:.1f}",
                flush=True,
            )

    won = bool(info.get("won", False)) and (reward >= 90.0)
    print(
        f"\nNight {expected_night} Result: {'6 AM WIN!' if won else 'Jumpscare'} | "
        f"Steps: {step_count} | Total Reward: {ep_reward:.1f}",
        flush=True,
    )
    return won

attempt = 0
try:
    while True:
        attempt += 1
        print(f"\n=== Attempt #{attempt}: Starting fresh Night 1 ===", flush=True)

        # Start a clean recording for each attempt so failed Night 1 runs are discarded
        writer = cv2.VideoWriter(RAW_VIDEO, FOURCC, 4.0, VIDEO_SIZE)

        obs, info = env.reset()
        if info.get("night") != 1:
            raise RuntimeError(f"Expected Night 1 on initial reset, got Night {info.get('night')}")

        night1_won = run_night(obs, expected_night=1, deterministic=True)

        if not night1_won:
            print("Died on Night 1. Discarding recording and retrying from a fresh Night 1...", flush=True)
            writer.release()
            writer = None
            continue

        # Night 1 survived -> env.reset() waits 10s for Night 2 to go live without reloading the menu
        print("\nNight 1 cleared! Waiting for Night 2 to start...", flush=True)
        obs, info = env.reset()
        if info.get("night") != 2:
            raise RuntimeError(f"Expected Night 2 after transition, got Night {info.get('night')}")

        # Continue recording Night 2 onto the exact same unbroken video stream
        run_night(obs, expected_night=2, deterministic=True)
        break
finally:
    if writer is not None:
        writer.release()
    env.close()

    if os.path.exists(RAW_VIDEO):
        print(f"\nRaw video saved cleanly to: {RAW_VIDEO}", flush=True)
        print(f"Converting {RAW_VIDEO} to H.264 {HF_VIDEO}...", flush=True)
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
        print(f"Converted H.264 MP4 saved to: {HF_VIDEO}", flush=True)