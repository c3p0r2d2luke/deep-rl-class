import os
import subprocess
from huggingface_hub import HfApi

REPO_ID = "Frenchwizz/MineSweeper"
INPUT_VIDEO = "agent_test.webm"
OUTPUT_VIDEO = "replay.mp4"

# 1. Convert .webm to H.264 .mp4 (required for Hugging Face's Video Preview widget)
if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(f"Could not find {INPUT_VIDEO} in the current directory.")

print(f"Converting {INPUT_VIDEO} to {OUTPUT_VIDEO}...")
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i", INPUT_VIDEO,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        OUTPUT_VIDEO,
    ],
    check=True,
)

# 2. Upload replay.mp4 to the root of Frenchwizz/MineSweeper
api = HfApi()
print(f"Uploading {OUTPUT_VIDEO} to {REPO_ID}...")
api.upload_file(
    path_or_fileobj=OUTPUT_VIDEO,
    path_in_repo="replay.mp4",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add replay.mp4 for model video preview",
)

print(f"Done! Check your video preview at: https://huggingface.co{REPO_ID}")
