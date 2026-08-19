"""
Pulls ONLY the cross_camera_log.txt (and kernel stdout log) from the most recent
completed run of the Kaggle kernel - fast, skips the ~500MB weights file and the
two output videos. Use this first to check event counts before deciding whether
a full video download is even needed.

Requires: pip install kaggle gdown (gdown only needed by fetch_videos.py)
Requires: %USERPROFILE%\.kaggle\kaggle.json (Kaggle API token) to already exist.

Usage: python fetch_log.py
"""
import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

OWNER_SLUG = "aimanshahidhuff"
KERNEL_SLUG = "daycare-cross-camera-gpu-test"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "kaggle_output_crosscam_run2")

api = KaggleApi()
api.authenticate()

os.makedirs(TARGET_DIR, exist_ok=True)

with api.build_kaggle_client() as kaggle:
    request = ApiListKernelSessionOutputRequest()
    request.user_name = OWNER_SLUG
    request.kernel_slug = KERNEL_SLUG
    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

print("Files in output:")
for item in response.files:
    print(" -", item.file_name)

for item in response.files:
    if "log" in item.file_name.lower():
        outfile = os.path.join(TARGET_DIR, item.file_name)
        r = requests.get(item.url, stream=True)
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded log -> {outfile}")

if response.log:
    outfile = os.path.join(TARGET_DIR, KERNEL_SLUG + ".stdout.log")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(response.log)
    print(f"Downloaded kernel stdout log -> {outfile}")
