"""
Downloads the two annotated output videos from the most recent completed run of
the Kaggle kernel. Chunked + retried (Kaggle's direct download links cut off
mid-transfer often) - plain `kaggle kernels output` was found to silently produce
truncated/corrupt files under flaky network conditions, this is the fix for that.

Requires: pip install kaggle gdown
Requires: %USERPROFILE%\.kaggle\kaggle.json (Kaggle API token) to already exist.

Usage: python fetch_videos.py
"""
import os
import time
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

targets = [item for item in response.files if item.file_name.endswith(".mp4")]

for item in targets:
    outfile = os.path.join(TARGET_DIR, item.file_name)
    for attempt in range(1, 6):
        try:
            with requests.get(item.url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                written = 0
                with open(outfile, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                print(f"{item.file_name}: downloaded {written}/{total} bytes (attempt {attempt})")
                if total == 0 or written == total:
                    print(f"OK: {outfile}")
                    break
                else:
                    print(f"Incomplete, retrying {item.file_name} (attempt {attempt})")
                    time.sleep(5)
        except Exception as e:
            print(f"Attempt {attempt} failed for {item.file_name}: {e}")
            time.sleep(5)
    else:
        print(f"FAILED after retries: {item.file_name}")

print("Done.")
