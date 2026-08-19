# Testing tools

Helper scripts for the Kaggle GPU testing workflow. All three assume the Kaggle
Python package is installed (`pip install kaggle gdown`) and `%USERPROFILE%\.kaggle\kaggle.json`
already exists (Kaggle account API token).

- `fetch_log.py` — pulls just `cross_camera_log.txt` from the latest completed
  kernel run. Fast, use this first.
- `fetch_videos.py` — pulls the two annotated output `.mp4`s, chunked with
  retries (plain `kaggle kernels output` was unreliable on flaky connections).
- `extract_frames.py` — grabs 6 sample frames spread across each downloaded
  video, for a quick visual check without opening a full video player.

All three write into `../../kaggle_output_crosscam_run2/` (project root).

## Kaggle CLI path

On this machine the `kaggle` command isn't on PATH by default; the working
executable is at:
```
C:/Users/maham/AppData/Local/Programs/Python/Python310/Scripts/kaggle.exe
```

## Standard test-a-new-pair workflow

1. Check if both clips are already in the dataset:
   `kaggle.exe datasets files aimanshahidhuff/daycare-fyp-test-clips`
2. If not, copy the missing clip into `C:/kg_ds_upload/` (already mirrors the
   dataset - `kaggle datasets version` REPLACES the whole dataset with
   whatever's in this folder, so don't upload from a folder missing the
   existing clips) and run:
   `kaggle.exe datasets version -p C:/kg_ds_upload -m "<message>"`
3. Edit `CAM1_VIDEO_NAME` / `CAM2_VIDEO_NAME` at the top of `../script.py`.
4. `cd ..` then `kaggle.exe kernels push -p .`
5. Poll `kaggle.exe kernels status aimanshahidhuff/daycare-cross-camera-gpu-test`
   every ~20s until it says `COMPLETE`.
6. `python fetch_log.py`, check event counts (`NEW ID MINTED`,
   `SPATIAL RESCUE`, `CROSS-CAMERA MATCH`, `CROSS-CAMERA MERGE`) before
   deciding if a video download is even needed.
7. If needed: `python fetch_videos.py` then `python extract_frames.py`.

## Windows-specific gotcha

If `datasets version` fails with a path error mentioning `.kaggle/uploads`,
pre-create this folder once (the CLI doesn't create it itself):
```
mkdir "%TEMP%\.kaggle\uploads\C_"
```

## Golden rule

Always verify a claimed bug against the actual log lines or exact video
frames - never confirm or deny from memory alone. Every real bug found this
project (girl/girl mismatch, border-kid exclusion, gallery contamination) was
only caught by literally grepping the log for the exact X-ID/gid involved.
