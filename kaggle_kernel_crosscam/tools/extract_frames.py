"""
Pulls a handful of frames spread across each output video for a quick visual
check, without needing to open the full video in a player.

Usage: python extract_frames.py
"""
import cv2
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "kaggle_output_crosscam_run2")
VIDEOS = {
    "camera1": os.path.join(BASE, "output_tracked_camera1_crosscam.mp4"),
    "camera2": os.path.join(BASE, "output_tracked_camera2_crosscam.mp4"),
}

for name, path in VIDEOS.items():
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"{name}: fps={fps:.2f}, frames={frame_count}, duration={duration:.1f}s")
    timestamps_sec = [round(duration * f, 1) for f in [0.05, 0.25, 0.45, 0.65, 0.85, 0.97]]
    for t in timestamps_sec:
        frame_idx = min(int(t * fps), frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  {t}s -> failed to read frame {frame_idx}")
            continue
        outfile = os.path.join(BASE, f"{name}_t{t}s.jpg")
        cv2.imwrite(outfile, frame)
        print(f"  {t}s -> frame {frame_idx} saved to {outfile}")
    cap.release()
