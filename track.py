"""
Step A: Single-Camera Persistent Tracking with Re-ID Memory Bank.

Plays a video (camera2.mp4 or live webcam SOURCE=0), detects people using YOLOv8s,
and tracks each child using DeepOCSORT + PersistentIDManager (OSNet Re-ID Memory Bank).

The PersistentIDManager prevents ID switches by maintaining a long-term feature gallery
and matching reappearing children against inactive tracks.
"""

from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from boxmot import DeepOCSORT, ReIDDetectMultiBackend
from persistent_tracker import PersistentIDManager

# ---- Change SOURCE to 0 for live webcam, or "camera1.mp4" / "camera2.mp4" ----
SOURCE = "camera5.mp4"
# -----------------------------------------------------------------------------

DETECTOR_MODEL = "yolov8s.pt"
REID_WEIGHTS = Path("osnet_x1_0_msmt17.pt")
MAX_DISPLAY_WIDTH = 960  # max display width for popup window


def resize_for_display(frame):
    h, w = frame.shape[:2]
    if w <= MAX_DISPLAY_WIDTH:
        return frame
    scale = MAX_DISPLAY_WIDTH / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def main():
    print("Initializing YOLOv8s Detector & OSNet Re-ID Model...")
    detector = YOLO(DETECTOR_MODEL)
    base_tracker = DeepOCSORT(
        model_weights=REID_WEIGHTS,
        device="cpu",
        fp16=False,
        max_age=200,
    )
    reid_backend = ReIDDetectMultiBackend(
        weights=REID_WEIGHTS,
        device=torch.device("cpu"),
        fp16=False,
    )

    persistent_manager = PersistentIDManager(
        reid_backend=reid_backend,
        sim_threshold=0.55,
        max_gallery_size=25,
        max_inactive_age=100000,
        verbose=True,
    )

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Unable to open video source '{SOURCE}'")
        return

    frame_idx = -1
    print(f"Starting tracking on '{SOURCE}'... Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        det_result = detector.predict(frame, classes=[0], conf=0.15, verbose=False)[0]
        boxes = det_result.boxes
        if len(boxes) == 0:
            dets = torch.empty((0, 6))
        else:
            xyxy = boxes.xyxy.cpu()
            conf = boxes.conf.cpu().reshape(-1, 1)
            cls = boxes.cls.cpu().reshape(-1, 1)
            dets = torch.hstack([xyxy, conf, cls])

        # Base tracker update
        raw_tracks = base_tracker.update(dets, frame)

        # Persistent Re-ID Memory Bank update
        tracks = persistent_manager.update(raw_tracks, frame, frame_idx)

        # Draw bounding boxes and Persistent IDs
        for t in tracks:
            x1, y1, x2, y2, persistent_id = t[0], t[1], t[2], t[3], int(t[4])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {persistent_id}",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        display_frame = resize_for_display(frame)
        cv2.imshow("Persistent Single-Camera Tracking (press q to quit)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
