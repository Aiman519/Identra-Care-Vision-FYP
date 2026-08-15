"""
Step B: Cross-Camera Persistent Tracking (two cameras at once).

Plays TWO video sources side by side (or two live cameras later - just change
SOURCE_CAM1 / SOURCE_CAM2, exactly like track.py's single-camera SOURCE), each
with its own independent single-camera tracker (unchanged from Step A). A
CrossCameraMatcher links matching people between the two cameras into a shared
"X-ID" once it has seen enough of them on both.

Each window shows: local ID (this camera's own ID) and X-ID (the shared
cross-camera ID once matched, or "?" while still unmatched/pending).
"""

from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from boxmot import DeepOCSORT, ReIDDetectMultiBackend
from persistent_tracker import PersistentIDManager
from cross_camera_matcher import CrossCameraMatcher

# ---- Change these to 0 / 1 for live webcams, or a video file path -----------
SOURCE_CAM1 = "camera1.mp4"
SOURCE_CAM2 = "camera2.mp4"
# -------------------------------------------------------------------------

DETECTOR_MODEL = "yolov8s.pt"
REID_WEIGHTS = Path("kaggle_output_crosscam/clip_market1501.pt")
MAX_DISPLAY_WIDTH = 720  # smaller since two windows share the screen


def resize_for_display(frame):
    h, w = frame.shape[:2]
    if w <= MAX_DISPLAY_WIDTH:
        return frame
    scale = MAX_DISPLAY_WIDTH / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


class CameraPipeline:
    def __init__(self, name, source, detector, reid_backend):
        self.name = name
        self.source = source
        self.detector = detector
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source '{source}' for {name}")

        self.base_tracker = DeepOCSORT(
            model_weights=REID_WEIGHTS,
            device="cpu",
            fp16=False,
            max_age=200,
        )
        self.persistent_manager = PersistentIDManager(
            reid_backend=reid_backend,
            sim_threshold=0.55,
            max_gallery_size=25,
            max_inactive_age=100000,
            verbose=True,
        )
        self.frame_idx = -1
        self.done = False

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            self.done = True
            return None
        self.frame_idx += 1

        det_result = self.detector.predict(frame, classes=[0], conf=0.15, verbose=False)[0]
        boxes = det_result.boxes
        if len(boxes) == 0:
            dets = torch.empty((0, 6))
        else:
            xyxy = boxes.xyxy.cpu()
            conf = boxes.conf.cpu().reshape(-1, 1)
            cls = boxes.cls.cpu().reshape(-1, 1)
            dets = torch.hstack([xyxy, conf, cls])

        raw_tracks = self.base_tracker.update(dets, frame)
        tracks = self.persistent_manager.update(raw_tracks, frame, self.frame_idx)
        return frame, tracks


def main():
    print("Initializing YOLOv8s Detector & OSNet Re-ID Model (shared across both cameras)...")
    detector = YOLO(DETECTOR_MODEL)
    reid_backend = ReIDDetectMultiBackend(
        weights=REID_WEIGHTS,
        device=torch.device("cpu"),
        fp16=False,
    )

    cam1 = CameraPipeline("camera1", SOURCE_CAM1, detector, reid_backend)
    cam2 = CameraPipeline("camera2", SOURCE_CAM2, detector, reid_backend)

    matcher = CrossCameraMatcher(
        camera_managers={cam1.name: cam1.persistent_manager, cam2.name: cam2.persistent_manager},
        sim_threshold=0.58,
        min_gallery_size=3,
    )

    print(f"Starting cross-camera tracking on '{SOURCE_CAM1}' + '{SOURCE_CAM2}'... Press 'q' to quit.")

    while not (cam1.done and cam2.done):
        for cam in (cam1, cam2):
            if cam.done:
                continue
            result = cam.step()
            if result is None:
                continue
            frame, tracks = result

            matcher.update(verbose=True, frame_idx=cam.frame_idx)

            for t in tracks:
                x1, y1, x2, y2, local_id = t[0], t[1], t[2], t[3], int(t[4])
                x_id = matcher.get_cross_id(cam.name, local_id)
                label = f"ID {x_id}" if x_id is not None else "ID ..."
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            cv2.imshow(f"{cam.name} (press q to quit)", resize_for_display(frame))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam1.cap.release()
    cam2.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
