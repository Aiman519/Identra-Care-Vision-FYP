"""
Kaggle GPU runner for the daycare-fyp person tracking pipeline.
Self-contained: installs deps, runs YOLOv8s + DeepOCSORT + PersistentIDManager
on camera2.mp4, writes an annotated output video (no live window) plus a
verbose decision log, both saved to /kaggle/working/ for download.
"""

import subprocess
import sys
import glob

subprocess.check_call([sys.executable, "-m", "pip", "install", "boxmot==13.0.17", "ultralytics"])

import cv2
import copy
import torch
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR


def _patched_unfreeze(self):
    """
    Upstream boxmot bug fix (KalmanFilterXYSR.unfreeze, boxmot 13.0.17): the original
    assumes at least 2 real observations exist in history_obs before a gap, and crashes
    with IndexError when a track had only 1 real hit before going quiet (indices[-2] on
    a length-1 array). This is what upstream GitHub issues #1369/#1480 describe, and it
    surfaces specifically when max_age is raised above ~60 (more time for a 1-hit track
    to linger). We keep max_age=200 as intended and just make unfreeze() tolerate the
    1-observation case by skipping interpolation instead of crashing - everything else
    is an exact copy of the original method.
    """
    if self.attr_saved is not None:
        new_history = copy.deepcopy(list(self.history_obs))
        self.__dict__ = self.attr_saved
        self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
        occur = [int(d is None) for d in new_history]
        indices = np.where(np.array(occur) == 0)[0]
        if len(indices) < 2:
            # Not enough real observations to interpolate a gap - just drop it safely.
            self.history_obs.pop()
            return
        index1, index2 = indices[-2], indices[-1]
        box1, box2 = new_history[index1], new_history[index2]
        x1, y1, s1, r1 = box1
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        x2, y2, s2, r2 = box2
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        time_gap = index2 - index1
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

        for i in range(index2 - index1):
            x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
            w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
            s, r = w * h, w / float(h)
            new_box = np.array([x, y, s, r]).reshape((4, 1))
            self.update(new_box)
            if not i == (index2 - index1 - 1):
                self.predict()
                self.history_obs.pop()
        self.history_obs.pop()


KalmanFilterXYSR.unfreeze = _patched_unfreeze


# ============================================================================
# PersistentIDManager (identical to the project's persistent_tracker.py,
# including the spatial-rescue appearance-sanity-check fix)
# ============================================================================

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


class PersistentIDManager:
    def __init__(
        self,
        reid_backend,
        sim_threshold: float = 0.73,
        duplicate_iou_threshold: float = 0.40,
        max_gallery_size: int = 25,
        max_inactive_age: int = 100000,
        verbose: bool = False,
    ):
        self.reid_backend = reid_backend
        self.sim_threshold = sim_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.max_gallery_size = max_gallery_size
        self.max_inactive_age = max_inactive_age
        self.verbose = verbose

        self.next_global_id = 1
        self.raw_to_global = {}
        self.global_gallery = {}
        self.last_seen_frame = {}
        self.last_seen_box = {}
        self.last_seen_border = {}

    def _extract_embeddings(self, frame, boxes):
        # NOTE: Kaggle-only adaptation. boxmot 13.0.17's ReID backend exposes a
        # public get_features(xyxys, img) that does crop+preprocess+forward+L2-norm
        # in one call (the private _preprocess()/__call__() pair used locally with
        # boxmot 10.0.12 no longer exists in newer boxmot versions).
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)

        boxes_np = np.asarray(boxes, dtype=np.float32)[:, :4]
        feats = self.reid_backend.get_features(boxes_np, frame)
        feats = np.asarray(feats, dtype=np.float32)

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        feats_norm = feats / norms
        return feats_norm

    def _match_inactive(self, embedding, active_gids):
        best_id = None
        best_sim = -1.0

        for gid, gallery in self.global_gallery.items():
            if gid in active_gids or len(gallery) == 0:
                continue

            centroid = np.mean(gallery, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            sim = float(np.dot(centroid, embedding))

            if sim > best_sim:
                best_sim = sim
                best_id = gid

        return best_id, best_sim

    def _find_spatial_rescue(self, box, emb, frame_idx, active_gids, frame_shape):
        bx1, by1, bx2, by2 = box[:4]
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        h_frame, w_frame = frame_shape[:2]

        is_near_border = (bx1 < 30) or (by1 < 30) or (bx2 > w_frame - 30) or (by2 > h_frame - 30)

        MIN_SANITY_SIM = 0.35

        candidates = []
        for gid, last_f in self.last_seen_frame.items():
            if gid in active_gids:
                continue

            frame_gap = frame_idx - last_f
            if not (0 < frame_gap <= 90):
                continue

            last_box = self.last_seen_box.get(gid)
            if last_box is None:
                continue

            was_border = self.last_seen_border.get(gid, False)
            lx1, ly1, lx2, ly2 = last_box
            lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            d = float(np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2))

            close_enough = (is_near_border and was_border and d < 220.0) or (d < 160.0)
            if not close_enough:
                continue

            gallery = self.global_gallery.get(gid, [])
            if len(gallery) == 0:
                candidates.append((d, gid, None))
            else:
                centroid = np.mean(gallery, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                sim = float(np.dot(centroid, emb))
                if sim >= MIN_SANITY_SIM:
                    candidates.append((d, gid, sim))
                elif self.verbose:
                    print(
                        f"[frame {frame_idx}] SPATIAL RESCUE REJECTED: gid {gid} was closest "
                        f"(d={d:.0f}px) but appearance sim={sim:.2f} < {MIN_SANITY_SIM} -> skipped"
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]

    def update(self, raw_tracks, frame, frame_idx):
        if len(raw_tracks) == 0:
            return []

        boxes = raw_tracks[:, :4]
        raw_ids = [int(t[4]) for t in raw_tracks]
        confs = [float(t[5]) if len(t) > 5 else 1.0 for t in raw_tracks]
        clss = [int(t[6]) if len(t) > 6 else 0 for t in raw_tracks]

        embeddings = self._extract_embeddings(frame, boxes)

        active_global_ids_this_frame = set()
        active_boxes_this_frame = []
        updated_tracks = []

        unmapped_indices = []
        mapped_indices = []
        for i, raw_id in enumerate(raw_ids):
            box = boxes[i]
            is_duplicate = False
            for active_box in active_boxes_this_frame:
                if compute_iou(box, active_box) >= self.duplicate_iou_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            if raw_id in self.raw_to_global:
                gid = self.raw_to_global[raw_id]
                if gid not in active_global_ids_this_frame:
                    active_global_ids_this_frame.add(gid)
                    active_boxes_this_frame.append(box)
                    mapped_indices.append(i)
                else:
                    unmapped_indices.append(i)
            else:
                unmapped_indices.append(i)

        if len(mapped_indices) >= 2:
            for idx_a in range(len(mapped_indices)):
                for idx_b in range(idx_a + 1, len(mapped_indices)):
                    i_a = mapped_indices[idx_a]
                    i_b = mapped_indices[idx_b]
                    raw_a, raw_b = raw_ids[i_a], raw_ids[i_b]
                    gid_a, gid_b = self.raw_to_global[raw_a], self.raw_to_global[raw_b]

                    gallery_a = self.global_gallery.get(gid_a, [])
                    gallery_b = self.global_gallery.get(gid_b, [])

                    if len(gallery_a) > 0 and len(gallery_b) > 0:
                        emb_a, emb_b = embeddings[i_a], embeddings[i_b]

                        sim_a_own = float(np.max(np.dot(gallery_a, emb_a)))
                        sim_a_other = float(np.max(np.dot(gallery_b, emb_a)))
                        sim_b_own = float(np.max(np.dot(gallery_b, emb_b)))
                        sim_b_other = float(np.max(np.dot(gallery_a, emb_b)))

                        if (sim_a_other > sim_a_own + 0.08) and (sim_b_other > sim_b_own + 0.08):
                            if self.verbose:
                                print(
                                    f"[frame {frame_idx}] ANTI-SWAP fired: "
                                    f"gid {gid_a}<->{gid_b} (raw {raw_a}<->{raw_b}) | "
                                    f"a: own={sim_a_own:.2f} other={sim_a_other:.2f} | "
                                    f"b: own={sim_b_own:.2f} other={sim_b_other:.2f}"
                                )
                            self.raw_to_global[raw_a] = gid_b
                            self.raw_to_global[raw_b] = gid_a

        for i in unmapped_indices:
            raw_id = raw_ids[i]
            box = boxes[i]
            emb = embeddings[i]

            spatial_gid = self._find_spatial_rescue(box, emb, frame_idx, active_global_ids_this_frame, frame.shape)

            if spatial_gid is not None:
                gid = spatial_gid
                self.raw_to_global[raw_id] = gid
                if self.verbose:
                    print(f"[frame {frame_idx}] SPATIAL RESCUE: raw {raw_id} -> gid {gid}")
            else:
                best_gid, best_sim = self._match_inactive(emb, active_global_ids_this_frame)

                if best_gid is not None and best_sim >= 0.68:
                    gid = best_gid
                    self.raw_to_global[raw_id] = gid
                    if self.verbose:
                        print(f"[frame {frame_idx}] CENTROID MATCH: raw {raw_id} -> gid {gid} (sim={best_sim:.2f})")
                else:
                    gid = self.next_global_id
                    self.next_global_id += 1
                    self.raw_to_global[raw_id] = gid
                    self.global_gallery[gid] = []
                    if self.verbose:
                        best_str = f"{best_sim:.2f}" if best_gid is not None else "n/a"
                        print(
                            f"[frame {frame_idx}] NEW ID MINTED: raw {raw_id} -> gid {gid} "
                            f"(best inactive candidate={best_gid}, sim={best_str})"
                        )

            active_global_ids_this_frame.add(gid)
            active_boxes_this_frame.append(box)

        for i, raw_id in enumerate(raw_ids):
            if raw_id not in self.raw_to_global:
                continue

            box = boxes[i]
            emb = embeddings[i]
            gid = self.raw_to_global[raw_id]

            if not any(np.array_equal(box, b) for b in active_boxes_this_frame):
                continue

            if gid not in self.global_gallery:
                self.global_gallery[gid] = []

            max_other_iou = 0.0
            for other_box in active_boxes_this_frame:
                if not np.array_equal(box, other_box):
                    iou_val = compute_iou(box, other_box)
                    if iou_val > max_other_iou:
                        max_other_iou = iou_val

            h_frame, w_frame = frame.shape[:2]
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            is_near_border = (box[0] < 15) or (box[1] < 15) or (box[2] > w_frame - 15) or (box[3] > h_frame - 15)
            is_high_quality = (confs[i] >= 0.50) and (box_area >= 1200.0) and (not is_near_border)

            if max_other_iou < 0.25 and is_high_quality:
                gallery = self.global_gallery[gid]
                if len(gallery) < self.max_gallery_size:
                    gallery.append(emb)
                else:
                    sims = np.dot(gallery, emb)
                    if np.max(sims) < 0.95:
                        gallery.pop(0)
                        gallery.append(emb)

            self.last_seen_frame[gid] = frame_idx
            self.last_seen_box[gid] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            self.last_seen_border[gid] = bool(is_near_border)

            updated_tracks.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), gid, confs[i], clss[i]))

        stale_gids = [
            gid for gid, last_f in self.last_seen_frame.items()
            if gid not in active_global_ids_this_frame and (frame_idx - last_f) > self.max_inactive_age
        ]
        for gid in stale_gids:
            self.global_gallery.pop(gid, None)
            self.last_seen_frame.pop(gid, None)
            self.last_seen_box.pop(gid, None)

        return updated_tracks


# ============================================================================
# Main: headless run -> annotated output video + log file
# ============================================================================

TARGET_VIDEO_NAME = "camera5.mp4"


def main():
    import os
    print("Contents of /kaggle/input:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")
    video_candidates = glob.glob(f"/kaggle/input/**/{TARGET_VIDEO_NAME}", recursive=True)
    if not video_candidates:
        raise FileNotFoundError(f"{TARGET_VIDEO_NAME} not found under /kaggle/input/ - check the dataset was attached.")
    source = video_candidates[0]
    print(f"Using video source: {source}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"torch version: {torch.__version__}")
        print(f"torch CUDA arch list: {torch.cuda.get_arch_list()}")
        try:
            probe = torch.randn(4, 4, device="cuda:0")
            probe = probe @ probe
            torch.cuda.synchronize()
            print("Basic GPU matmul probe: OK")
        except Exception as e:
            print(f"Basic GPU matmul probe FAILED: {e}")
            print("Falling back to CPU for this run due to GPU/torch incompatibility.")
            device = "cpu"

    detector = YOLO("yolov8s.pt")
    reid_weights = Path("osnet_x1_0_msmt17.pt")

    base_tracker = DeepOcSort(
        reid_weights=reid_weights,
        device=torch.device(device),
        half=False,
        max_age=200,
    )
    # Reuse the tracker's own internal ReID backend instead of loading the
    # checkpoint a second time - same object, same weights.
    reid_backend = base_tracker.model

    persistent_manager = PersistentIDManager(
        reid_backend=reid_backend,
        sim_threshold=0.55,
        max_gallery_size=25,
        max_inactive_age=100000,
        verbose=True,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = Path(TARGET_VIDEO_NAME).stem
    out_path = f"/kaggle/working/output_tracked_{stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    log_path = f"/kaggle/working/tracking_log_{stem}.txt"
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    frame_idx = -1
    print(f"Starting tracking on '{source}' ({w}x{h} @ {fps:.1f}fps)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        det_result = detector.predict(frame, classes=[0], conf=0.15, verbose=False, device="cpu")[0]
        boxes = det_result.boxes
        if len(boxes) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, conf, cls]).astype(np.float32)

        raw_tracks = base_tracker.update(dets, frame)
        tracks = persistent_manager.update(raw_tracks, frame, frame_idx)

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

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"...processed frame {frame_idx}")

    cap.release()
    writer.release()

    print(f"Done. Wrote {frame_idx + 1} frames to {out_path}")
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
