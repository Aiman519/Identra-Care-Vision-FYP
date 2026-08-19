"""
Kaggle GPU runner for Step B: Cross-Camera Re-ID.

Self-contained: installs deps, runs YOLOv8s + DeepOCSORT + PersistentIDManager
independently on camera1.mp4 and camera2.mp4 (same children, two different
cameras), then a CrossCameraMatcher links matching people between the two
into a shared cross-camera ID ("X-ID"). Writes two annotated output videos
(labelled with each camera's local ID AND the shared X-ID) plus a combined
verbose decision log, all saved to /kaggle/working/ for download.
"""

import subprocess
import sys
import glob

# Kaggle's default container currently ships torch built for CUDA 12.8, whose
# precompiled kernels only cover compute capability sm_70+ (Volta and newer). The
# P100 GPU on the free tier is Pascal (sm_60), so CUDA calls fail with "no kernel
# image is available for execution on the device" and everything silently falls
# back to CPU. Install an older CUDA 12.1 build of torch/torchvision that still
# ships sm_60 kernels, pinned in the SAME install command as boxmot/ultralytics so
# pip's resolver can't quietly pull the newer (incompatible) torch back in.
print("Installing torch==2.4.1 / torchvision==0.19.1 (cu121) - older build that still supports the P100's Pascal (sm_60) architecture")

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "boxmot==13.0.17", "ultralytics",
    "torch==2.4.1", "torchvision==0.19.1",
    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
])

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
    Upstream boxmot bug fix (KalmanFilterXYSR.unfreeze, boxmot 13.0.17) - see
    single-camera script.py for the full explanation. Keeps max_age=200 safe.
    """
    if self.attr_saved is not None:
        new_history = copy.deepcopy(list(self.history_obs))
        self.__dict__ = self.attr_saved
        self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
        occur = [int(d is None) for d in new_history]
        indices = np.where(np.array(occur) == 0)[0]
        if len(indices) < 2:
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
# PersistentIDManager (identical to the project's persistent_tracker.py)
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
        cam_name: str = "",
    ):
        self.reid_backend = reid_backend
        self.sim_threshold = sim_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.max_gallery_size = max_gallery_size
        self.max_inactive_age = max_inactive_age
        self.verbose = verbose
        # Every camera runs its OWN independent gid counter starting at 1, so gid
        # numbers collide between cameras in a shared log. Prefixing every printed
        # line with cam_name removes that ambiguity.
        self.cam_name = f"{cam_name} " if cam_name else ""

        self.next_global_id = 1
        self.raw_to_global = {}
        self.global_gallery = {}
        self.last_seen_frame = {}
        self.last_seen_box = {}
        self.last_seen_border = {}

    def _extract_embeddings(self, frame, boxes):
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
            if not (0 < frame_gap <= 150):  # was 90/3s, widened to 5s
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
                        f"[frame {frame_idx}] {self.cam_name}SPATIAL RESCUE REJECTED: gid {gid} was closest "
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
                                    f"[frame {frame_idx}] {self.cam_name}ANTI-SWAP fired: "
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
                    print(f"[frame {frame_idx}] {self.cam_name}SPATIAL RESCUE: raw {raw_id} -> gid {gid}")
            else:
                best_gid, best_sim = self._match_inactive(emb, active_global_ids_this_frame)

                if best_gid is not None and best_sim >= 0.55:  # was 0.68, lowered
                    gid = best_gid
                    self.raw_to_global[raw_id] = gid
                    if self.verbose:
                        print(f"[frame {frame_idx}] {self.cam_name}CENTROID MATCH: raw {raw_id} -> gid {gid} (sim={best_sim:.2f})")
                else:
                    gid = self.next_global_id
                    self.next_global_id += 1
                    self.raw_to_global[raw_id] = gid
                    self.global_gallery[gid] = []
                    if self.verbose:
                        best_str = f"{best_sim:.2f}" if best_gid is not None else "n/a"
                        print(
                            f"[frame {frame_idx}] {self.cam_name}NEW ID MINTED: raw {raw_id} -> gid {gid} "
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
            # near_border is still tracked (used by spatial rescue for border re-entries below),
            # but no longer blocks the gallery: debug logging showed kids simply standing at the
            # edge of a wide group shot were being permanently excluded despite high confidence
            # and full-size boxes - they were never actually cut off, just near the frame edge.
            is_high_quality = (confs[i] >= 0.50) and (box_area >= 1200.0)

            if max_other_iou < 0.25 and is_high_quality:  # reverted from 0.45 - was letting overlapping crops contaminate the gallery, causing high-confidence wrong matches
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
# CrossCameraMatcher (identical to the project's cross_camera_matcher.py)
# ============================================================================

class CrossCameraMatcher:
    """
    A match is never permanently locked in until "confirmed" (linked local IDs
    from 2+ different cameras). Until confirmed, a local ID keeps being
    re-evaluated on every update() call as its gallery grows, so an early thin
    gallery that just misses the threshold can still be merged into the right
    match later instead of being stuck as a wrong new ID forever.
    """

    def __init__(self, camera_managers: dict, sim_threshold: float = 0.65, min_gallery_size: int = 3):
        self.camera_managers = camera_managers
        self.sim_threshold = sim_threshold
        self.min_gallery_size = min_gallery_size

        self.next_cross_id = 1
        self.local_to_cross = {}   # (cam_name, local_gid) -> cross_id (can change until confirmed)
        self.cross_members = {}    # cross_id -> {cam_name: local_gid}, at most one local_gid per camera

    def _centroid_for(self, cam_name, local_gid):
        gallery = self.camera_managers[cam_name].global_gallery.get(local_gid, [])
        if len(gallery) == 0:
            return None
        c = np.mean(gallery, axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 1e-6 else c

    def _is_confirmed(self, cross_id):
        return len(self.cross_members.get(cross_id, {})) >= 2

    def update(self, verbose: bool = False, frame_idx=None):
        eligible = []
        for cam_name, manager in self.camera_managers.items():
            for local_gid, gallery in manager.global_gallery.items():
                if len(gallery) < self.min_gallery_size:
                    continue
                key = (cam_name, local_gid)
                current_cross_id = self.local_to_cross.get(key)
                if current_cross_id is not None and self._is_confirmed(current_cross_id):
                    continue
                eligible.append((cam_name, local_gid, current_cross_id))

        for cam_name, local_gid, current_cross_id in eligible:
            centroid = self._centroid_for(cam_name, local_gid)
            if centroid is None:
                continue

            best_cross_id, best_sim = None, -1.0
            for cross_id, members in self.cross_members.items():
                if cross_id == current_cross_id:
                    continue
                if cam_name in members:
                    continue
                for other_cam, other_gid in members.items():
                    other_centroid = self._centroid_for(other_cam, other_gid)
                    if other_centroid is None:
                        continue
                    sim = float(np.dot(centroid, other_centroid))
                    if sim > best_sim:
                        best_sim = sim
                        best_cross_id = cross_id

            key = (cam_name, local_gid)
            if best_cross_id is not None and best_sim >= self.sim_threshold:
                if current_cross_id is not None:
                    old_members = self.cross_members.get(current_cross_id, {})
                    old_members.pop(cam_name, None)
                    if not old_members:
                        self.cross_members.pop(current_cross_id, None)
                    if verbose:
                        print(
                            f"[frame {frame_idx}] CROSS-CAMERA MERGE: {cam_name} local-ID {local_gid} "
                            f"moved from X-ID {current_cross_id} -> X-ID {best_cross_id} (sim={best_sim:.2f})"
                        )
                elif verbose:
                    print(
                        f"[frame {frame_idx}] CROSS-CAMERA MATCH: {cam_name} local-ID {local_gid} "
                        f"-> X-ID {best_cross_id} (sim={best_sim:.2f})"
                    )
                self.local_to_cross[key] = best_cross_id
                self.cross_members.setdefault(best_cross_id, {})[cam_name] = local_gid

            elif current_cross_id is None:
                cross_id = self.next_cross_id
                self.next_cross_id += 1
                self.local_to_cross[key] = cross_id
                self.cross_members[cross_id] = {cam_name: local_gid}
                if verbose:
                    best_str = f"{best_sim:.2f}" if best_cross_id is not None else "n/a"
                    print(
                        f"[frame {frame_idx}] NEW X-ID: {cam_name} local-ID {local_gid} "
                        f"-> X-ID {cross_id} (best existing candidate sim={best_str})"
                    )

    def get_cross_id(self, cam_name: str, local_gid: int):
        # Only returns a Global number once CONFIRMED (2+ cameras) - never shown
        # until sure, so it can never change once shown. Caller falls back to the
        # local per-camera ID (separate, always available, already stable).
        cross_id = self.local_to_cross.get((cam_name, local_gid))
        if cross_id is None or not self._is_confirmed(cross_id):
            return None
        return cross_id


# ============================================================================
# Main: headless dual-camera run -> two annotated output videos + shared log
# ============================================================================

CAM1_VIDEO_NAME = "camera13.mp4"
CAM2_VIDEO_NAME = "camera14.mp4"


class CameraPipeline:
    def __init__(self, name, video_path, detector, base_tracker, persistent_manager, out_path):
        self.name = name
        self.detector = detector
        self.base_tracker = base_tracker
        self.persistent_manager = persistent_manager
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.w, self.h))
        self.frame_idx = -1
        self.done = False
        print(f"[{name}] source={video_path} ({self.w}x{self.h} @ {self.fps:.1f}fps) -> {out_path}")

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            self.done = True
            return None
        self.frame_idx += 1

        det_result = self.detector.predict(frame, classes=[0], conf=0.15, verbose=False, device="cpu")[0]
        boxes = det_result.boxes
        if len(boxes) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, conf, cls]).astype(np.float32)

        raw_tracks = self.base_tracker.update(dets, frame)
        tracks = self.persistent_manager.update(raw_tracks, frame, self.frame_idx)
        return frame, tracks

    def finish(self):
        self.cap.release()
        self.writer.release()


def main():
    import os
    print("Contents of /kaggle/input:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")

    cam1_candidates = glob.glob(f"/kaggle/input/**/{CAM1_VIDEO_NAME}", recursive=True)
    cam2_candidates = glob.glob(f"/kaggle/input/**/{CAM2_VIDEO_NAME}", recursive=True)
    if not cam1_candidates or not cam2_candidates:
        raise FileNotFoundError(f"{CAM1_VIDEO_NAME} and/or {CAM2_VIDEO_NAME} not found under /kaggle/input/")
    cam1_path, cam2_path = cam1_candidates[0], cam2_candidates[0]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        try:
            probe = torch.randn(4, 4, device="cuda:0")
            probe = probe @ probe
            torch.cuda.synchronize()
            print("Basic GPU matmul probe: OK")
        except Exception as e:
            print(f"Basic GPU matmul probe FAILED: {e}")
            print("Falling back to CPU for this run due to GPU/torch incompatibility.")
            device = "cpu"

    print("Loading YOLOv8s detector (shared) and two independent DeepOcSort+PersistentIDManager pairs...")
    detector = YOLO("yolov8s.pt")
    reid_weights = Path("clip_market1501.pt")

    # iou_threshold lowered from boxmot's default (0.3) to tolerate bigger frame-to-frame
    # jumps during fast motion (e.g. dancing) before giving up and minting a new ID -
    # trade-off: more tolerant of fast movement, but also more willing to latch onto a
    # nearby WRONG person in a tight crowd, so this is a balance, not a pure win.
    base_tracker_1 = DeepOcSort(reid_weights=reid_weights, device=torch.device(device), half=False, max_age=200, iou_threshold=0.2)
    base_tracker_2 = DeepOcSort(reid_weights=reid_weights, device=torch.device(device), half=False, max_age=200, iou_threshold=0.2)

    manager_1 = PersistentIDManager(
        reid_backend=base_tracker_1.model, sim_threshold=0.55, max_gallery_size=25,
        max_inactive_age=100000, verbose=True, cam_name="camera1",
    )
    manager_2 = PersistentIDManager(
        reid_backend=base_tracker_2.model, sim_threshold=0.55, max_gallery_size=25,
        max_inactive_age=100000, verbose=True, cam_name="camera2",
    )

    matcher = CrossCameraMatcher(
        camera_managers={"camera1": manager_1, "camera2": manager_2},
        sim_threshold=0.65,
        min_gallery_size=3,
    )

    cam1 = CameraPipeline("camera1", cam1_path, detector, base_tracker_1, manager_1, "/kaggle/working/output_tracked_camera1_crosscam.mp4")
    cam2 = CameraPipeline("camera2", cam2_path, detector, base_tracker_2, manager_2, "/kaggle/working/output_tracked_camera2_crosscam.mp4")

    log_path = "/kaggle/working/cross_camera_log.txt"
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

    print("Starting cross-camera tracking...")
    tick = 0
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
                cv2.putText(frame, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cam.writer.write(frame)

        tick += 1
        if tick % 50 == 0:
            print(f"...processed tick {tick} (cam1 frame {cam1.frame_idx}, cam2 frame {cam2.frame_idx})")

    cam1.finish()
    cam2.finish()

    print(f"Done. cam1: {cam1.frame_idx + 1} frames, cam2: {cam2.frame_idx + 1} frames.")
    print(f"Total distinct cross-camera identities (X-IDs) found: {matcher.next_cross_id - 1}")
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
