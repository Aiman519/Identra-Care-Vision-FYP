"""
Persistent Single-Camera MOT Tracker with Long-Term Inactive Re-ID Memory Bank.

This module wraps standard online MOT trackers (DeepOCSORT / BoTSORT) and adds:
1. Feature Gallery: Stores up to 25 diverse appearance embeddings per child.
2. Inactive Memory Bank: Keeps lost tracks in memory for up to 600 frames (20s).
3. Re-ID Matching Engine: Matches re-appearing tracks ONLY against INACTIVE galleries.
4. Duplicate Suppression: Filters duplicate overlapping detections (IoU > 0.4).
"""

from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOCSORT, ReIDDetectMultiBackend


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
    """
    Manages persistent global IDs across long-term occlusions, reappearances, and duplicate detections.
    """

    def __init__(
        self,
        reid_backend: ReIDDetectMultiBackend,
        sim_threshold: float = 0.73,
        duplicate_iou_threshold: float = 0.40,
        max_gallery_size: int = 25,
        max_inactive_age: int = 100000,  # Never expire galleries during video session!
    ):
        self.reid_backend = reid_backend
        self.sim_threshold = sim_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.max_gallery_size = max_gallery_size
        self.max_inactive_age = max_inactive_age

        self.next_global_id = 1
        self.raw_to_global = {}  # raw_id -> global_id
        self.global_gallery = {}  # global_id -> list of L2-normalized numpy vectors [512]
        self.last_seen_frame = {}  # global_id -> frame_idx
        self.last_seen_box = {}  # global_id -> (x1, y1, x2, y2)

    def _extract_embeddings(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Crops bounding boxes and extracts L2-normalized 512-dim Re-ID embeddings.
        """
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)

        crops = []
        h_frame, w_frame = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)

            if x2 - x1 < 10 or y2 - y1 < 10:
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]

            crops.append(crop)

        with torch.no_grad():
            tensor = self.reid_backend._preprocess(crops)
            feats = self.reid_backend(tensor)
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        feats_norm = feats / norms
        return feats_norm

    def _match_inactive(self, embedding: np.ndarray, active_gids: set) -> tuple[int | None, float]:
        """
        Finds the best matching INACTIVE global_id using gallery CENTROID (average feature vector).
        Prevents a single corrupted crop in a gallery from matching random new people.
        """
        best_id = None
        best_sim = -1.0

        for gid, gallery in self.global_gallery.items():
            if gid in active_gids or len(gallery) == 0:
                continue  # CANNOT match an ID that is already active!

            # Calculate average appearance vector (centroid) of clean gallery frames
            centroid = np.mean(gallery, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            sim = float(np.dot(centroid, embedding))

            if sim > best_sim:
                best_sim = sim
                best_id = gid

        return best_id, best_sim

    def _find_spatial_rescue(self, box: np.ndarray, frame_idx: int, active_gids: set) -> int | None:
        """
        Position Memory Rescue: Reconnects lost tracks if a new detection appears
        within 100 pixels of where an inactive ID was lost in the last 45 frames.
        """
        bx1, by1, bx2, by2 = box[:4]
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        best_gid = None
        min_dist = 100.0  # max 100 pixels threshold

        for gid, last_f in self.last_seen_frame.items():
            if gid in active_gids:
                continue

            frame_gap = frame_idx - last_f
            if 0 < frame_gap <= 45:  # lost within last 1.5 seconds
                last_box = self.last_seen_box.get(gid)
                if last_box is not None:
                    lx1, ly1, lx2, ly2 = last_box
                    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                    d = float(np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2))

                    if d < min_dist:
                        min_dist = d
                        best_gid = gid

        return best_gid

    def update(self, raw_tracks: np.ndarray, frame: np.ndarray, frame_idx: int) -> list[tuple]:
        """
        Processes raw tracks from the base MOT tracker, applies Re-ID persistent mapping,
        suppresses duplicate boxes, and returns updated tracks as list of (x1, y1, x2, y2, global_id, conf, cls).
        """
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

        # Pass 1: Assign IDs to raw_ids that ALREADY have an established mapping, ensuring NO DUPLICATE GIDs on screen!
        unmapped_indices = []
        mapped_indices = []
        for i, raw_id in enumerate(raw_ids):
            box = boxes[i]
            # Check duplicate box first
            is_duplicate = False
            for active_box in active_boxes_this_frame:
                if compute_iou(box, active_box) >= self.duplicate_iou_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            if raw_id in self.raw_to_global:
                gid = self.raw_to_global[raw_id]
                # Enforce STRICT UNIQUE ID rule: Two active detections in same frame CANNOT share the same GID!
                if gid not in active_global_ids_this_frame:
                    active_global_ids_this_frame.add(gid)
                    active_boxes_this_frame.append(box)
                    mapped_indices.append(i)
                else:
                    unmapped_indices.append(i)
            else:
                unmapped_indices.append(i)

        # Anti-Swap Check: Detect if the base tracker accidentally swapped IDs between active tracks during a collision
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

                        # If both tracks look significantly closer to the OTHER's gallery, correct the swap!
                        if (sim_a_other > sim_a_own + 0.08) and (sim_b_other > sim_b_own + 0.08):
                            self.raw_to_global[raw_a] = gid_b
                            self.raw_to_global[raw_b] = gid_a

        # Pass 2: Handle unmapped raw_ids using Spatial Position Rescue FIRST, then Strict Re-ID Centroid Matching
        for i in unmapped_indices:
            raw_id = raw_ids[i]
            box = boxes[i]
            emb = embeddings[i]

            # 1. Try Spatial Rescue (reconnect lost ID if near previous position)
            spatial_gid = self._find_spatial_rescue(box, frame_idx, active_global_ids_this_frame)

            if spatial_gid is not None:
                gid = spatial_gid
                self.raw_to_global[raw_id] = gid
            else:
                # 2. Try Strict Centroid Re-ID Matching (Threshold >= 0.78)
                best_gid, best_sim = self._match_inactive(emb, active_global_ids_this_frame)

                if best_gid is not None and best_sim >= 0.78:
                    gid = best_gid
                    self.raw_to_global[raw_id] = gid
                else:
                    # 3. Mint a BRAND NEW ID (Never recycle old IDs for new people!)
                    gid = self.next_global_id
                    self.next_global_id += 1
                    self.raw_to_global[raw_id] = gid
                    self.global_gallery[gid] = []

            active_global_ids_this_frame.add(gid)
            active_boxes_this_frame.append(box)

        # Pass 3: Update galleries and construct output list
        for i, raw_id in enumerate(raw_ids):
            if raw_id not in self.raw_to_global:
                continue

            box = boxes[i]
            emb = embeddings[i]
            gid = self.raw_to_global[raw_id]

            # Only process if this box was retained (not duplicate suppressed)
            if not any(np.array_equal(box, b) for b in active_boxes_this_frame):
                continue

            if gid not in self.global_gallery:
                self.global_gallery[gid] = []

            # Occlusion & Quality Check: Only update gallery if box is NOT overlapping (IoU < 0.25), confidence >= 0.50, and area >= 1200px^2
            max_other_iou = 0.0
            for other_box in active_boxes_this_frame:
                if not np.array_equal(box, other_box):
                    iou_val = compute_iou(box, other_box)
                    if iou_val > max_other_iou:
                        max_other_iou = iou_val

            box_area = (box[2] - box[0]) * (box[3] - box[1])
            is_high_quality = (confs[i] >= 0.50) and (box_area >= 1200.0)

            # Only add features to memory bank when the child is isolated and clearly detected
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

            updated_tracks.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), gid, confs[i], clss[i]))

        # Cleanup stale inactive tracks
        stale_gids = [
            gid for gid, last_f in self.last_seen_frame.items()
            if gid not in active_global_ids_this_frame and (frame_idx - last_f) > self.max_inactive_age
        ]
        for gid in stale_gids:
            self.global_gallery.pop(gid, None)
            self.last_seen_frame.pop(gid, None)
            self.last_seen_box.pop(gid, None)

        return updated_tracks
