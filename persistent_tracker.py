"""
Persistent Single-Camera MOT Tracker with Long-Term Inactive Re-ID Memory Bank.

This module wraps standard online MOT trackers (DeepOCSORT / BoTSORT) and adds:
1. Feature Gallery: Stores up to 25 diverse appearance embeddings per child.
2. Inactive Memory Bank: Keeps lost tracks in memory for up to 600 frames (20s).
3. Re-ID Matching Engine: Matches re-appearing tracks ONLY against INACTIVE galleries.
4. Duplicate Suppression: Filters duplicate overlapping detections (IoU > 0.4).
"""

import copy
from collections import deque
from pathlib import Path
import numpy as np
import torch
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR


def _patched_unfreeze(self):
    """
    Upstream boxmot bug fix (KalmanFilterXYSR.unfreeze, boxmot 13.0.17): the
    original method assumes at least 2 real observations exist in history_obs
    before a gap, and crashes with IndexError when a track had only 1 real hit
    before going quiet (indices[-2] on a length-1 array). Surfaces specifically
    when max_age is raised above ~60 (more time for a 1-hit track to linger),
    which is why max_age=200 here needs this patch (mirrors the fix already
    applied in kaggle_kernel_crosscam/script.py).
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
        box1 = np.asarray(new_history[index1], dtype=np.float64).reshape(-1)
        box2 = np.asarray(new_history[index2], dtype=np.float64).reshape(-1)
        x1, y1, s1, r1 = (float(v) for v in box1[:4])
        w1, h1 = float(np.sqrt(s1 * r1)), float(np.sqrt(s1 / r1))
        x2, y2, s2, r2 = (float(v) for v in box2[:4])
        w2, h2 = float(np.sqrt(s2 * r2)), float(np.sqrt(s2 / r2))
        time_gap = float(index2 - index1)
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

        for i in range(index2 - index1):
            x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
            w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
            s, r = w * h, w / h
            new_box = np.array([x, y, s, r]).reshape((4, 1))
            self.update(new_box)
            if not i == (index2 - index1 - 1):
                self.predict()
                self.history_obs.pop()
        self.history_obs.pop()


KalmanFilterXYSR.unfreeze = _patched_unfreeze


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
        reid_backend,
        sim_threshold: float = 0.73,
        duplicate_iou_threshold: float = 0.40,
        max_gallery_size: int = 25,
        max_inactive_age: int = 100000,  # Never expire galleries during video session!
        verbose: bool = False,
    ):
        self.reid_backend = reid_backend
        self.sim_threshold = sim_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.max_gallery_size = max_gallery_size
        self.max_inactive_age = max_inactive_age
        self.verbose = verbose

        self.next_global_id = 1
        self.raw_to_global = {}  # raw_id -> global_id
        self.global_gallery = {}  # global_id -> list of L2-normalized numpy vectors [512]
        self.last_seen_frame = {}  # global_id -> frame_idx
        self.last_seen_box = {}  # global_id -> (x1, y1, x2, y2)
        self.last_seen_border = {}  # global_id -> bool

    def _extract_embeddings(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Extracts L2-normalized Re-ID embeddings for each box via the tracker's own
        ReID backend (base_tracker.model), which crops + preprocesses internally.
        """
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)

        boxes_np = np.asarray(boxes, dtype=np.float32)[:, :4]
        feats = self.reid_backend.get_features(boxes_np, frame)
        feats = np.asarray(feats, dtype=np.float32)

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

    def _find_spatial_rescue(
        self, box: np.ndarray, emb: np.ndarray, frame_idx: int, active_gids: set, frame_shape: tuple
    ) -> int | None:
        """
        Expanded Spatial & Border Rescue: Reconnects lost tracks during multi-child collisions
        or edge-of-frame half-body re-entries.

        Position alone is NOT enough: when two kids cross paths, the nearest lost track by
        position is often the OTHER kid (that's exactly when their positions overlap). So any
        candidate with an existing gallery must also pass a low appearance-similarity sanity
        check before being accepted; pure distance is only trusted when the candidate has no
        gallery yet (nothing to check against).
        """
        bx1, by1, bx2, by2 = box[:4]
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        h_frame, w_frame = frame_shape[:2]

        is_near_border = (bx1 < 30) or (by1 < 30) or (bx2 > w_frame - 30) or (by2 > h_frame - 30)

        MIN_SANITY_SIM = 0.35  # below this, appearance clearly disagrees with position -> reject

        candidates = []  # (dist, gid, sim_or_none)
        for gid, last_f in self.last_seen_frame.items():
            if gid in active_gids:
                continue

            frame_gap = frame_idx - last_f
            if not (0 < frame_gap <= 90):  # lost within last 3 seconds
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

        candidates.sort(key=lambda c: c[0])  # nearest first among appearance-plausible candidates
        return candidates[0][1]

    def update(self, raw_tracks: np.ndarray, frame: np.ndarray, frame_idx: int) -> list[tuple]:
        """
        Processes raw tracks from the base MOT tracker, applies Re-ID persistent mapping,
        suppresses duplicate boxes, and returns updated tracks as list of (x1, y1, x2, y2, global_id, conf, cls).
        """
        if len(raw_tracks) == 0:
            return []

        h_frame, w_frame = frame.shape[:2]
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
                            if self.verbose:
                                print(
                                    f"[frame {frame_idx}] ANTI-SWAP fired: "
                                    f"gid {gid_a}<->{gid_b} (raw {raw_a}<->{raw_b}) | "
                                    f"a: own={sim_a_own:.2f} other={sim_a_other:.2f} | "
                                    f"b: own={sim_b_own:.2f} other={sim_b_other:.2f}"
                                )
                            self.raw_to_global[raw_a] = gid_b
                            self.raw_to_global[raw_b] = gid_a

        # Pass 2: Handle unmapped raw_ids using Spatial & Border Rescue FIRST, then Re-ID Centroid Matching
        for i in unmapped_indices:
            raw_id = raw_ids[i]
            box = boxes[i]
            emb = embeddings[i]

            # 1. Try Spatial & Border Rescue (reconnect lost ID if near previous position or border exit)
            spatial_gid = self._find_spatial_rescue(box, emb, frame_idx, active_global_ids_this_frame, frame.shape)

            if spatial_gid is not None:
                gid = spatial_gid
                self.raw_to_global[raw_id] = gid
                if self.verbose:
                    print(f"[frame {frame_idx}] SPATIAL RESCUE: raw {raw_id} -> gid {gid}")
            else:
                # 2. Try Centroid Re-ID Matching (Threshold >= 0.68 for rescue, >= 0.78 for new)
                best_gid, best_sim = self._match_inactive(emb, active_global_ids_this_frame)

                if best_gid is not None and best_sim >= 0.68:
                    gid = best_gid
                    self.raw_to_global[raw_id] = gid
                    if self.verbose:
                        print(f"[frame {frame_idx}] CENTROID MATCH: raw {raw_id} -> gid {gid} (sim={best_sim:.2f})")
                else:
                    # 3. Mint a BRAND NEW ID (Never recycle old IDs for new people!)
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

            h_frame, w_frame = frame.shape[:2]
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            is_near_border = (box[0] < 15) or (box[1] < 15) or (box[2] > w_frame - 15) or (box[3] > h_frame - 15)
            is_high_quality = (confs[i] >= 0.50) and (box_area >= 1200.0) and (not is_near_border)

            # Only add features to memory bank when the child is isolated, clear, and fully inside the frame
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
