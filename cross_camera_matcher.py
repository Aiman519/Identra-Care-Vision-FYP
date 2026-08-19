"""
Cross-Camera Identity Matcher (Step B: Cross-Camera Re-ID).

Each camera keeps running its OWN independent PersistentIDManager exactly as
before (single-camera tracking is untouched). This module adds one extra layer
on top: it watches each camera's local ID galleries and tries to match them
against people already seen on OTHER cameras.

Important: a match decision is NEVER permanently locked in until it is
"confirmed" (has a linked local ID from at least 2 different cameras). Until
confirmed, a local ID keeps being re-evaluated on every update() call as its
own gallery grows and as other cameras' galleries grow - if a better match
appears later, it gets retroactively merged into that match instead of being
stuck with an early wrong guess. This matters because the very first few
appearance samples of a person are often not representative enough to cross
the similarity threshold, even though the same person really is a match.

Works identically for recorded video files or live camera streams - it only
ever looks at embeddings each camera's manager has already collected, never at
the video source itself.
"""

import numpy as np


class CrossCameraMatcher:
    def __init__(self, camera_managers: dict, sim_threshold: float = 0.65, min_gallery_size: int = 3):
        """
        camera_managers: {camera_name: PersistentIDManager} - one entry per camera,
        each already created and being updated independently as usual.
        """
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
        # A cross-ID is "confirmed" once it has been linked from 2+ different
        # cameras - that's real cross-camera evidence, safe to stop revisiting.
        return len(self.cross_members.get(cross_id, {})) >= 2

    def update(self, verbose: bool = False, frame_idx=None):
        """
        Call periodically (e.g. once per processed frame) to look for new or
        improved cross-camera matches. Cheap to call often - it skips any
        local ID whose cross-ID is already confirmed.
        """
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
                    continue  # a cross-ID can only have one contributing local ID per camera
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
            # else: still unconfirmed with no better match found yet - left as is,
            # will be re-evaluated again on the next update() call.

    def get_cross_id(self, cam_name: str, local_gid: int):
        """
        Only returns a Global number once it's CONFIRMED (linked from 2+ cameras)
        - an unconfirmed guess is never shown, so a displayed Global number can
        never later change. Before confirmation this returns None, so the caller
        falls back to showing only the local per-camera ID (which is separate,
        always available, and already stable). Internally the matcher may still
        have a tentative best guess it keeps re-evaluating each update() call -
        it just isn't surfaced here until confirmed.
        """
        cross_id = self.local_to_cross.get((cam_name, local_gid))
        if cross_id is None or not self._is_confirmed(cross_id):
            return None
        return cross_id
