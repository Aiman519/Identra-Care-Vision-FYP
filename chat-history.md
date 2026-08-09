# Project Chat History

Running log of work sessions for the AI Child-Safety Monitoring FYP (person pipeline module).
Newest entries are appended at the bottom. Never overwritten.

---

## 2026-08-03 / 2026-08-04 — Environment setup, Step A tracking, and ID-switch investigation

**What we worked on:**
- Set up the Python environment and built the first version of Step A (single-camera tracking, per CLAUDE.md's plan).
- Spent most of the session diagnosing and reducing ID-switching (the core problem: tracker assigning a new ID number to a child who was already being tracked).

**Key decisions made:**
- Detector: `yolov8s.pt` (small) — chosen over `yolov8n.pt` (nano) after measuring that nano missed detections more often, causing more ID churn.
- Tracker: switched from ultralytics' built-in trackers (ByteTrack, then BoT-SORT+ReID) to **DeepOCSORT** from the `boxmot` package, using a real person-ReID model (`osnet_x0_25_msmt17.pt`, same OSNet family already planned for the cross-camera Re-ID step). This was chosen because it measurably created far fewer spurious new IDs (21 unique IDs vs. 40 with BoT-SORT, for the same ~7-8 real kids in camera2.mp4).
- Root cause of ID switching (explained to user): (1) fast/sudden motion (e.g. a child going from standing to running) breaks the tracker's position prediction, and (2) full occlusion (one child hidden behind another) means there's no detection at all for those frames. Both are known, unsolved-in-general limitations of frame-by-frame tracking, not a setup mistake — perfect ID stability through fast running + overlap is not realistic with this class of approach.
- Confirmed via `wmic`/PowerShell that the laptop has only integrated AMD Radeon graphics, no dedicated GPU — so no CUDA acceleration is available; current preview runs on CPU only and is slower than real-time. This is a separate issue from ID-switching correctness.

**Files created or modified:**
- `venv/` — Python virtual environment created in the project root.
- `track.py` — main Step A script. Final version: reads `SOURCE` video (currently `camera1.mp4`; change this one line for a live camera later), runs YOLO (`yolov8s.pt`) detection + DeepOCSORT tracking, draws box + ID per person, resizes the popup window (`MAX_DISPLAY_WIDTH = 960`) so it's not too large to read.
- `botsort_reid.yaml` — custom BoT-SORT+ReID tracker config, created during the investigation. **No longer used** by `track.py` (superseded by the DeepOCSORT switch); left in the repo as a record of what was tried.
- `chat-history.md` — this file, created per user request to persist session context.

**Commands executed (important):**
- `python -m venv venv`
- `./venv/Scripts/python.exe -m pip install ultralytics opencv-python`
- `./venv/Scripts/python.exe -m pip install boxmot==10.0.12` (pinned to 10.0.12 after the default install, v22.0.0, turned out to have a broken internal import — `ModuleNotFoundError: No module named 'boxmot.data'`)
- `./venv/Scripts/python.exe -m pip install "setuptools<81"` (newer setuptools removed `pkg_resources`, which `boxmot` depends on)

**Issues encountered and how they were resolved:**
1. Popup window too large to read → added `resize_for_display()` capping width at 960px.
2. ID switching even in a single camera → root-caused via headless diagnostic scripts (not guessing) that counted unique ID numbers created over each clip. Tried, in order: nano→small detector (improved), BoT-SORT with `with_reid: True` (improved), a dedicated ONNX ReID model in place of "auto" features (no improvement), looser `track_buffer`/thresholds (no improvement), a lowered `proximity_thresh` to let appearance rescue fast-motion cases (made it *worse* — more false appearance matches between different kids). Final working improvement: switching the whole tracker to DeepOCSORT (`boxmot`), which handles sudden acceleration and appearance matching better than BoT-SORT's constant-velocity assumption.
3. `boxmot` package install/import issues (see commands above) — resolved by pinning `boxmot==10.0.12` and downgrading `setuptools`.
4. `DeepOCSORT.update()` docstring says it accepts a numpy array but the actual code calls `.numpy()` on the input (a bug/inconsistency in that version) — resolved by passing a `torch.Tensor` instead of a numpy array.
5. Video now plays slower than real-time in the preview window — root cause is CPU-only inference (laptop has no dedicated GPU, confirmed via `wmic`/`Get-CimInstance Win32_VideoController`). Not yet resolved; open decision point below.

**Next steps:**
- Decide how to handle preview speed: (a) accept slow/laggy preview for now since only visual correctness is being checked, not real-time performance, or (b) shrink the frame fed to the detector to trade a little accuracy for speed. User has not yet decided.
- Once Step A is visually confirmed acceptable on both `camera1.mp4` and `camera2.mp4`, move to Step B (cross-camera Re-ID) per CLAUDE.md's plan — note the appearance embeddings used there (OSNet/Market-1501 family) are the same family already in use for tracking in `track.py`, so there's continuity between the two stages.
- Longer-term: live camera input (already designed for — just change the `SOURCE` line), and real-time speed (would need GPU, e.g. Google Colab, or a lighter model) once correctness is settled.

---

## 2026-08-07 — Persistent ID Memory Bank Refinements & Codebase Cleanup

**What was worked on:**
- Addressed advanced ID switching scenarios during collisions, fast motion, and long-term re-entries.
- Researched MOT Re-ID methods (centroid feature representation, spatial-temporal continuity, persistent gallery lifespan).
- Upgraded `PersistentIDManager` in [`persistent_tracker.py`](file:///C:/Users/maham/Documents/daycare-fyp/persistent_tracker.py):
  1. **Anti-Swap Verification:** Cross-checks active tracks crossing paths against stored appearance galleries; if swapped, global IDs are swapped back.
  2. **Spatial Position Memory Rescue:** Reconnects lost tracks if a new detection appears within 100px of where an inactive ID was lost in the last 45 frames (1.5s).
  3. **Gallery Centroid Matching:** Computes Cosine Similarity against the L2-normalized mean feature vector (centroid) of past clean crops rather than single-frame max, preventing single corrupted crops from matching new people.
  4. **Session-Lifetime Persistent Gallery (`max_inactive_age = 100,000`):** Replaced short 20s expiration with session-lifetime persistence so children re-entering after extended periods keep their original IDs.
  5. **Occlusion & High-Quality Crop Filtering:** Freezes gallery updates when bounding box overlap IoU >= 0.25 or when detection confidence < 0.50 / area < 1200px², preventing boundary noise from corrupting feature galleries.
  6. **Strict Unique Active IDs & Gate (>= 0.78):** Ensures two active detections never share an ID, and new people entering get monotonically fresh IDs (8, 9, 10...).
- **Codebase Cleanup:** Removed temporary diagnostic scripts (`diagnose_*.py`), obsolete tracker config (`botsort_reid.yaml`), and unused model weights (`yolov8n.pt`, `yolo26s-reid.onnx`).

**Current Status:**
- Workspace is clean and fully self-contained with core tracking pipeline ([`track.py`](file:///C:/Users/maham/Documents/daycare-fyp/track.py) and [`persistent_tracker.py`](file:///C:/Users/maham/Documents/daycare-fyp/persistent_tracker.py)).
- Step A single-camera persistent tracking is stabilized.


