# FYP: AI Child-Safety Monitoring for Indoor Play Areas

## What this project is
Real-time child safety monitoring using multiple CCTV cameras in an indoor play area (jungle gyms, soft play zones, slides, trampolines).
The system watches children, tracks each child's identity across cameras, and (later)
raises an alert when a child enters a restricted zone with no adult (staff) nearby.

## My module (what I am building)
The full PERSON PIPELINE — modules 1 to 4:
1. Person Detection — find every person in each frame.
2. Assign a unique ID to each person.
3. Single-camera tracking — keep the same ID when a person moves / is briefly hidden.
4. Cross-camera Re-ID — keep the SAME global ID when the same person appears in a
   different camera.
5. Child vs Adult classification (staff = adult, kids = child). DOING THIS LAST.

The alert / safety-decision logic is NOT my part. Someone else handles that.

## IMPORTANT ORDER OF WORK
Build and verify the tracking pipeline FIRST. Do child-vs-adult classification LAST.
Detection, tracking, and Re-ID do not care about age — they just see "person".
Classification is a label added on top at the very end.
Build ONE layer at a time and watch it work before adding the next. Do not stack
five things at once.

## MY TEST DATA (important)
I have TWO short recorded video clips of children playing.
- They show the SAME children filmed by TWO DIFFERENT cameras.
- So the same child appears in BOTH clips -> these are used to test cross-camera Re-ID.
- The clips are SHORT. That is fine for watching it work visually.
Suggested filenames in this folder: camera1.mp4 and camera2.mp4

## WHAT I WANT TO CHECK, AND IN WHAT ORDER
Right now the priority is to SEE it working with my eyes (visual check), not metrics.

Step A — Single-camera tracking (run on ONE clip at a time):
  Play camera1.mp4, draw a box + ID on each child, and confirm each child's ID stays
  stable (ID 12 stays 12) as they move, cross each other, and get briefly hidden.
  Then do the same on camera2.mp4.

Step B — Cross-camera Re-ID (use BOTH clips):
  The same child should get the SAME global ID in both camera1.mp4 and camera2.mp4,
  instead of a new ID in the second clip. Watch both windows and confirm the global
  ID matches for the same child across the two cameras.

Numbers come LATER: after the visual check passes, I also want proper metrics
(MOTA / IDF1 / ID-switches for tracking; Rank-1 / mAP for Re-ID). Not now — visual first.

## AFTER MY OWN CLIPS
Once it works on my two clips, it must ALSO work on live real-time camera footage.
The same script should switch from a video file to a live camera by changing the input
source only (one line). Live camera is the final target.

## Chosen approach (subject to change)
- Detection + single-camera tracking: YOLO via `ultralytics`, using
  `model.track(persist=True, classes=[0], tracker="bytetrack.yaml")`. class 0 = person.
- Cross-camera Re-ID: `torchreid` with a model pre-trained on Market-1501.
  For each person-crop, produce an embedding (fingerprint); match embeddings BETWEEN the
  two cameras. If two fingerprints are close enough -> same person -> same global ID.
- Child vs Adult (LAST): prefer simplest reliable option for a play area —
  (A) staff wear a colored vest/badge and detect that, or
  (B) height/size heuristic since kids are clearly smaller,
  (C) only train a real image classifier if A and B are rejected.

## Environment
- Windows laptop. Python installed. I can code in Python. GPU status unknown
  (may use Google Colab for any training later).
- My two test clips are already in this project folder.

## Current status / next step
Basic detection+tracking already ran once on a simple clip.
NEXT: a clean script that plays a video source and shows boxes + stable IDs, so I can
watch ID consistency on camera1.mp4 and camera2.mp4 (Step A). Then add cross-camera
Re-ID (Step B) using both clips. Then make it work on a live camera.

## How I like help
- Explain simply, in plain wording.
- Build one piece, verify it works visually, THEN add the next.
- Don't stack multiple new things at once.
