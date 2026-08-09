# TECHNICAL SYSTEM SPECIFICATION
## Single-Camera Multi-Object Tracking & Persistent Identity Pipeline

**Module:** Person Pipeline — Stage A: Single-Camera Detection, MOT & Persistent Identity Maintenance  
**Domain:** AI Child-Safety Monitoring for Indoor Play Areas  
**Document Version:** 1.0.0  
**Status:** Production Ready  

---

## 1. System Overview

This specification details the design, algorithmic logic, and software architecture of the **Single-Camera Person Detection and Persistent Tracking Pipeline**. 

The system is designed for indoor play zones, daycare centers, and soft play parks. It continuously tracks children and staff under challenging conditions, including:
1. **High-Density Collisions:** Multi-person occlusions and intersecting paths during fast activity.
2. **Abrupt Motion Dynamics:** Non-linear speed changes, running accelerations, and sharp turns.
3. **Camera Frame Edge Transitions:** Partial body entry and exit at image borders.
4. **Extended Disappearances:** Invisibility for extended periods (15–20+ minutes) inside play tunnels, structures, or ball pits.

---

## 2. System Architecture & Data Flow

```
                                [ VIDEO INPUT ]
                       (Recorded File / Live RTSP Camera)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: PERSON DETECTION                                                   │
│ • Architecture: YOLOv8s (Ultralytics PyTorch Engine)                        │
│ • Input Frame Resolution: Capped at Max Display Width 960px                 │
│ • Class Filter: Class 0 (Person Only)                                       │
│ • Output: Bounding Boxes B = [x1, y1, x2, y2] & Detection Confidence c       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SHORT-TERM MOT TRACKING                                            │
│ • Engine: DeepOCSORT (Observation-Centric SORT)                             │
│ • Motion Model: Dynamic Velocity Kalman Filter                              │
│ • Appearance Extractor: OSNet (osnet_x1_0_msmt17, 512-dim embedding)        │
│ • Output: Frame-by-Frame Tracklets T_raw = (B, raw_id, c)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: PERSISTENT IDENTITY MEMORY ENGINE                                  │
│          (PersistentIDManager Module)                                       │
│                                                                             │
│  ├── 3.1 Monotonic Global Identity Registry (No ID Reuse)                   │
│  ├── 3.2 Anti-Swap Active Track Cross-Verification                          │
│  ├── 3.3 Spatial Position Rescue (160px Search Radius)                      │
│  ├── 3.4 Border Track Anchor (Edge Re-entry Recovery)                       │
│  ├── 3.5 Centroid Cosine Similarity Matching (sim >= 0.68 / 0.78)           │
│  └── 3.6 Occlusion-Aware & High-Quality Crop Memory Filter                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                               [ OUTPUT STREAM ]
               (Annotated Buffer: Bounding Boxes + Global IDs)
```

---

## 3. Mathematical & Algorithmic Specifications

### 3.1 Feature Vector Normalization & Cosine Similarity
For every detected person crop, OSNet extracts a 512-dimensional feature vector $\mathbf{f} \in \mathbb{R}^{512}$. Each feature vector is $L_2$-normalized:

$$\mathbf{f}_{\text{norm}} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$$

Because vectors are $L_2$-normalized, the Cosine Similarity between a crop feature $\mathbf{e}$ and a gallery centroid $\mathbf{c}$ simplifies to the vector dot product:

$$\text{Sim}(\mathbf{c}, \mathbf{e}) = \mathbf{c} \cdot \mathbf{e} = \sum_{k=1}^{512} c_k e_k$$

### 3.2 Gallery Centroid Representation
Rather than matching against individual noisy single-frame crops, the system computes the normalized mean centroid vector $\mathbf{c}_i$ for global ID $i$ over its stored gallery $\mathcal{G}_i = \{\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_N\}$ ($N \le 25$):

$$\mathbf{c}_i = \frac{\sum_{k=1}^N \mathbf{f}_k}{\left\| \sum_{k=1}^N \mathbf{f}_k \right\|_2}$$

### 3.3 Spatial Position & Border-Anchor Rescue
When a tracklet breaks due to occlusion or frame-edge exit, the system applies a two-stage spatial recovery prior to visual matching:

1. **Spatial Rescue Radius ($R = 160\text{px}$):** For any unmapped tracklet detected at centroid $(c_x, c_y)$ within $T_{\text{gap}} \le 90$ frames (3 seconds) of track loss, the spatial distance $d$ to the last known position $(l_x, l_y)$ is evaluated:
   $$d = \sqrt{(c_x - l_x)^2 + (c_y - l_y)^2}$$
   If $d \le 160\text{px}$, the tracklet is re-anchored to the lost global ID.

2. **Border Track Anchor:** If an individual is lost within $30\text{px}$ of the image boundary ($\text{is\_near\_border} = \text{True}$), any subsequent half-body detection within $220\text{px}$ of that boundary is automatically re-anchored to the exiting global ID.

### 3.4 Anti-Swap Verification
For active overlapping tracks, current embeddings $\mathbf{e}_A, \mathbf{e}_B$ are continuously evaluated against their galleries $\mathbf{c}_A, \mathbf{c}_B$. If a base-tracker swap occurs, the system detects the mutual similarity inversion:

$$\text{Sim}(\mathbf{e}_A, \mathbf{c}_B) > \text{Sim}(\mathbf{e}_A, \mathbf{c}_A) + 0.08 \quad \text{and} \quad \text{Sim}(\mathbf{e}_B, \mathbf{c}_A) > \text{Sim}(\mathbf{e}_B, \mathbf{c}_B) + 0.08$$

and automatically corrects the global ID assignment.

### 3.5 Quality-Gated Memory Maintenance
A feature vector $\mathbf{e}$ is appended to gallery $\mathcal{G}_i$ **only** when all of the following criteria pass:
* **Occlusion Isolation:** $\max_{j \neq i} \text{IoU}(B_i, B_j) < 0.25$
* **Detection Confidence:** $c \ge 0.50$
* **Target Area:** $\text{Area}(B) \ge 1200\text{px}^2$
* **Boundary Margin:** $x_1 \ge 15, y_1 \ge 15, x_2 \le W - 15, y_2 \le H - 15$

---

## 4. Software Component & File Structure

```
daycare-fyp/
├── track.py                           # Application Entry Point & GUI Rendering Loop
├── persistent_tracker.py              # Core PersistentIDManager Class & Memory Engine
├── yolov8s.pt                         # YOLOv8 Object Detection Weights
├── osnet_x1_0_msmt17.pt               # OSNet Re-Identification Weights
│
├── persistent_tracker_v1_backup.py    # System Baseline Module Backup
├── track_v1_backup.py                 # System Baseline Script Backup
│
└── SINGLE_CAMERA_TRACKING_PIPELINE.md # Technical System Specification Document
```

---

## 5. Execution Blueprint

To launch the tracking pipeline:

```powershell
.\venv\Scripts\python.exe track.py
```
