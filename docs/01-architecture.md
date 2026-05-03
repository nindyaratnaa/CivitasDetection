# Multi-Face Civitas Detection System — Architecture

## Overview
Sistem deteksi multi-wajah dengan tracking ringan dan per-person state management untuk Jetson Nano (~20 FPS target).

---

## Component Hierarchy

```
JetsonCivitasSystem
├── FaceTracker              # Lightweight IOU+centroid tracker
├── PersonStateRegistry      # person_id → CivitasTemporalAveraging
├── CivitasDetector          # ORB logo + color detection
└── FPSMonitor               # Performance metrics
```

---

## 1. FaceTracker

**Purpose:** Assign persistent IDs to detected faces across frames.

**Algorithm:**
- **Stage 1 — IOU matching:** Match existing tracks to new detections via spatial overlap (IOU ≥ 0.25).
- **Stage 2 — Centroid fallback:** For unmatched tracks, use centroid distance < 80px.
- **Track lifecycle:**
  - `age=1` → new track spawned
  - `age≥MIN_AGE=2` → confirmed, eligible for rendering
  - `lost>0` → no detection this frame, coasting on last bbox
  - `lost>MAX_LOST=6` → pruned from memory

**Key parameters:**
```python
MAX_FACES      = 3   # Only track top-3 largest faces
MAX_LOST       = 6   # Frames before track deletion
MIN_AGE        = 2   # Frames before track is rendered (anti-flicker)
IOU_THRESHOLD  = 0.25
DIST_THRESHOLD = 80  # pixels
```

**Output:**
```python
update(detections) → [(track_id, x, y, w, h), ...]
# Only returns tracks where lost==0 AND age>=MIN_AGE
```

---

## 2. PersonStateRegistry

**Purpose:** Maintain independent `CivitasTemporalAveraging` instance per person.

### Lifecycle States

| State | Condition | Behavior |
|-------|-----------|----------|
| **BORN** | New ID appears in `tracker.tracks` | Allocate fresh averager |
| **ALIVE** | `lost==0` (visible this frame) | Feed detection → averager |
| **LOST** | `lost>0` but `lost≤MAX_LOST` | Freeze averager, keep in memory |
| **DEAD** | ID removed from `tracker.tracks` | Purge averager |

### Key Methods

**`sync(tracker_track_ids: set)`**
- Call once per frame AFTER `tracker.update()`.
- Input: `set(face_tracker.tracks.keys())` — **all** live+lost tracks.
- Allocates averager for new IDs, purges averager for dead IDs.
- Critical: uses full `tracks` dict, NOT just the confirmed-visible subset, so brief occlusions don't destroy history.

**`feed(person_id, score, is_civitas)`**
- Push one frame's detection into the person's averager.
- Only call for tracks where `lost==0` (actively detected this frame).

**`query(person_id) → (status, confidence)`**
- Return smoothed civitas status for a person.
- Safe to call even if person is `lost>0` — returns last known state.

**`reset_all()`**
- Hard-clear all averagers (scene change / long absence).
- Triggered when `no_face_counter > 40` frames.

### Why This Design?

**Problem with old approach:**
```python
# ❌ BAD: cleanup based on tracked (confirmed-visible only)
active_ids = {tid for tid, *_ in tracked}
for old_id in list(track_civitas.keys()):
    if old_id not in active_ids:
        del track_civitas[old_id]  # ← destroys state too early!
```
If a person turns their head for 1 frame → `lost=1` → not in `tracked` → state deleted → smoothing history lost.

**Solution:**
```python
# ✅ GOOD: sync based on tracker.tracks (all live tracks)
self.person_states.sync(set(self.face_tracker.tracks.keys()))
```
State survives brief occlusions (up to `MAX_LOST=6` frames), only purged when tracker fully drops the ID.

---

## 3. CivitasDetector

**Purpose:** Detect UB logo + navy/gold colors in chest ROI.

**Pipeline:**
1. Extract chest ROI from face bbox (1.6× width, 2.2× height below face).
2. **Color detection:** HSV thresholding for navy/gold ratios.
3. **ORB logo matching:** Multi-scale template matching with homography validation.
4. **Classification logic:**
   - Strong logo (conf>0.45) → Civitas UB
   - Weak logo + navy → Civitas UB
   - Weak logo + no navy → Non-Civitas UB (false positive suppression)

**Optimizations:**
- Blurry-frame cache: reuse last result for up to 6 blurry frames (skip ORB).
- Reused CLAHE instance (no per-frame allocation).

---

## 4. CivitasTemporalAveraging

**Purpose:** Smooth noisy per-frame predictions over time.

**Algorithm:**
- Exponential recency weighting: recent frames matter more.
- Hysteresis: state must hold for `STATE_HOLD_FRAMES=10` before switching.
- Thresholds:
  - `w_status ≥ 0.55` AND `w_score ≥ 0.65` → Civitas UB
  - `w_status ≤ 0.32` AND `w_score < 0.65` → Non-Civitas UB
  - Otherwise → Uncertain

**Buffer size:** 20 frames (~1 second at 20 FPS).

---

## Main Loop Flow

```python
while True:
    # 1. Detect faces (Haar cascade)
    raw_faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(80,80))
    
    # 2. Update tracker → get confirmed-visible tracks
    tracked = face_tracker.update(raw_faces)  # [(id,x,y,w,h), ...]
    
    # 3. Sync state registry with ALL tracker IDs (live+lost)
    person_states.sync(set(face_tracker.tracks.keys()))
    
    # 4. Process each confirmed-visible track
    for (tid, x, y, w, h) in tracked:
        inst_status, inst_conf, _, _ = civitas_detector.detect_civitas_status(frame, x,y,w,h)
        person_states.feed(tid, inst_conf, inst_status == "Civitas UB")
        smooth_status, smooth_conf = person_states.query(tid)
        draw_face(frame, tid, x, y, w, h, ...)
    
    # 5. Handle no-face scenario
    if not tracked:
        no_face_counter += 1
        if no_face_counter > 40:
            person_states.reset_all()
```

---

## Key Design Decisions

### 1. Why sync with `tracker.tracks` instead of `tracked`?

`tracked` = only confirmed-visible tracks (`lost==0` AND `age≥MIN_AGE`).  
`tracker.tracks` = all live tracks including those temporarily lost.

If we sync with `tracked`, a person who blinks or turns away for 1 frame loses their entire smoothing history. Syncing with `tracker.tracks` preserves state during brief occlusions.

### 2. Why `MAX_LOST=6` instead of 15?

Lower `MAX_LOST` = faster cleanup of ghost tracks (false positives that disappear).  
6 frames (~0.3s at 20 FPS) is enough to handle brief occlusions but not long enough to create persistent "ghost" bboxes.

### 3. Why `MIN_AGE=2`?

Haar cascade produces occasional 1-frame false positives. Requiring a track to survive 2 consecutive frames before rendering eliminates most flicker.

### 4. Why `minNeighbors=8` instead of 5?

Higher `minNeighbors` = stricter Haar cascade, fewer false positives from shadows/textures. Trade-off: slightly lower recall on difficult angles.

---

## Performance Characteristics

**Target:** 20 FPS on Jetson Nano  
**Bottleneck:** ORB feature matching (3× per frame for 3 faces)

**Optimizations applied:**
- Top-3 face limit (no wasted ORB on background faces)
- Blurry-frame cache (skip ORB on motion blur)
- Reused CLAHE/sharpen kernels
- `minSize=(80,80)` on Haar cascade (ignore tiny faces)

**Expected FPS:**
- 1 face: 22-25 FPS
- 2 faces: 18-22 FPS
- 3 faces: 15-20 FPS

---

## State Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PersonStateRegistry                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   ID not seen    │
                    │   (no state)     │
                    └──────────────────┘
                              │
                              │ sync() sees new ID
                              ▼
                    ┌──────────────────┐
                    │   BORN (age=1)   │
                    │  allocate avg    │
                    └──────────────────┘
                              │
                              │ age≥MIN_AGE
                              ▼
                    ┌──────────────────┐
                    │  ALIVE (lost=0)  │◄──┐
                    │  feed() updates  │   │
                    └──────────────────┘   │
                              │            │
                              │ no detect  │ detect again
                              ▼            │
                    ┌──────────────────┐   │
                    │  LOST (lost>0)   │───┘
                    │  state frozen    │
                    └──────────────────┘
                              │
                              │ lost>MAX_LOST
                              ▼
                    ┌──────────────────┐
                    │  DEAD (pruned)   │
                    │  sync() purges   │
                    └──────────────────┘
```

---

## Extending the System

### Add new per-person attributes
```python
class PersonStateRegistry:
    def __init__(self):
        self._states = {}
        self._metadata = {}  # NEW: track extra info per person
    
    def set_metadata(self, person_id, key, value):
        if person_id not in self._metadata:
            self._metadata[person_id] = {}
        self._metadata[person_id][key] = value
```

### Add face recognition
```python
# In main loop after civitas detection:
face_embedding = face_recognizer.extract(face_roi)
person_states.set_metadata(tid, 'embedding', face_embedding)
```

### Add pose estimation
```python
pose = pose_estimator.detect(frame, x, y, w, h)
person_states.set_metadata(tid, 'pose', pose)
```

---

## Troubleshooting

**Problem:** Bbox flickers on/off rapidly.  
**Solution:** Increase `MIN_AGE` to 3-4 frames.

**Problem:** State lost during brief occlusions.  
**Solution:** Increase `MAX_LOST` to 10-12 frames.

**Problem:** Ghost bboxes persist after person leaves.  
**Solution:** Decrease `MAX_LOST` to 4-5 frames.

**Problem:** Too many false positives.  
**Solution:** Increase `minNeighbors` to 10-12 in Haar cascade.

**Problem:** FPS drops below 15.  
**Solution:** Reduce `MAX_FACES` to 2, or increase `minSize` to (100,100).

---

## File Structure

```
Detection/
├── orv-rev3.py              # Main system
├── ARCHITECTURE.md          # This file
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── templates/
│   ├── ub_logo_colored.png
│   └── ub_logo_bw.png
└── src/
    ├── 1.mp4                # Test videos
    └── 2.mp4
```

---

## Usage

```bash
# Webcam
python orv-rev3.py

# Video file
python orv-rev3.py 1              # src/1.mp4
python orv-rev3.py path/to/video.mp4

# Quit
Press 'q'
```

---

## Performance Metrics

System tracks and displays:
- Current/Avg/Min/Max FPS
- Frame time (ms)
- Stability score (0-100%)
- Active tracks vs states
- Runtime

Final statistics printed on exit.
