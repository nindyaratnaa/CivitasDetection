import cv2
import numpy as np
import os
import time
from collections import deque
import statistics
import argparse

# ==================== CONFIGURATION ====================
class Config:
    CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(CASCADE_PATH):
        CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    CIVITAS_LABELS = ['Non-Civitas UB', 'Civitas UB']
    UB_LOGO_TEMPLATES = {
        'colored': 'templates/ub_logo_colored.png',
        'bw':      'templates/ub_logo_bw.png'
    }
    UB_COLORS = {
        'navy':       ([ 98,  80,  40], [125, 255, 210]),
        'gold':       ([  8,  80,  80], [ 35, 255, 255]),
        'dark_navy':  ([100, 100,  20], [118, 255, 160]),
        'light_navy': ([ 95,  60,  60], [125, 255, 230]),
    }

    # Brightness thresholds
    BRIGHT_DARK   = 60
    BRIGHT_NORMAL = 180

    # Full power mode — no throttling, max quality
    ORB_EVERY_N    = {'Dark': 1, 'Normal': 1, 'Bright': 1}
    ORB_NFEATURES  = {'Dark': 800, 'Normal': 1000, 'Bright': 1000}
    ORB_ROI_SIZE   = {'Dark': 120, 'Normal': 160, 'Bright': 160}
    DETECT_EVERY_N = {'Dark': 1, 'Normal': 1, 'Bright': 1}

# ==================== FPS MONITOR ====================
class FPSMonitor:
    def __init__(self, buffer_size=100):
        self.fps_buffer = deque(maxlen=buffer_size)
        self.frame_times = deque(maxlen=buffer_size)
        self.start_time = time.time()
        self.frame_count = 0
        self.smoothed_fps = 0.0
        self.alpha = 0.1  # Smoothing factor (lower = smoother)
        
    def update(self, fps):
        # Exponential moving average for smoother FPS
        if self.smoothed_fps == 0.0:
            self.smoothed_fps = fps
        else:
            self.smoothed_fps = self.alpha * fps + (1 - self.alpha) * self.smoothed_fps
        
        self.fps_buffer.append(self.smoothed_fps)
        self.frame_times.append(time.time())
        self.frame_count += 1
    
    def get_stats(self):
        if len(self.fps_buffer) < 10:
            return None
            
        fps_list = list(self.fps_buffer)
        
        # Calculate frame time variations for stability
        if len(self.frame_times) >= 2:
            time_diffs = []
            for i in range(1, len(self.frame_times)):
                time_diffs.append(self.frame_times[i] - self.frame_times[i-1])
            
            avg_frame_time = statistics.mean(time_diffs) if time_diffs else 0
            frame_time_std = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0
            stability_score = max(0, 100 - (frame_time_std * 1000))  # Convert to percentage
        else:
            avg_frame_time = 0
            stability_score = 0
        
        return {
            'avg_fps': statistics.mean(fps_list),
            'min_fps': min(fps_list),
            'max_fps': max(fps_list),
            'current_fps': self.smoothed_fps,  # Use smoothed value
            'avg_frame_time_ms': avg_frame_time * 1000,
            'stability_score': stability_score,
            'total_frames': self.frame_count,
            'runtime_seconds': time.time() - self.start_time
        }

# ==================== BRIGHTNESS ANALYZER ====================
class BrightnessAnalyzer:
    """
    Real-time brightness analysis using smoothed mean of gray frame.

    Category  | mean gray | Effect on pipeline
    ----------|-----------|--------------------------------------------
    Dark      | < 60      | ORB every 5 frames, 300 features, ROI 80px
    Normal    | 60-179    | ORB every 2 frames, 600 features, ROI 120px
    Bright    | >= 180    | ORB every 1 frame,  1000 features, ROI 160px

    Smoothing: EMA over last N frames prevents rapid category flipping
    from single noisy frames (e.g. flash, reflection).
    """
    EMA_ALPHA = 0.15  # lower = smoother, slower to react

    def __init__(self):
        self._smoothed: float = 128.0  # start at neutral
        self.category: str    = 'Normal'
        self.raw_value: float = 128.0

    def update(self, gray_frame: np.ndarray):
        """Call once per frame with the full grayscale frame."""
        self.raw_value  = float(np.mean(gray_frame))
        self._smoothed  = self.EMA_ALPHA * self.raw_value + (1 - self.EMA_ALPHA) * self._smoothed
        self.category   = self._classify(self._smoothed)

    def _classify(self, value: float) -> str:
        if value < Config.BRIGHT_DARK:
            return 'Dark'
        if value < Config.BRIGHT_NORMAL:
            return 'Normal'
        return 'Bright'

    # Convenience accessors used by FrameScheduler
    @property
    def orb_every_n(self) -> int:
        return Config.ORB_EVERY_N[self.category]

    @property
    def orb_nfeatures(self) -> int:
        return Config.ORB_NFEATURES[self.category]

    @property
    def orb_roi_size(self) -> int:
        return Config.ORB_ROI_SIZE[self.category]

    @property
    def detect_every_n(self) -> int:
        return Config.DETECT_EVERY_N[self.category]


# ==================== FRAME SCHEDULER ====================
class FrameScheduler:
    """
    Controls WHEN expensive operations run, driven by BrightnessAnalyzer.
    - should_detect() gates Haar cascade frequency.
    - should_run_orb() gates ORB per track.
    Both thresholds are pulled live from the brightness category each frame.
    """
    def __init__(self):
        self._detect_counter     = 0
        self._orb_counters: dict = {}

    def should_run_orb(self, track_id: int, chest_gray: np.ndarray,
                       brightness: BrightnessAnalyzer) -> bool:
        every_n = brightness.orb_every_n
        counter = self._orb_counters.get(track_id, every_n)
        counter += 1
        self._orb_counters[track_id] = counter
        if counter < every_n:
            return False
        self._orb_counters[track_id] = 0
        return True

    def remove(self, track_id: int):
        self._orb_counters.pop(track_id, None)


# ==================== FACE TRACKER ====================
class FaceTracker:
    """
    Multi-face tracker: IOU → centroid → appearance (histogram) matching.

    Appearance fingerprint (HSV histogram) dipakai sebagai tiebreaker
    saat 2+ track saling berdekatan (potensi swap posisi).
    Histogram di-update secara EMA supaya adaptif terhadap perubahan cahaya.
    """
    MAX_FACES        = 3
    MAX_LOST         = 15
    MIN_AGE          = 1
    IOU_THRESHOLD    = 0.15
    DIST_THRESHOLD   = 500
    # Jarak antar centroid (px) di bawah ini dianggap "zona swap" → aktifkan histogram
    SWAP_ZONE_DIST   = 180
    # Bobot appearance vs posisi saat di zona swap (0=posisi saja, 1=appearance saja)
    APPEAR_WEIGHT    = 0.55
    HIST_EMA_ALPHA   = 0.25   # seberapa cepat histogram update (lebih kecil = lebih stabil)

    def __init__(self):
        self.tracks   = {}   # id -> {'bbox', 'centroid', 'lost', 'age', 'hist'}
        self._next_id = 1

    @staticmethod
    def _centroid(bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2], dtype=np.float32)

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx); iy = max(ay, by)
        ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
        inter = max(0, ix2-ix) * max(0, iy2-iy)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _compute_hist(frame, bbox):
        """Hitung histogram HSV (H+S channel) dari face ROI, return None jika ROI invalid."""
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w < 10 or h < 10:
            return None
        roi  = frame[y:y+h, x:x+w]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _hist_sim(h1, h2):
        """Bhattacharyya similarity: 0=tidak mirip, 1=identik."""
        if h1 is None or h2 is None:
            return 0.0
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        return float(1.0 - dist)

    def _in_swap_zone(self):
        """
        Cek apakah ada 2 track yang centroid-nya saling berdekatan
        (di bawah SWAP_ZONE_DIST). Kalau iya, aktifkan appearance matching.
        """
        tids = list(self.tracks.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                d = float(np.linalg.norm(
                    self.tracks[tids[i]]['centroid'] - self.tracks[tids[j]]['centroid']
                ))
                if d < self.SWAP_ZONE_DIST:
                    return True
        return False

    def update(self, detections, frame=None):
        """
        detections : list of (x,y,w,h)
        frame      : BGR frame — diperlukan untuk histogram appearance matching
        Returns    : list of (track_id, x, y, w, h)
        """
        detections = sorted(detections, key=lambda b: b[2]*b[3], reverse=True)[:self.MAX_FACES]

        # Hitung histogram deteksi baru (hanya kalau frame tersedia)
        det_hists = []
        if frame is not None:
            for det in detections:
                det_hists.append(self._compute_hist(frame, det))
        else:
            det_hists = [None] * len(detections)

        use_appearance = (frame is not None) and self._in_swap_zone()

        matched_ids  = set()
        matched_dets = set()
        track_ids    = list(self.tracks.keys())

        # --- Bangun cost matrix: gabungan posisi + appearance ---
        # cost[tid_idx][di] = skor gabungan (lebih tinggi = lebih cocok)
        def _match_score(tid, di):
            det  = detections[di]
            iou  = self._iou(self.tracks[tid]['bbox'], det)
            dist = float(np.linalg.norm(self.tracks[tid]['centroid'] - self._centroid(det)))
            pos_score = iou + max(0.0, 1.0 - dist / self.DIST_THRESHOLD)
            if use_appearance:
                app_score = self._hist_sim(self.tracks[tid].get('hist'), det_hists[di])
                return (1.0 - self.APPEAR_WEIGHT) * pos_score + self.APPEAR_WEIGHT * app_score
            return pos_score

        # Greedy matching berdasarkan skor tertinggi
        candidates = []
        for tid in track_ids:
            for di in range(len(detections)):
                score = _match_score(tid, di)
                candidates.append((score, tid, di))
        candidates.sort(reverse=True)

        for score, tid, di in candidates:
            if tid in matched_ids or di in matched_dets:
                continue
            det  = detections[di]
            dist = float(np.linalg.norm(self.tracks[tid]['centroid'] - self._centroid(det)))
            iou  = self._iou(self.tracks[tid]['bbox'], det)
            # Terima match jika IOU cukup ATAU jarak cukup dekat
            if iou >= self.IOU_THRESHOLD or dist < self.DIST_THRESHOLD:
                self._update_track(tid, det, det_hists[di])
                matched_ids.add(tid)
                matched_dets.add(di)

        # --- Spawn new tracks untuk deteksi yang tidak ter-match ---
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            if len(self.tracks) < self.MAX_FACES:
                self.tracks[self._next_id] = {
                    'bbox':     det,
                    'centroid': self._centroid(det),
                    'lost':     0,
                    'age':      1,
                    'hist':     det_hists[di],
                }
                self._next_id += 1

        # --- Age out lost tracks ---
        for tid in track_ids:
            if tid not in matched_ids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.MAX_LOST:
                    del self.tracks[tid]

        return [
            (tid, *t['bbox'])
            for tid, t in self.tracks.items()
            if t['lost'] == 0 and t['age'] >= self.MIN_AGE
        ]

    def _update_track(self, tid, bbox, new_hist):
        self.tracks[tid]['bbox']     = bbox
        self.tracks[tid]['centroid'] = self._centroid(bbox)
        self.tracks[tid]['lost']     = 0
        self.tracks[tid]['age']     += 1
        # EMA update histogram — blend histogram lama dengan yang baru
        old_hist = self.tracks[tid].get('hist')
        if old_hist is None or new_hist is None:
            self.tracks[tid]['hist'] = new_hist
        else:
            self.tracks[tid]['hist'] = (
                (1 - self.HIST_EMA_ALPHA) * old_hist +
                self.HIST_EMA_ALPHA * new_hist
            )

    def remove(self, track_id):
        self.tracks.pop(track_id, None)


# ==================== CIVITAS TEMPORAL AVERAGING ====================
class CivitasTemporalAveraging:
    def __init__(self, window_size=30, confidence_threshold=0.60):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)  # (score, is_civitas)
        self.current_state = None
        self.state_hold_counter = 0
        # Hold frames proporsional ke window — makin besar buffer, makin stabil transisi
        self.STATE_HOLD_FRAMES = max(15, window_size // 5)

    def add_prediction(self, civitas_score, is_civitas):
        self.buffer.append((civitas_score, 1.0 if is_civitas else 0.0))

    def get_averaged_civitas(self):
        if len(self.buffer) < 5:
            return "Detecting...", 0.0

        n = len(self.buffer)
        weights = np.array([np.exp(0.1 * i) for i in range(n)], dtype=np.float32)
        weights /= weights.sum()

        scores   = np.array([item[0] for item in self.buffer], dtype=np.float32)
        statuses = np.array([item[1] for item in self.buffer], dtype=np.float32)

        w_score  = float(np.dot(weights, scores))
        w_status = float(np.dot(weights, statuses))

        # Dead-band: zona Uncertain dipersempit supaya tidak terlalu sering muncul
        if w_status >= 0.50 and w_score >= self.confidence_threshold:
            raw = "Civitas UB"
        elif w_status <= 0.35 and w_score < self.confidence_threshold:
            raw = "Non-Civitas UB"
        else:
            raw = "Uncertain"

        # Hysteresis — state hanya berubah setelah raw konsisten N frame
        if self.current_state is None:
            self.current_state = raw
            self.state_hold_counter = 0
        elif raw == self.current_state:
            self.state_hold_counter = 0
        else:
            self.state_hold_counter += 1
            if self.state_hold_counter >= self.STATE_HOLD_FRAMES:
                self.current_state = raw
                self.state_hold_counter = 0

        return self.current_state, w_score

    def soft_reset(self):
        keep = self.window_size // 3
        recent = list(self.buffer)[-keep:]
        self.buffer.clear()
        self.buffer.extend(recent)
        self.state_hold_counter = 0

    def reset(self):
        self.buffer.clear()
        self.current_state = None
        self.state_hold_counter = 0


# ==================== PER-PERSON STATE REGISTRY ====================
class PersonStateRegistry:
    """
    Owns the mapping: person_id → CivitasTemporalAveraging.

    Lifecycle:
    BORN  → sync() sees new ID in tracker.tracks  → allocate averager
    ALIVE → lost==0, feed() pushes data each frame
    LOST  → lost>0 but ≤MAX_LOST, averager frozen (history preserved)
    DEAD  → ID pruned from tracker.tracks → sync() purges averager
    """

    def __init__(self, window_size=30, confidence_threshold=0.60):
        self._window_size    = window_size
        self._conf_threshold = confidence_threshold
        self._states: dict   = {}  # id → CivitasTemporalAveraging

    def sync(self, tracker_track_ids: set):
        current = set(self._states.keys())
        for tid in tracker_track_ids - current:
            self._states[tid] = CivitasTemporalAveraging(
                window_size=self._window_size,
                confidence_threshold=self._conf_threshold
            )
        for tid in current - tracker_track_ids:
            del self._states[tid]

    def feed(self, person_id: int, score: float, is_civitas: bool):
        if person_id in self._states:
            self._states[person_id].add_prediction(score, is_civitas)

    def query(self, person_id: int):
        if person_id not in self._states:
            return "Detecting...", 0.0
        return self._states[person_id].get_averaged_civitas()

    def soft_reset(self, person_id: int):
        if person_id in self._states:
            self._states[person_id].soft_reset()

    def reset_all(self):
        for avg in self._states.values():
            avg.reset()

    @property
    def active_count(self) -> int:
        return len(self._states)

# ==================== CIVITAS DETECTION (ORB) ====================
class CivitasDetector:
    def __init__(self):
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=15,
            edgeThreshold=20,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE
        )
        self.bf          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self._sharpen_k  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        self.templates   = []
        self._track_cache: dict = {}  # track_id -> last (status, score, chest_box, logo_box)

        for template_type, path in Config.UB_LOGO_TEMPLATES.items():
            if os.path.exists(path):
                img = cv2.imread(path, 0)
                for size in [80, 120, 160]:
                    resized = cv2.resize(img, (size, size))
                    resized = cv2.equalizeHist(resized)
                    kp, des = self.orb.detectAndCompute(resized, None)
                    if des is not None:
                        self.templates.append({"image": resized, "kp": kp, "des": des, "size": size})
                print(f"✓ Loaded ORB template: {template_type} (3 scales)")

    def detect_ub_logo(self, chest_roi_gray: np.ndarray, roi_size: int):
        """ORB logo matching. Returns logo location in chest_roi_gray coordinate space."""
        orig_h, orig_w = chest_roi_gray.shape
        if orig_h < 50 or orig_w < 50:
            return False, 0.0, None

        scale = 1.0
        working = chest_roi_gray
        if orig_h > roi_size or orig_w > roi_size:
            scale = roi_size / max(orig_h, orig_w)
            working = cv2.resize(chest_roi_gray, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        proc_h, proc_w = working.shape
        sharpened = cv2.filter2D(working, -1, self._sharpen_k)
        variants  = [
            cv2.equalizeHist(working),
            self.clahe.apply(working),
            cv2.equalizeHist(sharpened),
        ]

        best_matches  = 0
        best_location = None
        best_score    = 0.0

        # Logo almamater UB ada di dada kiri orang (kanan frame karena flip)
        # Batasi area pencarian: 40%-100% lebar, 0%-70% tinggi chest ROI
        search_x_min = int(proc_w * 0.40)
        search_x_max = proc_w
        search_y_min = 0
        search_y_max = int(proc_h * 0.70)

        for processed in variants:
            kp2, des2 = self.orb.detectAndCompute(processed, None)
            if des2 is None or len(kp2) < 3:
                continue
            for template in self.templates:
                if template["des"] is None or len(template["des"]) < 3:
                    continue
                try:
                    matches = self.bf.knnMatch(template["des"], des2, k=2)
                except:
                    continue
                good = [m for pair in matches if len(pair) == 2
                        for m, n in [pair] if m.distance < 0.75 * n.distance]
                if len(good) < 4:
                    continue

                # Filter keypoints yang berada di luar zona logo yang diharapkan
                dst_pts_all = np.float32([kp2[m.trainIdx].pt for m in good])
                valid_mask = (
                    (dst_pts_all[:, 0] >= search_x_min) & (dst_pts_all[:, 0] <= search_x_max) &
                    (dst_pts_all[:, 1] >= search_y_min) & (dst_pts_all[:, 1] <= search_y_max)
                )
                good_filtered = [m for m, v in zip(good, valid_mask) if v]

                if len(good_filtered) < 4:
                    continue

                if len(good_filtered) > best_matches:
                    best_matches = len(good_filtered)

                    # Gunakan centroid keypoints sebagai pusat logo — jauh lebih stabil dari homography
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_filtered])
                    cx_kp = float(np.median(dst_pts[:, 0]))
                    cy_kp = float(np.median(dst_pts[:, 1]))

                    # Estimasi ukuran box dari spread keypoints, dengan batas minimum
                    spread_x = float(np.percentile(dst_pts[:, 0], 90) - np.percentile(dst_pts[:, 0], 10))
                    spread_y = float(np.percentile(dst_pts[:, 1], 90) - np.percentile(dst_pts[:, 1], 10))
                    box_size = max(spread_x, spread_y, template["size"] * 0.4)
                    box_size = min(box_size, template["size"] * 1.5)

                    bx = int(cx_kp - box_size / 2)
                    by = int(cy_kp - box_size / 2)
                    bw = int(box_size)
                    bh = int(box_size)

                    # Scale balik ke ruang chest_roi_gray asli
                    inv = 1.0 / scale
                    best_location = (
                        int(bx * inv), int(by * inv),
                        int(bw * inv), int(bh * inv)
                    )
                    best_score = len(good_filtered) / max(len(good), 1)

        match_score = min(best_matches / 8.0, 1.0)
        final_score = match_score * 0.6 + best_score * 0.4
        is_logo     = best_matches >= 4 and final_score > 0.20
        return is_logo, final_score, best_location

    def remove_track(self, track_id: int):
        self._track_cache.pop(track_id, None)

    def detect_civitas_status(self, frame, track_id, x, y, w, h,
                               scheduler: 'FrameScheduler',
                               brightness: 'BrightnessAnalyzer'):
        chest_y = y + int(h * 0.5)
        chest_h = max(int(h * 2.5), 160)
        chest_x = max(0, x - int(w * 0.8))
        chest_w = max(int(w * 2.6), 160)
        chest_y = min(chest_y, frame.shape[0] - chest_h)
        chest_x = min(chest_x, frame.shape[1] - chest_w)
        chest_h = min(chest_h, frame.shape[0] - chest_y)
        chest_w = min(chest_w, frame.shape[1] - chest_x)
        if chest_h <= 0 or chest_w <= 0:
            return "Non-Civitas UB", 0.0, None, None

        chest_roi  = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
        chest_gray = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)

        run_orb = scheduler.should_run_orb(track_id, chest_gray, brightness)
        if run_orb:
            hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
            navy_lower, navy_upper = Config.UB_COLORS['navy']
            navy_mask = cv2.inRange(hsv, np.array(navy_lower), np.array(navy_upper))
            navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
            dark_navy_lower, dark_navy_upper = Config.UB_COLORS['dark_navy']
            dark_navy_mask = cv2.inRange(hsv, np.array(dark_navy_lower), np.array(dark_navy_upper))
            dark_navy_ratio = np.sum(dark_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
            light_navy_lower, light_navy_upper = Config.UB_COLORS['light_navy']
            light_navy_mask = cv2.inRange(hsv, np.array(light_navy_lower), np.array(light_navy_upper))
            light_navy_ratio = np.sum(light_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
            gold_lower, gold_upper = Config.UB_COLORS['gold']
            gold_mask = cv2.inRange(hsv, np.array(gold_lower), np.array(gold_upper))
            gold_ratio = np.sum(gold_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
            navy_total_ratio = max(navy_ratio, dark_navy_ratio, light_navy_ratio)

            has_logo, logo_confidence, logo_location = self.detect_ub_logo(chest_gray, brightness.orb_roi_size)

            if has_logo and logo_confidence > 0.25 and navy_total_ratio > 0.10:
                civitas_score = 0.90 + (logo_confidence * 0.10)
                status = "Civitas UB"
            elif has_logo and logo_confidence > 0.35:
                civitas_score = 0.70
                status = "Civitas UB"
            elif has_logo and logo_confidence > 0.20 and navy_total_ratio > 0.08:
                civitas_score = 0.60
                status = "Civitas UB"
            elif navy_total_ratio > 0.25 and gold_ratio > 0.02:
                civitas_score = 0.55
                status = "Civitas UB"
            elif has_logo and navy_total_ratio < 0.06:
                civitas_score = 0.35
                status = "Non-Civitas UB"
            elif navy_total_ratio > 0.35 and gold_ratio > 0.03:
                civitas_score = 0.38
                status = "Non-Civitas UB"
            elif navy_total_ratio > 0.25:
                civitas_score = 0.28
                status = "Non-Civitas UB"
            elif navy_total_ratio > 0.12:
                civitas_score = 0.18
                status = "Non-Civitas UB"
            else:
                civitas_score = 0.10
                status = "Non-Civitas UB"

            result = (status, civitas_score, (chest_x, chest_y, chest_w, chest_h), logo_location)
            self._track_cache[track_id] = result
            return result
        else:
            if track_id in self._track_cache:
                return self._track_cache[track_id]
            return "Non-Civitas UB", 0.0, (chest_x, chest_y, chest_w, chest_h), None

# ==================== JETSON CIVITAS SYSTEM ====================
class JetsonCivitasSystem:
    def __init__(self):
        print(f"--- CIVITAS UB DETECTION SYSTEM (ORB + MULTI-FACE) ---")
        
        self.face_cascade      = cv2.CascadeClassifier(Config.CASCADE_PATH)
        self.civitas_detector   = CivitasDetector()
        self.face_tracker       = FaceTracker()
        self.scheduler          = FrameScheduler()
        self.brightness         = BrightnessAnalyzer()
        self.person_states      = PersonStateRegistry(window_size=30, confidence_threshold=0.60)
        self.no_face_counter    = 0
        self.metrics            = None

        self.prev_frame_time = 0
        self.fps = 0
        self.fps_monitor = FPSMonitor()

    def draw_static_ui(self, frame):
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width//2 - 120, 0), (width//2 + 120, 40), (20, 20, 20), -1)
        cv2.putText(frame, "Civitas UB Detection (ORB)", (width//2 - 110, 28), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_fps_panel(self, frame):
        """Draw FPS stats panel (top-left)."""
        white  = (255, 255, 255)
        orange = (0, 165, 255)
        green  = (0, 255, 0)
        red    = (0, 0, 255)
        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale  = 0.55
        sx, sy, sp = 15, 35, 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (310, 230), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        fps_stats = self.fps_monitor.get_stats()
        if fps_stats:
            fps_color = green if fps_stats['current_fps'] > 20 else orange if fps_stats['current_fps'] > 15 else red
            lines = [
                (f"FPS: {fps_stats['current_fps']:.1f}",          fps_color),
                (f"Avg FPS: {fps_stats['avg_fps']:.1f}",           white),
                (f"Min/Max: {fps_stats['min_fps']:.1f}/{fps_stats['max_fps']:.1f}", white),
                (f"Frame Time: {fps_stats['avg_frame_time_ms']:.1f}ms", white),
                (f"Stability: {fps_stats['stability_score']:.1f}%",
                 green if fps_stats['stability_score'] > 80 else orange if fps_stats['stability_score'] > 60 else red),
                (f"Runtime: {fps_stats['runtime_seconds']:.1f}s",  white),
                (f"Tracks: {len(self.face_tracker.tracks)} | States: {self.person_states.active_count}", white),
                (f"Brightness: {self.brightness.category} ({self.brightness.raw_value:.0f})",
                 red if self.brightness.category == 'Dark' else
                 green if self.brightness.category == 'Bright' else white),
            ]
            for i, (txt, col) in enumerate(lines):
                cv2.putText(frame, txt, (sx, sy + sp * i), font, scale, col, 1, cv2.LINE_AA)
        return frame

    def draw_face(self, frame, track_id, x, y, w, h, instant_info, smooth_info):
        """Draw bounding box + label for one tracked face."""
        gold   = (0, 215, 255)
        orange = (0, 165, 255)
        gray   = (128, 128, 128)
        font   = cv2.FONT_HERSHEY_SIMPLEX

        smooth_status = smooth_info[0] if smooth_info else None
        face_color = gold if smooth_status == "Civitas UB" else orange if smooth_status == "Uncertain" else gray

        cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)

        conf_str = f"{smooth_info[1]:.2f}" if smooth_info else "?"
        if smooth_status == "Civitas UB":
            label = f"ID{track_id} UB ({conf_str})"
        elif smooth_status == "Uncertain":
            label = f"ID{track_id} ? ({conf_str})"
        else:
            label = f"ID{track_id} Non-UB"

        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x, y - 28), (x + tw + 8, y), face_color, -1)
        cv2.putText(frame, label, (x + 4, y - 8), font, 0.6, (0, 0, 0), 2)
        return frame

    def draw_dashboard(self, frame, x, y, w, h, instant_civitas_info=None, smooth_civitas_info=None):
        """Legacy single-face dashboard — kept for compatibility."""
        self.draw_fps_panel(frame)
        self.draw_face(frame, 1, x, y, w, h, instant_civitas_info, smooth_civitas_info)
        return frame

    def print_final_stats(self):
        """Print final performance statistics"""
        stats = self.fps_monitor.get_stats()
        if stats:
            print("\n" + "="*50)
            print("📊 PERFORMANCE STATISTICS")
            print("="*50)
            print(f"🎯 Average FPS: {stats['avg_fps']:.2f}")
            print(f"⬇️  Minimum FPS: {stats['min_fps']:.2f}")
            print(f"⬆️  Maximum FPS: {stats['max_fps']:.2f}")
            print(f"⏱️  Average Frame Time: {stats['avg_frame_time_ms']:.2f}ms")
            print(f"📈 Frame Rate Stability: {stats['stability_score']:.1f}%")
            print(f"🕐 Total Runtime: {stats['runtime_seconds']:.1f} seconds")
            print(f"🎬 Total Frames: {stats['total_frames']}")
            print("="*50)
            
            # Performance evaluation
            if stats['avg_fps'] >= 25:
                print("✅ EXCELLENT: High performance system")
            elif stats['avg_fps'] >= 20:
                print("✅ GOOD: Acceptable performance")
            elif stats['avg_fps'] >= 15:
                print("⚠️  FAIR: Consider optimization")
            else:
                print("❌ POOR: Performance issues detected")
                
            if stats['stability_score'] >= 80:
                print("✅ STABLE: Consistent frame rate")
            elif stats['stability_score'] >= 60:
                print("⚠️  MODERATE: Some frame rate variation")
            else:
                print("❌ UNSTABLE: High frame rate variation")

    def run(self, source=None, enable_metrics=False, metrics_prefix=None, metrics_dir='metrics'):
        if enable_metrics:
            try:
                from metrics_graph import MetricsCollector
                os.makedirs(metrics_dir, exist_ok=True)
                self.metrics = MetricsCollector(algorithm='ORB')
                print(f"📊 Metrics AKTIF — graph disimpan ke '{metrics_dir}/' saat selesai")
            except ImportError:
                print("⚠️  metrics_graph.py tidak ditemukan, metrics dinonaktifkan")

        # source=None -> kamera, source='1' atau '2' -> video src/1.mp4 atau src/2.mp4
        if source is None:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            source_label = "Webcam"
        else:
            video_path = f"src/{source}.mp4" if not os.path.exists(source) else source
            cap = cv2.VideoCapture(video_path)
            source_label = f"Video: {video_path}"
        
        if not cap.isOpened():
            print(f"❌ Gagal membuka {source_label}.")
            return

        print(f"🎥 {source_label} dimulai. Tekan 'q' untuk keluar.")
        print("📊 FPS monitoring aktif - statistik akan ditampilkan setelah program selesai")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                new_frame_time = time.time()
                diff = new_frame_time - self.prev_frame_time
                self.fps = 1 / diff if diff > 0 else 0
                self.prev_frame_time = new_frame_time
                self.fps_monitor.update(self.fps)

                frame = cv2.flip(frame, 1)
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Update brightness every frame — cheap (single np.mean)
                self.brightness.update(gray)

                # Haar cascade every frame — no throttling
                raw_faces  = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
                detections = [tuple(f) for f in raw_faces] if len(raw_faces) > 0 else []

                # Update tracker
                tracked = self.face_tracker.update(detections)  # [(id,x,y,w,h), ...]

                frame = self.draw_static_ui(frame)

                # Sync registry with tracker BEFORE processing
                # Uses full tracker.tracks keyset so lost-but-not-dead IDs keep their history
                self.person_states.sync(set(self.face_tracker.tracks.keys()))

                people_data  = []
                orb_ran_this = False

                if tracked:
                    self.no_face_counter = 0

                    for (tid, x, y, w, h) in tracked:
                        try:
                            inst_status, inst_conf, civitas_box, logo_box = \
                                self.civitas_detector.detect_civitas_status(
                                    frame, tid, x, y, w, h,
                                    self.scheduler, self.brightness)

                            # Feed only confirmed-visible tracks
                            self.person_states.feed(tid, inst_conf, inst_status == "Civitas UB")
                            smooth_status, smooth_conf = self.person_states.query(tid)

                            if self.metrics:
                                people_data.append({
                                    'instant_conf': inst_conf,
                                    'smooth_conf':  smooth_conf,
                                    'status':       smooth_status,
                                })
                                if inst_conf > 0:
                                    orb_ran_this = True

                            # Chest area box
                            if civitas_box:
                                cx, cy, cw, ch = civitas_box
                                c_col = (0, 215, 255) if smooth_status == "Civitas UB" else (128, 128, 128)
                                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), c_col, 1)
                                cv2.putText(frame, "Chest", (cx, cy-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_col, 1)

                            self.draw_face(frame, tid, x, y, w, h,
                                           (inst_status, inst_conf),
                                           (smooth_status, smooth_conf))

                        except Exception as e:
                            print(f"Track {tid} error: {e}")

                else:
                    self.no_face_counter += 1
                    if self.no_face_counter > 40:
                        self.person_states.reset_all()
                    cv2.putText(frame, "Mencari Wajah...", (25, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

                if self.metrics:
                    self.metrics.record(
                        fps=self.fps,
                        people_data=people_data,
                        brightness=self.brightness.category,
                        orb_ran=orb_ran_this,
                        track_count=len(self.face_tracker.tracks)
                    )

                # Cleanup scheduler + detector cache for dropped tracks
                live_ids = set(self.face_tracker.tracks.keys())
                for old_id in list(self.scheduler._orb_counters.keys()):
                    if old_id not in live_ids:
                        self.scheduler.remove(old_id)
                        self.civitas_detector.remove_track(old_id)

                self.draw_fps_panel(frame)
                cv2.imshow('Civitas UB Detection - ORB', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        except KeyboardInterrupt:
            print("\n⏹️  Program dihentikan oleh user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()
            if self.metrics:
                self.metrics.save_graph(output_dir=metrics_dir, prefix=metrics_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Civitas UB Detection System')
    parser.add_argument('source', nargs='?', default=None,
                        help='Video source: kosong=webcam, 1/2/3=src/N.mp4, atau path/to/video.mp4')
    parser.add_argument('--metrics', action='store_true',
                        help='Aktifkan metrics collection dan simpan graph di akhir')
    parser.add_argument('--metrics-file', type=str, default=None, dest='metrics_file',
                        help='Prefix nama file output graph (contoh: orv-rev3)')
    parser.add_argument('--metrics-dir', type=str, default='metrics', dest='metrics_dir',
                        help='Folder output graph (default: metrics/)')
    args = parser.parse_args()

    app = JetsonCivitasSystem()
    app.run(source=args.source, enable_metrics=args.metrics,
            metrics_prefix=args.metrics_file, metrics_dir=args.metrics_dir)