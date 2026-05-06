# Face Tracker Documentation

## Overview

`FaceTracker` adalah komponen inti dalam sistem deteksi Civitas UB yang bertanggung jawab untuk mempertahankan identitas konsisten wajah-wajah yang terdeteksi di sepanjang waktu. Modul ini mengimplementasikan multi-face tracking dengan strategi hybrid: kombinasi Intersection over Union (IOU), jarak centroid, dan appearance matching menggunakan histogram HSV sebagai tiebreaker.

### Tujuan Utama
- **Konsistensi ID**: Memastikan wajah yang sama mempertahankan ID yang sama meskipun ada occlusion sementara atau pergerakan cepat.
- **Efisiensi**: Membatasi tracking ke maksimal 3 wajah terbesar untuk optimasi performa pada Jetson Nano.
- **Robustness**: Menangani noise deteksi Haar Cascade dengan toleransi occlusion hingga 15 frame.

### Arsitektur
FaceTracker menggunakan struktur data dictionary untuk menyimpan state setiap track:
- `bbox`: Bounding box (x, y, w, h)
- `centroid`: Pusat geometris bounding box
- `lost`: Counter frame tanpa deteksi
- `age`: Usia track dalam frame
- `hist`: Histogram HSV untuk appearance matching

## Parameters

Berikut adalah parameter konfigurasi utama FaceTracker beserta dampak perubahan nilai:

| Parameter | Default Value | Deskripsi | Efek Jika Nilai Naik | Efek Jika Nilai Turun |
|-----------|---------------|-----------|----------------------|-----------------------|
| `MAX_FACES` | 3 | Jumlah maksimal wajah yang dapat dilacak secara simultan | Lebih banyak wajah terdeteksi, beban komputasi meningkat, risiko ID swap lebih tinggi | Kurang wajah terdeteksi, lebih efisien, tapi mungkin kehilangan track penting |
| `MAX_LOST` | 15 | Toleransi maksimal frame tanpa deteksi sebelum track dihapus | Track bertahan lebih lama saat occlusion, lebih stabil tapi memori lebih banyak | Track hilang lebih cepat, lebih responsif tapi rentan false negative |
| `MIN_AGE` | 1 | Usia minimum track sebelum dianggap valid untuk output | Track baru muncul lebih lambat, mengurangi false positive | Track muncul lebih cepat, tapi meningkatkan noise |
| `IOU_THRESHOLD` | 0.15 | Threshold minimum IOU untuk matching otomatis | Matching lebih ketat, mengurangi false positive tapi meningkatkan lost track | Matching lebih longgar, lebih toleran tapi risiko ID swap |
| `DIST_THRESHOLD` | 500 | Jarak maksimal centroid (px) untuk matching alternatif | Matching lebih toleran jarak jauh, lebih stabil | Matching lebih ketat, lebih akurat tapi rentan lost track |
| `SWAP_ZONE_DIST` | 180 | Jarak centroid untuk mengaktifkan appearance matching | Zona swap lebih luas, appearance matching aktif lebih sering | Zona swap lebih sempit, appearance matching jarang aktif |
| `APPEAR_WEIGHT` | 0.55 | Bobot appearance vs posisi dalam zona swap (0-1) | Appearance lebih dominan, lebih akurat untuk swap tapi sensitif cahaya | Posisi lebih dominan, lebih stabil tapi kurang akurat |
| `HIST_EMA_ALPHA` | 0.25 | Faktor smoothing histogram (0-1, lebih kecil = lebih stabil) | Histogram update lebih cepat, adaptif cahaya tapi noise | Histogram lebih stabil, kurang noise tapi lambat adaptasi |

## Core Functionality

### Metode Utama

#### `__init__()`
- Inisialisasi struktur data tracks dan counter ID berikutnya
- Tidak ada parameter input, menggunakan konstanta kelas

#### `update(detections, frame=None)`
- **Input**: `detections` (list of (x,y,w,h)), `frame` (BGR frame opsional untuk appearance)
- **Proses**:
  1. Sort detections berdasarkan ukuran (terbesar dulu), batasi ke MAX_FACES
  2. Hitung histogram deteksi baru jika frame tersedia
  3. Cek apakah dalam zona swap (centroid berdekatan)
  4. Lakukan greedy matching berdasarkan skor gabungan posisi + appearance
  5. Spawn track baru untuk deteksi unmatched
  6. Age out track yang lost > MAX_LOST
- **Output**: List of (track_id, x, y, w, h) untuk track visible (lost=0, age>=MIN_AGE)

#### `remove(track_id)`
- Hapus track tertentu dari dictionary
- Digunakan saat cleanup di main loop

### Metode Helper

#### `_centroid(bbox)`
- Hitung pusat geometris bounding box
- Return: numpy array [cx, cy]

#### `_iou(a, b)`
- Hitung Intersection over Union antara dua bounding box
- Return: float 0.0-1.0

#### `_compute_hist(frame, bbox)`
- Ekstrak ROI dari frame, konversi HSV, hitung histogram H+S channel
- Return: normalized histogram atau None jika ROI invalid

#### `_hist_sim(h1, h2)`
- Hitung similarity Bhattacharyya antara dua histogram
- Return: float 0.0-1.0 (1.0 = identik)

#### `_in_swap_zone()`
- Cek apakah ada dua track dengan centroid < SWAP_ZONE_DIST
- Return: boolean

#### `_update_track(tid, bbox, new_hist)`
- Update state track: bbox, centroid, reset lost, increment age
- EMA update histogram jika tersedia

## Algorithm Details

### Matching Strategy
FaceTracker menggunakan **hybrid matching** dengan prioritas:

1. **IOU Matching**: Jika IOU >= threshold, langsung match
2. **Distance Matching**: Jika jarak centroid < threshold, match sebagai alternatif
3. **Appearance Matching**: Jika dalam zona swap, gunakan histogram sebagai tiebreaker

### Greedy Assignment
- Bangun cost matrix semua kombinasi track-detection
- Sort berdasarkan skor tertinggi
- Assign secara greedy tanpa konflik

### Appearance Fingerprint
- Histogram HSV 16x16 bin untuk channel Hue (0-180) dan Saturation (0-256)
- Normalized ke range 0-1
- EMA smoothing untuk adaptasi cahaya gradual
- Bhattacharyya distance untuk similarity

### Lifecycle Management
- **New Track**: age=1, lost=0, hist dari deteksi pertama
- **Matched Track**: lost=0, age+=1, hist EMA update
- **Lost Track**: lost+=1, jika > MAX_LOST maka delete
- **Visible Track**: lost==0 AND age>=MIN_AGE

## Usage Examples

### Basic Usage
```python
tracker = FaceTracker()
detections = [(100, 50, 80, 80), (200, 60, 70, 90)]  # x,y,w,h
visible_tracks = tracker.update(detections, frame=bgr_frame)
for tid, x, y, w, h in visible_tracks:
    print(f"Track {tid}: {x},{y},{w},{h}")
```

### Integration with Main Loop
```python
# Dalam JetsonCivitasSystem.run()
tracked = face_tracker.update(raw_faces, frame)
person_states.sync(set(face_tracker.tracks.keys()))
for tid, x, y, w, h in tracked:
    # Process per track
    pass
```

## Performance Considerations

### Kompleksitas
- **Time**: O(N×M) dimana N=tracks, M=detections (greedy matching)
- **Space**: O(N) untuk tracks dictionary
- **GPU**: Tidak menggunakan CUDA, semua CPU-based

### Optimasi
- MAX_FACES=3 membatasi kompleksitas
- Histogram hanya dihitung jika frame tersedia dan dalam zona swap
- EMA smoothing mengurangi komputasi histogram per frame

### Trade-offs
- **Stabilitas vs Responsivitas**: MAX_LOST tinggi = stabil tapi lambat re-acquire
- **Akurasi vs Efisiensi**: Appearance matching akurat tapi mahal
- **Robustness vs Simplicity**: Hybrid approach kompleks tapi efektif

### Monitoring
Track state dapat diakses via `tracker.tracks` untuk debugging:
- Jumlah active tracks
- Lost counters
- Age distribution
- Swap zone status

## Code Snippets

Berikut adalah snippet kode utama dari implementasi FaceTracker di `civitas_detection_cuda.py`:

### Definisi Kelas dan Konstanta
```python
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
```

### Metode Helper
```python
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
```

### Metode Utama: update()
```python
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
```

### Metode Update Track
```python
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
```