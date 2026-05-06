# Civitas Detector Documentation

## Overview

`CivitasDetector` adalah komponen pusat untuk menentukan apakah seorang track wajah termasuk `Civitas UB` atau tidak. Modul ini menggabungkan:
- 
ORB logo matching untuk mendeteksi tanda UB di area dada,
- analisis warna navy/gold berdasarkan HSV,
- pemrosesan gambar adaptif dengan CLAHE dan sharpen,
- caching per track untuk menghindari penghitungan berulang bila ORB tidak dijalankan.

### Tujuan Utama
- **Mengidentifikasi logo Almamater UB** pada area dada.
- **Mengevaluasi kombinasi warna** navy/gold untuk memverifikasi atribut seragam Civitas.
- **Menghasilkan score dan status** yang digunakan oleh smoothing temporal.
- **Meminimalkan overhead** melalui cache hasil saat scheduler menghentikan ORB.

## Parameter dan Efeknya

| Parameter | Lokasi | Default / Tipe | Deskripsi | Efek Jika Nilai Naik | Efek Jika Nilai Turun |
|-----------|--------|----------------|-----------|----------------------|-----------------------|
| `Config.UB_LOGO_TEMPLATES` | `Config` | dict path | Daftar template logo UB yang di-load | Menambahkan template meningkatkan robustness, tapi menambah waktu inisialisasi | Mengurangi template menurunkan kemungkinan cocok |
| `Config.UB_COLORS['navy']` | `Config` | HSV range | Ambang warna navy utama | Makin luas, deteksi warna lebih longgar; risiko false positive naik | Makin sempit, deteksi lebih ketat; bisa melewatkan variasi warna asli |
| `Config.UB_COLORS['dark_navy']` | `Config` | HSV range | Ambang warna navy gelap | Sama seperti di atas |
| `Config.UB_COLORS['light_navy']` | `Config` | HSV range | Ambang warna navy muda | Sama seperti di atas |
| `Config.UB_COLORS['gold']` | `Config` | HSV range | Ambang warna gold | Lebih luas meningkatkan deteksi atribut gold, tapi bisa menerima noise | Lebih sempit menurunkan false positive namun riskan melewatkan highlight gold |
| `Config.ORB_NFEATURES['Dark']` | `Config` | 800 | Jumlah fitur ORB saat frame gelap | Lebih banyak fitur memperbaiki matching tapi mahal | Lebih sedikit fitur mempercepat ORB tapi akurasi turun |
| `Config.ORB_NFEATURES['Normal']` | `Config` | 1000 | Jumlah fitur ORB saat kondisi normal | Sama seperti di atas |
| `Config.ORB_NFEATURES['Bright']` | `Config` | 1300 | Jumlah fitur ORB saat terang | Sama seperti di atas |
| `Config.ORB_ROI_SIZE['Dark']` | `Config` | 120 | Ukuran max chest ROI saat gelap | Lebih besar area pencocokan, lebih mahal, source image lebih representatif | Lebih kecil area, kurang akurat tapi lebih cepat |
| `Config.ORB_ROI_SIZE['Normal']` | `Config` | 160 | Ukuran max chest ROI saat normal | Sama seperti di atas |
| `Config.ORB_ROI_SIZE['Bright']` | `Config` | 180 | Ukuran max chest ROI saat terang | Sama seperti di atas |
| `brightness.orb_every_n` | `BrightnessAnalyzer` property | Dinamis | Interval ORB berdasarkan kecerahan | ORB dijalankan jarang, efisiensi naik, akurasi matching turun | ORB dijalankan sering, akurasi naik, beban komputasi naik |
| `FrameScheduler._orb_counters` | `FrameScheduler` internal | dict | Scheduler per-track ORB interval | Lebih sedikit eksekusi ORB per track jika interval besar | Lebih banyak eksekusi jika interval kecil |
| `self._cuda_orb` | `CivitasDetector init` | bool | Menggunakan ORB CUDA jika tersedia | ORB hardware acceleration meningkatkan kecepatan | Fallback CPU lebih lambat, tetapi stabil jika CUDA tidak tersedia |
| `self._cuda_eq` | `CivitasDetector init` | bool | Mode equalizeHist CUDA | Mempercepat preprocessing GPU | Gunakan CPU equalizeHist |
| `self._track_cache` | `CivitasDetector internal` | dict | Cache hasil per track saat ORB skip | Cache menyimpan hasil lama, mengurangi beban | Cache kecil memaksa hitungan ulang lebih sering |

> Note: Parameter `Config.ORB_NFEATURES`, `Config.ORB_ROI_SIZE`, dan `brightness.orb_every_n` berdampak langsung pada kualitas ORB matching, sedangkan `Config.UB_COLORS` mempengaruhi sensitivitas deteksi warna.

## Fungsi Inti

### 1. Inisialisasi dan Template Loading
`CivitasDetector.__init__()` membangun:
- ORB detector (CUDA jika tersedia, fallback CPU)
- BFMatcher untuk Hamming distance
- CLAHE dan sharpen kernel
- Linear CUDA filter untuk preprocessing jika tersedia
- Template logo UB pada skala 80, 120, 160 pixel

### 2. Preprocessing Image
- `self._detect_and_compute()` melakukan ORB detect+compute, menggunakan CUDA bila mungkin.
- `_equalize_cuda()` dan `_sharpen_cuda()` memperkuat kontras dan detail logo.

### 3. Logo Matching
- `detect_ub_logo(chest_roi_gray, roi_size)` melakukan:
  - resizing chest ROI ke `roi_size`
  - sharpening + equalize + CLAHE variant
  - ORB descriptor extraction untuk setiap variant
  - knnMatch + ratio test (0.75)
  - filter keypoint berdasarkan area logo UB yang diharapkan pada dada kiri
  - estimasi lokasi logo dari centroid keypoint
  - scoring dan pemilihan best match

### 4. Color Ratio Evaluation
- `_color_ratio_cuda(hsv_cpu, lower, upper)` menghitung fraksi pixel yang berada dalam rentang HSV.
- Mendukung CuPy pada GPU untuk mempercepat mask dan aggregasi.
- Warna yang dievaluasi: `navy`, `dark_navy`, `light_navy`, `gold`.

### 5. Prediksi Civitas
`detect_civitas_status(frame, track_id, x, y, w, h, scheduler, brightness)` bertugas:
- ekstrak chest ROI dari bounding box wajah
- konversi ROI ke grayscale
- panggil `scheduler.should_run_orb()` untuk memutuskan apakah ORB akan dijalankan
- jika `run_orb == True`, hitung HSV color ratios dan deteksi logo
- jalankan aturan scoring untuk menentukan status dan confidence
- cache hasil per track dan kembalikan `(status, score, chest_box, logo_box)`
- jika ORB tidak dijalankan, gunakan cache terakhir untuk menghindari false negative akibat skipped frame

## Detail Algoritma

### Chest ROI Extraction
- Area dada ditentukan dari posisi wajah:
  - `chest_y = y + int(h * 0.5)`
  - `chest_x = max(0, x - int(w * 0.8))`
  - `chest_w = max(int(w * 2.6), 160)`
  - `chest_h = max(int(h * 2.5), 160)`
- ROI dibatasi agar tidak keluar frame.
- Jika ROI invalid, kembali `Non-Civitas UB`.

### Color Logic
- `navy_total_ratio = max(navy_ratio, dark_navy_ratio, light_navy_ratio)`
- Status Civitas berhasil lebih kuat saat:
  - logo terdeteksi dan ratio navy > threshold,
  - atau logo terdeteksi dengan confidence cukup tinggi,
  - atau kombinasi navy + gold memadai.
- Beberapa fallback `Non-Civitas UB` diaktifkan ketika warna mendekati namun logo tidak kuat.

### Scoring Rules
- `has_logo && logo_confidence > 0.25 && navy_total_ratio > 0.10` → score 0.90
- `has_logo && logo_confidence > 0.35` → score 0.70
- `has_logo && logo_confidence > 0.20 && navy_total_ratio > 0.08` → score 0.60
- `navy_total_ratio > 0.25 && gold_ratio > 0.02` → score 0.55
- `has_logo && navy_total_ratio < 0.06` → score 0.35 (Non-Civitas UB)
- `navy_total_ratio > 0.35 && gold_ratio > 0.03` → score 0.38 (Non-Civitas UB)
- `navy_total_ratio > 0.25` → score 0.28 (Non-Civitas UB)
- `navy_total_ratio > 0.12` → score 0.18 (Non-Civitas UB)
- else → score 0.10 (Non-Civitas UB)

### Cache Behavior
- `self._track_cache[track_id]` menyimpan hasil terakhir untuk track tertentu.
- Saat ORB dibatalkan, cache dipakai untuk mempertahankan output stabil.
- `remove_track(track_id)` membersihkan cache ketika track hilang.

## Code Snippets

### Kelas dan Inisialisasi
```python
class CivitasDetector:
    def __init__(self):
        if USE_CUDA:
            try:
                self.orb = cv2.cuda.ORB_create(
                    nfeatures=1000,
                    scaleFactor=1.2,
                    nlevels=15,
                    edgeThreshold=20,
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE
                )
                self._cuda_orb = True
                print("✅ ORB menggunakan CUDA")
            except Exception:
                self.orb = cv2.ORB_create(
                    nfeatures=1000, scaleFactor=1.2, nlevels=15,
                    edgeThreshold=20, firstLevel=0, WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE
                )
                self._cuda_orb = False
                print("⚠️  CUDA ORB gagal, fallback ke CPU ORB")
        else:
            self.orb = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=15,
                edgeThreshold=20, firstLevel=0, WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE
            )
            self._cuda_orb = False

        self.bf          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self._sharpen_k  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        self.templates   = []
        self._track_cache: dict = {}
```

### Logo Matching
```python
def detect_ub_logo(self, chest_roi_gray: np.ndarray, roi_size: int):
    orig_h, orig_w = chest_roi_gray.shape
    if orig_h < 50 or orig_w < 50:
        return False, 0.0, None

    scale = 1.0
    working = chest_roi_gray
    if orig_h > roi_size or orig_w > roi_size:
        scale = roi_size / max(orig_h, orig_w)
        working = cv2.resize(chest_roi_gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)

    sharpened = self._sharpen_cuda(working)
    variants  = [
        self._equalize_cuda(working),
        self.clahe.apply(working),
        self._equalize_cuda(sharpened),
    ]

    ...

    return is_logo, final_score, best_location
```

### Color Ratio
```python
def _color_ratio_cuda(self, hsv_cpu: np.ndarray, lower, upper) -> float:
    total = hsv_cpu.shape[0] * hsv_cpu.shape[1]
    if USE_CUPY:
        try:
            import cupy as cp
            hsv_cp  = cp.asarray(hsv_cpu)
            lo      = cp.array(lower, dtype=cp.uint8)
            hi      = cp.array(upper, dtype=cp.uint8)
            mask    = cp.all((hsv_cp >= lo) & (hsv_cp <= hi), axis=2)
            return float(cp.sum(mask).get()) / total
        except Exception:
            pass
    mask = cv2.inRange(hsv_cpu, np.array(lower), np.array(upper))
    return np.sum(mask > 0) / total
```

### Civitas Status Prediction
```python
def detect_civitas_status(self, frame, track_id, x, y, w, h,
                        scheduler: 'FrameScheduler',
                        brightness: 'BrightnessAnalyzer'):
    chest_y = y + int(h * 0.5)
    chest_h = max(int(h * 2.5), 160)
    chest_x = max(0, x - int(w * 0.8))
    chest_w = max(int(w * 2.6), 160)
    ...
    run_orb = scheduler.should_run_orb(track_id, chest_gray, brightness)
    if run_orb:
        hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        navy_ratio       = self._color_ratio_cuda(hsv, *Config.UB_COLORS['navy'])
        dark_navy_ratio  = self._color_ratio_cuda(hsv, *Config.UB_COLORS['dark_navy'])
        light_navy_ratio = self._color_ratio_cuda(hsv, *Config.UB_COLORS['light_navy'])
        gold_ratio       = self._color_ratio_cuda(hsv, *Config.UB_COLORS['gold'])
        navy_total_ratio = max(navy_ratio, dark_navy_ratio, light_navy_ratio)

        has_logo, logo_confidence, logo_location = self.detect_ub_logo(chest_gray, brightness.orb_roi_size)

        ...

        self._track_cache[track_id] = result
        return result
    else:
        if track_id in self._track_cache:
            return self._track_cache[track_id]
        return "Non-Civitas UB", 0.0, (chest_x, chest_y, chest_w, chest_h), None
```

## Flow Diagram

Diagram berikut menggambarkan alur evaluasi logika `CivitasDetector` dari ekstraksi dada hingga output status:

```
Input: frame + track bbox
        |
        v
Extract chest ROI berdasarkan wajah
        |
        v
Convert chest ROI ke grayscale
        |
        v
Panggil scheduler.should_run_orb(track_id, chest_gray, brightness)
        |
        +-----------------------------+
        | run_orb == True             |
        +-----------------------------+
         /                           \
        v                             v
  Convert chest ROI ke HSV       Gunakan cache terakhir jika ada
        |                             |
        v                             v
  Hitung color ratios:             Return cached (status, score, chest_box, logo_box)
   - navy                          atau Non-Civitas default
   - dark_navy
   - light_navy
   - gold
        |
        v
  detect_ub_logo(chest_gray, brightness.orb_roi_size)
        |
        v
  Evaluasi scoring rules
        |
        v
  Simpan ke cache per track
        |
        v
  Return (status, civitas_score, chest_box, logo_box)
```

## Integration Notes

- `CivitasDetector` adalah jantung keputusan Civitas dalam `JetsonCivitasSystem`.
- Ia tidak hanya memutuskan status, tetapi juga menyediakan bounding box chest dan logo untuk overlay.
- Cache memungkinkan `FrameScheduler` menghemat beban ORB tanpa kehilangan konsistensi output.

## Professional Documentation Summary

`CivitasDetector` menggabungkan deteksi visual logo UB dan analisis warna seragam untuk memberikan prediksi `Civitas UB` yang lebih tahan noise. Desainnya fokus pada:
- fleksibilitas perangkat keras (CUDA fallback),
- penyesuaian kualitas berdasarkan kecerahan,
- dan penggunaan cache untuk kestabilan temporal.

Dengan parameter yang dapat diubah pada `Config` dan `BrightnessAnalyzer`, tim dapat melakukan tuning operasi ORB dan deteksi warna sesuai target performa hardware.
