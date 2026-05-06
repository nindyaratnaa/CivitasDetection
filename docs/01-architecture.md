# Multi-Face Civitas Detection System — Architecture

## Overview
Sistem ini dirancang untuk mendeteksi wajah dan menentukan status `Civitas UB` secara real-time pada platform Jetson Nano. Arsitektur ini menggabungkan:
- multi-face tracking ringan
- smoothing status per orang
- deteksi logo UB + warna dada
- penyesuaian beban komputasi berdasarkan kecerahan frame
- monitoring performa dan stabilitas

Target performa adalah sekitar 20 FPS pada hardware Jetson Nano.

---

## Component Hierarchy

```
JetsonCivitasSystem
├── FaceTracker
├── BrightnessAnalyzer
├── FrameScheduler
├── CivitasDetector
├── PersonStateRegistry
└── FPSMonitor
```

---

## 1. JetsonCivitasSystem

`JetsonCivitasSystem` adalah orchestrator utama. Modul ini menjalankan loop video dan mengkoordinasikan seluruh subkomponen:

1. baca frame dari webcam atau file video
2. flip frame horizontal dan konversi ke grayscale
3. update kecerahan frame dengan `BrightnessAnalyzer`
4. deteksi wajah menggunakan Haar Cascade
5. update `FaceTracker`
6. sinkronisasikan `PersonStateRegistry`
7. jalankan `CivitasDetector` untuk setiap track yang visible
8. beri hasil smoothing status ke overlay UI
9. bersihkan state dan cache untuk track yang hilang
10. tampilkan frame dan update statistik FPS

Modul ini juga menangani fallback CUDA/CPU, pengelolaan metric bila diaktifkan, dan laporan performa akhir.

---

## 2. FaceTracker

**Tujuan:** mempertahankan ID yang konsisten untuk setiap wajah yang terdeteksi.

**Strategi utama:**
- batasi pelacakan ke 3 wajah terbesar saja
- pencocokan track dilakukan dengan kombinasi:
  - IOU ≥ 0.15
  - atau jarak centroid < 500 pixel
- saat track berdekatan, gunakan histogram HSV sebagai tiebreaker
- setiap track menyimpan atribut:
  - `bbox`
  - `centroid`
  - `lost`
  - `age`
  - histogram `hist`

**Lifecycle:**
- `age=1` → track baru dibuat
- `lost==0` → track terlihat
- jika tidak ada deteksi, `lost += 1`
- `lost > 15` → track dihapus

**Output:**
`FaceTracker.update()` mengembalikan track yang visible pada frame saat ini:
- `lost == 0`
- `age >= MIN_AGE`

---

## 3. BrightnessAnalyzer

**Tujuan:** menentukan kategori kecerahan frame untuk adaptasi beban komputasi.

`BrightnessAnalyzer` menghitung rata-rata nilai grayscale dan melakukan smoothing dengan Exponential Moving Average. Kategorinya adalah:
- `Dark` untuk frame gelap
- `Normal` untuk kondisi umum
- `Bright` untuk frame terang

Kategori ini mempengaruhi:
- frekuensi eksekusi ORB per track
- jumlah fitur ORB
- ukuran ROI ORB
- frekuensi deteksi kasar

Dengan cara ini, sistem mempertahankan kualitas deteksi tanpa berlebihan pada beban GPU/CPU.

---

## 4. FrameScheduler

**Tujuan:** mengatur kapan operasi ORB yang mahal dijalankan untuk setiap track.

`FrameScheduler` menyimpan counter per track. ORB hanya dijalankan ketika counter mencapai ambang yang bergantung pada kategori kecerahan. Saat track hilang, counter tersebut dibersihkan.

Pendekatan ini menjaga kestabilan performa dengan menyesuaikan penggunaan ORB pada tiap target.

---

## 5. CivitasDetector

**Tujuan:** menentukan apakah satu track adalah `Civitas UB`.

**Pipeline deteksi:**
1. ekstrak chest ROI dari face bounding box
2. konversi ROI ke grayscale dan HSV
3. hitung rasio warna untuk `navy`, `dark_navy`, `light_navy`, `gold`
4. jalankan ORB logo matching pada chest ROI
5. cocokkan dengan template UB pada beberapa skala
6. filter keypoint menurut area dada yang diharapkan
7. hitung skor final dan tentukan status

**Fitur utama:**
- dukungan ORB CUDA jika tersedia
- fallback ke ORB CPU bila CUDA tidak tersedia
- CuPy opsional untuk perhitungan mask warna HSV
- cache hasil per track agar framebuffer berikutnya dapat menggunakan hasil sebelumnya saat ORB tidak dijalankan
- CLAHE dan sharpening reuse untuk stabilitas deteksi

**Keluaran:**
- status (`Civitas UB` / `Non-Civitas UB`)
- confidence score
- chest box
- logo box

---

## 6. PersonStateRegistry

**Tujuan:** menyimpan history dan melakukan smoothing status per person ID.

`PersonStateRegistry` membuat satu instance `CivitasTemporalAveraging` untuk setiap `track_id`. Ia menggunakan `sync()` untuk:
- menambahkan ID baru saat muncul
- menghapus ID yang sudah mati

`feed()` hanya dipanggil untuk track yang benar-benar terlihat. `query()` dapat mengembalikan status terakhir bahkan bila track sedang hilang sesaat. `reset_all()` membersihkan semua riwayat saat tidak ada wajah dalam durasi panjang.

Sinkronisasi berdasarkan `tracker.tracks` menjaga history ketika track mengalami occlusion singkat.

---

## 7. CivitasTemporalAveraging

**Tujuan:** meredam noise per-frame untuk menghasilkan status yang lebih stabil.

**Implementasi utama:**
- buffer hingga 30 frame
- bobot eksponensial untuk menekankan frame terbaru
- threshold confidence default 0.60
- status akhir menjadi:
  - `Civitas UB`
  - `Non-Civitas UB`
  - `Uncertain`

**Logika transisi:**
- `Civitas UB` bila `w_status >= 0.50` dan `w_score >= 0.60`
- `Non-Civitas UB` bila `w_status <= 0.35` dan `w_score < 0.60`
- selain itu, status menjadi `Uncertain`

Status hanya berganti setelah kondisi konsisten selama beberapa frame untuk menghindari fluktuasi cepat.

---

## 8. Main Loop Flow

```python
while True:
    frame = capture_frame()
    frame = flip(frame)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness.update(gray)

    raw_faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
    detections = [tuple(f) for f in raw_faces]

    tracked = face_tracker.update(detections)

    person_states.sync(set(face_tracker.tracks.keys()))

    if tracked:
        no_face_counter = 0
        for tid, x, y, w, h in tracked:
            inst_status, inst_conf, civitas_box, logo_box = civitas_detector.detect_civitas_status(
                frame, tid, x, y, w, h,
                scheduler, brightness
            )
            person_states.feed(tid, inst_conf, inst_status == "Civitas UB")
            smooth_status, smooth_conf = person_states.query(tid)
            draw_face(...)
    else:
        no_face_counter += 1
        if no_face_counter > 40:
            person_states.reset_all()

    cleanup_lost_tracks()
    draw_fps_panel(frame)
    show_frame(frame)
```

---

## 9. Design Rationale

- `FaceTracker` memprioritaskan konsistensi ID dengan kombinasi IOU, jarak centroid, dan appearance.
- `MAX_LOST = 15` memberi toleransi occlusion sementara sebelum track dihapus.
- `MIN_AGE = 1` memungkinkan track segera tampil karena false positive dikendalikan di lapisan lain.
- `tracker.tracks` adalah sumber kebenaran untuk `PersonStateRegistry`, sehingga riwayat tidak hilang saat track sementara lost.
- `BrightnessAnalyzer` dan `FrameScheduler` menjaga keseimbangan antara deteksi yang andal dan efisiensi komputasi.
- `CivitasDetector` menggabungkan deteksi logo dan deteksi warna agar false positive lebih rendah.

---

## 10. File Structure

```
Detection/
├── civitas_detection.py
├── civitas_detection_cuda.py
├── metrics_graph.py
├── requirements.txt
├── docs/
│   ├── 01-architecture.md
│   ├── 02-face-tracker.md
│   ├── 03-logo-detection.md
│   ├── 04-color-detection.md
│   ├── 05-brightness-analyzer.md
│   ├── 06-temporal-averaging.md
│   ├── 07-civitas-detector.md
│   └── 08-deployment-jetson.md
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── metrics/
├── src/
│   ├── 1.mp4
│   └── 2.mp4
├── templates/
│   ├── ub_logo_colored.png
│   └── ub_logo_bw.png
└── unused/
```

---

## 11. Usage

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

## 12. Performance Metrics

Aplikasi menampilkan dan merekam:
- Current / Average / Min / Max FPS
- Frame time (ms)
- Stability score (0-100%)
- Active tracks vs states
- Runtime

Statistik akhir ditampilkan saat aplikasi berhenti.

---

## 13. Data Flow Illustration

Berikut adalah ilustrasi alur data dari input hingga output dalam sistem `civitas_detection_cuda.py`. Diagram ini menunjukkan bagaimana data mengalir melalui komponen utama dalam satu iterasi loop utama.

```
Input Frame (Webcam/Video)
    |
    v
Flip Frame Horizontal
    |
    v
Konversi ke Grayscale
    |
    v
Update BrightnessAnalyzer
    |
    v
Deteksi Wajah (Haar Cascade)
    |
    v
Update FaceTracker
    |
    v
Sync PersonStateRegistry
    |
    +---------------------+
    | Ada track visible?  |
    +---------------------+
          | Ya
          v
    Loop untuk setiap track:
    - Jalankan CivitasDetector
    - Feed ke PersonStateRegistry
    - Query status smoothed
    - Draw Face dengan status
          |
          v
    Cleanup track hilang
          |
          v
    Draw FPS Panel & Display Frame
          |
          v
    Loop kembali ke Input
          | Tidak
          v
    Increment no_face_counter
    |
    +---------------------+
    | no_face_counter > 40? |
    +---------------------+
          | Ya
          v
    Reset PersonStateRegistry
          |
          v
    Cleanup track hilang
          | Tidak
          v
    Cleanup track hilang
```

### Penjelasan Alur Data:

1. **Input Frame**: Sistem menerima frame BGR dari webcam atau file video.

2. **Preprocessing**: Frame di-flip horizontal (untuk mirror effect) dan dikonversi ke grayscale untuk deteksi wajah.

3. **Brightness Analysis**: `BrightnessAnalyzer` menghitung kategori kecerahan (Dark/Normal/Bright) untuk mengatur beban komputasi ORB.

4. **Face Detection**: Haar Cascade mendeteksi wajah pada grayscale frame, menghasilkan list bounding box.

5. **Tracking**: `FaceTracker` mempertahankan ID konsisten untuk wajah, menggunakan IOU, centroid distance, dan histogram appearance sebagai tiebreaker.

6. **State Synchronization**: `PersonStateRegistry` disinkronkan dengan semua track (live + lost) untuk menjaga history smoothing.

7. **Per-Track Processing** (jika ada track visible):
   - `CivitasDetector` mengekstrak chest ROI, mendeteksi warna navy/gold, dan menjalankan ORB logo matching.
   - Hasil per-frame di-feed ke `CivitasTemporalAveraging` untuk smoothing.
   - Status smoothed di-query dan digunakan untuk rendering overlay.

8. **No-Face Handling**: Jika tidak ada wajah, counter diincrement. Jika counter > 40 frame, semua state di-reset.

9. **Cleanup**: Track yang hilang dihapus dari scheduler dan detector cache.

10. **Output Rendering**: FPS panel dan face overlays ditambahkan ke frame, lalu frame ditampilkan.

11. **Loop**: Proses berulang untuk frame berikutnya.

Diagram ini menunjukkan bagaimana data mengalir secara sequential, dengan branching untuk kondisi presence/absence wajah, dan parallel processing per track.
