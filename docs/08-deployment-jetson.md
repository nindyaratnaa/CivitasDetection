# Deployment Civitas Detection ke NVIDIA Jetson

## 1. Tujuan
Dokumen ini menjelaskan langkah-langkah deployment project deteksi Civitas pada platform NVIDIA Jetson. Fokusnya adalah:
- persyaratan hardware/software
- pemindahan kode via GitHub
- setup lingkungan Jetson
- struktur direktori project
- cara pengujian
- troubleshooting umum

Dokumen ini dibuat agar deployment menjadi terstandarisasi dan mudah diulang pada perangkat Jetson Nano / Xavier / Orin.

---

## 2. Prasyarat

### 2.1 Hardware
- NVIDIA Jetson Nano, Xavier NX, Orin Nano, atau perangkat Jetson lain.
- Kamera USB atau kamera CSI yang kompatibel.
- Penyimpanan cukup untuk repo, dataset kecil, dan dependency.
- Daya yang stabil sesuai kebutuhan Jetson.

### 2.2 Software
- NVIDIA JetPack SDK yang sesuai dengan perangkat Jetson.
  - JetPack 5.x → CUDA 11.x / OpenCV JetPack
  - JetPack 6.x → CUDA 12.x / OpenCV JetPack
- Python 3.10 / 3.11 yang kompatibel dengan JetPack.
- `git` untuk sinkronisasi ke GitHub.
- `pip` untuk instalasi dependensi Python.

### 2.3 Dependensi Project
Project ini membutuhkan paket Python berikut:
- `numpy`
- `matplotlib`
- `cupy-cuda11x` atau `cupy-cuda12x` sesuai JetPack

> Penting: jangan install `opencv-python` via pip jika Anda menggunakan OpenCV yang sudah terpasang dari JetPack. Paket pip akan menimpa atau membuat konflik dengan OpenCV CUDA yang disertakan JetPack.

---

## 3. Persiapan Repository dan GitHub

### 3.1 Siapkan repository GitHub
1. Buat repository baru di GitHub.
2. Untuk repo baru, salin URL HTTPS atau SSH.
3. Pastikan `requirements.txt` telah ada di root repo.

### 3.2 Upload atau sinkronisasi kode
Jika project sudah ada secara lokal:
```bash
cd /path/to/Detection
git init
git add .
git commit -m "Initial commit for Jetson Civitas Detection"
git branch -M main
git remote add origin <GITHUB_URL>
git push -u origin main
```

Jika repo sudah ada di GitHub dan Anda ingin memindahkan ke Jetson:
```bash
cd /home/ubuntu
git clone <GITHUB_URL>
cd Detection
```

### 3.3 Rekomendasi GitHub workflow
- Gunakan branch terpisah untuk fitur baru: `feature/jetson-deploy`.
- Tetap commit perubahan konfigurasi environment di file teks, bukan di binary.
- Tambahkan `.gitignore` untuk:
  - `__pycache__/`
  - `*.pyc`
  - `venv/`
  - `*.log`
  - `*.sqlite`

---

## 4. Setup dan Instalasi di Jetson

### 4.1 Update sistem
Jalankan update awal di Jetson:
```bash
sudo apt update
sudo apt upgrade -y
```

### 4.2 Pastikan OpenCV JetPack dan CUDA
JetPack biasanya sudah memasang OpenCV yang mendukung CUDA. Verifikasi dengan Python:
```python
import cv2
print(cv2.getBuildInformation())
print(cv2.cuda.getCudaEnabledDeviceCount())
```
Jika `cv2` berhasil diimport dan CUDA device count > 0, JetPack OpenCV sudah aktif.

### 4.3 Buat virtual environment (opsional tapi direkomendasikan)
```bash
cd ~/Detection
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 4.4 Instal dependensi Python
Instal paket utama dari `requirements.txt`:
```bash
pip install -r requirements.txt
```

Kemudian instal CuPy sesuai JetPack:
- JetPack 5.x / CUDA 11: `pip install cupy-cuda11x`
- JetPack 6.x / CUDA 12: `pip install cupy-cuda12x`

Jika JetPack Anda belum menyediakan OpenCV dengan CUDA support dan project ingin menggunakan CPU fallback, gunakan:
```bash
pip install opencv-python
```
Tapi ingat, ini tidak direkomendasikan untuk Jetson yang sudah terpasang OpenCV JetPack.

### 4.5 Pastikan file dependency lokal tersedia
Repo ini memerlukan beberapa file pendukung:
- `haarcascades/haarcascade_frontalface_default.xml`
- `templates/` untuk logo UB
- `src/` jika gunakan video sample

Pastikan file-file tersebut ada di dalam folder repo setelah clone.

---

## 5. Struktur Direktori yang Disarankan

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
└── venv/               # jika virtualenv digunakan
```

### File utama
- `civitas_detection_cuda.py`: skrip utama untuk Jetson dengan dukungan CUDA.
- `civitas_detection.py`: versi fallback CPU.
- `requirements.txt`: daftar dependensi Python.
- `docs/`: dokumentasi arsitektur dan modul.
- `haarcascades/`: model Haar Cascade wajah.
- `templates/`: asset logo/template yang digunakan detector.

---

## 6. Deployment Workflow ke Jetson

### 6.1 Dari GitHub ke Jetson
1. Clone repo pada Jetson:
```bash
git clone <GITHUB_URL>
cd Detection
```
2. Checkout branch yang benar:
```bash
git checkout main
```
3. Update repo saat ada perubahan:
```bash
git pull origin main
```

### 6.2 Jalankan project
Untuk menjalankan versi CUDA:
```bash
python civitas_detection_cuda.py
```
Jika ingin pakai kamera USB tertentu atau file video:
```bash
python civitas_detection_cuda.py 0
python civitas_detection_cuda.py src/1.mp4
```

### 6.3 Deployment ulang setelah perubahan
Jika Anda melakukan perubahan di PC lokal, push ke GitHub lalu di Jetson jalankan:
```bash
git pull origin main
```
Jika ada dependency baru, jalankan ulang:
```bash
pip install -r requirements.txt
```

---

## 7. Pengujian pada Jetson

### 7.1 Verifikasi OpenCV CUDA
Buka Python dan jalankan:
```python
import cv2
print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())
```
Hasil yang diharapkan:
- versi OpenCV muncul
- CUDA device count ≥ 1

### 7.2 Uji Webcam
Jalankan skrip utama dan perhatikan output di layar:
```bash
python civitas_detection_cuda.py
```
Pastikan:
- kamera terbuka tanpa error
- deteksi wajah muncul pada frame
- overlay status Civitas ditampilkan
- FPS berada di kisaran harapan Jetson Nano (~10–20 FPS) atau lebih baik pada Jetson Xavier/Orin

### 7.3 Uji file video
```bash
python civitas_detection_cuda.py src/1.mp4
```
Verifikasi bahwa file diputar dan deteksi tetap berjalan.

### 7.4 Uji fallback CPU (opsional)
Jika OpenCV CUDA tidak tersedia, pakai `civitas_detection.py`:
```bash
python civitas_detection.py
```
Ini berguna untuk memastikan fungsi logika deteksi tetap berjalan pada lingkungan non-GPU.

---

## 8. Troubleshooting Umum

### 8.1 `ImportError: No module named cv2`
- Pastikan JetPack OpenCV tersedia.
- Jika menggunakan virtualenv, pastikan sudah aktif.
- Jangan install `opencv-python` jika JetPack OpenCV sudah ada; gunakan OpenCV JetPack yang sudah terpasang.

### 8.2 `cv2.cuda.getCudaEnabledDeviceCount()` menghasilkan 0
- Periksa apakah JetPack dan driver CUDA sudah terpasang.
- Pastikan `LIBTORCH` atau environment GPU Jetson tidak rusak.
- Jika Jetson tidak memiliki GPU, gunakan `civitas_detection.py` tanpa CUDA.

### 8.3 Error CuPy / `cupy.cuda.runtime.CUDARuntimeError`
- Pastikan versi CuPy sesuai versi CUDA JetPack:
  - JetPack 5.x → `cupy-cuda11x`
  - JetPack 6.x → `cupy-cuda12x`
- Jika CuPy tidak diperlukan, hapus import atau gunakan fallback CPU pada skrip.

### 8.4 FPS terlalu rendah
- Coba turunkan resolusi kamera atau ukuran ROI.
- Pastikan hanya `civitas_detection_cuda.py` yang dijalankan, bukan versi debug tertentu.
- Nonaktifkan metric logging jika tidak diperlukan.
- Periksa apakah `BrightnessAnalyzer` dan `FrameScheduler` bekerja dengan benar: kategori kecerahan harus memungkinkan interval ORB yang adaptif.

### 8.5 Kamera tidak terdeteksi
- Pastikan kamera terhubung dan dikenali oleh sistem:
```bash
v4l2-ctl --list-devices
```
- Coba `ls /dev/video*`.
- Gunakan argument index kamera yang benar, misal `0`, `1`, atau path device.

### 8.6 File `haarcascade_frontalface_default.xml` tidak ditemukan
- Pastikan folder `haarcascades/` ada di root repository.
- Jalankan dari root project, bukan dari folder lain.
- Jika perlu, modifikasi path dalam skrip agar menjadi relatif terhadap root repo.

### 8.7 Asset template logo UB tidak muncul atau error
- Pastikan folder `templates/` dan file logo UB tersedia di repo.
- Periksa nama file dan path case-sensitive jika di Jetson Linux.

---

## 9. Best Practice Deployment

- Simpan semua konfigurasi lingkungan di `requirements.txt`.
- Catat versi JetPack dan paket penting pada `README.md` atau issue tracker.
- Jalankan benchmark sederhana setiap kali deploy ke Jetson baru.
- Pastikan GitHub branch stabil (`main` atau `release`) selalu bisa di-clone dan dijalankan.
- Gunakan virtual environment agar dependensi tidak bercampur dengan sistem Jetson.
- Dokumentasikan perubahan environment Jetson apabila upgrade JetPack dilakukan.

---

## 10. Ringkasan
Deployment project Civitas Detection ke Jetson terdiri dari:
1. menyiapkan JetPack dan CUDA
2. meng-clone repo dari GitHub
3. membuat virtualenv dan menginstal dependensi
4. memastikan OpenCV CUDA tersedia
5. menjalankan `civitas_detection_cuda.py`
6. menguji kamera dan video
7. memecahkan masalah import, CUDA, dan file path

Dengan mengikuti alur ini, deployment menjadi lebih terstruktur dan dapat direproduksi oleh tim lain.
