# Civitas Detection

## Ringkasan Proyek
`Civitas Detection` adalah sistem deteksi real-time yang dirancang untuk mengidentifikasi status `Civitas UB` dari wajah dan atribut pakaian pada video atau input kamera. Proyek ini fokus pada:
- deteksi multi-wajah
- pelacakan wajah dengan ID konsisten
- analisis logo dan warna pakaian
- adaptasi beban komputasi berdasarkan kondisi frame
- dukungan NVIDIA Jetson dengan fallback CPU

## Tujuan
Sistem ini dibuat untuk membantu tim memantau dan mengklasifikasikan kehadiran civitas di lingkungan kampus atau event, dengan stabilitas status yang ditingkatkan melalui smoothing temporal dan optimasi performa untuk platform embedded.

## Fitur Utama
- Deteksi wajah multi-face menggunakan Haar Cascade
- Face tracking dengan ID yang konsisten
- Deteksi `Civitas UB` via logo dan analisis warna
- Adaptive scheduling untuk operasi berat seperti ORB
- Monitoring FPS dan performa
- Dokumentasi deployment khusus Jetson

## Struktur Utama Repository
- `civitas_detection.py` — versi CPU / fallback
- `civitas_detection_cuda.py` — versi Jetson/CUDA
- `requirements.txt` — daftar dependensi Python
- `docs/` — dokumentasi teknis dan deployment
- `haarcascades/` — model deteksi wajah
- `templates/` — aset template logo
- `metrics/` — output metrik dan benchmark

## Daftar Isi Dokumentasi
Berikut adalah dokumen referensi utama yang tersedia di folder `docs/`:

- [01-architecture.md](docs/01-architecture.md) — Arsitektur sistem dan komponen utama
- [02-face-tracker.md](docs/02-face-tracker.md) — Detil algoritma pelacakan wajah
- [03-brightness-analyzer.md](docs/03-brightness-analyzer.md) — Analisis kecerahan frame untuk adaptasi beban
- [04-frame-scheduler.md](docs/04-frame-scheduler.md) — Penjadwalan eksekusi operasi berat
- [05-civitas-detector.md](docs/05-civitas-detector.md) — Logika deteksi Civitas UB
- [06-person-state-registry.md](docs/06-person-state-registry.md) — Manajemen histori status per orang
- [07-fps-monitor.md](docs/07-fps-monitor.md) — Monitoring performa dan frame rate
- [08-deployment-jetson.md](docs/08-deployment-jetson.md) — Panduan deployment ke NVIDIA Jetson

## Cara Memulai
1. Clone repository ini.
2. Buat dan aktifkan virtual environment Python.
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan versi CUDA (Jetson):
   ```bash
   python civitas_detection_cuda.py
   ```
5. Jalankan versi CPU jika CUDA tidak tersedia:
   ```bash
   python civitas_detection.py
   ```

## Catatan Deployment
Untuk deployment di platform Jetson, ikuti panduan lengkap pada [docs/08-deployment-jetson.md](docs/08-deployment-jetson.md). Dokumen tersebut berisi instruksi persiapan JetPack, instalasi dependensi, dan pengujian kamera.

## Kontak dan Pengembangan
Gunakan dokumentasi di `docs/` sebagai panduan teknis untuk memahami setiap modul. Tambahkan issue atau commit baru ketika menambahkan fitur atau memperbaiki pipeline deteksi.
