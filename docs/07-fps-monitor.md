# FPS Monitor Documentation

## Overview

`FPSMonitor` adalah komponen observability yang mengukur dan menyimpan statistik performa frame dalam sistem deteksi Civitas. Modul ini bertanggung jawab untuk menghitung nilai FPS yang dismooth, statistik frame time, dan stability score yang dapat digunakan untuk monitoring runtime dan tuning performa.

### Tujuan Utama
- **Melacak kecepatan frame** secara real-time.
- **Memberikan statistik historis** untuk analisis performa jangka pendek.
- **Menghitung stabilitas frame** berdasarkan deviasi waktu antar frame.
- **Mendukung UI overlay** untuk menampilkan runtime metrics.

## Parameters dan Efeknya

| Parameter | Lokasi | Default / Tipe | Deskripsi | Efek Jika Nilai Naik | Efek Jika Nilai Turun |
|-----------|--------|----------------|-----------|----------------------|-----------------------|
| `buffer_size` | `FPSMonitor.__init__` | 100 | Jumlah frame terakhir yang disimpan untuk statistik | Statistik lebih stabil dan historis lebih panjang; memori bertambah | Statistik lebih responsif terhadap perubahan; memori berkurang |
| `alpha` | `FPSMonitor.__init__` | 0.1 | Faktor smoothing untuk `smoothed_fps` | Smoothing lebih kuat; nilai FPS berubah lebih lambat | Smoothing lebih lemah; nilai FPS lebih responsif |
| `fps_buffer.maxlen` | derived | 100 | Batas jumlah sampel FPS yang disimpan | Lebih banyak sampel memperhalus rata-rata; lebih butuh memori | Lebih sedikit sampel membuat rata-rata lebih variatif |
| `frame_times.maxlen` | derived | 100 | Batas jumlah timestamps frame | Lebih banyak data meningkatkan akurasi stability score | Lebih sedikit data membuat stability score lebih volatile |

> Catatan: `buffer_size` terhubung langsung ke window statistik, sedangkan `alpha` mengontrol respons per-frame untuk FPS yang ditampilkan.

## Fungsi Inti

### `__init__(buffer_size=100)`
- Membuat buffer internal untuk FPS dan frame times.
- Mengatur `start_time` untuk hitung runtime total.
- Menyimpan counter `frame_count` dan `smoothed_fps`.
- Menetapkan smoothing factor `alpha`.

### `update(fps)`
- Update `smoothed_fps` dengan Exponential Moving Average (EMA):
  - jika `smoothed_fps` = 0.0, inisialisasi langsung dengan nilai `fps`
  - jika tidak, `smoothed_fps = alpha * fps + (1 - alpha) * smoothed_fps`
- Menambahkan nilai `smoothed_fps` ke `fps_buffer`.
- Menambahkan timestamp saat ini ke `frame_times`.
- Menambah `frame_count`.

### `get_stats()`
- Mengembalikan `None` bila buffer belum memiliki 10 sampel.
- Menghitung rata-rata, min, max FPS dari `fps_buffer`.
- Menghitung rata-rata frame time dan stability score berdasarkan deviasi antar timestamp.
- Mengembalikan dictionary:
  - `avg_fps`
  - `min_fps`
  - `max_fps`
  - `current_fps`
  - `avg_frame_time_ms`
  - `stability_score`
  - `total_frames`
  - `runtime_seconds`

## Detail Algoritma

### Exponential Moving Average
- EMA memberikan FPS yang lebih halus dibanding nilai mentah per-frame.
- Rumus: `smoothed_fps = alpha * fps + (1 - alpha) * smoothed_fps`.
- `alpha=0.1` membuat `smoothed_fps` bereaksi secara perlahan terhadap lonjakan FPS.

### Statistik Frame Time
- `frame_times` menyimpan timestamps setiap update.
- `time_diffs` dihitung antar timestamp berturut-turut.
- `avg_frame_time_ms` adalah rata-rata durasi frame dalam milidetik.
- `stability_score` dihitung sebagai `max(0, 100 - (frame_time_std * 1000))`.
- Score ini menurunkan nilai jika variasi waktu antar frame tinggi.

### Stabilitas
- `stability_score` mendeteksi apakah frame time konsisten.
- Nilai mendekati 100 mengindikasikan frame time stabil.
- Nilai rendah menandakan jitter atau variasi rendering.

## Flow Diagram

```
Frame selesai dirender
        |
        v
Hitung instantaneous FPS
        |
        v
Panggil FPSMonitor.update(fps)
        |
        v
Update smoothed_fps (EMA)
        |
        v
Simpan smoothed_fps ke fps_buffer
        |
        v
Simpan timestamp ke frame_times
        |
        v
get_stats() -> hitung avg/min/max, avg_frame_time_ms, stability_score
        |
        v
Return metrics untuk overlay UI
```

## Code Snippets

```python
class FPSMonitor:
    def __init__(self, buffer_size=100):
        self.fps_buffer = deque(maxlen=buffer_size)
        self.frame_times = deque(maxlen=buffer_size)
        self.start_time = time.time()
        self.frame_count = 0
        self.smoothed_fps = 0.0
        self.alpha = 0.1  # Smoothing factor (lower = smoother)

    def update(self, fps):
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

        if len(self.frame_times) >= 2:
            time_diffs = []
            for i in range(1, len(self.frame_times)):
                time_diffs.append(self.frame_times[i] - self.frame_times[i-1])

            avg_frame_time = statistics.mean(time_diffs) if time_diffs else 0
            frame_time_std = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0
            stability_score = max(0, 100 - (frame_time_std * 1000))
        else:
            avg_frame_time = 0
            stability_score = 0

        return {
            'avg_fps': statistics.mean(fps_list),
            'min_fps': min(fps_list),
            'max_fps': max(fps_list),
            'current_fps': self.smoothed_fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'stability_score': stability_score,
            'total_frames': self.frame_count,
            'runtime_seconds': time.time() - self.start_time
        }
```

## Integration Notes

- `FPSMonitor` dipanggil setiap frame dari loop utama setelah FPS dihitung.
- Data `get_stats()` ideal untuk overlay panel dan logging performa.
- Buffer size dan alpha bisa di-tune untuk trade-off antara kestabilan statistik dan responsif terhadap perubahan beban.

## Professional Documentation Summary

`FPSMonitor` memberikan metrik performa real-time yang penting untuk debugging dan tuning sistem Civitas. Desainnya sederhana namun efektif, memisahkan penghimpunan data FPS dari logika deteksi utama sehingga monitoring tidak mengganggu pipeline utama.
