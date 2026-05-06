# Frame Scheduler Documentation

## Overview

`FrameScheduler` adalah komponen kontrol yang mengatur kapan operasi ORB yang mahal dijalankan pada setiap track wajah. Pada `civitas_detection_cuda.py`, scheduler ini bekerja bersama `BrightnessAnalyzer` untuk menyeimbangkan performa dan kualitas deteksi dengan memanfaatkan interval eksekusi berbasis kategori kecerahan.

### Tujuan Utama
- **Mengurangi beban komputasi** dengan menunda ORB matching pada track yang sama sampai interval yang tepat.
- **Menjaga responsif** dengan memprioritaskan ORB pada frame awal saat track baru terbentuk.
- **Menjaga stabilitas** melalui counter track-specific, sehingga setiap wajah memiliki jadwal eksekusi sendiri.

## Parameter dan Efeknya

| Parameter | Lokasi | Default / Tipe | Deskripsi | Efek Jika Nilai Naik | Efek Jika Nilai Turun |
|-----------|--------|----------------|-----------|----------------------|-----------------------|
| `Config.ORB_EVERY_N['Dark']` | `Config` | 1 | Interval ORB untuk kategori Dark | ORB lebih jarang, efisiensi naik, akurasi logo turun | ORB lebih sering, kualitas naik, beban GPU/CPU naik |
| `Config.ORB_EVERY_N['Normal']` | `Config` | 1 | Interval ORB untuk kategori Normal | Sama seperti di atas |
| `Config.ORB_EVERY_N['Bright']` | `Config` | 1 | Interval ORB untuk kategori Bright | Sama seperti di atas |
| `BrightnessAnalyzer.orb_every_n` | `BrightnessAnalyzer` property | Dinamis | Nilai interval ORB saat ini berdasarkan kategori kecerahan | Semakin besar, ORB dijalankan lebih jarang | Semakin kecil, ORB dijalankan lebih sering |
| `FrameScheduler._orb_counters` | `FrameScheduler` internal | dict | Counter track-specific untuk interval ORB | Tidak langsung, tetapi jika counter melejit, ORB dijalankan jarang sekali | Jika di-reset sering, ORB jalankan lebih banyak |
| `Config.ORB_NFEATURES[...]` | `Config` | 800 / 1000 / 1300 | Jumlah fitur ORB per kategori | Perframe ORB lebih mahal tapi lebih banyak deteksi | Perframe ORB lebih ringan tapi bisa kehilangan match |
| `Config.ORB_ROI_SIZE[...]` | `Config` | 120 / 160 / 180 | Ukuran ROI untuk ORB matching | Area pencocokan lebih luas, lebih mahal | Area lebih sempit, lebih cepat |
| `Config.DETECT_EVERY_N[...]` | `Config` | 1 / 5 / 1 | Frekuensi Haar cascade deteksi wajah | Deteksi lebih jarang, efisiensi naik, responsif turun | Deteksi lebih sering, responsif naik, beban lebih besar |

> Catatan: `FrameScheduler` sendiri tidak menyimpan parameter interval global. Ia bergantung pada nilai dinamis dari `BrightnessAnalyzer` dan `Config.ORB_EVERY_N`.

## Core Functionality

### `__init__()`
- Inisialisasi scheduler dengan counter internal:
  - `_detect_counter`: reservasi untuk deteksi kasar (tidak dipakai secara eksplisit di versi saat ini)
  - `_orb_counters`: dictionary track-specific untuk menjadwalkan ORB

### `should_run_orb(track_id, chest_gray, brightness)`
- **Input**:
  - `track_id`: ID unik track wajah
  - `chest_gray`: chest region grayscale (opsional untuk logika berikutnya)
  - `brightness`: instance `BrightnessAnalyzer`
- **Proses**:
  1. Ambil interval `every_n = brightness.orb_every_n`
  2. Ambil counter track dari `_orb_counters`; jika tidak ada, mulai dari `every_n`
  3. Increment counter
  4. Jika counter < every_n, simpan dan kembalikan `False`
  5. Jika counter >= every_n, reset counter ke 0 dan kembalikan `True`
- **Output**: `True` bila ORB seharusnya dijalankan untuk track ini pada frame saat ini.

### `remove(track_id)`
- Menghapus counter ORB untuk `track_id` yang tidak lagi aktif.
- Mencegah akumulasi state lama saat track dihapus atau hilang.

## Behavior and Workflow

1. Ketika track baru dibuat, `_orb_counters` tidak memiliki entry untuk `track_id`.
2. Pada pemanggilan pertama `should_run_orb()`, counter akan diset ke `every_n`, lalu diincrement, menghasilkan nilai `every_n + 1`.
3. Karena `counter >= every_n`, ORB akan dijalankan pada frame pertama untuk track baru, kemudian counter direset.
4. Setelah itu, ORB hanya dijalankan ulang setelah `every_n` panggilan lagi.
5. Jika `every_n` besar, ORB akan dijalankan lebih jarang sehingga meningkatkan performa, tetapi bisa menurunkan akurasi matching.
6. `remove(track_id)` digunakan saat track hilang untuk membersihkan state dan menghindari penggunaan counter usang.

## Code Snippets

Berikut adalah potongan kode utama dari implementasi `FrameScheduler` di `civitas_detection_cuda.py`:

### Definisi Kelas dan Constructor
```python
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
```

### ORB Execution Control
```python
def should_run_orb(self, track_id: int, chest_gray: np.ndarray, brightness: BrightnessAnalyzer) -> bool:
    every_n = brightness.orb_every_n
    counter = self._orb_counters.get(track_id, every_n)
    counter += 1
    self._orb_counters[track_id] = counter
    if counter < every_n:
        return False
    self._orb_counters[track_id] = 0
    return True
```

### Cleanup Method
```python
def remove(self, track_id: int):
    self._orb_counters.pop(track_id, None)
```

## Integration Notes

- `FrameScheduler` adalah bagian kunci dari loop utama untuk menjaga beban ORB tetap terkendali.
- Meskipun class saat ini tidak memanggil `should_detect()` secara eksplisit, ia dioptimalkan untuk menjadi titik tunggal kontrol frekuensi operasi mahal.
- Pengaturan `Config.DETECT_EVERY_N` dan `BrightnessAnalyzer` bekerja bersama `FrameScheduler` untuk memberikan adaptasi beban secara end-to-end.

## Flow Diagram

Diagram berikut memperlihatkan bagaimana `FrameScheduler` menentukan kapan ORB dijalankan untuk setiap track:

```
Track ID masuk ke FrameScheduler
        |
        v
Ambil interval every_n dari BrightnessAnalyzer
        |
        v
Ambil counter track dari _orb_counters
        |
        v
Counter += 1
        |
        v
Counter < every_n ?
   /           \
  v             v
False         True
  |             |
  v             v
Reset counter  Simpan counter
kembali ke 0   kembali
  |             |
  v             v
Return True   Return False
```

## Professional Documentation Summary

`FrameScheduler` adalah mekanisme pengaturan beban komputasi yang memisahkan jadwal eksekusi ORB per track dari logika deteksi wajah. Desain ini menjadikan sistem lebih scalable pada target hardware Jetson Nano, karena tidak semua track mengeksekusi ORB secara penuh setiap frame. Dengan menambah atau mengurangi interval `every_n`, Anda dapat men-tune trade-off antara performa dan akurasi deteksi secara granular.
