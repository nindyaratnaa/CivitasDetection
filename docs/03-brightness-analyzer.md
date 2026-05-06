# Brightness Analyzer Documentation

## Overview

`BrightnessAnalyzer` adalah komponen yang bertanggung jawab untuk analisis kecerahan frame secara real-time dalam sistem deteksi Civitas UB. Modul ini menghitung rata-rata nilai grayscale frame dengan smoothing menggunakan Exponential Moving Average (EMA) untuk mencegah fluktuasi kategori yang terlalu cepat akibat noise seperti flash atau refleksi cahaya.

### Tujuan Utama
- **Adaptasi Beban Komputasi**: Menentukan kategori kecerahan (Dark/Normal/Bright) untuk mengatur frekuensi dan intensitas operasi ORB dan deteksi wajah.
- **Stabilitas Sistem**: Smoothing EMA mencegah perubahan kategori yang drastis antar frame.
- **Optimasi Performa**: Mengurangi beban GPU/CPU pada kondisi gelap dengan mengurangi frekuensi ORB matching.

### Kategori Kecerahan
| Kategori | Rentang Mean Gray | Dampak pada Pipeline |
|----------|-------------------|----------------------|
| Dark     | < 60              | ORB setiap 1 frame, 800 fitur, ROI 120px, deteksi setiap 1 frame |
| Normal   | 60-179            | ORB setiap 1 frame, 1000 fitur, ROI 160px, deteksi setiap 5 frame |
| Bright   | >= 180            | ORB setiap 1 frame, 1300 fitur, ROI 180px, deteksi setiap 1 frame |

### Mekanisme Smoothing
- Menggunakan EMA dengan alpha 0.15 (lebih kecil = lebih stabil, lebih lambat bereaksi)
- Mencegah kategori berubah-ubah karena frame noise (flash, bayangan)

## Parameters

Berikut adalah parameter konfigurasi utama BrightnessAnalyzer beserta dampak perubahan nilai:

| Parameter | Default Value | Deskripsi | Efek Jika Nilai Naik | Efek Jika Nilai Turun |
|-----------|---------------|-----------|----------------------|-----------------------|
| `EMA_ALPHA` | 0.15 | Faktor smoothing EMA (0-1, lebih kecil = lebih stabil) | Smoothing lebih kuat, kategori lebih stabil tapi lambat adaptasi cahaya | Smoothing lebih lemah, kategori lebih responsif tapi rentan noise |
| `BRIGHT_DARK` | 60 | Threshold bawah untuk kategori Dark | Lebih banyak frame dikategorikan Normal/Bright, beban komputasi meningkat | Lebih banyak frame dikategorikan Dark, efisiensi lebih baik tapi akurasi turun |
| `BRIGHT_NORMAL` | 180 | Threshold atas untuk kategori Normal | Lebih banyak frame dikategorikan Normal, transisi ke Bright lebih awal | Lebih banyak frame dikategorikan Bright, beban komputasi maksimal |
| `ORB_EVERY_N['Dark']` | 1 | Frekuensi ORB pada kondisi Dark (frame ke-) | ORB lebih jarang, efisiensi lebih baik tapi akurasi turun | ORB lebih sering, akurasi lebih baik tapi beban naik |
| `ORB_EVERY_N['Normal']` | 1 | Frekuensi ORB pada kondisi Normal | ORB lebih jarang, efisiensi lebih baik tapi akurasi turun | ORB lebih sering, akurasi lebih baik tapi beban naik |
| `ORB_EVERY_N['Bright']` | 1 | Frekuensi ORB pada kondisi Bright | ORB lebih jarang, efisiensi lebih baik tapi akurasi turun | ORB lebih sering, akurasi lebih baik tapi beban naik |
| `ORB_NFEATURES['Dark']` | 800 | Jumlah fitur ORB pada kondisi Dark | Deteksi lebih akurat, beban komputasi naik | Deteksi kurang akurat, efisiensi lebih baik |
| `ORB_NFEATURES['Normal']` | 1000 | Jumlah fitur ORB pada kondisi Normal | Deteksi lebih akurat, beban komputasi naik | Deteksi kurang akurat, efisiensi lebih baik |
| `ORB_NFEATURES['Bright']` | 1300 | Jumlah fitur ORB pada kondisi Bright | Deteksi lebih akurat, beban komputasi naik | Deteksi kurang akurat, efisiensi lebih baik |
| `ORB_ROI_SIZE['Dark']` | 120 | Ukuran ROI ORB pada kondisi Dark (px) | ROI lebih besar, deteksi lebih luas tapi beban naik | ROI lebih kecil, efisiensi lebih baik tapi akurasi turun |
| `ORB_ROI_SIZE['Normal']` | 160 | Ukuran ROI ORB pada kondisi Normal (px) | ROI lebih besar, deteksi lebih luas tapi beban naik | ROI lebih kecil, efisiensi lebih baik tapi akurasi turun |
| `ORB_ROI_SIZE['Bright']` | 180 | Ukuran ROI ORB pada kondisi Bright (px) | ROI lebih besar, deteksi lebih luas tapi beban naik | ROI lebih kecil, efisiensi lebih baik tapi akurasi turun |
| `DETECT_EVERY_N['Dark']` | 1 | Frekuensi deteksi wajah pada kondisi Dark | Deteksi lebih jarang, efisiensi lebih baik tapi tracking kurang stabil | Deteksi lebih sering, tracking lebih stabil tapi beban naik |
| `DETECT_EVERY_N['Normal']` | 5 | Frekuensi deteksi wajah pada kondisi Normal | Deteksi lebih jarang, efisiensi lebih baik tapi tracking kurang stabil | Deteksi lebih sering, tracking lebih stabil tapi beban naik |
| `DETECT_EVERY_N['Bright']` | 1 | Frekuensi deteksi wajah pada kondisi Bright | Deteksi lebih jarang, efisiensi lebih baik tapi tracking kurang stabil | Deteksi lebih sering, tracking lebih stabil tapi beban naik |

## Core Functionality

### Metode Utama

#### `__init__()`
- Inisialisasi state internal
- `self._smoothed = 128.0` (nilai netral grayscale)
- `self.category = 'Normal'` (kategori awal)
- `self.raw_value = 128.0` (nilai raw terakhir)

#### `update(gray_frame)`
- **Input**: `gray_frame` (np.ndarray atau cv2.cuda.GpuMat)
- **Proses**:
  1. Hitung mean grayscale (menggunakan CUDA jika tersedia)
  2. Update EMA smoothing
  3. Klasifikasi kategori berdasarkan nilai smoothed
- **Output**: Update state internal (tidak return value)

#### `_classify(value: float) -> str`
- Klasifikasi kategori berdasarkan threshold
- Return: 'Dark', 'Normal', atau 'Bright'

### Properties
- `orb_every_n`: Frekuensi eksekusi ORB berdasarkan kategori
- `orb_nfeatures`: Jumlah fitur ORB berdasarkan kategori
- `orb_roi_size`: Ukuran ROI ORB berdasarkan kategori
- `detect_every_n`: Frekuensi deteksi wajah berdasarkan kategori

## Algorithm Details

### Perhitungan Mean Grayscale
- **CPU**: `np.mean(gray_frame)`
- **CUDA**: `cv2.cuda.meanStdDev(gray_frame)[0]` untuk GPU acceleration
- Mendukung kedua mode untuk fleksibilitas hardware

### Exponential Moving Average
- Formula: `smoothed = alpha * raw + (1 - alpha) * smoothed`
- Alpha = 0.15 memberikan smoothing yang cukup tanpa terlalu lambat
- Mencegah kategori berubah drastis akibat frame outlier

### Threshold Classification
- Dark: < 60 (frame gelap, tingkatkan akurasi deteksi)
- Normal: 60-179 (kondisi standar)
- Bright: >= 180 (frame terang, tingkatkan frekuensi untuk stabilitas)

### Dampak pada Pipeline
BrightnessAnalyzer mempengaruhi:
- **FrameScheduler**: Mengatur kapan ORB dijalankan
- **CivitasDetector**: Mengatur parameter ORB (fitur, ROI size)
- **Face Detection**: Mengatur frekuensi Haar Cascade

## Usage Examples

### Basic Usage
```python
brightness = BrightnessAnalyzer()

# Dalam loop utama
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
brightness.update(gray)

print(f"Kategori: {brightness.category}")
print(f"ORB setiap {brightness.orb_every_n} frame")
```

### Integration dengan Scheduler
```python
scheduler = FrameScheduler()
brightness = BrightnessAnalyzer()

# Update brightness setiap frame
brightness.update(gray_frame)

# Gunakan untuk kontrol ORB
if scheduler.should_run_orb(track_id, chest_gray, brightness):
    # Jalankan ORB matching
    pass
```

## Performance Considerations

### Kompleksitas
- **Time**: O(1) per frame (mean calculation + EMA)
- **Space**: O(1) (hanya beberapa float)
- **GPU**: Mendukung CUDA untuk mean calculation

### Optimasi
- EMA smoothing mengurangi false positive kategori
- Threshold adaptif berdasarkan kondisi lighting
- Fallback CPU jika CUDA tidak tersedia

### Trade-offs
- **Stabilitas vs Responsivitas**: Alpha tinggi = responsif tapi noise
- **Akurasi vs Efisiensi**: Parameter tinggi = akurat tapi mahal
- **Adaptasi vs Konsistensi**: Threshold ketat = konsisten tapi kurang adaptif

### Monitoring
State dapat diakses untuk debugging:
- `brightness.category`: Kategori current
- `brightness.raw_value`: Mean grayscale terakhir
- `brightness._smoothed`: Nilai smoothed

## Code Snippets

Berikut adalah snippet kode utama dari implementasi BrightnessAnalyzer di `civitas_detection_cuda.py`:

### Definisi Kelas
```python
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
```

### Metode Update
```python
def update(self, gray_frame):
    """Call once per frame — accepts np.ndarray or cv2.cuda.GpuMat."""
    if USE_CUDA and isinstance(gray_frame, cv2.cuda.GpuMat):
        mean, _ = cv2.cuda.meanStdDev(gray_frame)
        self.raw_value = float(mean[0])
    else:
        self.raw_value = float(np.mean(gray_frame))
    self._smoothed  = self.EMA_ALPHA * self.raw_value + (1 - self.EMA_ALPHA) * self._smoothed
    self.category   = self._classify(self._smoothed)
```

### Metode Klasifikasi
```python
def _classify(self, value: float) -> str:
    if value < Config.BRIGHT_DARK:
        return 'Dark'
    if value < Config.BRIGHT_NORMAL:
        return 'Normal'
    return 'Bright'
```

### Properties
```python
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
```