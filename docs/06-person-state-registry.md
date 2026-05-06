# Person State Registry Documentation

## Overview

`PersonStateRegistry` adalah modul state management yang menjaga histori prediksi Civitas per user track. Ia memetakan `person_id` dari `FaceTracker` ke instance `CivitasTemporalAveraging`, sehingga output `CivitasDetector` dapat di-smoothed secara temporal dan dikembalikan bahkan ketika track berfluktuasi.

### Tujuan Utama
- **Mempertahankan state per identity**: setiap track wajah memiliki histori prediksi sendiri.
- **Menghindari fluktuasi status**: output tidak langsung berubah saat satu frame memberikan score berbeda.
- **Mendukung occlusion sementara**: state tetap tersimpan meskipun track sempat hilang.
- **Membersihkan state lama**: delete state saat ID track hilang permanen.

## Parameters dan Efeknya

| Parameter | Lokasi | Default / Tipe | Deskripsi | Efek Jika Nilai Naik | Efek Jika Nilai Turun |
|-----------|--------|----------------|-----------|----------------------|-----------------------|
| `window_size` | `PersonStateRegistry.__init__` | 30 | Ukuran buffer history frame untuk smoothing | Lebih banyak buffer, transisi status lebih stabil, latency respons naik | Lebih sedikit buffer, respons lebih cepat, status lebih fluktuatif |
| `confidence_threshold` | `PersonStateRegistry.__init__` | 0.60 | Ambang confidence untuk memutuskan status raw | Threshold lebih tinggi memerlukan bukti lebih kuat untuk `Civitas UB`, mencegah false positive | Threshold lebih rendah membuat `Civitas UB` lebih mudah tercapai, risiko false positive naik |
| `STATE_HOLD_FRAMES` | `CivitasTemporalAveraging` internal | `max(15, window_size // 5)` | Jumlah frame yang diperlukan sebelum transisi state baru diterima | Lebih besar membuat transisi lebih lambat dan stabil | Lebih kecil membuat state berubah lebih cepat dan lebih sensitif |

> Catatan: `window_size` dan `confidence_threshold` disetel di `PersonStateRegistry`, sementara `STATE_HOLD_FRAMES` adalah turunan internal di `CivitasTemporalAveraging`.

## Fungsi Inti

### `PersonStateRegistry`

#### `__init__(window_size=30, confidence_threshold=0.60)`
- Inisialisasi registry dengan parameter window size dan ambang confidence.
- Membangun dictionary `_states` untuk menyimpan objek `CivitasTemporalAveraging` per `person_id`.

#### `sync(tracker_track_ids: set)`
- Menambahkan state baru untuk `person_id` yang muncul di `tracker_tracks` tetapi belum ada di registry.
- Menghapus state untuk `person_id` yang sudah tidak ada lagi di tracker.
- Menjaga kesesuaian lifecycle: BORN → ALIVE → LOST → DEAD.

#### `feed(person_id: int, score: float, is_civitas: bool)`
- Meneruskan prediksi per-frame ke state yang sesuai.
- Hanya memasukkan data bila `person_id` aktif di registry.
- Data disimpan di buffer `CivitasTemporalAveraging` untuk smoothing.

#### `query(person_id: int)`
- Mengembalikan status smoothing terakhir dan score rata-rata.
- Bila `person_id` tidak ada, mengembalikan `("Detecting...", 0.0)`.

#### `soft_reset(person_id: int)`
- Memotong history buffer hingga sepertiga paling akhir.
- Menahan transisi status sejenak tanpa menghapus semua histori.

#### `reset_all()`
- Menghapus semua historis dan state per-person.
- Biasanya dipakai saat tidak ada wajah dalam durasi panjang.

#### `active_count`
- Properti read-only yang mengembalikan jumlah active state.

### `CivitasTemporalAveraging`

#### `__init__(window_size=30, confidence_threshold=0.60)`
- Membuat buffer `deque(maxlen=window_size)` untuk menyimpan `(score, is_civitas)`.
- Mengatur `current_state` dan counter hold internal.
- Menghitung `STATE_HOLD_FRAMES` sebagai `max(15, window_size // 5)`.

#### `add_prediction(civitas_score, is_civitas)`
- Menambahkan entry baru ke buffer.
- Entry berupa score dan boolean civitas.

#### `get_averaged_civitas()`
- Mengembalikan `("Detecting...", 0.0)` jika buffer kurang dari 5 frame.
- Menghitung weighted average dengan bobot eksponensial untuk memberi prioritas pada frame terbaru.
- Menentukan `raw` state berdasarkan `w_status` dan `w_score`.
- Menerapkan hysteresis: state hanya berubah setelah `STATE_HOLD_FRAMES` frame konsisten.

#### `soft_reset()`
- Menjaga sepertiga terakhir buffer.
- Menghapus counter hold.

#### `reset()`
- Menghapus semua history dan mengembalikan state ke awal.

## Detail Algoritma

### Buffer dan Weighted Average
- Buffer menyimpan hingga `window_size` prediksi.
- Bobot dihitung dengan `exp(0.1 * i)` untuk memberi prioritas frame terbaru.
- `w_score` adalah dot product bobot dengan nilai score.
- `w_status` adalah dot product bobot dengan flag civitas.

### Klasifikasi Raw
- `Civitas UB` jika `w_status >= 0.50` dan `w_score >= confidence_threshold`.
- `Non-Civitas UB` jika `w_status <= 0.35` dan `w_score < confidence_threshold`.
- Lainnya menjadi `Uncertain`.

### Hysteresis / Stabilitas
- `current_state` hanya berubah bila `raw` baru konsisten selama `STATE_HOLD_FRAMES` frame.
- Ini mencegah perubahan status cepat akibat satu frame noisy.

### Soft Reset vs Full Reset
- `soft_reset()` menjaga sebagian history agar transisi tidak brutal saat occlusion singkat.
- `reset()` membersihkan seluruh state saat track menghilang atau sistem restart.

## Flow Diagram

```
FaceTracker memberikan track IDs
        |
        v
PersonStateRegistry.sync(tracker_track_ids)
        |
        v
Buat/hapus CivitasTemporalAveraging sesuai tracker
        |
        v
Untuk setiap frame per track:
    PersonStateRegistry.feed(person_id, score, is_civitas)
        |
        v
  CivitasTemporalAveraging.add_prediction()
        |
        v
  get_averaged_civitas() -> weighted average
        |
        v
  Hysteresis -> state final
        |
        v
  PersonStateRegistry.query(person_id)
        |
        v
  Output status, score ke overlay
```

## Code Snippets

### `CivitasTemporalAveraging`
```python
class CivitasTemporalAveraging:
    def __init__(self, window_size=30, confidence_threshold=0.60):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)  # (score, is_civitas)
        self.current_state = None
        self.state_hold_counter = 0
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

        if w_status >= 0.50 and w_score >= self.confidence_threshold:
            raw = "Civitas UB"
        elif w_status <= 0.35 and w_score < self.confidence_threshold:
            raw = "Non-Civitas UB"
        else:
            raw = "Uncertain"

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
```

### `PersonStateRegistry`
```python
class PersonStateRegistry:
    def __init__(self, window_size=30, confidence_threshold=0.60):
        self._window_size    = window_size
        self._conf_threshold = confidence_threshold
        self._states: dict   = {}

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
```

## Integration Notes

- `PersonStateRegistry` adalah jembatan antara `FaceTracker` dan `CivitasDetector`.
- `sync()` harus dipanggil setiap frame agar registry hanya menyimpan ID lain yang masih ada.
- `feed()` hanya berjalan untuk track yang visible sehingga history tidak tercampur dengan ghost track.
- `query()` memberikan status smoothed untuk overlay UI.
- `soft_reset()` cocok untuk occlusion singkat, sedangkan `reset_all()` cocok saat tidak ada wajah lagi.

## Professional Documentation Summary

`PersonStateRegistry` menjaga histori prediksi Civitas per face track dan mendukung stabilitas output lewat weighted temporal smoothing. Dengan parameter `window_size` dan `confidence_threshold`, tim dapat melakukan tuning antara latency respons dan ketahanan terhadap fluktuasi deteksi.
