# Dimension Reduction dengan Autoencoder — Overhead-MNIST (kelas `ship`)

Proyek ini melakukan **dimension reduction** pada citra satelit kapal (`ship`) berukuran 28×28 grayscale dari dataset [Overhead-MNIST](https://www.kaggle.com/datasets/datamunge/overheadmnist/data) menggunakan **Autoencoder** (784 dimensi → latent space). Dua arsitektur dibandingkan: baseline (sesuai diagram soal) dan versi improved dengan tuning hyperparameter. Kualitas rekonstruksi dievaluasi dengan SSIM, PSNR, dan MSE.

## Dataset & Preprocessing

- Sumber: Overhead-MNIST (versi 2), kelas **`ship`**. Data `train` + `test` digabung menjadi 8.012 citra (7.124 + 888).
- **Cek duplikat** dengan hash MD5 dari isi piksel: ditemukan **12 duplikat (0,15%)** dan dihapus, sehingga tersisa **8.000 citra**.
- **Scaling** piksel ke `[0, 1]` (dibagi 255), karena output decoder memakai `sigmoid`. Data di-reshape ke `(N, 28, 28, 1)`.
- **Split 80/10/10** dengan `train_test_split` dua tahap (`random_state=42`):

| Train | Validation | Test |
|---|---|---|
| 6.400 | 800 | 800 |

Autoencoder bersifat unsupervised, jadi label tidak dipakai saat training.

## Model

### Baseline (sesuai diagram)

- **Encoder:** `Conv2D(32, 3×3)` → `MaxPooling2D(2×2)` → `Flatten` (6272) → `Dense(128)` sebagai latent
- **Decoder:** `Dense(6272)` → `Reshape(14,14,32)` → `UpSampling2D(2×2)` → `Conv2D(32, 3×3)` → `Conv2D(1, 3×3, sigmoid)`
- Loss `binary_crossentropy`, Adam `lr=1e-3`, `batch_size=32`, 30 epoch

### Improved

| Aspek | Baseline | Improved |
|---|---|---|
| Downsampling | `MaxPooling2D` (28→14) | `Conv2D` strided, 2 tahap (28→14→7) |
| Upsampling | `UpSampling2D` (tidak dilatih) | `Conv2DTranspose` strided (trainable) |
| Normalisasi | — | `BatchNormalization` setelah tiap Conv |
| Aktivasi | `ReLU` | `LeakyReLU(0.2)` |
| Regularisasi | — | `Dropout` (rate ikut di-tuning) |
| Latent dim | Tetap 128 | Di-tuning |
| Early stopping | — | `EarlyStopping(monitor='val_loss', patience=7)` |

### Tuning hyperparameter (grid search, 25 epoch per kandidat)

Grid: `learning_rate` ∈ {1e-3, 5e-4, 1e-4} × `latent_dim` ∈ {32, 64, 128} × `dropout_rate` ∈ {0.0, 0.2, 0.3} (27 kombinasi), dipilih berdasarkan `val_loss` terendah.

| Peringkat | Learning rate | Latent dim | Dropout | Val loss |
|---|---|---|---|---|
| **1** | **1e-3** | **128** | **0.2** | **0.4496** |
| 2 | 1e-3 | 128 | 0.3 | 0.4504 |
| 3 | 5e-4 | 128 | 0.2 | 0.4506 |

Latent dim 128 konsisten lebih baik daripada 64 dan 32 di semua kombinasi. Konfigurasi terbaik lalu dilatih ulang dari awal selama maksimal 100 epoch (`batch_size=32`).

## Hasil (Test Set, 800 citra)

| Model | Latent dim | Test loss (BCE) | SSIM ↑ | PSNR (dB) ↑ | MSE ↓ |
|---|---|---|---|---|---|
| Baseline | 128 | 0.5146 | 0.5298 | 13.58 | 0.04605 |
| **Improved** | 128 | **0.4423** | **0.7957** | **17.31** | **0.01985** |

SSIM naik dari 0.53 menjadi 0.80, PSNR naik sekitar 3,7 dB, dan MSE turun lebih dari separuh. Secara visual, rekonstruksi improved memiliki garis objek yang lebih tajam dibanding baseline yang cenderung blur.

## Kesimpulan

- Layer downsampling/upsampling yang bisa dilatih (`Conv2D` strided / `Conv2DTranspose`) memungkinkan model belajar cara meringkas dan mengembalikan informasi, tidak seperti `MaxPooling`/`UpSampling` yang bersifat tetap.
- Kurva loss train dan validation menurun stabil tanpa tanda overfitting.

## Keterbatasan

- Pada training final, jumlah epoch mencapai batas 100, jadi `EarlyStopping` tidak terpicu. Menambah jumlah epoch bisa menunjukkan apakah early stopping benar-benar bekerja.
- Grid search memakai 25 epoch per kandidat, sehingga peringkat kandidat yang selisihnya sangat kecil (misalnya 0.4496 vs 0.4504) belum tentu bermakna.
- Latent dim 128 berada di batas atas grid, jadi nilai yang lebih besar belum diuji.
