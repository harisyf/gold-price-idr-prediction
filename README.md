# Machine Learning Project Report - Haris Yafie

A project for my submission of my learning in Dicoding Machine Learning Terapan course. This project is about prediction on global gold price to IDR. The data used spans from 24 April 2023 until 22 April 2025.

## Global Gold Price to IDR Prediction

Harga emas dunia merupakan salah satu aset yang banyak digunakan sebagai lindung nilai terhadap inflasi dan ketidakstabilan ekonomi global. Di Indonesia, pergerakan harga emas dalam Rupiah (IDR) sangat dipengaruhi oleh dinamika global seperti kebijakan moneter, fluktuasi kurs, dan krisis ekonomi dunia.

Namun, memprediksi harga emas terhadap IDR bukanlah hal yang mudah karena sifatnya yang sangat volatil dan dipengaruhi oleh berbagai variabel eksternal. Oleh karena itu, proyek ini bertujuan membangun model machine learning berbasis time series yang dapat memprediksi harga emas dalam IDR secara akurat, dengan memanfaatkan data historis.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Masalah ini penting untuk diselesaikan karena dapat membantu individu, pelaku pasar, dan investor dalam membuat keputusan keuangan yang lebih terinformasi.
- Referensi:
  - [World Gold Council - Gold Market Insights](https://www.gold.org/)
  - [IMF Global Economic Outlook](https://www.imf.org/)
  - [Investing.com - GAU/IDR](https://id.investing.com/currencies/gau-idr-historical-data)

## Business Understanding

### Problem Statements
- Sulitnya memprediksi harga emas terhadap Rupiah karena tingginya volatilitas dan pengaruh faktor eksternal seperti kurs USD, inflasi global, dan ketegangan geopolitik.
- Peningkatan permintaan terhadap emas yang menyebabkan kenaikan harga emas secara drastis

### Goals
- Membangun model machine learning untuk memprediksi harga emas dalam IDR berdasarkan data historis.
- Menggunakan pendekatan berbasis deep learning yang mampu memahami pola temporal dari pergerakan harga emas.
- Membandingkan performa model LSTM dan GRU sebagai algoritma time-series terbaik untuk kasus ini.

### Solution Statements
- Model 1: LSTM Neural Network — memanfaatkan kemampuan untuk mengingat pola jangka panjang pada data sekuensial.
- Model 2: GRU Neural Network — alternatif yang lebih ringan dan efisien dengan performa yang sebanding.
- Kinerja masing-masing model akan dievaluasi menggunakan metrik RMSE dan MAE untuk menentukan model terbaik.

## Data Understanding

Data yang digunakan dalam proyek ini diambil dari [Investing.com - GAU/IDR](https://id.investing.com/currencies/gau-idr-historical-data), mencakup periode 24 April 2023 hingga 22 April 2025. Data ini merepresentasikan harga emas dunia yang dikonversikan ke dalam Rupiah Indonesia (IDR).

Dataset mencakup kolom-kolom berikut (berdasarkan struktur umumnya dari Investing.com):
- Tanggal: tanggal pengamatan
- Terakhir: harga terakhir emas pada hari tersebut
- Open: harga pembukaan
- High: harga tertinggi dalam sehari
- Low: harga terendah dalam sehari
- Change %: persentase perubahan harga
Namun yang digunakan pada proyek ini yaitu 'Tanggal' dan 'Terakhir'

**Exploratory Data Analysis**:
**Statistika Deskriptif**
### 📊 Descriptive Statistics for Closing Gold Price (IDR)

| Metric           | Value           |
|------------------|-----------------|
| Count            | 527             |
| Mean             | 1,178,196       |
| Std Dev          | 217,310         |
| Min              | 912,144         |
| Q1 (25%)         | 975,623         |
| Median (50%)     | 1,201,496       |
| Q3 (75%)         | 1,331,908       |
| Max              | 1,872,926       |


Dataset ini berisi 527 data harian harga emas dunia dalam Rupiah Indonesia (IDR), dari April 2023 hingga April 2025. Berikut adalah interpretasi dari statistik deskriptifnya:
1. Rata-rata (Mean): Harga penutupan emas berada di kisaran Rp1.178.196, yang mencerminkan tren umum selama dua tahun terakhir.
2. Standar Deviasi (Std Dev): Sebesar Rp217.310, menunjukkan adanya fluktuasi yang cukup signifikan (~18%), wajar untuk instrumen komoditas seperti emas.
3. Harga Minimum: Harga terendah tercatat sebesar Rp912.144, kemungkinan saat terjadi penurunan pasar.
4. Harga Maksimum: Harga tertinggi mencapai Rp1.872.926, menunjukkan adanya lonjakan kuat di waktu tertentu.
5. Kuartil:
     - Q1 (25%): 25% data berada di bawah Rp975.623
     - Median (50%): Titik tengah berada di Rp1.201.496, sedikit lebih tinggi dari rata-rata.
     - Q3 (75%): 75% data berada di bawah Rp1.331.908
6. Distribusi harga terlihat sedikit condong ke kanan (right-skewed), namun tetap relatif seimbang. Ini baik untuk modeling karena tidak terlalu ekstrem.

Visualisasi awal yang dilakukan:
- Plot tren harga emas IDR sepanjang 2 tahun
- Korelasi antara harga pembukaan, tertinggi, terendah, dan penutupan
- Seasonal decomposition (optional)

  
## Data Preparation

Data preparation dilakukan dalam beberapa tahapan:

1. **Data Cleaning**  
   - Menghilangkan missing values
   - Mengubah format tanggal menjadi datetime
   - Mengubah kolom harga menjadi tipe numerik

2. **Feature Engineering**  
   - Membuat fitur `price_diff` (perubahan harga harian)
   - Membuat fitur `rolling_mean_7` dan `rolling_std_7` sebagai konteks jangka pendek
   - Mengubah data menjadi bentuk window sequences untuk input ke LSTM dan GRU

3. **Scaling**  
   - MinMaxScaler digunakan untuk menskalakan harga agar sesuai dengan input layer model neural network

4. **Splitting**  
   - Data di-split menjadi training (80%) dan testing (20%) berdasarkan waktu (bukan random)

**Rubrik Tambahan**:
- Penjelasan kenapa scaling dibutuhkan: input LSTM/GRU sensitif terhadap skala fitur
- Transformasi menjadi windowed sequence sangat penting agar data bisa dipelajari sebagai urutan (sequence)
## Modeling

Model yang digunakan:

### 1. LSTM Neural Network
- Arsitektur: 2 layer LSTM dengan dropout
- Loss Function: MSE
- Optimizer: Adam
- Epochs: 50, batch size: 32
- Kelebihan: Cocok untuk time-series dengan pola panjang
- Kekurangan: Waktu training relatif lama

### 2. GRU Neural Network
- Arsitektur: 2 layer GRU dengan dropout
- Loss Function: MSE
- Optimizer: Adam
- Epochs: 50, batch size: 32
- Kelebihan: Lebih ringan dari LSTM, cocok untuk perangkat terbatas
- Kekurangan: Mungkin tidak sekuat LSTM pada pola yang sangat kompleks

Model terbaik dipilih berdasarkan performa pada data uji menggunakan metrik RMSE dan MAE.

## Evaluation

Metrik yang digunakan:
- **Root Mean Squared Error (RMSE)**:  
  $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
- **Mean Absolute Error (MAE)**:  
  $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

Hasil evaluasi:
- LSTM: RMSE = ..., MAE = ...
- GRU: RMSE = ..., MAE = ...

Model dengan RMSE dan MAE paling rendah dipilih sebagai model terbaik. Hasil prediksi juga divisualisasikan dalam bentuk:
- Plot harga aktual vs harga prediksi
- Plot residual error

**Rubrik Tambahan**:
- Penjelasan formula metrik dan konteksnya dalam time-series regression
