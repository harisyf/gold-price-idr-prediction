# Machine Learning Project Report - Haris Yafie

Proyek untuk pembelajaran saya dalam kursus Dicoding Machine Learning Terapan. Proyek ini tentang prediksi harga emas dunia terhadap IDR. Data yang digunakan berkisar dari 24 April 2023 hingga 22 April 2025.

ID Dicoding: harisyafie

Email: yafie345@gmail.com

# Table of Contents

- [Global Gold Price to IDR Prediction](#global-gold-price-to-idr-prediction)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
  - [Data Collecting and Loading](#data-collecting-and-loading)
  - [Data Checking](#data-checking)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Statistika Deskriptif](#statistika-deskriptif)
    - [Data Visualization](#data-visualization)
- [Data Preparation](#data-preparation)
  - [Import Library Data Preparation](#import-library-data-preparation)
  - [Data Cleaning](#data-cleaning)
  - [Data Scaling](#data-scaling)
  - [Sequence Generation (Windowing)](#sequence-generation-windowing)
  - [Data Train-Test Splitting](#data-train-test-splitting)
- [Model Development](#model-development)
  - [1. GRU Neural Network](#1-gru-neural-network)
  - [2. LSTM Neural Network](#2-lstm-neural-network)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
  - [Interpretasi Hasil Evaluasi Model GRU](#interpretasi-hasil-evaluasi-model-gru)
  - [Interpretasi Hasil Evaluasi Model LSTM](#interpretasi-hasil-evaluasi-model-lstm)
  - [Kesimpulan Evaluasi Model](#kesimpulan-evaluasi-model)
- [Prediction](#prediction)
  - [Prediksi Harga Emas 30 Hari ke Depan](#prediksi-harga-emas-30-hari-ke-depan)
    - [Grafik Prediksi Harga Emas](#grafik-prediksi-harga-emas)
    - [Forecast Data Preview](#forecast-data-preview)
    - [Interpretasi Tabel Hasil Prediksi](#interpretasi-tabel-hasil-prediksi)
- [Kesimpulan](#kesimpulan)


## Global Gold Price to IDR Prediction

Emas merupakan logam mulia yang memiliki berbagai bentuk dan fungsi. Selain digunakan untuk perhiasan, emas juga sering dijadikan sebagai salah satu instrumen investasi yang populer. Investasi emas tergolong sebagai investasi dengan risiko menengah (*middle risk*), yang menjadikannya sebagai pilihan yang relatif aman namun tetap menjanjikan keuntungan (Fachruddin, 2019).

Secara global, harga emas dihitung menggunakan satuan *troy ounce (Oz)* yang setara dengan 31,1034 gram. Seperti instrumen investasi lainnya, harga emas bersifat fluktuatif dari waktu ke waktu, meskipun secara jangka panjang cenderung mengalami kenaikan. Beberapa faktor utama yang memengaruhi harga emas antara lain:

1. **Ketidakpastian Global**  
   Peristiwa politik, ekonomi, krisis, resesi, hingga perang sering memicu kenaikan harga emas, karena investor menganggap emas sebagai aset lindung nilai (*safe haven*).

2. **Penawaran dan Permintaan**  
   Ketika permintaan lebih tinggi dari penawaran, harga emas akan naik, dan sebaliknya.

3. **Kebijakan Moneter The Fed (AS)**  
   Penurunan suku bunga oleh The Fed cenderung mendorong investor untuk beralih ke emas, sehingga harga emas naik.

4. **Inflasi**  
   Semakin tinggi inflasi, harga emas biasanya ikut naik karena nilai uang fiat menurun.

5. **Nilai Tukar Dolar Amerika Serikat (USD)**  
   Harga emas global dikonversi ke IDR berdasarkan kurs USD. Jika Rupiah melemah terhadap USD, maka harga emas lokal akan meningkat.

---

### Mengapa Prediksi Harga Emas Penting?

Prediksi harga emas menjadi penting karena dapat membantu investor dalam menentukan waktu yang tepat untuk membeli atau menjual emas. Dengan menggunakan model prediksi yang efektif, seperti LSTM dan GRU Neural Network, kita dapat memahami pola pergerakan harga emas dan memberikan estimasi yang akurat terhadap harga di masa mendatang. Hal ini akan membantu meningkatkan kualitas pengambilan keputusan dalam berinvestasi emas, terutama dalam situasi ekonomi yang tidak pasti.

**Referensi:**  

- Fachruddin, M. A. (2019). *Implementasi Metode Elman Recurrent Neural Network (ERNN) Untuk Prediksi Harga Emas*. Thesis, Universitas Islam Negeri Sultan Syarif Kasim, Riau.  
- World Gold Council. (n.d.). *Gold Market Insights*. Retrieved from [https://www.gold.org/](https://www.gold.org/)
- International Monetary Fund. (n.d.). *Global Economic Outlook*. Retrieved from [https://www.imf.org/](https://www.imf.org/)
- Investing.com. (n.d.). *GAU/IDR Historical Data*. Retrieved from [https://id.investing.com/currencies/gau-idr-historical-data](https://id.investing.com/currencies/gau-idr-historical-data)


## Business Understanding

### Problem Statements
1. Sulitnya memprediksi harga emas terhadap Rupiah karena tingginya volatilitas dan pengaruh faktor eksternal seperti kurs USD, inflasi global, dan ketegangan geopolitik.
2. Peningkatan permintaan terhadap emas yang menyebabkan kenaikan harga emas secara drastis.

### Goals
1. Membangun model machine learning untuk memprediksi harga emas dalam IDR secara akurat agar dapat membantu mengatasi ketidakpastian akibat volatilitas harga.
2. Menganalisis tren permintaan emas melalui pola historis untuk memahami faktor-faktor yang mendorong kenaikan harga secara drastis.

### Solution Statements
- Model 1: LSTM Neural Network — memanfaatkan kemampuan untuk mengingat pola jangka panjang pada data sekuensial.
- Model 2: GRU Neural Network — alternatif yang lebih ringan dan efisien dengan performa yang sebanding.
- Kinerja masing-masing model akan dievaluasi menggunakan metrik RMSE dan MAE untuk menentukan model terbaik.

## Data Understanding

Sebelum membangun model prediksi, penting untuk memahami terlebih dahulu karakteristik dataset yang digunakan. Pada tahap *data understanding*, dilakukan eksplorasi terhadap struktur data, kondisi kualitas data, serta pemahaman terhadap fitur-fitur yang tersedia.

Langkah ini bertujuan untuk memastikan bahwa data yang digunakan benar-benar representatif, relevan, dan siap untuk diproses lebih lanjut dalam tahap modeling. Selain itu, melalui pemahaman awal terhadap data, potensi masalah seperti missing values, outlier, atau duplikasi dapat diidentifikasi dan ditangani dengan tepat.

### Data Collecting and Loading

Data yang digunakan dalam proyek ini diperoleh dari Investing.com, dan dapat diakses melalui tautan berikut:  
[Investing.com - GAU/IDR Historical Data](https://id.investing.com/currencies/gau-idr-historical-data)

Dataset yang digunakan mencakup periode **24 April 2023 hingga 22 April 2025**, dengan detail sebagai berikut:

![Dataset Awal](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/dataset-new.png)

- **Jumlah Data**:  
  - **527 baris** (record harian)  
  - **7 kolom** (fitur)
 
- **Fitur pada Dataset**:
  - **Tanggal**: Tanggal pengamatan harga emas.
  - **Terakhir**: Harga penutupan emas pada hari tersebut (IDR).
  - **Pembukaan (Open)**: Harga pembukaan emas di hari tersebut.
  - **Tertinggi (High)**: Harga tertinggi emas yang tercapai dalam sehari.
  - **Terendah (Low)**: Harga terendah emas yang tercapai dalam sehari.
  - **Vol. (Volume)**: Volume perdagangan emas dalam satuan transaksi.
  - **Perubahan% (Change%)**: Persentase perubahan harga emas dibandingkan dengan hari sebelumnya.

### Data Checking

Setelah data dimuat, langkah selanjutnya adalah memeriksa kualitas data.

Pertama, kita akan mengubah format kolom `'Tanggal'` menjadi format datetime untuk memastikan bahwa data dapat diurutkan dan diproses sebagai deret waktu (*time series*). Setelah itu, data akan diurutkan berdasarkan kolom `'Tanggal'` untuk menjaga urutan kronologis.

Selanjutnya, dilakukan pengecekan terhadap:
- **Jumlah data** (baris dan kolom) untuk mengetahui ukuran dataset,
- **Missing values** pada setiap kolom untuk mengidentifikasi apakah terdapat data yang hilang,
- **Data duplikat** untuk memastikan tidak ada entri yang tercatat lebih dari satu kali,
- **Outlier** pada kolom harga (`'Terakhir'`) menggunakan metode Interquartile Range (IQR), guna memahami apakah terdapat data yang ekstrem yang mungkin mempengaruhi analisis.

Pengecekan kualitas data ini penting untuk memastikan bahwa dataset yang akan digunakan dalam modeling bersih, valid, dan representatif terhadap kondisi sebenarnya.

![Dataset Quality Check](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/dataset-quality.png)

- **Kondisi Data**:  
  - **Missing Values**: Tidak ditemukan missing value pada kolom yang digunakan (`Tanggal`, `Terakhir`). Kolom lain (`Vol.`) ada beberapa missing values.
  - **Duplikat**: Tidak ditemukan data duplikat.
  - **Outlier**: Teridentifikasi terdapat 1 outlier harga, namun tidak dilakukan penghapusan karena merupakan bagian dari pergerakan harga riil.

### **Exploratory Data Analysis**:
   #### **Statistika Deskriptif**

| Metric           | Value           |
|------------------|-----------------|
| Count            | 527             |
| Mean             | 1.178.196       |
| Std Dev          | 217.310         |
| Min              | 912.144         |
| Q1 (25%)         | 975.623         |
| Median (50%)     | 1.201.496       |
| Q3 (75%)         | 1.331.908       |
| Max              | 1.872.926       |


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

#### **Data Visualization**

Visualisasi yang dilakukan:

Setelah melakukan analisis statistik deskriptif, langkah selanjutnya adalah membuat visualisasi data untuk mendapatkan pemahaman yang lebih mendalam.
Visualisasi akan dilakukan menggunakan plot time series untuk melihat bagaimana tren harga emas berubah dari waktu ke waktu.

Dengan melihat visualisasi ini, kita dapat:

- Mengidentifikasi pola tren jangka panjang
- Melihat fluktuasi atau volatilitas harga
- Menemukan titik-titik ekstrem seperti lonjakan atau penurunan tajam

Visualisasi ini juga menjadi langkah awal yang penting sebelum membangun model prediksi berbasis data historis.

**Line Plot – Harga Penutupan Emas dari Waktu ke Waktu**

![Line Plot Harga Emas Dunia](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/line-plot-gold-price.png)

Visualisasi ini menunjukkan tren harga emas dunia dalam Rupiah dari April 2023 hingga April 2025. Terlihat adanya:

- Tren kenaikan secara bertahap
- Beberapa lonjakan signifikan di akhir periode
- Fluktuasi tetap ada, tapi tren jangka panjang meningkat

**Boxplot – Distribusi Harga Emas per Bulan**

![Boxplot Distribusi Harga Emas per Bulan](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/box-plot-gold-price.png)

Boxplot ini mengungkap:

- Perbedaan harga antar bulan
- Outlier di bulan tertentu
- IQR makin besar di bulan akhir → volatilitas meningkat

## Data Preparation

Setelah memahami karakteristik dan kondisi data, langkah berikutnya adalah melakukan tahapan *data preparation* untuk menyiapkan dataset agar siap digunakan dalam proses pelatihan model machine learning.

Pada tahap ini, dilakukan beberapa proses penting seperti data cleaning (drop fitur), normalisasi data (*scaling*), pembentukan urutan (*sequence generation*), dan pembagian dataset menjadi data pelatihan dan pengujian (*train-test split*). Setiap tahapan dirancang untuk memastikan bahwa model dapat mempelajari pola data secara efektif dan menghasilkan prediksi yang akurat. Data preparation dilakukan dalam beberapa tahapan:

### Import Library Data Preparation

- `numpy`  
  Digunakan untuk manipulasi data. `numpy` digunakan untuk operasi numerik dan array.

- `sklearn.preprocessing.MinMaxScaler`  
  Digunakan untuk melakukan **normalisasi** data harga emas ke rentang 0–1 sebelum dimasukkan ke dalam model.

### Data Cleaning
 
Lalu dilanjutkan dengan melakukan pembersihan data. Kita akan menghapus kolom `'Vol.'`, `'Pembukaan'`, `'Tertinggi'`, `'Terendah'`, dan `'Perubahan%'` karena kolom-kolom tersebut tidak digunakan dalam proses prediksi. Serta kolom `'Vol.'` juga terdapat missing values sehingga harus dikeluarkan dari dataset

![Dataset Final](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/dataset-final-new.png)
   
### Data Scaling

Langkah penting yang dilakukan adalah **Data Scaling**. Pada proyek ini, kolom `Terakhir` (harga emas harian) dinormalisasi menggunakan `MinMaxScaler` agar berada dalam rentang **0 hingga 1**.

Tujuan dari scaling ini adalah:

- **Menstabilkan proses pelatihan model**  
  Nilai harga emas dalam IDR bisa mencapai ratusan ribu hingga jutaan. Jika langsung dimasukkan ke model tanpa normalisasi, nilai yang besar dapat membuat proses training tidak stabil dan memperlambat konvergensi.

- **Mempercepat konvergensi model**  
  Model deep learning (seperti LSTM/GRU) lebih cepat belajar jika nilai input berada dalam skala yang konsisten.

- **Mengoptimalkan fungsi aktivasi**  
  Fungsi aktivasi seperti `tanh` dan `sigmoid` bekerja paling optimal jika input berada dalam kisaran yang kecil (biasanya antara -1 hingga 1 atau 0 hingga 1).

- **Fokus pada pola, bukan skala**  
  Dengan normalisasi, model dapat lebih fokus pada pola pergerakan harga (naik/turun) daripada nilai absolut harga emas.

Oleh karena itu, normalisasi menjadi tahap penting agar model dapat mempelajari tren dengan lebih efisien dan akurat.
   
### Sequence Generation (Windowing)

Setelah data dinormalisasi, tahap selanjutnya adalah mengubah data historis harga emas menjadi bentuk **sequence** yang dapat diproses oleh model LSTM dan GRU.

   Dalam masalah prediksi deret waktu (*time series*), model tidak hanya melihat nilai terakhir saja, tetapi juga perlu memahami pola perubahan harga dalam beberapa hari ke belakang. Untuk itu, dilakukan **pembentukan window sequence**, yaitu mengelompokkan data ke dalam blok-blok kecil sepanjang waktu.

   Proses sequence generation dilakukan dengan teknik berikut:
   - Menggunakan **30 data harga terakhir** sebagai input (X) untuk memprediksi harga **1 hari ke depan** (y).
   - Membentuk pasangan input-output sebanyak mungkin dari seluruh dataset.
   - Contohnya:
     - Input: Harga emas dari hari ke-1 hingga hari ke-30
     - Output: Harga emas pada hari ke-31
     - Input berikutnya: Hari ke-2 hingga ke-31 → Prediksi hari ke-32, dst.
   
   Teknik ini bertujuan agar model dapat:
   - **Mempelajari pola urutan harga** (misal: tren naik, pola naik-turun musiman)
   - **Menghubungkan informasi historis** untuk memprediksi harga di masa depan
   - **Mengadaptasi terhadap dinamika pasar** yang bersifat sekuensial dan berurutan

   Sequence generation menjadi **fondasi utama** bagi model LSTM dan GRU, karena kedua arsitektur ini dirancang khusus untuk memproses data yang berbentuk urutan (*sequential data*). Tanpa pembentukan window sequences, model tidak akan mampu memahami dependensi jangka pendek maupun jangka panjang dari pergerakan harga emas.

### Data Train-Test Splitting

Dalam pemodelan machine learning, membagi data menjadi data pelatihan (*training*) dan data pengujian (*testing*) merupakan langkah penting untuk mengukur kemampuan generalisasi model. Pada kasus prediksi deret waktu (time series), pembagian data dilakukan **berdasarkan urutan waktu** — bukan secara acak — agar struktur temporal dari data tetap terjaga.

Dalam proyek ini, data historis harga emas yang telah dinormalisasi dibagi dengan proporsi:
- **80%** untuk data pelatihan (*training set*)
- **20%** untuk data pengujian (*test set*)

Dengan pendekatan ini, model akan belajar dari pola harga di masa lalu (training) dan diuji menggunakan data masa depan yang belum pernah dilihat sebelumnya (testing). Hal ini memastikan bahwa proses evaluasi mencerminkan kemampuan model dalam memprediksi data baru secara realistis.


## Model Development

Setelah data dipersiapkan, langkah selanjutnya adalah membangun arsitektur model deep learning untuk melakukan prediksi harga emas. Pada proyek ini, digunakan dua jenis model Recurrent Neural Network (RNN) yang populer untuk data deret waktu, yaitu **GRU (Gated Recurrent Unit)** dan **LSTM (Long Short-Term Memory)**.

### 1. GRU Neural Network

GRU merupakan varian dari RNN yang dirancang untuk menangani masalah *vanishing gradient* dan memperbaiki performa pada data sekuensial jangka panjang. GRU menggunakan mekanisme gate sederhana namun efektif untuk mengontrol informasi yang mengalir dalam jaringan.

**Cara Kerja GRU:**

![GRU Workflow](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/gru-model-illustration.png)

- **Update Gate**: Mengontrol seberapa banyak informasi dari masa lalu yang harus dipertahankan ke langkah saat ini.
- **Reset Gate**: Mengontrol seberapa banyak informasi lama yang boleh dilupakan.
- **Memori**: Kombinasi kedua gate ini membuat GRU dapat mempelajari ketergantungan jangka pendek maupun jangka panjang tanpa kehilangan informasi penting.
- GRU lebih ringan dibanding LSTM karena hanya memiliki dua gate, sehingga lebih cepat dalam proses pelatihan.
  
  **Arsitektur GRU pada proyek ini:**
  
- Dua layer GRU berurutan, dengan parameter `return_sequences=True` pada layer pertama agar output dapat diteruskan ke layer berikutnya.
- Layer `Dropout` digunakan untuk mengurangi risiko overfitting dengan cara mengabaikan sejumlah unit selama pelatihan.
- Layer output `Dense(1)` digunakan untuk menghasilkan prediksi harga pada 1 hari ke depan.
- Aktivasi menggunakan aktivasi `tanh` karena menghasilkan output di rentang **-1 hingga 1**, cocok untuk data yang sudah diskalakan, dapat membantu menjaga **stabilitas memori** dalam proses sequence, dan efektif menangkap **pola positif dan negatif** dalam data time-series.
- Model dikompilasi dengan **optimizer Adam**, fungsi loss **Mean Absolute Error (MAE)**, dan metrik evaluasi yang sama.
- Kelebihan: Lebih ringan dari LSTM, cocok untuk perangkat terbatas
- Kekurangan: Mungkin tidak sekuat LSTM pada pola yang sangat kompleks

**Parameter Model yang Dibuat:**
- **Units**: 32 neuron digunakan pada masing-masing layer LSTM/GRU
- **Dropout Rate**: 0.1, digunakan untuk mencegah overfitting
- **Learning Rate**: 0.001, diterapkan dalam optimizer Adam untuk mengontrol laju training

**Summary Model GRU**

![Summary Model GRU](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/gru-model-summary.png)

### 2. LSTM Neural Network

LSTM merupakan arsitektur RNN yang dikembangkan untuk mengatasi keterbatasan RNN standar dalam mengingat informasi jangka panjang. LSTM memperkenalkan konsep cell state dan mekanisme gate yang lebih kompleks untuk mengatur aliran informasi.

**Cara Kerja LSTM:**

![LSTM Workflow](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/lstm-model-illustration.jpg)

- **Forget Gate**: Menentukan informasi mana yang akan dibuang dari cell state.
- **Input Gate**: Menentukan informasi baru yang akan disimpan ke cell state.
- **Output Gate**: Mengatur apa yang akan dikeluarkan dari cell state ke output jaringan.
- **Cell State**: Bertindak seperti jalur utama informasi yang dimodifikasi sedikit di setiap langkah waktu, memungkinkan LSTM untuk mengingat informasi jangka panjang dengan stabil.

**Arsitektur LSTM pada proyek ini:**

- Dua layer LSTM berurutan, dengan parameter `return_sequences=True` pada layer pertama agar output dapat diteruskan ke layer berikutnya.
- Layer `Dropout` digunakan untuk mengurangi risiko overfitting dengan cara mengabaikan sejumlah unit selama pelatihan.
- Layer output `Dense(1)` digunakan untuk menghasilkan prediksi harga pada 1 hari ke depan.
- Aktivasi menggunakan aktivasi `tanh` karena menghasilkan output di rentang **-1 hingga 1**, cocok untuk data yang sudah diskalakan, dapat membantu menjaga **stabilitas memori** dalam proses sequence, dan efektif menangkap **pola positif dan negatif** dalam data time-series.
- Model dikompilasi dengan **optimizer Adam**, fungsi loss **Mean Absolute Error (MAE)**, dan metrik evaluasi yang sama.
- Kelebihan: Cocok untuk time-series dengan pola panjang
- Kekurangan: Waktu training relatif lama

**Parameter Model yang Dibuat:**
- **Units**: 32 neuron digunakan pada masing-masing layer LSTM/GRU
- **Dropout Rate**: 0.1, digunakan untuk mencegah overfitting
- **Learning Rate**: 0.001, diterapkan dalam optimizer Adam untuk mengontrol laju training

**Summary Model LSTM**

![Summary Model LSTM](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/lstm-model-summary.png)

Model terbaik dipilih berdasarkan performa pada data uji menggunakan metrik RMSE dan MAE.

**Berikut snippet code dari Model Building**
```python
def build(model_type, units=32, in_shape=(WIN, 1), dropout_rate=0.1, lr=0.001):
    model = Sequential()

    if model_type == 'LSTM':
        model.add(LSTM(units, return_sequences=True, activation='tanh', input_shape=in_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units, activation='tanh'))
        model.add(Dropout(dropout_rate))

    elif model_type == 'GRU':
        model.add(GRU(units, return_sequences=True, activation='tanh', input_shape=in_shape))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units, activation='tanh'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss='mae',
              metrics=['mae'])

    return model
```

## Model Training

Setelah model GRU dan LSTM berhasil dibangun dan diinisialisasi, langkah selanjutnya adalah melakukan proses **pelatihan (training)**. Pada tahap ini, model akan mempelajari pola historis dari harga emas berdasarkan data yang telah dibagi sebelumnya (training set).

Model akan dilatih menggunakan fungsi `fit()`, dengan parameter sebagai berikut:
- **Epochs**: jumlah iterasi pelatihan penuh terhadap seluruh data training.
- **Batch size**: jumlah sampel yang digunakan sebelum parameter model diperbarui.
- **Validation split**: sebagian kecil dari data training yang digunakan untuk mengevaluasi performa model selama pelatihan.

**Parameter Training Model:**
- **Epochs**: 100
- **Batch Size**: 32
- **Validation Split**: 10% dari data training digunakan untuk validasi model

**Berikut snippet code untuk model training GRU**

```python
history_gru = model_gru.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

**Berikut snippet code untuk model training LSTM**

Selama proses training, model akan menghitung loss dan metrik MAE (Mean Absolute Error) pada data pelatihan dan validasi untuk mengukur seberapa baik model belajar. Nilai-nilai ini dapat digunakan untuk memantau overfitting dan convergence model.

Proses ini dilakukan secara terpisah untuk model GRU dan model LSTM, sehingga nantinya performa keduanya dapat dibandingkan secara objektif.

```python
history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

## Model Evaluation

**Perhitungan metrik evaluasi**, yaitu:
   - **MAE (Mean Absolute Error)**: rata-rata selisih absolut antara nilai aktual dan prediksi

     $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
   
   - **RMSE (Root Mean Squared Error)**: menghitung error dengan penalti lebih besar terhadap prediksi yang jauh meleset

     $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$


Hasil evaluasi:
- **GRU**: RMSE = 22.599, MAE = 17.412
- **LSTM**: RMSE = 39.230, MAE = 27.721

Model dengan RMSE dan MAE paling rendah dipilih sebagai model terbaik. Hasil prediksi juga divisualisasikan dalam bentuk plot harga aktual vs harga prediksi

### **Interpretasi Hasil Evaluasi Model GRU**

**Plot Prediksi Harga Emas dengan GRU**

![Plot Prediksi Harga Emas dengan GRU](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/gold-pred-gru.png)

Berdasarkan hasil evaluasi terhadap model GRU, diperoleh metrik sebagai berikut:

- **RMSE (Root Mean Squared Error)**: 22.599,44
- **MAE (Mean Absolute Error)**: 17.412,55

Nilai MAE dan RMSE yang relatif rendah menunjukkan bahwa model GRU memiliki tingkat kesalahan yang kecil dalam memprediksi harga emas dibandingkan dengan nilai aktual. MAE mengukur rata-rata selisih absolut antara prediksi dan data aktual, sementara RMSE memberikan penalti lebih besar terhadap prediksi yang jauh meleset.

Pada grafik di atas, terlihat bahwa **garis prediksi (orange)** berhasil mengikuti tren **garis aktual (biru)** dengan cukup baik. Model GRU mampu menangkap pola kenaikan harga emas secara umum, meskipun terdapat beberapa deviasi kecil terutama pada fluktuasi tajam.

Secara keseluruhan, model GRU menunjukkan performa yang baik dan dapat diandalkan dalam memprediksi harga emas dalam IDR berdasarkan data historis. Hasil ini menjadi salah satu kandidat kuat untuk digunakan dalam prediksi 30 hari ke depan.

### **Interpretasi Hasil Evaluasi Model LSTM**

**Plot Prediksi Harga Emas dengan LSTM**

![Plot Prediksi Harga Emas dengan LSTM](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/gold-pred-lstm.png)

Berdasarkan hasil evaluasi terhadap model LSTM, diperoleh metrik sebagai berikut:

- **RMSE (Root Mean Squared Error)**: 39.229,94
- **MAE (Mean Absolute Error)**: 27.721,11

Dibandingkan dengan GRU, model LSTM menghasilkan nilai error yang lebih tinggi, baik pada RMSE maupun MAE. Hal ini menunjukkan bahwa LSTM memiliki akurasi yang lebih rendah dalam memprediksi harga emas pada data uji.

Dari grafik, terlihat bahwa garis prediksi model LSTM (orange) cenderung **lebih halus dan konservatif**, dan **kurang responsif terhadap perubahan tajam** pada tren harga aktual (biru). Model ini sedikit tertinggal dalam mengikuti kenaikan tajam harga emas yang terjadi pada periode akhir.

Meskipun LSTM berhasil menangkap arah tren secara umum, ia cenderung menghasilkan prediksi yang terlalu rata pada periode volatil. Hal ini mengindikasikan bahwa model LSTM mungkin lebih cocok untuk data yang tidak terlalu fluktuatif, namun kurang optimal untuk kondisi pasar yang dinamis seperti pergerakan harga emas.

### **Kesimpulan Evaluasi Model**
Secara keseluruhan, performa LSTM dalam eksperimen ini **kurang baik dibandingkan GRU** dilihat berdasarkan nilai RMSE dan MAE serta garis tren prediksi pada plot prediction. Sehingga model yang dipilih pada proyek ini yaitu:

***MODEL GRU***

## **Prediction**

Setelah melalui proses pelatihan dan evaluasi terhadap dua arsitektur model, yaitu GRU dan LSTM, model **GRU dipilih sebagai model terbaik** berdasarkan nilai error yang lebih rendah serta kemampuannya dalam mengikuti tren harga emas dengan lebih akurat.

Pada bagian ini, model GRU yang telah dilatih akan digunakan untuk melakukan **prediksi harga emas selama 30 hari ke depan**. Proses prediksi dilakukan secara bertahap dengan menggunakan pendekatan rekursif, di mana prediksi sebelumnya akan digunakan sebagai input untuk memprediksi hari berikutnya. Hasil prediksi akan ditampilkan dalam bentuk grafik serta diringkas dalam bentuk statistik untuk memberikan gambaran tren harga emas dalam waktu dekat.

### **Prediksi Harga Emas 30 Hari ke Depan**

Untuk melakukan prediksi harga emas 30 hari ke depan, digunakan fungsi `forecast_future()` yang bekerja secara **rekursif**.  
Artinya, prediksi untuk hari ke-1 digunakan sebagai input untuk memprediksi hari ke-2, dan seterusnya hingga jumlah hari yang diinginkan (dalam hal ini 30 hari).

Berikut langkah-langkah yang dilakukan:

1. **Mengambil urutan terakhir** dari data historis sebagai input awal untuk prediksi.
2. Menggunakan model GRU yang telah dilatih untuk memprediksi satu hari ke depan, lalu memasukkan hasil prediksi tersebut sebagai bagian dari input selanjutnya.
3. Proses ini diulang sebanyak 30 kali untuk membentuk prediksi berurutan selama 30 hari.
4. Hasil prediksi yang masih dalam bentuk ter-normalisasi dikembalikan ke skala IDR asli menggunakan `inverse_transform()`.
5. Grafik dibuat untuk menampilkan:
   - Harga historis 60 hari terakhir
   - Hasil prediksi harga emas selama 30 hari ke depan

Selain visualisasi, dicetak juga ringkasan statistik dari hasil prediksi:
- Harga terakhir (day-30)
- Nilai minimum, maksimum, dan rata-rata dari prediksi


### Grafik Prediksi Harga Emas

![Plot Prediksi Harga Emas Selama 30 Hari ke Depan](https://raw.githubusercontent.com/harisyf/gold-price-idr-prediction/main/images/gold-forecast-gru.png)

### **Interpretasi Grafik Prediksi**

Grafik di atas menunjukkan hasil prediksi harga emas selama 30 hari ke depan (garis merah putus-putus) dibandingkan dengan tren historis 60 hari terakhir (garis biru).

Terlihat bahwa model GRU memproyeksikan tren harga emas cenderung **stabil naik** dengan kenaikan yang lebih moderat dibandingkan fluktuasi tajam di masa lalu. Model menangkap arah tren secara umum namun tidak terlalu agresif dalam mengikuti pola lonjakan harga yang ekstrem.

Prediksi ini dapat memberikan gambaran awal bagi investor atau pengambil keputusan untuk mengantisipasi pergerakan harga dalam waktu dekat.

#### 30-Day Gold Price Forecast

---

**Forecast Period:**
- **Start Index:** 527
- **End Index:** 556

---

**GRU Forecast (Last Day):**
- 1.850.999,62 IDR

---

**GRU Forecast Summary:**
- **Min:** 1.838.954,88 IDR
- **Max:** 1.850.999,62 IDR
- **Mean:** 1.843.954,00 IDR

---

### Forecast Data Preview:

| Index | GRU_Forecast (IDR) |
|:-----:|:------------------:|
| 527   | 1.845.125,625       |
| 528   | 1.844.144,750       |
| 529   | 1.841.061,125       |
| 530   | 1.839.760,500       |
| 531   | 1.839.237,000       |

---


**Interpretasi Tabel Hasil Prediksi**

Tabel di atas menampilkan hasil prediksi harga emas selama 30 hari ke depan yang dihasilkan oleh model GRU. Setiap baris merepresentasikan nilai prediksi untuk satu hari ke depan berdasarkan indeks waktu yang berurutan.

Berikut beberapa ringkasan statistik dari hasil prediksi:

- **Harga prediksi pada hari ke-30**: Rp 1.850.999,62
- **Harga maksimum selama 30 hari**: Rp 1.850.999,62
- **Harga minimum selama 30 hari**: Rp 1.838.954,88
- **Rata-rata prediksi**: Rp 1.843.954,00

Rentang harga yang relatif sempit menunjukkan bahwa model memproyeksikan **stabilitas harga** dalam jangka pendek. Tidak ada lonjakan atau penurunan tajam yang terdeteksi, sehingga proyeksi ini dapat digunakan sebagai dasar awal untuk pengambilan keputusan yang bersifat konservatif.

## Kesimpulan

Dalam proyek ini, dilakukan proses prediksi harga emas global dalam IDR menggunakan dua jenis model deep learning untuk data deret waktu, yaitu **LSTM** dan **GRU**. Berikut adalah poin-poin utama dari hasil yang diperoleh:

1. **Data Preparation dan Normalisasi**  
   Data harga emas dari 24 April 2023 hingga 22 April 2025 diproses dan dinormalisasi untuk memastikan stabilitas dan efisiensi dalam pelatihan model.

2. **Model Building dan Training**  
   Dua arsitektur model dibangun dan dilatih menggunakan data historis:
   - LSTM: mampu mengikuti tren secara umum, namun menghasilkan error yang lebih besar.
   - GRU: memberikan hasil prediksi yang lebih akurat dan efisien, serta lebih baik dalam mengikuti pola harga emas.

3. **Evaluasi Model**  
   Berdasarkan metrik evaluasi:
   - **GRU**: RMSE = 22.599, MAE = 17.412
   - **LSTM**: RMSE = 39.230, MAE = 27.721 
   Dengan hasil tersebut, **model GRU dinyatakan sebagai model terbaik** untuk digunakan dalam prediksi harga emas selanjutnya.

4. **Prediksi 30 Hari ke Depan**  
   Model GRU digunakan untuk memprediksi harga emas selama 30 hari ke depan. Hasilnya menunjukkan tren harga yang cenderung stabil dan naik secara moderat, dengan nilai rata-rata prediksi sebesar **Rp 1.843.954,00**.

**Final Insight**

Model GRU berhasil menunjukkan performa yang solid dalam memprediksi harga emas dan dapat digunakan sebagai dasar analisis untuk keputusan finansial dalam jangka pendek. Namun, untuk aplikasi di dunia nyata, disarankan untuk mempertimbangkan faktor eksternal lain seperti nilai tukar, inflasi, dan sentimen global untuk prediksi yang lebih komprehensif.




