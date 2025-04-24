# Machine Learning Project Report - Haris Yafie

Proyek untuk pembelajaran saya dalam kursus Dicoding Machine Learning Terapan. Proyek ini tentang prediksi harga emas dunia terhadap IDR. Data yang digunakan berkisar dari 24 April 2023 hingga 22 April 2025.

ID Dicoding: harisyafie

Email: yafie345@gmail.com

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

Data yang digunakan dalam proyek ini diperoleh dari [Investing.com - GAU/IDR](https://id.investing.com/currencies/gau-idr-historical-data), mencakup periode 24 April 2023 hingga 22 April 2025. Data ini merepresentasikan harga emas dunia yang dikonversikan ke dalam Rupiah Indonesia (IDR).

Dataset mencakup kolom-kolom berikut (berdasarkan struktur umumnya dari Investing.com):
- Tanggal: tanggal pengamatan
- Terakhir: harga terakhir emas pada hari tersebut
- Open: harga pembukaan
- High: harga tertinggi dalam sehari
- Low: harga terendah dalam sehari
- Change %: persentase perubahan harga
Namun yang digunakan pada proyek ini yaitu 'Tanggal' dan 'Terakhir'


## Data Preparation

Data preparation dilakukan dalam beberapa tahapan:

1. **Data Cleaning**  
   - Menghilangkan kolom yang tidak diperlukan
   - Mengubah format tanggal menjadi datetime
   - Mengurutkan data berdasarkan tanggal
  
2. **Feature Engineering**  
   - Mengubah data menjadi bentuk window sequences untuk input ke LSTM dan GRU

3. **Scaling**  
   - MinMaxScaler digunakan untuk menskalakan harga agar sesuai dengan input layer model neural network
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

4. **Splitting**  
    Dalam pemodelan machine learning, membagi data menjadi data pelatihan (*training*) dan data pengujian (*testing*) merupakan langkah penting untuk mengukur kemampuan generalisasi model. Pada kasus prediksi deret waktu (time series), pembagian data dilakukan **berdasarkan urutan waktu** — bukan secara acak — agar struktur temporal dari data tetap terjaga.
    
    Dalam proyek ini, data historis harga emas yang telah dinormalisasi dibagi dengan proporsi:
    - **80%** untuk data pelatihan (*training set*)
    - **20%** untuk data pengujian (*test set*)
    
    Dengan pendekatan ini, model akan belajar dari pola harga di masa lalu (training) dan diuji menggunakan data masa depan yang belum pernah dilihat sebelumnya (testing). Hal ini memastikan bahwa proses evaluasi mencerminkan kemampuan model dalam memprediksi data baru secara realistis.

## **Exploratory Data Analysis**:
1. **Statistika Deskriptif**
### Descriptive Statistics for Closing Gold Price (IDR)

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

2. **Data Visualization**

Visualisasi yang dilakukan:

Setelah melakukan analisis statistik deskriptif, langkah selanjutnya adalah membuat visualisasi data untuk mendapatkan pemahaman yang lebih mendalam.
Visualisasi akan dilakukan menggunakan plot time series untuk melihat bagaimana tren harga emas berubah dari waktu ke waktu.

Dengan melihat visualisasi ini, kita dapat:

- Mengidentifikasi pola tren jangka panjang
- Melihat fluktuasi atau volatilitas harga
- Menemukan titik-titik ekstrem seperti lonjakan atau penurunan tajam

Visualisasi ini juga menjadi langkah awal yang penting sebelum membangun model prediksi berbasis data historis.

**Line Plot – Harga Penutupan Emas dari Waktu ke Waktu**

Visualisasi ini menunjukkan tren harga emas dunia dalam Rupiah dari April 2023 hingga April 2025. Terlihat adanya:

- Tren kenaikan secara bertahap
- Beberapa lonjakan signifikan di akhir periode
- Fluktuasi tetap ada, tapi tren jangka panjang meningkat

**Boxplot – Distribusi Harga Emas per Bulan**

Boxplot ini mengungkap:

- Perbedaan harga antar bulan
- Outlier di bulan tertentu
- IQR makin besar di bulan akhir → volatilitas meningkat


## Modeling

Setelah data dipersiapkan, langkah selanjutnya adalah membangun arsitektur model deep learning untuk melakukan prediksi harga emas. Pada proyek ini, digunakan dua jenis model Recurrent Neural Network (RNN) yang populer untuk data deret waktu, yaitu **GRU (Gated Recurrent Unit)** dan **LSTM (Long Short-Term Memory)**.

### 1. GRU Neural Network
- Dua layer GRU berurutan, dengan parameter `return_sequences=True` pada layer pertama agar output dapat diteruskan ke layer berikutnya.
- Layer `Dropout` digunakan untuk mengurangi risiko overfitting dengan cara mengabaikan sejumlah unit selama pelatihan.
- Layer output `Dense(1)` digunakan untuk menghasilkan prediksi harga pada 1 hari ke depan.
- Aktivasi menggunakan aktivasi `tanh` karena menghasilkan output di rentang **-1 hingga 1**, cocok untuk data yang sudah diskalakan, dapat membantu menjaga **stabilitas memori** dalam proses sequence, dan efektif menangkap **pola positif dan negatif** dalam data time-series.
- Model dikompilasi dengan **optimizer Adam**, fungsi loss **Mean Absolute Error (MAE)**, dan metrik evaluasi yang sama.
- Epochs: 100, batch size: 32
- Kelebihan: Lebih ringan dari LSTM, cocok untuk perangkat terbatas
- Kekurangan: Mungkin tidak sekuat LSTM pada pola yang sangat kompleks

**Summary Model GRU**



### 2. LSTM Neural Network
- Dua layer LSTM berurutan, dengan parameter `return_sequences=True` pada layer pertama agar output dapat diteruskan ke layer berikutnya.
- Layer `Dropout` digunakan untuk mengurangi risiko overfitting dengan cara mengabaikan sejumlah unit selama pelatihan.
- Layer output `Dense(1)` digunakan untuk menghasilkan prediksi harga pada 1 hari ke depan.
- Aktivasi menggunakan aktivasi `tanh` karena menghasilkan output di rentang **-1 hingga 1**, cocok untuk data yang sudah diskalakan, dapat membantu menjaga **stabilitas memori** dalam proses sequence, dan efektif menangkap **pola positif dan negatif** dalam data time-series.
- Model dikompilasi dengan **optimizer Adam**, fungsi loss **Mean Absolute Error (MAE)**, dan metrik evaluasi yang sama.
- Epochs: 100, batch size: 32
- Kelebihan: Cocok untuk time-series dengan pola panjang
- Kekurangan: Waktu training relatif lama

**Summary Model LSTM**



Model terbaik dipilih berdasarkan performa pada data uji menggunakan metrik RMSE dan MAE.

## Evaluation

**Perhitungan metrik evaluasi**, yaitu:
   - **MAE (Mean Absolute Error)**: rata-rata selisih absolut antara nilai aktual dan prediksi

     $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
   
   - **RMSE (Root Mean Squared Error)**: menghitung error dengan penalti lebih besar terhadap prediksi yang jauh meleset

     $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
  
Hasil evaluasi:
- GRU: RMSE = 20.821,06, MAE = 15.176,30
- LSTM: RMSE = 41.024,68, MAE = 28.934,15

Model dengan RMSE dan MAE paling rendah dipilih sebagai model terbaik. Hasil prediksi juga divisualisasikan dalam bentuk:
- Plot harga aktual vs harga prediksi
