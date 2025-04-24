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

### **Kesimpulan Evaluasi Model**
Secara keseluruhan, performa LSTM dalam eksperimen ini **kurang baik dibandingkan GRU**. Sehingga model yang dipilih pada proyek ini yaitu:

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
1859472.38 IDR

---

**GRU Forecast Summary:**
- **Min:** 1,841,465.75 IDR  
- **Max:** 1,859,472.38 IDR  
- **Mean:** 1,850,020.75 IDR

---

### Forecast Data Preview:

| Index | GRU_Forecast   |
|-------|----------------|
| 527   | 1,845,663.125  |
| 528   | 1,841,963.750  |
| 529   | 1,841,465.750  |
| 530   | 1,841,749.500  |
| 531   | 1,842,222.250  |

**Interpretasi Tabel Hasil Prediksi**

Tabel di atas menampilkan hasil prediksi harga emas selama 30 hari ke depan yang dihasilkan oleh model GRU. Setiap baris merepresentasikan nilai prediksi untuk satu hari ke depan berdasarkan indeks waktu yang berurutan.

Berikut beberapa ringkasan statistik dari hasil prediksi:

- **Harga prediksi pada hari ke-30**: Rp 1.859.472,38
- **Harga maksimum selama 30 hari**: Rp 1.859.472,38
- **Harga minimum selama 30 hari**: Rp 1.841.465,75
- **Rata-rata prediksi**: Rp 1.850.020,75

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
   - **GRU**: RMSE = 20.821, MAE = 15.176
   - **LSTM**: RMSE = 41.024, MAE = 28.934  
   Dengan hasil tersebut, **model GRU dinyatakan sebagai model terbaik** untuk digunakan dalam prediksi harga emas selanjutnya.

4. **Prediksi 30 Hari ke Depan**  
   Model GRU digunakan untuk memprediksi harga emas selama 30 hari ke depan. Hasilnya menunjukkan tren harga yang cenderung stabil dan naik secara moderat, dengan nilai rata-rata prediksi sebesar **Rp 1.850.020,75**.

**Final Insight**

Model GRU berhasil menunjukkan performa yang solid dalam memprediksi harga emas dan dapat digunakan sebagai dasar analisis untuk keputusan finansial dalam jangka pendek. Namun, untuk aplikasi di dunia nyata, disarankan untuk mempertimbangkan faktor eksternal lain seperti nilai tukar, inflasi, dan sentimen global untuk prediksi yang lebih komprehensif.




