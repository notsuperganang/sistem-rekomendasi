# Laporan Proyek Machine Learning - Ganang Setyo Hadi

## Project Overview

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film yang memberikan rekomendasi personal dan akurat kepada pengguna berdasarkan preferensi dan historical ratings mereka. Sistem ini dibangun menggunakan **MovieLens Latest-Small Dataset**, sebuah dataset standar industri yang dikembangkan oleh GroupLens Research, yang sering digunakan untuk penelitian dan evaluasi sistem rekomendasi. Dataset ini mencakup data dari Maret 1996 hingga September 2018, dengan total 610 pengguna, 9.742 film unik, 100.836 rating dalam skala 0.5-5.0, dan 3.683 tag yang dihasilkan pengguna.

Permasalahan yang diangkat dalam proyek ini sangat relevan di era digital, di mana platform streaming film menghadapi tantangan seperti **cold start problem** (sulitnya memberikan rekomendasi untuk pengguna baru) dan kebutuhan untuk meningkatkan **user engagement** serta **content discovery**. Sistem rekomendasi yang efektif dapat membantu pengguna menemukan film yang sesuai dengan selera mereka, sehingga meningkatkan kepuasan dan retensi pengguna pada platform. Menurut sebuah studi sistem rekomendasi hybrid dapat meningkatkan akurasi dan kepuasan pengguna hingga 20% dibandingkan pendekatan tunggal (Zhang et al., 2015). Oleh karena itu, proyek ini mengimplementasikan dua pendekatan yang saling melengkapi untuk mengatasi permasalahan tersebut.

**Referensi:**
<div style="padding-left: 2em; text-indent: -2em;">
Zhang, H.-R., Min, F., He, X., & Xu, Y.-Y. (2015). A hybrid recommender system based on user-recommender interaction. <i>Mathematical Problems in Engineering</i>, <i>2015</i>, Article ID 145636. https://doi.org/10.1155/2015/145636
</div>

## Business Understanding

### Problem Statements
- **Pernyataan Masalah 1**: Pengguna baru atau pengguna dengan sedikit rating (cold start problem) kesulitan mendapatkan rekomendasi film yang relevan, sehingga menurunkan engagement mereka di platform streaming.
- **Pernyataan Masalah 2**: Pengguna sering kali tidak menemukan film yang sesuai dengan preferensi mereka karena kurangnya personalisasi dalam rekomendasi, yang menyebabkan rendahnya content discovery dan user satisfaction.

### Goals
- **Jawaban Pernyataan Masalah 1**: Mengembangkan sistem rekomendasi hybrid yang menggabungkan Content-based Filtering dan Collaborative Filtering untuk mengatasi cold start problem dengan memanfaatkan metadata film dan pola rating pengguna.
- **Jawaban Pernyataan Masalah 2**: Menyediakan rekomendasi top-N yang personal dan relevan untuk setiap pengguna berdasarkan preferensi mereka, sehingga meningkatkan content discovery dan user satisfaction.

### Solution Approach
- **Pendekatan 1**: Menggunakan **Content-based Filtering** berbasis TF-IDF dan cosine similarity untuk merekomendasikan film berdasarkan kesamaan konten (genre dan tag), yang efektif untuk pengguna dengan sedikit rating karena tidak bergantung pada data interaksi pengguna lain.
- **Pendekatan 2**: Menggunakan **Collaborative Filtering** berbasis Singular Value Decomposition (SVD) untuk merekomendasikan film berdasarkan pola rating pengguna lain, yang cocok untuk pengguna dengan banyak rating dan dapat menangkap preferensi laten.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens Latest-Small Dataset**, yang dapat diunduh dari [MovieLens Latest-Small Dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip). Dataset ini terdiri dari 100.836 rating dari 610 pengguna untuk 9.742 film unik, dengan rentang rating 0.5 hingga 5.0. Dataset ini juga mencakup 3.683 tag yang dihasilkan pengguna dan metadata film seperti genre. Kondisi data cukup baik, tanpa missing values, tetapi memiliki tingkat sparsity 98.30%, yang umum dalam dataset rekomendasi.

**Variabel-variabel pada MovieLens Latest-Small Dataset adalah sebagai berikut:**
- **ratings.csv**:
  - `userId`: ID unik untuk setiap pengguna (1 hingga 610).
  - `movieId`: ID unik untuk setiap film (1 hingga 9.742).
  - `rating`: Nilai rating yang diberikan pengguna (0.5 hingga 5.0).
  - `timestamp`: Waktu rating diberikan (dalam detik sejak 1 Januari 1970 UTC).
- **movies.csv**:
  - `movieId`: ID unik untuk setiap film.
  - `title`: Judul film beserta tahun rilis (misalnya "Toy Story (1995)").
  - `genres`: Genre film, dipisahkan oleh tanda `|` (misalnya "Adventure|Comedy").
- **tags.csv**:
  - `userId`: ID pengguna yang memberikan tag.
  - `movieId`: ID film yang diberi tag.
  - `tag`: Tag yang diberikan pengguna pada film (misalnya "funny", "action").
  - `timestamp`: Waktu tag diberikan.
- **links.csv**:
  - `movieId`: ID film.
  - `imdbId`: ID film di IMDb.
  - `tmdbId`: ID film di TMDb.

**Exploratory Data Analysis (EDA):**
- Distribusi rating menunjukkan bahwa rating 4.0 (26.818) dan 3.0 (20.047) adalah yang paling umum, mengindikasikan kecenderungan pengguna memberikan rating tinggi.
- Analisis genre menunjukkan bahwa Drama (4.361 film) dan Comedy (3.756 film) adalah genre paling populer, memberikan wawasan tentang preferensi umum pengguna.

## Data Preparation

Proses data preparation dilakukan untuk memastikan data siap digunakan oleh kedua model. Tahapan yang dilakukan meliputi:

1. **Pembagian Data Train-Test**:
   - Data rating dibagi menjadi set pelatihan (80%) dan uji (20%) per pengguna berdasarkan timestamp untuk menjaga konsistensi temporal. Pengguna dengan kurang dari 5 rating diabaikan untuk memastikan kualitas data. Hasilnya adalah 80.419 rating untuk pelatihan dan 20.417 untuk uji.

2. **Persiapan Fitur Konten untuk Content-based Filtering**:
   - Genre dan tag per film digabungkan menjadi satu kolom teks (`content`), dengan normalisasi teks (huruf kecil, penghapusan karakter khusus) untuk memastikan konsistensi dalam analisis kesamaan konten.

3. **Pembentukan Matriks User-Item untuk Collaborative Filtering**:
   - Matriks user-item dibuat dari data pelatihan dengan dimensi 610 pengguna dan 8.231 film, diisi dengan rating dan 0 untuk data yang hilang. Sparsity matriks adalah 98.40%, yang konsisten dengan sifat dataset rekomendasi.

**Alasan Data Preparation**:
- Pembagian train-test diperlukan untuk melatih dan mengevaluasi model secara terpisah, memastikan hasil evaluasi realistis.
- Penggabungan genre dan tag meningkatkan kualitas fitur untuk Content-based Filtering, memungkinkan model menangkap kesamaan konten dengan lebih baik.
- Matriks user-item diperlukan untuk Collaborative Filtering agar model dapat menangkap pola interaksi pengguna-film.

## Modeling

Dua pendekatan sistem rekomendasi telah diimplementasikan untuk memberikan rekomendasi top-N kepada pengguna:

1. **Content-based Filtering**:
   - Model ini menggunakan TF-IDF untuk mengubah fitur konten (genre dan tag) menjadi vektor, lalu menghitung kesamaan antar film dengan cosine similarity. Rekomendasi dibuat berdasarkan film yang disukai pengguna (rating ≥ 4.0).
   - **Kelebihan**: Efektif untuk pengguna baru (cold start problem) karena hanya bergantung pada metadata film.
   - **Kekurangan**: Catalog coverage rendah (1.9%), sehingga cenderung merekomendasikan film serupa dalam subset kecil katalog.
   - Contoh rekomendasi untuk User 1: "Batman: Mask of the Phantasm (1993)" (score: 4.73).

2. **Collaborative Filtering (SVD)**:
   - Model ini menggunakan SVD untuk mereduksi matriks user-item menjadi faktor laten (50 komponen), lalu memprediksi rating dengan mempertimbangkan bias pengguna dan film.
   - **Kelebihan**: Akurat dalam menangkap pola preferensi laten, dengan RMSE 0.925 dan MAE 0.711.
   - **Kekurangan**: Cenderung memberikan prediksi rating maksimum (5.0), yang dapat mengindikasikan overfitting.
   - Contoh rekomendasi untuk User 1: "Persuasion (1995)" (predicted rating: 5.0).

**Top-N Recommendation Demo**:
- **User 1** (preferensi: Adventure, Action, Comedy):
  - Content-based: "Swan Princess, The (1994)" (score: 4.73).
  - Collaborative Filtering: "Usual Suspects, The (1995)" (predicted: 5.0).
- **User 15** (preferensi: Drama, Sci-Fi, Adventure):
  - Content-based: "Juror, The (1996)" (score: 5.00).
  - Collaborative Filtering: "Star Wars: Episode IV - A New Hope (1977)" (predicted: 5.0).
- **User 50** (preferensi: Drama, Thriller, Crime):
  - Content-based: "Platoon (1986)" (score: 4.32).
  - Collaborative Filtering: "Pulp Fiction (1994)" (predicted: 4.5).

## Evaluation

**Metrik Evaluasi dan Formula:**

1. **Collaborative Filtering**:
   - **RMSE (Root Mean Squared Error)**: Mengukur rata-rata kuadrat error antara rating prediksi dan aktual.
     ![Rumus RMSE](https://arize.com/wp-content/uploads/2023/08/RMSE-equation.png "Persamaan Root Mean Square Error")
   - **MAE (Mean Absolute Error)**: Mengukur rata-rata error absolut.
     ![Rumus RMSE](https://arize.com/wp-content/uploads/2024/04/mean-absolute-error-formula.png)
   - Hasil: RMSE 0.925 dan MAE 0.711 menunjukkan akurasi prediksi yang baik, dengan error rata-rata di bawah 1 poin pada skala 0.5-5.0.

2. **Content-based Filtering**:
   - **Success Rate**: Persentase pengguna yang berhasil mendapatkan rekomendasi top-10.
   - **Genre Diversity**: Rata-rata jumlah genre unik per daftar rekomendasi.
   - **Catalog Coverage**: Persentase film unik yang direkomendasikan dari total katalog.
   - Hasil: Success rate 100%, genre diversity 6.0 genres per pengguna, dan catalog coverage 1.9%. Ini menunjukkan keandalan dalam memberikan rekomendasi yang relevan dan beragam, meskipun eksplorasi katalog masih terbatas.

**Kesimpulan Evaluasi**:
Kedua model menunjukkan performa yang kuat, dengan Collaborative Filtering unggul dalam prediksi rating dan Content-based Filtering efektif untuk content discovery. Sistem ini siap untuk implementasi hybrid, meskipun perlu peningkatan pada catalog coverage untuk Content-based Filtering dan penyesuaian parameter SVD untuk mengurangi overfitting.    