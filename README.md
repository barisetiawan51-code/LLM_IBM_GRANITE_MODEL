# LLM_IBM_GRANITE_MODEL

## Project Overview

Data lowongan pekerjaan merupakan salah satu sumber informasi penting untuk memahami kebutuhan pasar tenaga kerja. informasi seperti judul pekerjaan, kualifikasi, pengalaman, lokasi, gaji, keterampilan serta deskripsi pekerjaan dapat membantu hal-hal sebagai berikut:
*   Membantu perusahaan dalam strategi rekrutmen dan perencanaan SDM.
*   Membantu kandidat dalam memahami  tren keterampilan yang paling dibutuhkan.
*   Membantu peneliti dan praktisi AI dalam pengembangan model NLP untuk klasifikasi teks, pencarian lowongan kerja dan sistem rekomendasi.
*   Analisis pasar kerja global menggunakan dataset lowongan kerja dan IBM Granite LLM.
*   Mengeksplorasi skill yang paling dibutuhkan, kisaran gaji, serta perbandingan antar profesi, sekaligus mengevaluasi performa LLM dengan metrik otomatis dan penilaian berbasis AI.
Namun, data semacam ini biasanya tidak terstruktur dengan baik, bervariasi formatnya, serta mengandung noise (contoh: gaji yang tidak konsisten, deskripsi terlalu panjang/pendek). Maka diperlukan pendekatan analisis data dan NLP agar informasi yang ada bisa diolah menjadi pengetahuan yang berguna.

## Raw Dataset (Link Dataset: [Job Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset/data))
Dataset yang digunakan berasal dari Kaggle dengan judul Job Dataset from Kaggle: Ravender Singh Rana. Dataset ini berisi informasi seperti berikut:
1. Job ID: Pengidentifikasi unik untuk setiap lowongan pekerjaan.
2. Experience: Lamanya tahun pengalaman yang dibutuhkan atau diutamakan untuk pekerjaan tersebut.
3. Qualifications: Kualifikasi pendidikan yang dibutuhkan untuk pekerjaan tersebut.
4. Salary Range: Kisaran gaji atau kompensasi yang ditawarkan untuk posisi tersebut.
5. Location: Kota atau wilayah tempat lowongan pekerjaan berada.
6. Country: Negara tempat lowongan pekerjaan berada.
7. Latitude: Koordinat lintang lokasi lowongan pekerjaan.
8. Longitude: Koordinat bujur lokasi lowongan pekerjaan.
9. Work Type: Jenis pekerjaan (misalnya, penuh waktu, paruh waktu, kontrak).
10. Company Size: Perkiraan ukuran atau skala perusahaan yang merekrut.
11. Job Posting Date: Tanggal saat lowongan pekerjaan dipublikasikan.
12. Preference: Preferensi atau persyaratan khusus untuk pelamar (misalnya, Hanya Pria atau Hanya Wanita, atau Keduanya)
13. Contact Person: Nama narahubung atau perekrut untuk lowongan pekerjaan tersebut.
14. Contact: Informasi kontak untuk pertanyaan pekerjaan.
15. Job Title: Jabatan atau posisi yang diiklankan.
16. Role: Peran atau kategori pekerjaan (misalnya, pengembang perangkat lunak, manajer pemasaran).
17. Job Portal: Platform atau situs web tempat lowongan pekerjaan diposting.
18. Job Description: Deskripsi detail tentang tanggung jawab dan persyaratan pekerjaan.
19. Benefits: Informasi tentang tunjangan yang ditawarkan dalam pekerjaan (misalnya, asuransi kesehatan, program pensiun).
20. Skills: Keterampilan atau kualifikasi yang dibutuhkan untuk pekerjaan tersebut.
21. Responsibilities: Tanggung jawab dan tugas spesifik yang terkait dengan pekerjaan.
22. Company Name: Nama perusahaan yang merekrut.
23. Company Profile: Gambaran singkat tentang latar belakang dan misi perusahaan.

## Insights & Findings
Beberapa temuan penting dari analisis awal yaitu:
1. 

## AI Support Explanation
Dalam proyek ini, AI (IBM Granite LLM) digunakan untuk:
1. Menjawab pertanyaan berbasis konteks seputar lowongan kerja yang jumlahnya sangat besar.
2. Membatasi input dokumen agar tetap efisien (hanya beberapa dokumen relevan yang dipakai).
3. Membantu mengatasi keterbatasan manusia dalam membaca jutaan baris data secara manual.
4. Evaluasi otomatis hasil jawaban dengan metrik tekstual & embedding.
5. Memberikan evaluasi kualitas jawaban agar hasil analisis tetap terukur (bukan hanya opini model).
