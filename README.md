# LLM_IBM_GRANITE_MODEL

## Project Overview

Data lowongan pekerjaan merupakan salah satu sumber informasi penting untuk memahami kebutuhan pasar tenaga kerja. informasi seperti judul pekerjaan, kualifikasi, pengalaman, lokasi, gaji, keterampilan serta deskripsi pekerjaan dapat membantu hal-hal sebagai berikut:
1. Membantu perusahaan dalam strategi rekrutmen dan perencanaan SDM.
2. Membantu kandidat dalam memahami  tren keterampilan yang paling dibutuhkan.
3. Membantu peneliti dan praktisi AI dalam pengembangan model NLP untuk klasifikasi teks, pencarian lowongan kerja dan sistem rekomendasi.
4. Analisis pasar kerja global menggunakan dataset lowongan kerja dan IBM Granite LLM.
5. Mengeksplorasi skill yang paling dibutuhkan, kisaran gaji, serta perbandingan antar profesi, sekaligus mengevaluasi performa LLM dengan metrik otomatis dan penilaian berbasis AI.

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

## Permasalahan Data
1. Ukuran Dataset yang Sangat Besar
   * Dataset berisi lebih dari 1,6 juta job postings.
   * Tidak semua informasi bisa diproses LLM karena keterbatasan panjang input (context window).
2. Keterbatasan Dokumen pada LLM
   * IBM Granite hanya bisa menerima beberapa ribu karakter konteks.
   * Akibatnya, hanya sebagian kecil lowongan bisa dipakai dalam sekali query yang berpotensi membuat jawaban kurang lengkap.
3. Variasi Format & Inkonstistensi
   * Pengalaman kerja ditulis dalam range berbeda (contoh: “0 to 15 Years”, “3 to 13 Years”).
   * Deskripsi lowongan bervariasi seperti ada yang detail sekali dan ada juga yang sangat singkat.
5. Dominasi Kata Kunci Umum
   * Word cloud memperlihatkan kata seperti “social media”, “user interface”, dan “compliance” sangat dominan.
   * Kata-kata umum ini bisa menutupi skill teknis yang lebih spesifik (misalnya Python, TensorFlow, Docker).
     
## Insights & Findings
Beberapa temuan penting dari analisis awal yaitu:
1. Kualifikasi pendidikan ini menunjukkan bahwa jumlah lowongan kerja berdasarkan kualifikasi relatif merata di semua tingkat pendidikan (BBA, BA, BCA, M.Tech, PhD, MBA, B.Tech, M.Com, B.Com, MCA). Tidak ada jurusan atau gelar yang dominan secara signifikan, menandakan bahwa pasar kerja membuka peluang luas untuk berbagai latar belakang pendidikan. Hal ini juga mengindikasikan bahwa faktor lain seperti pengalaman, skill, dan peran pekerjaan lebih berpengaruh dibandingkan hanya tingkat pendidikan formal.
2. Minimum salary cenderung stabil dengan rentang yang relatif kecil (sekitar 55K–65K), sedangkan maximum salary memiliki variasi yang jauh lebih besar (sekitar 80K–130K). Artinya, batas bawah gaji cukup konsisten, tetapi batas atas gaji sangat bervariasi tergantung pada posisi, pengalaman, atau industri. Hal ini menandakan adanya peluang besar untuk peningkatan gaji pada level yang lebih tinggi, meskipun titik masuknya relatif seragam.
3. Banyak lowongan kerja menuntut fokus pada user interface, user experience, dan social media, serta kepatuhan terhadap compliance. Selain itu, keterampilan dalam digital marketing, web application, dan interaction design juga muncul dominan. Hal ini menunjukkan bahwa pasar kerja modern sangat menekankan kombinasi antara teknologi, desain, dan pemasaran digital untuk meningkatkan pengalaman pengguna sekaligus menjaga kepatuhan bisnis.
4. perusahaan sangat menekankan soft skills seperti communication skills, problem-solving, dan attention to detail, disamping hard skills seperti data analysis, design principles, software proficiency, dan market research. Hal ini menunjukkan bahwa kandidat yang menguasai keterampilan teknis sekaligus memiliki kemampuan komunikasi dan kolaborasi akan lebih diminati di pasar kerja.
5. Pekerjaan di bidang teknologi seperti UX/UI Designer, Software Engineer, Network Engineer, dan Software Tester mendominasi pasar kerja. Namun, posisi non-teknis seperti Digital Marketing Specialist, Key Account Manager, Procurement Coordinator, dan Personal Assistant juga tetap menunjukkan permintaan yang signifikan.
6. Posting pekerjaan paling banyak terjadi pada awal tahun (Q1) dan pertengahan tahun (Q3), dengan frekuensi lebih tinggi pada hari kerja dibanding akhir pekan. Hal ini menunjukkan pola musiman dalam perekrutan.


## AI Support Explanation
1. LLM (IBM Granite) digunakan untuk:
    * Menjawab pertanyaan seputar skill, gaji, dan perbandingan pekerjaan.
    * Memberi evaluasi kualitas jawaban (LLM-as-a-Judge).
2. Karena keterbatasan input panjang, digunakan strategi batasi jumlah dokumen & panjang teks agar tetap efisien.
3. Membantu mengatasi keterbatasan manusia dalam membaca jutaan baris data secara manual.
4. Evaluasi otomatis hasil jawaban dengan metrik tekstual & embedding serta memberikan evaluasi kualitas jawaban agar hasil analisis tetap terukur (bukan hanya opini model). Evaluasi dilakukan dengan kombinasi:
    * Cosine Similarity (TF-IDF)
    * ROUGE Score
    * Semantic Embedding Similarity
    * LLM-as-a-Judge
