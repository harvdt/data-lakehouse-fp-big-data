# XYZ E-Commerce Data Lake House

Proyek ini merupakan implementasi data lakehouse untuk menganalisis data penjualan dari platform e-commerce XYZ menggunakan arsitektur mediated dan teknologi Delta Lake.

## Anggota Kelompok 5

| Nama                            | NRP        |
| ------------------------------- | ---------- |
| Samuel Yuma Krismata            | 5027221029 |
| Muhammad Harvian Dito Syahputra | 5027221039 |
| Hafiz Akmaldi Santosa           | 5027221061 |
| Nur Azka Rahadiansyah           | 5027221064 |
| Naufan Zaki Luqmanulhakim       | 5027221065 |

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun data lakehouse yang memproses data e-commerce untuk menghasilkan beberapa analisis:

### Output Analisis
1. **Analisis Performa Penjualan**
   - Revenue per kategori produk
   - Impact discount terhadap sales volume

2. **Analisis Strategi Harga**
   - Analisis efektivitas diskon
   - Korelasi harga dengan rating

3. **Metrik Kepuasan Pelanggan**
   - Korelasi harga dengan kepuasan
   - Rasio review terhadap penjualan

### Arsitektur
Proyek ini menggunakan beberapa komponen utama:
- **Apache Kafka**: Untuk mengumpulkan dan mentransfer data secara real-time dan batch
- **Delta Lake**: Untuk menyimpan data dalam berbagai tahap processing:
  - Bronze: Raw data tanpa modifikasi
  - Silver: Data yang sudah dibersihkan dan divalidasi
  - Gold: Data yang sudah diagregasi dan siap untuk analisis
- **Apache Spark**: Untuk transformasi dan analisis data
- **Web Layer**: Frontend menggunakan ReactJS dan Backend menggunakan Python + FastAPI

## Prasyarat

Sebelum menjalankan proyek, pastikan sistem Anda memiliki:
1. Python 3.11
2. Docker dan Docker Compose
3. Java Runtime Environment (JRE)
4. pip (Python package manager)

## Cara Menjalankan Proyek

### 1. Install Python 3.11 (Jika belum terinstall)
```bash
bash scripts/python.sh
```

### 2. Menjalankan Seluruh Pipeline
```bash
bash scripts/run.sh
```
Script ini akan:
- Membuat virtual environment Python
- Menginstall semua dependencies yang diperlukan
- Menjalankan container Docker yang dibutuhkan
- Memproses data melalui layer bronze, silver, dan gold

### 3. Memulai Training Model dan API endpoints
```bash
bash scripts/run_ml.sh
```

### 3. Monitoring
Setelah pipeline berjalan, Anda dapat memonitor:
- Proses ingestion data di Kafka
- Transformasi data di setiap layer
- Metrik kualitas data
- Output analisis

### Deskripsi Endpoint (Not Fixed)
```r
# Frontend example (React)
const getPrediction = async (productData) => {
  const response = await fetch('http://localhost:8000/predict/sales', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(productData)
  });
  return await response.json();
};

// Category analytics
const getCategoryAnalytics = async (category) => {
  const response = await fetch(`http://localhost:8000/analytics/category/${category}`);
  return await response.json();
};
```