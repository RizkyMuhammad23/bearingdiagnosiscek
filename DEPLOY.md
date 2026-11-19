# Panduan deploy

Yang dinilai adalah aplikasi yang bisa dibuka lewat tautan. Cara paling sederhana adalah Streamlit Community
Cloud, karena aplikasi Streamlit di repositori ini memuat model langsung dan tidak bergantung pada API. API
FastAPI bisa ditaruh terpisah di Render jika tim ingin menunjukkan arsitektur data → model → API → aplikasi
secara utuh.

## 1. Unggah repositori ke GitHub

```bash
cd bearing-diagnosis
git init
git add .
git commit -m "Proyek diagnosis bearing: EDA, model, API, aplikasi"
git branch -M main
git remote add origin https://github.com/<akun-atau-organisasi>/<nama-repo>.git
git push -u origin main
```

Buat repositorinya publik, lalu tambahkan semua anggota sebagai *collaborator* (Settings → Collaborators).
Setiap anggota sebaiknya punya *commit* sendiri, misalnya untuk bagian yang dikerjakannya.

`artifacts/model.joblib` berukuran sekitar 0,5 MB, jadi aman masuk Git biasa tanpa Git LFS.

## 2. Aplikasi di Streamlit Community Cloud

1. Masuk ke https://share.streamlit.io dengan akun GitHub.
2. Pilih **Create app** → **Deploy a public app from GitHub**.
3. Isi *Repository* dengan repositori tim, *Branch* `main`, dan *Main file path* `app/ui/streamlit_app.py`.
4. Buka **Advanced settings** dan pilih Python **3.13**. Model disimpan dengan scikit-learn 1.9.1; versi
   pustaka di `requirements.txt` sudah dikunci agar sama.
5. Klik **Deploy**. Pemasangan pertama memakan waktu beberapa menit.
6. Salin tautan `https://<nama>.streamlit.app` ke README dan laporan, lalu coba buka dari ponsel.

Jika muncul galat saat memuat model (misalnya `InconsistentVersionWarning` atau `ModuleNotFoundError`),
periksa versi Python dan pastikan `requirements.txt` ada di akar repositori.

## 3. API di Render (opsional)

1. Masuk ke https://render.com dengan akun GitHub, lalu pilih **New** → **Blueprint** dan arahkan ke
   repositori tim. Render membaca `render.yaml` dan membuat layanan `bearing-diagnosis-api`.
   Cara manualnya: **New** → **Web Service**, *Build command* `pip install -r requirements.txt`, *Start
   command* `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT`.
2. Tambahkan variabel lingkungan `PYTHON_VERSION` = `3.13.0` agar versi Python sama dengan saat pelatihan.
3. Setelah aktif, cek `https://<nama-api>.onrender.com/health` dan `https://<nama-api>.onrender.com/docs`.
4. API ini berdiri sendiri untuk sistem lain (misalnya integrasi dengan sistem getaran klien). Aplikasi
   Streamlit tidak membutuhkannya dan tetap berjalan tanpa API.

Layanan gratis Render tertidur setelah tidak dipakai beberapa saat, sehingga permintaan pertama bisa lambat
hingga sekitar satu menit. Bangunkan API sebelum merekam video demo.

## 4. Sebelum mengumpulkan

- Isi tautan aplikasi, API, dan video di bagian atas `README.md`.
- Isi nama tim, nama anggota, NIM, dan pembagian kerja di `README.md` dan lampiran laporan.
- Coba aplikasi dari ponsel, termasuk tombol contoh "Tidak wajar" dan "Mustahil".
