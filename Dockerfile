# Gunakan image Python dasar yang sudah termasuk OS
FROM python:3.12-slim

# Pasang system dependencies yang dibutuhkan OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja di server
WORKDIR /app

# Salin file requirements.txt dan instal library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek yang lain
COPY . .

# Perintah untuk menjalankan aplikasi
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]