# StudentDOA: Analisis Status Mahasiswa

Proyek ini menganalisis dataset mahasiswa untuk memahami distribusi status akhir studi (**Dropout**, **Enrolled**, dan **Graduate**) serta mengevaluasi model klasifikasi multi-kelas. Dashboard dibuat dengan Streamlit.

## Isi folder

- `dataset.csv`: data sumber; satu baris mewakili satu mahasiswa dan `Target` adalah status akhir.
- `data_prep.py`: validasi data, pembagian train/test stratified, dan laporan kualitas.
- `ml_model.py`: pipeline Random Forest, baseline kelas mayoritas, metrik evaluasi, serta artefak model.
- `vai_analyst.py`: interpretasi keluaran model melalui Ollama lokal.
- `App.py`: dashboard untuk ringkasan, eksplorasi, evaluasi, dan vAI Analyst.

## Instalasi dan persiapan

Jalankan dari root repository:

```bash
python -m pip install -r StudentDOA/requirements.txt
python StudentDOA/data_prep.py
python StudentDOA/ml_model.py
```

`data_prep.py` membuat `student_train.csv`, `student_test.csv`, `student_ml_dataset.csv`, dan `data_quality_report.json`.

`ml_model.py` membuat model `student_status_pipeline.pkl`, `feature_importance.csv`, `student_predictions.csv`, `confusion_matrix.csv`, dan `model_metrics.json`. Hasil model dibandingkan dengan baseline kelas mayoritas agar peningkatan kinerjanya dapat dinilai.

## Menjalankan dashboard

```bash
streamlit run StudentDOA/App.py
```

Dashboard melatih model ketika dibuka. Kategori yang disimpan sebagai angka diproses dengan one-hot encoding, sementara fitur numerik diimputasi dengan median. Split train/test memakai random state tetap dan distribusi kelas dijaga dengan stratifikasi.

## vAI Analyst (opsional)

Pasang Ollama secara terpisah. Jalankan `ollama serve` pada terminal lain, lalu:

```bash
ollama pull llama3.2
python StudentDOA/vai_analyst.py --prompt-only
python StudentDOA/vai_analyst.py
```

Model default dapat diganti dengan opsi `--model nama-model` atau variabel lingkungan `OLLAMA_MODEL`. vAI merangkum bukti dan prediksi; periksa konteks secara langsung sebelum mengambil tindakan.

## Batasan dan interpretasi

Dataset memuat hasil akademik semester pertama dan kedua. Karena itu, evaluasi ini mengklasifikasikan status menggunakan informasi yang tersedia dalam dataset dan **tidak menunjukkan kemampuan prediksi dini sebelum semester kedua**. Feature importance menunjukkan pola yang dipakai model, bukan penyebab status. Nilai dataset sumber dipertahankan apa adanya; pembersihan atau transformasi hanya dilakukan dalam pipeline dan tidak mengubah `dataset.csv`.
