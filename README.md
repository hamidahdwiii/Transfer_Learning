# ⚡ GridCast Panama: Benchmark Model Prediksi Beban Listrik

Dashboard Streamlit ini membandingkan **5 metode prediksi** beban listrik nasional Panama per jam (2015–2020):

| Model | Jenis | Pengembang |
|---|---|---|
| **Chronos-2** | Foundation model (zero-shot) | Amazon |
| **TimesFM 2.5** | Foundation model (zero-shot) | Google Research |
| **Moirai 2.0** | Foundation model (zero-shot) | Salesforce |
| **Lag-Llama** | Foundation model (zero-shot) | Rasul dkk., 2024 |
| **SARIMAX** | Statistik klasik + variabel kalender | – |
| Seasonal Naive | Patokan minimum (nilai seminggu lalu) | – |

Model dinilai dengan **MAPE** dan **RMSE** pada tiga horizon:

| Horizon | Jendela uji | Periode |
|---|---|---|
| 24 jam | 7 jendela | 7 hari terakhir |
| 168 jam | 4 jendela | 4 minggu terakhir |
| 720 jam | 1 jendela | 30 hari terakhir |

## 📁 Struktur repo

```text
beban-listrik-panama/
├── app.py                         # dashboard Streamlit
├── requirements.txt               # dependensi dashboard (untuk Streamlit Cloud)
├── .streamlit/config.toml         # tema
├── data/panama_hourly_load.csv    # data beban + cuaca + kalender per jam
├── results/
│   ├── metrics.csv                # metrik per model × horizon (dibaca dashboard)
│   ├── forecasts.csv              # semua prediksi (dibaca dashboard)
│   └── raw/                       # prediksi mentah per model
├── modeling/
│   ├── common.py                  # desain evaluasi bersama
│   ├── run_sarimax.py             # SARIMAX + Seasonal Naive
│   ├── run_chronos.py
│   ├── run_timesfm.py
│   ├── run_moirai.py
│   ├── run_lagllama.py
│   ├── evaluate.py                # gabungkan hasil + hitung MAPE/RMSE
│   └── requirements/              # dependensi tiap model (dipisah karena saling bentrok)
└── notebooks/benchmark_5_model.ipynb   # jalankan semua model di Google Colab
```

## 🚀 Cara pakai

1. **Unggah repo ini ke GitHub.**
2. **Jalankan model di Colab.** Buka `notebooks/benchmark_5_model.ipynb` di Google Colab (Runtime → T4 GPU) lalu *Run all*. Notebook akan meng-clone repo, membuat virtualenv terpisah untuk tiap model, menjalankan semuanya, lalu mengunduh `results.zip`.
3. **Unggah hasilnya.** Ekstrak `results.zip` dan unggah isinya ke folder `results/` di GitHub.
4. **Deploy dashboard.** Di [share.streamlit.io](https://share.streamlit.io) pilih repo ini dengan *main file* `app.py`.

Jalankan dashboard secara lokal:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Catatan metodologi

- Tidak ada kebocoran data: setiap model hanya melihat data **sebelum** jendela uji.
- Model foundation dipakai **zero-shot** (tanpa fine-tuning) dan univariat. Prediksi titiknya adalah **median (P50)**.
- SARIMAX(2,0,1)(1,1,1,24) dilatih ulang pada 6 minggu terakhir sebelum tiap jendela, dengan variabel X: Fourier mingguan, `holiday`, dan `school`. Data cuaca tidak dipakai secara default karena cuaca aktual periode uji tidak diketahui saat prediksi. Opsi `--weather` tersedia untuk eksperimen.
- Seluruh periode uji (Mei–Jun 2020) jatuh pada masa COVID-19, ketika beban turun ±15%. Hasil di periode ini belum tentu berlaku untuk kondisi normal.
