"""
SARIMAX + baseline Seasonal Naive.

SARIMAX(p,d,q)(P,D,Q,24) dengan variabel eksogen (X):
  - Fourier mingguan (periode 168 jam, K=3) -> menangkap pola hari kerja vs akhir pekan
  - holiday & school -> kalender yang sudah diketahui sebelumnya (adil dipakai saat prediksi)
  - (opsional, --weather) suhu 3 kota. PERHATIAN: memakai suhu AKTUAL periode uji
    berarti model "melihat" cuaca masa depan secara sempurna (oracle). Default MATI
    agar perbandingan dengan model foundation (univariat) tetap adil.

Seasonal Naive: prediksi = nilai pada jam yang sama seminggu sebelumnya (lag 168).
Ini patokan minimum; model yang tidak bisa mengalahkannya tidak layak dipakai.

Jalankan:  python modeling/run_sarimax.py
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (TARGET, Timer, base_argparser, collect_rows, load_data,
                    make_tasks, save_predictions)

warnings.filterwarnings("ignore")


def fourier(t: np.ndarray, period: float, k: int) -> np.ndarray:
    cols = []
    for i in range(1, k + 1):
        cols += [np.sin(2 * np.pi * i * t / period), np.cos(2 * np.pi * i * t / period)]
    return np.column_stack(cols)


def build_exog(df: pd.DataFrame, use_weather: bool) -> np.ndarray:
    t = np.arange(len(df))
    parts = [fourier(t, 168, 3), df[["holiday", "school"]].to_numpy(float)]
    if use_weather:
        parts.append(df[["T2M_toc", "T2M_san", "T2M_dav"]].to_numpy(float))
    return np.hstack(parts)


def main():
    p = base_argparser("SARIMAX + Seasonal Naive")
    p.add_argument("--train_days", type=int, default=42,
                   help="Panjang data latih SARIMAX (hari) sebelum tiap jendela uji")
    p.add_argument("--order", default="2,0,1")
    p.add_argument("--seasonal_order", default="1,1,1,24")
    p.add_argument("--weather", action="store_true", help="Tambahkan suhu aktual (oracle)")
    a = p.parse_args()

    df = load_data(a.data)
    y = df[TARGET].to_numpy(float)
    X = build_exog(df, a.weather)
    order = tuple(int(v) for v in a.order.split(","))
    sorder = tuple(int(v) for v in a.seasonal_order.split(","))
    tasks = make_tasks(len(df))
    win = a.train_days * 24

    # ---------- Seasonal Naive ----------
    with Timer() as tn:
        frames = []
        for t in tasks:
            hist = y[:t.start]
            pred = np.array([hist[-168 + (i % 168)] for i in range(t.horizon)])
            frames.append(collect_rows("Seasonal Naive", t, df, pred))
    save_predictions("Seasonal Naive", frames, a.out, tn.elapsed, {"family": "Baseline"})

    # ---------- SARIMAX ----------
    name = "SARIMAX"
    frames = []
    cache = {}   # satu fit per titik potong (cutoff), dipakai ulang lintas horizon
    with Timer() as ts:
        for t in tasks:
            need = max(tt.horizon for tt in tasks if tt.start == t.start)
            if t.start not in cache:
                s = t.start - win
                model = SARIMAX(y[s:t.start], exog=X[s:t.start], order=order,
                                seasonal_order=sorder, trend="n",
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False, maxiter=200)
                fc = res.get_forecast(steps=need, exog=X[t.start:t.start + need])
                ci = fc.conf_int(alpha=0.2)   # interval 80% -> P10..P90
                cache[t.start] = (np.asarray(fc.predicted_mean), np.asarray(ci)[:, 0], np.asarray(ci)[:, 1])
                print(f"  SARIMAX fit cutoff={df['datetime'].iloc[t.start]} AIC={res.aic:.0f}")
            m, lo, hi = cache[t.start]
            h = t.horizon
            frames.append(collect_rows(name, t, df, m[:h], lo[:h], hi[:h]))
    save_predictions(name, frames, a.out, ts.elapsed,
                     {"family": "Statistik", "config": f"SARIMAX{order}x{sorder} exog={'kalender+cuaca' if a.weather else 'kalender'}"})


if __name__ == "__main__":
    main()
