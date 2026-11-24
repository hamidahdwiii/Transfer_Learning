"""
Gabungkan semua prediksi di results/raw/*.csv lalu hitung metrik.

Output:
  results/forecasts.csv  -> semua prediksi (dibaca dashboard)
  results/metrics.csv    -> MAE, RMSE, MAPE, cakupan interval P10-P90 per model x horizon

Jalankan:  python modeling/evaluate.py
"""
import glob
import os

import numpy as np
import pandas as pd

from common import ROOT

RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")


def metrics(g: pd.DataFrame) -> pd.Series:
    err = g["actual"] - g["pred_mean"]
    has_pi = g["pred_p10"].notna().all()
    cover = ((g["actual"] >= g["pred_p10"]) & (g["actual"] <= g["pred_p90"])).mean() * 100 if has_pi else np.nan
    return pd.Series({
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE (%)": (err.abs() / g["actual"]).mean() * 100,
        "Bias": err.mean(),                     # + = model cenderung di bawah aktual
        "Cakupan P10-P90 (%)": cover,
        "Jumlah Titik": len(g),
        "Jumlah Jendela": g["window"].nunique(),
        "Waktu (dtk)": g["runtime_s"].iloc[0],
    })


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    if not files:
        raise SystemExit("Belum ada file di results/raw/. Jalankan skrip run_*.py dulu.")
    fc = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fc["config"] = fc.get("config", pd.Series(dtype=str)).fillna("")

    m = (fc.groupby(["model", "family", "config", "horizon", "horizon_h"])
         .apply(metrics, include_groups=False).reset_index())

    # Peningkatan relatif terhadap Seasonal Naive (patokan minimum)
    base = m[m["model"] == "Seasonal Naive"].set_index("horizon_h")
    for col in ["MAPE (%)", "RMSE"]:
        ref = m["horizon_h"].map(base[col]) if not base.empty else np.nan
        m[f"Lebih baik dari Naive ({col.split()[0]})"] = (1 - m[col] / ref) * 100

    m["Peringkat MAPE"] = m.groupby("horizon_h")["MAPE (%)"].rank(method="min").astype(int)
    m["Peringkat RMSE"] = m.groupby("horizon_h")["RMSE"].rank(method="min").astype(int)
    m = m.sort_values(["horizon_h", "MAPE (%)"]).reset_index(drop=True)

    fc.drop(columns=["runtime_s", "config"]).to_csv(os.path.join(OUT, "forecasts.csv"), index=False, float_format="%.3f")
    m.to_csv(os.path.join(OUT, "metrics.csv"), index=False, float_format="%.4f")

    pd.set_option("display.width", 160)
    print(m[["horizon", "model", "MAPE (%)", "RMSE", "MAE", "Cakupan P10-P90 (%)",
             "Peringkat MAPE", "Peringkat RMSE"]].round(2).to_string(index=False))
    print("\nModel terbaik per horizon:")
    for h, g in m.groupby("horizon_h"):
        print(f"  {h:>4} jam | MAPE: {g.loc[g['MAPE (%)'].idxmin(), 'model']:<15} | RMSE: {g.loc[g['RMSE'].idxmin(), 'model']}")


if __name__ == "__main__":
    main()
