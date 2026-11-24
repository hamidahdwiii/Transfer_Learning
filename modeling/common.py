"""
Utilitas bersama untuk semua skrip pemodelan.

Desain evaluasi (sama untuk SEMUA model, supaya perbandingannya adil):
  - Horizon 24 jam  : 7 jendela uji berurutan (7 hari terakhir data)
  - Horizon 168 jam : 4 jendela uji berurutan (4 minggu terakhir data)
  - Horizon 720 jam : 1 jendela uji (30 hari terakhir data)
Untuk setiap jendela, model hanya boleh melihat data SEBELUM jendela itu dimulai
(tidak ada kebocoran data masa depan).

File ini sengaja hanya bergantung pada numpy & pandas, karena dipakai di
lingkungan (virtualenv) yang berbeda-beda untuk tiap model.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data", "panama_hourly_load.csv")
DEFAULT_OUT = os.path.join(ROOT, "results", "raw")

TARGET = "nat_demand"
# horizon (jam) -> jumlah jendela uji
HORIZONS: dict[int, int] = {24: 7, 168: 4, 720: 1}


@dataclass
class Task:
    horizon: int          # panjang prediksi (jam)
    window: int           # nomor jendela uji, 1 = paling lama
    start: int            # indeks baris pertama periode uji
    end: int              # indeks (eksklusif) baris terakhir periode uji

    @property
    def label(self) -> str:
        return f"{self.horizon} Jam"


def load_data(path: str = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def make_tasks(n_rows: int, horizons: dict[int, int] = HORIZONS) -> list[Task]:
    tasks = []
    for h, n_win in horizons.items():
        for k in range(n_win):
            # k = 0 adalah jendela paling baru
            end = n_rows - k * h
            start = end - h
            tasks.append(Task(horizon=h, window=n_win - k, start=start, end=end))
    return sorted(tasks, key=lambda t: (t.horizon, t.window))


def collect_rows(model: str, task: Task, df: pd.DataFrame,
                 mean, p10=None, p90=None) -> pd.DataFrame:
    """Rapikan satu hasil prediksi menjadi format panjang yang seragam."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    assert len(mean) == task.horizon, f"{model}: panjang prediksi {len(mean)} != {task.horizon}"
    test = df.iloc[task.start:task.end]
    out = pd.DataFrame({
        "model": model,
        "horizon": task.label,
        "horizon_h": task.horizon,
        "window": task.window,
        "step": np.arange(1, task.horizon + 1),
        "timestamp": test["datetime"].values,
        "actual": test[TARGET].values,
        "pred_mean": mean,
        "pred_p10": np.nan if p10 is None else np.asarray(p10, dtype=float).reshape(-1),
        "pred_p90": np.nan if p90 is None else np.asarray(p90, dtype=float).reshape(-1),
    })
    return out


def save_predictions(model: str, frames: list[pd.DataFrame], out_dir: str,
                     runtime_s: float, extra: dict | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    res = pd.concat(frames, ignore_index=True)
    res["runtime_s"] = round(runtime_s, 1)
    for k, v in (extra or {}).items():
        res[k] = v
    slug = model.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    path = os.path.join(out_dir, f"{slug}.csv")
    res.to_csv(path, index=False, float_format="%.4f")
    mape = np.mean(np.abs(res["actual"] - res["pred_mean"]) / res["actual"]) * 100
    print(f"[OK] {model}: {len(res)} baris disimpan ke {path} | MAPE gabungan {mape:.2f}% | {runtime_s:.0f} dtk")
    return path


def base_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--context", type=int, default=2048,
                   help="Panjang konteks (jam) yang dilihat model foundation")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    return p


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
