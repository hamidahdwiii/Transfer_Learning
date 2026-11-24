"""
Lag-Llama (zero-shot).

Butuh kode sumber Lag-Llama (di-clone otomatis oleh notebook ke ./lag-llama)
dan checkpoint dari Hugging Face (diunduh otomatis).

Jalankan:  python modeling/run_lagllama.py --repo_dir lag-llama
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from common import (TARGET, Timer, base_argparser, collect_rows, load_data,
                    make_tasks, pick_device, save_predictions)

# Checkpoint Lag-Llama berisi objek GluonTS; PyTorch >= 2.6 menolaknya bila
# weights_only=True (default baru). Checkpoint ini resmi dari penulis model.
_orig_load = torch.load


def _load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _load


def main():
    p = base_argparser("Lag-Llama zero-shot")
    p.add_argument("--repo_dir", default="lag-llama", help="Folder hasil git clone lag-llama")
    p.add_argument("--ckpt", default=None, help="Path lag-llama.ckpt (kosong = unduh dari HF)")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.set_defaults(context=512)
    a = p.parse_args()
    device = torch.device(pick_device(a.device))

    sys.path.insert(0, os.path.abspath(a.repo_dir))
    from gluonts.dataset.common import ListDataset
    from lag_llama.gluon.estimator import LagLlamaEstimator

    ckpt_path = a.ckpt
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    mk = ckpt["hyper_parameters"]["model_kwargs"]

    df = load_data(a.data)
    y = df[TARGET].to_numpy(np.float32)
    tasks = make_tasks(len(df))
    history = a.context + 1200   # konteks + ruang untuk fitur lag (lag terjauh Lag-Llama ~1092 langkah)

    frames = []
    with Timer() as tm:
        for h in sorted({t.horizon for t in tasks}):
            ts = [t for t in tasks if t.horizon == h]
            est = LagLlamaEstimator(
                ckpt_path=ckpt_path,
                prediction_length=h,
                context_length=a.context,
                input_size=mk["input_size"],
                n_layer=mk["n_layer"],
                n_embd_per_head=mk["n_embd_per_head"],
                n_head=mk["n_head"],
                scaling=mk["scaling"],
                time_feat=mk["time_feat"],
                rope_scaling={"type": "linear",
                              "factor": max(1.0, (a.context + h) / mk["context_length"])},
                batch_size=a.batch,
                num_parallel_samples=a.samples,
                device=device,
            )
            module = est.create_lightning_module()
            predictor = est.create_predictor(est.create_transformation(), module)

            entries = []
            for t in ts:
                s = max(0, t.start - history)
                entries.append({"start": pd.Period(df["datetime"].iloc[s], freq="H"),
                                "target": y[s:t.start]})
            forecasts = list(predictor.predict(ListDataset(entries, freq="H")))
            for t, f in zip(ts, forecasts):
                frames.append(collect_rows("Lag-Llama", t, df, f.quantile(0.5), f.quantile(0.1), f.quantile(0.9)))
            print(f"  Lag-Llama horizon {h} jam selesai ({len(ts)} jendela)")
    save_predictions("Lag-Llama", frames, a.out, tm.elapsed,
                     {"family": "Foundation Model",
                      "config": f"Lag-Llama | konteks {a.context} jam + RoPE scaling | {a.samples} sampel | zero-shot"})


if __name__ == "__main__":
    main()
