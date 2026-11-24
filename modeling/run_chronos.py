"""
Amazon Chronos (zero-shot).

Default memakai Chronos-2 (amazon/chronos-2), generasi terbaru Chronos.
Bisa diganti ke Chronos-Bolt dengan:  --model_id amazon/chronos-bolt-base

Jalankan:  python modeling/run_chronos.py
"""
import numpy as np
import torch
from chronos import BaseChronosPipeline

from common import (TARGET, Timer, base_argparser, collect_rows, load_data,
                    make_tasks, pick_device, save_predictions)

QL = [0.1, 0.5, 0.9]


def to_numpy_quantiles(q) -> np.ndarray:
    """Seragamkan output Chronos-2 (list tensor) & Chronos-Bolt (tensor) -> (B, h, 3)."""
    if isinstance(q, (list, tuple)):
        return np.stack([x.float().cpu().numpy().reshape(x.shape[-2], x.shape[-1]) for x in q])
    return q.float().cpu().numpy()


def main():
    p = base_argparser("Chronos zero-shot")
    p.add_argument("--model_id", default="amazon/chronos-2")
    a = p.parse_args()
    device = pick_device(a.device)

    df = load_data(a.data)
    y = df[TARGET].to_numpy(np.float32)
    tasks = make_tasks(len(df))

    pipe = BaseChronosPipeline.from_pretrained(a.model_id, device_map=device)
    name = "Chronos-2" if "chronos-2" in a.model_id else "Chronos-Bolt"

    frames = []
    with Timer() as tm:
        for h in sorted({t.horizon for t in tasks}):
            ts = [t for t in tasks if t.horizon == h]
            ctx = [torch.tensor(y[max(0, t.start - a.context):t.start]) for t in ts]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=QL)
            q = to_numpy_quantiles(q)
            for i, t in enumerate(ts):
                frames.append(collect_rows(name, t, df, q[i, :, 1], q[i, :, 0], q[i, :, 2]))
            print(f"  {name} horizon {h} jam selesai ({len(ts)} jendela)")
    save_predictions(name, frames, a.out, tm.elapsed,
                     {"family": "Foundation Model", "config": f"{a.model_id} | konteks {a.context} jam | zero-shot"})


if __name__ == "__main__":
    main()
