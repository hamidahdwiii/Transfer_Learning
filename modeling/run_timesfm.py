"""
Google TimesFM 2.5 (zero-shot).

Jalankan:  python modeling/run_timesfm.py
"""
import numpy as np
import timesfm
import torch

from common import (TARGET, Timer, base_argparser, collect_rows, load_data,
                    make_tasks, save_predictions)


def main():
    p = base_argparser("TimesFM 2.5 zero-shot")
    p.add_argument("--model_id", default="google/timesfm-2.5-200m-pytorch")
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")

    df = load_data(a.data)
    y = df[TARGET].to_numpy(np.float32)
    tasks = make_tasks(len(df))
    max_h = max(t.horizon for t in tasks)

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(a.model_id)
    model.compile(timesfm.ForecastConfig(
        max_context=int(np.ceil(a.context / 32) * 32),
        max_horizon=int(np.ceil(max_h / 128) * 128),   # 720 -> 768
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))

    frames = []
    with Timer() as tm:
        for h in sorted({t.horizon for t in tasks}):
            ts = [t for t in tasks if t.horizon == h]
            ctx = [y[max(0, t.start - a.context):t.start] for t in ts]
            point, quant = model.forecast(horizon=h, inputs=ctx)
            # quant[..., 0] = mean, quant[..., 1..9] = kuantil 0.1 ... 0.9
            for i, t in enumerate(ts):
                frames.append(collect_rows("TimesFM", t, df, point[i, :h], quant[i, :h, 1], quant[i, :h, 9]))
            print(f"  TimesFM horizon {h} jam selesai ({len(ts)} jendela)")
    save_predictions("TimesFM", frames, a.out, tm.elapsed,
                     {"family": "Foundation Model", "config": f"{a.model_id} | konteks {a.context} jam | zero-shot"})


if __name__ == "__main__":
    main()
