"""
Salesforce Moirai 2.0 (zero-shot).

Jalankan:  python modeling/run_moirai.py
"""
import numpy as np
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from common import (TARGET, Timer, base_argparser, collect_rows, load_data,
                    make_tasks, pick_device, save_predictions)


def main():
    p = base_argparser("Moirai 2.0 zero-shot")
    p.add_argument("--model_id", default="Salesforce/moirai-2.0-R-small")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    device = pick_device(a.device)

    df = load_data(a.data)
    y = df[TARGET].to_numpy(np.float32)
    tasks = make_tasks(len(df))

    module = Moirai2Module.from_pretrained(a.model_id)
    qlev = np.asarray(module.quantile_levels, dtype=float)
    i10, i50, i90 = (int(np.argmin(np.abs(qlev - q))) for q in (0.1, 0.5, 0.9))

    frames = []
    with Timer() as tm:
        for h in sorted({t.horizon for t in tasks}):
            ts = [t for t in tasks if t.horizon == h]
            model = Moirai2Forecast(
                module=module, prediction_length=h, context_length=a.context,
                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
            ).to(device)
            for b in range(0, len(ts), a.batch):
                chunk = ts[b:b + a.batch]
                ctx = [y[max(0, t.start - a.context):t.start] for t in chunk]
                out = model.predict(past_target=ctx)        # (B, n_quantil, h)
                for i, t in enumerate(chunk):
                    frames.append(collect_rows("Moirai", t, df, out[i, i50], out[i, i10], out[i, i90]))
            print(f"  Moirai horizon {h} jam selesai ({len(ts)} jendela)")
    save_predictions("Moirai", frames, a.out, tm.elapsed,
                     {"family": "Foundation Model", "config": f"{a.model_id} | konteks {a.context} jam | zero-shot"})


if __name__ == "__main__":
    main()
