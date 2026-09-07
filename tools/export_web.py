"""Write the console's data file from the same code that writes results/.

The console must not carry numbers of its own. Everything it draws comes from
here, so the page and the repository cannot disagree.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbb.analysis import by_participant, per_session  # noqa: E402
from fbb.corpus import NATIVE_FPS, load_corpus  # noqa: E402
from fbb.estimators import duty_cycle_cost_ratio, estimate_work_rate  # noqa: E402

KS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 300]
SEEDS = 30
WINDOW_S, PERIOD_S = 45.0, 300.0


def main() -> None:
    sessions = load_corpus(ROOT / "data" / "inhard_gt")
    bench = json.loads((ROOT / "results" / "bench.json").read_text())
    corpus = json.loads((ROOT / "results" / "corpus.json").read_text())

    participants = sorted({s.participant for s in sessions})
    out_rows = []

    for k in KS:
        part = by_participant(per_session(sessions, [k])[k])

        # Corrected estimator, averaged over calibration draws.
        acc = defaultdict(lambda: {"c": 0.0, "sp": 0.0})
        ov_num = ov_den = 0.0
        for seed in range(SEEDS):
            for s in sessions:
                r = estimate_work_rate(
                    s, k, window_s=WINDOW_S, period_s=PERIOD_S, seed=seed
                )
                a = acc[r["participant"]]
                a["c"] += r["corrected_work_rate"] * r["span"]
                a["sp"] += r["span"]
                if seed == 0:
                    ov_num += r["calibration_overrun_s"]
                    ov_den += r["span"]

        rows = []
        for p in participants:
            rows.append(
                {
                    "id": p,
                    "true": part[p]["true_work_rate"],
                    "naive": part[p]["obs_work_rate"],
                    "corrected": acc[p]["c"] / acc[p]["sp"],
                    "recall": part[p]["action_recall"],
                    "share": part[p]["true_activity_share"],
                }
            )
        out_rows.append(
            {
                "k": k,
                "fps": NATIVE_FPS / k,
                "cost_naive": 1.0 / k,
                "cost_corrected": duty_cycle_cost_ratio(
                    k, WINDOW_S, PERIOD_S, ov_num / ov_den
                ),
                "operators": rows,
            }
        )

    ms = {r["provider"]: r["ms_per_frame"] for r in bench["runs"]}
    decode_floor = bench["decode_floor_ms_per_video_second"]
    payload = {
        "generated_from": "python -m fbb results && python tools/export_web.py",
        "corpus": {
            "sessions": corpus["sessions"],
            "participants": corpus["participants"],
            "segments": corpus["action_segments"],
            "hours": corpus["total_span_h"],
            "native_fps": NATIVE_FPS,
            "median_action_s": corpus["duration_s"]["median"],
        },
        "bench": {
            "model": bench["runs"][0]["model"],
            "ms_per_frame": ms,
            "default_ms": min(ms.values()),
            # Charged at every frame budget: the stream arrives in real time
            # whether or not every frame is analysed.
            "decode_floor_ms_per_video_second": decode_floor,
        },
        "calibration": {"window_s": WINDOW_S, "period_s": PERIOD_S, "seeds": SEEDS},
        "rows": out_rows,
    }
    dest = ROOT / "web" / "data.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {dest} ({dest.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
