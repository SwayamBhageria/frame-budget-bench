"""Entry points: `python -m fbb results` and `python -m fbb bench`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fbb")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("results", help="write every published number to results/")
    r.add_argument("--data", type=Path, default=Path("data/inhard_gt"))
    r.add_argument("--csv", type=Path, default=Path("data/InHARD_GT.csv"))
    r.add_argument("--out", type=Path, default=Path("results"))

    b = sub.add_parser("bench", help="measure inference cost on real footage")
    b.add_argument("--model", type=Path, default=Path("data/yolov8n-pose.onnx"))
    b.add_argument("--clips", type=Path, default=Path("data/clips"))
    b.add_argument("--frames", type=int, default=120)
    b.add_argument("--out", type=Path, default=Path("results/bench.json"))

    a = ap.parse_args(argv)
    if a.cmd == "results":
        from .pipeline import run_all

        run_all(a.data, a.csv, a.out)
        print(f"wrote {a.out}/*.json")
        return 0

    if a.cmd == "bench":
        import onnxruntime as ort

        from .bench import benchmark, deployment_math, load_frames, measure_decode

        frames = load_frames(a.clips, limit=a.frames)
        decode = measure_decode(a.clips)
        # The floor: decoding while skipping as much as possible.
        decode_floor = min(d["ms_per_video_second"] for d in decode)
        out = {
            "frames_decoded": int(len(frames)),
            "decode": decode,
            "decode_floor_ms_per_video_second": decode_floor,
            "runs": [],
        }
        print(
            f"decode: {decode[0]['ms_per_video_second']:.2f} ms/video-second at every "
            f"frame, floor {decode_floor:.2f}"
        )
        for prov in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
            if prov not in ort.get_available_providers():
                continue
            res = benchmark(a.model, frames, prov)
            res["deployment"] = [
                deployment_math(res["ms_per_frame"], f, decode_floor)
                for f in (29.97, 15.0, 6.0, 2.0, 1.0, 0.5, 0.33)
            ]
            out["runs"].append(res)
            print(f"{prov}: {res['ms_per_frame']:.2f} ms/frame")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {a.out}")
        return 0
    return 1
