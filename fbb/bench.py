"""Measured inference cost, so the frame budget converts to money.

The frontier's x-axis is frames analysed per camera-second. To turn that into
cameras-per-accelerator and rupees-per-camera-hour you need one measured number:
how long one frame takes. This measures it on real footage from the same corpus
with a real model, rather than quoting FLOPs.

The absolute milliseconds are a property of this machine and this model and do
not transfer. What transfers is that cost is exactly linear in analysed frames,
so a deployment can substitute its own measured per-frame time.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np


def load_frames(clip_dir: Path, limit: int = 240, size: int = 640) -> np.ndarray:
    """Decode real clip frames into one NCHW float batch."""
    import cv2

    frames = []
    for p in sorted(clip_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        while len(frames) < limit:
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.resize(img, (size, size))
            frames.append(img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
        cap.release()
        if len(frames) >= limit:
            break
    if not frames:
        raise FileNotFoundError(f"no decodable clips in {clip_dir}")
    return np.stack(frames)


def benchmark(
    model_path: Path,
    frames: np.ndarray,
    provider: str,
    warmup: int = 10,
    repeats: int = 3,
) -> dict:
    """Time one forward pass per frame, reporting the median of `repeats` runs."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), opts, providers=[provider])
    name = sess.get_inputs()[0].name

    for i in range(warmup):
        sess.run(None, {name: frames[i % len(frames)][None]})

    per_run = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for f in frames:
            sess.run(None, {name: f[None]})
        per_run.append((time.perf_counter() - t0) / len(frames))

    ms = statistics.median(per_run) * 1000.0
    return {
        "provider": provider,
        "model": model_path.name,
        "frames_timed": len(frames),
        "repeats": repeats,
        "ms_per_frame": ms,
        "ms_per_frame_best": min(per_run) * 1000.0,
        "ms_per_frame_worst": max(per_run) * 1000.0,
        "frames_per_second": 1000.0 / ms,
    }


def measure_decode(clip_dir: Path, strides=(1, 2, 5, 10, 30), size: int = 640) -> list[dict]:
    """Decode cost per second of video, which is the part that does not scale.

    Analysing fewer frames does not mean decoding fewer. The container still has
    to be demuxed and every frame still has to be walked past to stay in sync, so
    only the pixel-format conversion and resize are actually skipped. That leaves
    a floor, and below a low enough frame budget the floor is most of the bill.
    """
    import cv2

    clips = sorted(Path(clip_dir).glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"no clips in {clip_dir}")
    out = []
    for stride in strides:
        t0 = time.perf_counter()
        source_frames = 0
        for p in clips:
            cap = cv2.VideoCapture(str(p))
            i = 0
            while cap.grab():
                if i % stride == 0:
                    ok, img = cap.retrieve()
                    if ok:
                        cv2.resize(img, (size, size))
                i += 1
                source_frames += 1
            cap.release()
        elapsed = time.perf_counter() - t0
        video_seconds = source_frames / (30000.0 / 1001.0)
        out.append(
            {
                "stride": stride,
                "source_frames": source_frames,
                "ms_per_video_second": elapsed / video_seconds * 1000.0,
            }
        )
    return out


def deployment_math(
    ms_per_frame: float, analysed_fps: float, decode_ms_per_video_second: float = 0.0
) -> dict:
    """Cameras one accelerator can carry at a given frame budget.

    `decode_ms_per_video_second` is charged whatever the frame budget, because a
    camera streams in real time whether or not you look at every frame.
    """
    frames_per_camera_hour = analysed_fps * 3600.0
    infer_s = frames_per_camera_hour * ms_per_frame / 1000.0
    decode_s = 3600.0 * decode_ms_per_video_second / 1000.0
    total = infer_s + decode_s
    return {
        "analysed_fps": analysed_fps,
        "frames_per_camera_hour": frames_per_camera_hour,
        "inference_seconds_per_camera_hour": infer_s,
        "decode_seconds_per_camera_hour": decode_s,
        "device_seconds_per_camera_hour": total,
        "decode_share": decode_s / total if total else 0.0,
        "cameras_per_device": 3600.0 / total if total else float("inf"),
        "cameras_per_device_inference_only": 3600.0 / infer_s if infer_s else float("inf"),
    }
