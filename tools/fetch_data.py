"""Download everything the analysis needs. Nothing here is vendored.

About 17 MB: the InHARD temporal annotations, one session's independent copy in
seconds, a sample of the RGB clips for the cost measurement, and the ONNX model.

Every download asserts what it expected to get. A partial or silently-empty
fetch fails here rather than producing a plausible wrong number later.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

GT_REPO = "ToufikBenmessabih/GSK-C2F"
GT_PREFIX = "ground_truth/GT_bvh/"
SAMPLE_REPO = "Maeng-Su/InHARD-sample"
MODEL_URL = "https://huggingface.co/Xenova/yolov8n-pose/resolve/main/onnx/model.onnx"

EXPECTED_SESSIONS = 38
N_CLIPS = 36


def get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "frame-budget-bench"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def tree(repo: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    payload = json.loads(get(url))
    if "tree" not in payload:
        raise RuntimeError(f"{repo}: {payload.get('message', 'no tree returned')}")
    return payload["tree"]


def raw(repo: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/HEAD/" + urllib.parse.quote(path)


def fetch_annotations() -> int:
    out = DATA / "inhard_gt"
    out.mkdir(parents=True, exist_ok=True)
    paths = [t["path"] for t in tree(GT_REPO) if t["path"].startswith(GT_PREFIX)]
    paths = [p for p in paths if p.endswith(".json")]
    if len(paths) != EXPECTED_SESSIONS:
        raise RuntimeError(
            f"expected {EXPECTED_SESSIONS} session files, the repo now lists {len(paths)}"
        )
    for p in paths:
        dest = out / Path(p).name
        if dest.exists():
            continue
        body = get(raw(GT_REPO, p))
        parsed = json.loads(body)
        (sid,) = parsed.keys()
        if not parsed[sid]:
            raise RuntimeError(f"{p}: no segments")
        dest.write_bytes(body)
    print(f"  annotations: {len(list(out.glob('P*.json')))} sessions")
    return len(paths)


def fetch_sample() -> None:
    csv = DATA / "InHARD_GT.csv"
    if not csv.exists():
        body = get(raw(SAMPLE_REPO, "GT.csv"))
        if b"Meta_action_label" not in body:
            raise RuntimeError("GT.csv does not look like the InHARD annotation")
        csv.write_bytes(body)

    clips = DATA / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    have = len(list(clips.glob("*.mp4")))
    if have >= N_CLIPS:
        print(f"  clips: {have} already present")
        return
    paths = sorted(
        t["path"]
        for t in tree(SAMPLE_REPO)
        if t["path"].endswith("clip_V1.mp4")
    )
    if not paths:
        raise RuntimeError("no clips found in the sample repo")
    for p in paths[:N_CLIPS]:
        dest = clips / p.replace("gt_clips/", "").replace("/", "__")
        if dest.exists():
            continue
        body = get(raw(SAMPLE_REPO, p))
        if len(body) < 1000:
            raise RuntimeError(f"{p}: {len(body)} bytes, that is not a video")
        dest.write_bytes(body)
    print(f"  clips: {len(list(clips.glob('*.mp4')))} downloaded")


def fetch_model() -> None:
    dest = DATA / "yolov8n-pose.onnx"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  model: already present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    body = get(MODEL_URL, timeout=300)
    if len(body) < 1_000_000:
        raise RuntimeError(f"model download was {len(body)} bytes")
    dest.write_bytes(body)
    print(f"  model: {len(body)/1e6:.1f} MB")


def main() -> int:
    DATA.mkdir(exist_ok=True)
    try:
        print("InHARD annotations (CC-BY-4.0, see CITATION.md)")
        fetch_annotations()
        print("InHARD sample: second annotation copy and RGB clips")
        fetch_sample()
        print("ONNX model for the cost measurement")
        fetch_model()
    except (urllib.error.URLError, RuntimeError) as e:
        print(f"\nfailed: {e}", file=sys.stderr)
        return 1
    print("\nready: make results && make bench")
    return 0


if __name__ == "__main__":
    sys.exit(main())
