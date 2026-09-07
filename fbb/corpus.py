"""Load the InHARD ground-truth annotations as timed action segments.

The corpus is 38 recording sessions of 15 participants performing the same
assembly job (InHARD, Dallel et al., IEEE ICHMS 2020, CC-BY-4.0). Each session
is a list of annotated action segments with start and end frames on the motion
capture clock.

Frames are converted to seconds with a rate that is *fitted*, not assumed: an
independent second copy of session P01_R01 exists as a CSV in seconds, and
`fit_frame_rate` regresses one against the other. See tests/test_corpus.py.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

# Fitted against the independent per-second annotation of P01_R01; see
# fit_frame_rate() and results/corpus.json. The nominal capture rate is 120 Hz.
MOCAP_HZ = 119.03

# The rate the RGB cameras in this corpus actually deliver, from ffprobe on the
# published clips: 30000/1001. This is the ceiling on any frame budget, because
# you cannot analyse a frame the camera never sent.
NATIVE_FPS = 30000.0 / 1001.0


@dataclass(frozen=True)
class Segment:
    """One annotated action, in seconds from the start of the session."""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Session:
    """One recording of one participant working through the assembly job."""

    session_id: str  # e.g. "P01_R01"
    segments: tuple[Segment, ...]

    @property
    def participant(self) -> str:
        return self.session_id.split("_")[0]

    @property
    def span(self) -> float:
        """Wall-clock seconds from the first action starting to the last ending."""
        return self.segments[-1].end - self.segments[0].start

    @property
    def active_time(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def active_fraction(self) -> float:
        return self.active_time / self.span

    def gaps(self) -> list[tuple[float, float]]:
        """Unannotated intervals between consecutive actions."""
        out = []
        for a, b in zip(self.segments, self.segments[1:]):
            if b.start > a.end:
                out.append((a.end, b.start))
        return out


def load_session(path: Path, hz: float = MOCAP_HZ) -> Session:
    raw = json.loads(path.read_text())
    (session_id,) = raw.keys()
    segs = tuple(
        Segment(
            start=s["Action_start_bvh_frame"] / hz,
            end=s["Action_end_bvh_frame"] / hz,
            label=s["label"],
        )
        for s in raw[session_id]
    )
    if any(s.duration <= 0 for s in segs):
        raise ValueError(f"{session_id}: non-positive segment duration")
    return Session(session_id=session_id, segments=segs)


def load_corpus(root: Path, hz: float = MOCAP_HZ) -> list[Session]:
    paths = sorted(Path(root).glob("P*_R*.json"))
    if not paths:
        raise FileNotFoundError(f"no session files under {root}; run tools/fetch_data.py")
    return [load_session(p, hz) for p in paths]


def fit_frame_rate(json_path: Path, csv_path: Path) -> dict:
    """Regress mocap frames against the independent per-second annotation.

    Returns the least-squares rate through the origin plus the worst residual,
    so the caller can assert the two sources actually describe the same session
    rather than trusting that they do.
    """
    raw = json.loads(json_path.read_text())
    (sid,) = raw.keys()
    js = raw[sid]
    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) != len(js):
        raise ValueError(f"segment count differs: csv {len(rows)} vs json {len(js)}")

    mismatches = sum(
        1
        for r, j in zip(rows, js)
        if r["Meta_action_label"].split("] ")[-1] != j["label"]
    )

    pairs = []
    for r, j in zip(rows, js):
        for col, key in (
            ("Start Time (s)", "Action_start_bvh_frame"),
            ("End Time (s)", "Action_end_bvh_frame"),
        ):
            v = r[col].strip()
            if not v:
                continue  # the published CSV truncates the final end time
            t = float(v)
            if t > 0:
                pairs.append((t, float(j[key])))

    num = sum(t * f for t, f in pairs)
    den = sum(t * t for t, _ in pairs)
    hz = num / den
    resid = [abs(f - hz * t) for t, f in pairs]
    # The last segment is the truncated one; report the residual over the rest.
    trimmed = sorted(resid)[:-1]
    return {
        "session": sid,
        "segments": len(js),
        "label_mismatches": mismatches,
        "timing_pairs": len(pairs),
        "fitted_hz": hz,
        "max_residual_frames": max(trimmed),
        "max_residual_seconds": max(trimmed) / hz,
    }
