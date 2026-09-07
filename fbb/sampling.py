"""What a camera pipeline sees when it analyses only some of the frames.

The camera delivers frames at `NATIVE_FPS`. A deployed system analyses every
k-th of them, because inference costs money per frame. This module reproduces
what the analysed subset looks like, *assuming the classifier is perfect*: every
analysed frame is given its true ground-truth label.

That assumption is the point. Everything measured downstream is therefore a
floor on the error of any real system, whatever its accuracy. A vendor cannot
answer it with a better model, because the model here is already perfect.

Sampling has a phase: analysing frames 0, k, 2k... sees something slightly
different from 1, k+1, 2k+1... Rather than draw one phase at random, every
result here is averaged over all k phases, which makes it an exact expectation
rather than a sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import NATIVE_FPS, Session

IDLE = "__idle__"


@dataclass(frozen=True)
class Observation:
    """What the pipeline recorded for one session at one operating point."""

    analysed_fps: float
    n_analysed: int
    # Per phase, the number of true action segments that got at least one
    # analysed frame inside them.
    segments_seen: np.ndarray
    n_true_segments: int
    # Per phase, the share of analysed frames that carried an action label.
    active_fraction_est: np.ndarray
    true_active_fraction: float
    # Per phase, the number of distinct action runs the pipeline would report
    # after run-length encoding the analysed labels.
    runs_reported: np.ndarray
    # Per phase, summed duration of reported action runs, in seconds.
    active_time_est: np.ndarray
    true_active_time: float


def label_timeline(session: Session, times: np.ndarray) -> np.ndarray:
    """The true label at each instant in `times`, as an index into segments.

    Returns -1 where the instant falls in a gap between annotated actions.
    """
    starts = np.array([s.start for s in session.segments])
    ends = np.array([s.end for s in session.segments])
    idx = np.searchsorted(starts, times, side="right") - 1
    out = np.where((idx >= 0) & (times < ends[np.clip(idx, 0, None)]), idx, -1)
    return out


def observe(session: Session, k: int, native_fps: float = NATIVE_FPS) -> Observation:
    """Analyse every k-th camera frame of `session`, over every phase."""
    if k < 1:
        raise ValueError("k must be >= 1")
    t0 = session.segments[0].start
    t1 = session.segments[-1].end
    # Camera frame instants across the session.
    n_frames = int((t1 - t0) * native_fps)
    frame_times = t0 + np.arange(n_frames) / native_fps
    truth = label_timeline(session, frame_times)

    n_true = len(session.segments)
    seen, active_est, runs, active_time = [], [], [], []
    dt = k / native_fps  # seconds represented by one analysed frame

    for phase in range(k):
        lab = truth[phase::k]
        if lab.size == 0:
            continue
        hit = np.unique(lab[lab >= 0])
        seen.append(hit.size)
        is_action = lab >= 0
        active_est.append(float(is_action.mean()))
        # Run-length encode: how many separate stretches of work get reported.
        # A reported run ends when the label changes, which is what a system
        # counting cycles or work episodes would see.
        changes = np.flatnonzero(np.diff(lab)) + 1
        bounds = np.concatenate(([0], changes, [lab.size]))
        run_labels = lab[bounds[:-1]]
        runs.append(int((run_labels >= 0).sum()))
        active_time.append(float(is_action.sum() * dt))

    return Observation(
        analysed_fps=native_fps / k,
        n_analysed=int(np.ceil(n_frames / k)),
        segments_seen=np.array(seen, dtype=float),
        n_true_segments=n_true,
        active_fraction_est=np.array(active_est),
        true_active_fraction=session.active_fraction,
        runs_reported=np.array(runs, dtype=float),
        active_time_est=np.array(active_time),
        true_active_time=session.active_time,
    )


def segment_visibility(session: Session, k: int, native_fps: float = NATIVE_FPS) -> np.ndarray:
    """Per true segment, the share of sampling phases that see it at all.

    A segment of duration d sampled every dt seconds is seen in about
    min(1, d/dt) of phases. This measures it exactly on the real segments
    rather than assuming the approximation.
    """
    dt = k / native_fps
    out = np.empty(len(session.segments))
    for i, seg in enumerate(session.segments):
        # Phase offsets are multiples of one camera frame.
        offsets = (np.arange(k) / native_fps)[:, None]
        # First analysed instant at or after the session start for this phase.
        first = session.segments[0].start + offsets
        # Does any analysed instant land inside [seg.start, seg.end)?
        n0 = np.ceil((seg.start - first) / dt)
        t = first + np.maximum(n0, 0) * dt
        out[i] = float(np.mean((t >= seg.start) & (t < seg.end)))
    return out
