"""Getting the count back without buying the frames back.

The bias measured in analysis.py comes from counting: a work action is counted
only if a sampled frame lands inside it, and short actions land inside fewer
sampling phases. Fast operators do short actions, so fast operators are
undercounted.

Time-in-work does not have this problem. Uniform sampling estimates the share of
the shift spent working without bias at any rate, because it is an average over
instants rather than a count of events.

So estimate the count from the time instead of counting:

    actions ~= time observed working / mean duration of one action

which needs one number the cheap stream cannot supply: the operator's mean
action duration. Take it from a duty cycle - run at full rate for a short window
every so often, cheap the rest of the time - and pay for it in the cost model.

`duty_cycle_cost_ratio` accounts for those calibration frames, so the corrected
estimator is never credited with a saving it did not make.
"""

from __future__ import annotations

import numpy as np

from .corpus import NATIVE_FPS, Session
from .sampling import label_timeline


def calibration_windows(
    session: Session,
    window_s: float,
    period_s: float,
    jitter: bool = True,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Full-rate windows of `window_s` seconds once every `period_s` seconds.

    `jitter` places each window at a random offset inside its period instead of
    at the start of it. This matters more than it looks. The job on this line is
    cyclical, so a calibration schedule on a fixed period lands on the same
    phase of the work cycle every time and samples the same few actions. On this
    corpus that alone biases the calibrated mean duration by +5.4%; jittering
    the offset takes it to +0.2% while collecting *fewer* calibration actions.
    See results/calibration.json.
    """
    rng = rng or np.random.default_rng(0)
    t0 = session.segments[0].start
    t1 = session.segments[-1].end
    out = []
    t = t0
    while t < t1:
        # Room left in this period, and in the session, for a whole window.
        room = min(t + period_s, t1) - t - window_s
        off = rng.uniform(0.0, room) if (jitter and room > 0) else 0.0
        a = min(t + off, max(t1 - window_s, t0))
        b = min(a + window_s, t1)
        if b > a:
            out.append((a, b))
        t += period_s
    return out


def calibrated_mean_duration(
    session: Session,
    windows: list[tuple[float, float]],
    policy: str = "run_to_completion",
) -> float | None:
    """Mean action duration seen inside the full-rate windows.

    Two policies, because the obvious one is wrong:

    `contained` keeps only actions that both start and end inside a window. A
    long action is more likely to straddle a boundary than a short one, so this
    throws away long actions preferentially, underestimates the mean duration,
    and therefore *overestimates* the count. Measured in results/calibration.json.

    `run_to_completion` keeps every action that *starts* inside a window and
    holds the full frame rate until it ends. Selection then depends only on
    where the action started, which is independent of how long it runs, so the
    mean is unbiased. It costs the overrun frames, which
    `duty_cycle_cost_ratio` charges for.
    """
    if policy == "contained":
        durs = [
            s.duration
            for s in session.segments
            for a, b in windows
            if s.start >= a and s.end <= b
        ]
    elif policy == "run_to_completion":
        durs = [
            s.duration
            for s in session.segments
            if any(a <= s.start < b for a, b in windows)
        ]
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return float(np.mean(durs)) if durs else None


def calibration_overrun(
    session: Session, windows: list[tuple[float, float]]
) -> float:
    """Extra full-rate seconds spent letting a straddling action finish."""
    total = 0.0
    for s in session.segments:
        for a, b in windows:
            if a <= s.start < b and s.end > b:
                total += s.end - b
    return total


def estimate_work_rate(
    session: Session,
    k: int,
    window_s: float = 45.0,
    period_s: float = 300.0,
    native_fps: float = NATIVE_FPS,
    policy: str = "run_to_completion",
    jitter: bool = True,
    seed: int = 0,
) -> dict:
    """Naive counted work rate versus the time-based corrected one.

    Averaged over every sampling phase, as elsewhere.
    """
    t0 = session.segments[0].start
    t1 = session.segments[-1].end
    span = t1 - t0
    n_frames = int(span * native_fps)
    frame_times = t0 + np.arange(n_frames) / native_fps
    truth = label_timeline(session, frame_times)

    windows = calibration_windows(
        session, window_s, period_s, jitter, np.random.default_rng(seed)
    )
    mean_dur = calibrated_mean_duration(session, windows, policy)
    true_mean_dur = float(np.mean([s.duration for s in session.segments]))
    true_rate = 60.0 * len(session.segments) / span

    naive, corrected = [], []
    for phase in range(k):
        lab = truth[phase::k]
        if lab.size == 0:
            continue
        changes = np.flatnonzero(np.diff(lab)) + 1
        bounds = np.concatenate(([0], changes, [lab.size]))
        counted = int((lab[bounds[:-1]] >= 0).sum())
        naive.append(60.0 * counted / span)
        if mean_dur:
            active_time = float((lab >= 0).mean()) * span
            corrected.append(60.0 * (active_time / mean_dur) / span)

    return {
        "session": session.session_id,
        "participant": session.participant,
        "k": k,
        "analysed_fps": native_fps / k,
        "true_work_rate": true_rate,
        "naive_work_rate": float(np.mean(naive)),
        "corrected_work_rate": float(np.mean(corrected)) if corrected else float("nan"),
        "calibrated_mean_duration": mean_dur,
        "true_mean_duration": true_mean_dur,
        "n_calibration_windows": len(windows),
        "calibration_policy": policy,
        "calibration_jitter": jitter,
        "n_calibration_actions": sum(
            1
            for sg in session.segments
            if any(a <= sg.start < b for a, b in windows)
        ),
        "calibration_overrun_s": calibration_overrun(session, windows),
        "span": span,
    }


def duty_cycle_cost_ratio(
    k: int, window_s: float, period_s: float, overrun_fraction: float = 0.0
) -> float:
    """Share of native frames analysed, counting the full-rate calibration.

    Cheap frames cost 1/k of native. The calibration windows cost full rate for
    window_s out of every period_s, plus `overrun_fraction` of the session spent
    letting a straddling action finish.
    """
    duty = min(window_s / period_s + overrun_fraction, 1.0)
    return duty * 1.0 + (1.0 - duty) * (1.0 / k)
