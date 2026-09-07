"""Produce every number this repository publishes, into results/.

Nothing in the README or the console is typed by hand. `make results` writes
these files and `tools/inject.py` substitutes them into the prose, so a figure
cannot drift away from the code that produced it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .analysis import DEFAULT_KS, by_participant, frontier, per_session
from .corpus import MOCAP_HZ, NATIVE_FPS, fit_frame_rate, load_corpus
from .estimators import (
    calibrated_mean_duration,
    calibration_windows,
    duty_cycle_cost_ratio,
    estimate_work_rate,
)

SEEDS = 30


def corpus_report(root: Path, csv_path: Path) -> dict:
    sessions = load_corpus(root)
    segs = [s for ss in sessions for s in ss.segments]
    d = np.array([s.duration for s in segs])
    parts = sorted({s.participant for s in sessions})
    by_label = defaultdict(list)
    for s in segs:
        by_label[s.label].append(s.duration)
    return {
        "sessions": len(sessions),
        "participants": len(parts),
        "participant_ids": parts,
        "action_segments": len(segs),
        "total_span_s": sum(s.span for s in sessions),
        "total_span_h": sum(s.span for s in sessions) / 3600.0,
        "total_active_s": sum(s.active_time for s in sessions),
        "corpus_activity_share": sum(s.active_time for s in sessions)
        / sum(s.span for s in sessions),
        "action_classes": len(by_label),
        "duration_s": {
            "min": float(d.min()),
            "p10": float(np.quantile(d, 0.10)),
            "median": float(np.median(d)),
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "cv": float(d.std(ddof=1) / d.mean()),
            "p90": float(np.quantile(d, 0.90)),
            "max": float(d.max()),
        },
        "activity_share_range": [
            min(
                sum(x.active_time for x in sessions if x.participant == p)
                / sum(x.span for x in sessions if x.participant == p)
                for p in parts
            ),
            max(
                sum(x.active_time for x in sessions if x.participant == p)
                / sum(x.span for x in sessions if x.participant == p)
                for p in parts
            ),
        ],
        "native_fps": NATIVE_FPS,
        "mocap_hz_used": MOCAP_HZ,
        "frame_rate_fit": fit_frame_rate(root / "P01_R01.json", csv_path),
        "per_label": {
            k: {"n": len(v), "median_s": float(np.median(v)), "min_s": float(min(v))}
            for k, v in sorted(by_label.items(), key=lambda kv: -len(kv[1]))
        },
    }


def discrimination_report(root: Path, ks=(3, 5, 10, 15, 20, 30, 45, 60, 90, 120)) -> dict:
    sessions = load_corpus(root)
    rows = []
    for k in ks:
        part = by_participant(per_session(sessions, [k])[k])
        ps = sorted(part)
        rec = np.array([part[p]["action_recall"] for p in ps])
        rate = np.array([part[p]["true_work_rate"] for p in ps])
        share = np.array([part[p]["true_activity_share"] for p in ps])
        # At generous frame budgets every operator is at recall 1.0, so the
        # correlation is undefined rather than zero. Say so instead of emitting
        # a NaN that JSON cannot represent and a reader would misread.
        constant = float(rec.max() - rec.min()) < 1e-12
        sr = None if constant else stats.spearmanr(rate, rec)
        ss = None if constant else stats.spearmanr(share, rec)
        worst = ps[int(np.argmin(rec))]
        best = ps[int(np.argmax(rec))]
        # Quartile contrast: the claim is a trend over sixteen people, and the
        # two extreme individuals are a noisy way to state it.
        order = np.argsort(-rate)
        fastq = rec[order[:4]].mean()
        slowq = rec[order[-4:]].mean()
        rows.append(
            {
                "k": k,
                "analysed_fps": NATIVE_FPS / k,
                "recall_vs_work_rate_rho": None if sr is None else float(sr.statistic),
                "recall_vs_work_rate_p": None if sr is None else float(sr.pvalue),
                "recall_vs_activity_share_rho": None if ss is None else float(ss.statistic),
                "recall_vs_activity_share_p": None if ss is None else float(ss.pvalue),
                "recall_constant": constant,
                "worst_participant": worst,
                "worst_recall": float(rec.min()),
                "best_participant": best,
                "best_recall": float(rec.max()),
                "recall_gap_pp": float(100 * (rec.max() - rec.min())),
                "fastest_quartile_recall": float(fastq),
                "slowest_quartile_recall": float(slowq),
                "quartile_loss_ratio": float((1 - fastq) / (1 - slowq)) if slowq < 1 else None,
                "fastest_participant": ps[int(np.argmax(rate))],
                "fastest_recall": float(rec[int(np.argmax(rate))]),
                "slowest_participant": ps[int(np.argmin(rate))],
                "slowest_recall": float(rec[int(np.argmin(rate))]),
            }
        )
    return {"rows": rows}


def _pooled_bias(sessions, k, jitter, seed, **kw):
    rows = [estimate_work_rate(s, k, jitter=jitter, seed=seed, **kw) for s in sessions]
    agg = defaultdict(lambda: {"t": 0.0, "n": 0.0, "c": 0.0, "sp": 0.0, "ov": 0.0, "nc": 0})
    for r in rows:
        a = agg[r["participant"]]
        a["t"] += r["true_work_rate"] * r["span"]
        a["n"] += r["naive_work_rate"] * r["span"]
        a["c"] += r["corrected_work_rate"] * r["span"]
        a["sp"] += r["span"]
        a["ov"] += r["calibration_overrun_s"]
        a["nc"] += r["n_calibration_actions"]
    ps = sorted(agg)
    tr = np.array([agg[p]["t"] / agg[p]["sp"] for p in ps])
    nv = np.array([agg[p]["n"] / agg[p]["sp"] for p in ps])
    cw = np.array([agg[p]["c"] / agg[p]["sp"] for p in ps])
    ovf = sum(agg[p]["ov"] for p in ps) / sum(agg[p]["sp"] for p in ps)
    ncal = float(np.mean([r["n_calibration_actions"] for r in rows]))
    return tr, (nv - tr) / tr, (cw - tr) / tr, ovf, ncal


def correction_report(root: Path, ks=(30, 60, 90), window_s=45.0, period_s=300.0) -> dict:
    sessions = load_corpus(root)
    rows = []
    for k in ks:
        tr, bn, _, ovf, ncal = _pooled_bias(
            sessions, k, jitter=True, seed=0, window_s=window_s, period_s=period_s
        )
        bcs = [
            _pooled_bias(sessions, k, True, s, window_s=window_s, period_s=period_s)[2]
            for s in range(SEEDS)
        ]
        bc = np.mean(bcs, axis=0)
        sn = stats.spearmanr(tr, bn)
        sc = stats.spearmanr(tr, bc)
        rows.append(
            {
                "k": k,
                "analysed_fps": NATIVE_FPS / k,
                "seeds": SEEDS,
                "mean_calibration_actions": ncal,
                "cost_ratio_naive": 1.0 / k,
                "cost_ratio_corrected": duty_cycle_cost_ratio(k, window_s, period_s, ovf),
                "naive_bias_mean": float(bn.mean()),
                "naive_bias_worst": float(bn.min()),
                "naive_rho": float(sn.statistic),
                "naive_p": float(sn.pvalue),
                "corrected_bias_mean": float(bc.mean()),
                "corrected_bias_worst": float(bc.min()),
                "corrected_rho": float(sc.statistic),
                "corrected_p": float(sc.pvalue),
            }
        )
    return {"window_s": window_s, "period_s": period_s, "rows": rows}


def spread_report(root: Path, ks=DEFAULT_KS) -> dict:
    """How much of the gap between the best and worst operator survives.

    Separating operators is what the product is for, so this is the metric that
    matters commercially: a dashboard that makes everyone look alike has stopped
    doing its job even if every individual number is only slightly off.

    Retention is expressed against the true ratio, on the part above parity:
    (measured_ratio - 1) / (true_ratio - 1). A retention of 1 reproduces the
    real spread; 0 says every operator now looks identical.
    """
    sessions = load_corpus(root)
    participants = sorted({s.participant for s in sessions})
    rows = []
    for k in ks:
        part = by_participant(per_session(sessions, [k])[k])
        acc = defaultdict(lambda: {"c": 0.0, "sp": 0.0})
        for seed in range(SEEDS):
            for s in sessions:
                r = estimate_work_rate(s, k, seed=seed)
                a = acc[r["participant"]]
                a["c"] += r["corrected_work_rate"] * r["span"]
                a["sp"] += r["span"]
        true = np.array([part[p]["true_work_rate"] for p in participants])
        naive = np.array([part[p]["obs_work_rate"] for p in participants])
        corr = np.array([acc[p]["c"] / acc[p]["sp"] for p in participants])

        # Quartile means rather than the single fastest over the single slowest.
        # A max/min ratio over sixteen people is decided by two of them and is
        # not monotone in the frame budget even though the underlying loss is.
        order = np.argsort(-true)
        def ratio(v):
            return float(v[order[:4]].mean() / v[order[-4:]].mean())

        t_ratio = ratio(true)
        rows.append(
            {
                "k": k,
                "analysed_fps": NATIVE_FPS / k,
                "true_ratio": float(t_ratio),
                "naive_ratio": ratio(naive),
                "corrected_ratio": ratio(corr),
                "naive_retention": (ratio(naive) - 1) / (t_ratio - 1),
                "corrected_retention": (ratio(corr) - 1) / (t_ratio - 1),
            }
        )
    # Where the correction starts being worth its calibration frames.
    crossover = next(
        (r["analysed_fps"] for r in rows if r["naive_retention"] < r["corrected_retention"]),
        None,
    )
    return {"rows": rows, "crossover_fps": crossover}


def calibration_report(root: Path) -> dict:
    """The aliasing result: fixed-offset versus jittered calibration windows."""
    sessions = load_corpus(root)
    out = []
    for window_s, period_s in ((45.0, 300.0), (60.0, 300.0), (45.0, 150.0)):
        entry = {"window_s": window_s, "period_s": period_s}
        for jitter, seeds in ((False, 1), (True, SEEDS)):
            rel, ns = [], []
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                num = den = 0.0
                n = []
                for s in sessions:
                    w = calibration_windows(s, window_s, period_s, jitter, rng)
                    m = calibrated_mean_duration(s, w)
                    if not m:
                        continue
                    true_mean = float(np.mean([x.duration for x in s.segments]))
                    num += (true_mean / m - 1.0) * s.span
                    den += s.span
                    n.append(sum(1 for x in s.segments if any(a <= x.start < b for a, b in w)))
                rel.append(num / den)
                ns.append(float(np.mean(n)))
            entry["jittered" if jitter else "fixed"] = {
                "mean_duration_bias": float(np.mean(rel)),
                "mean_calibration_actions": float(np.mean(ns)),
                "seeds": seeds,
            }
        out.append(entry)

    # Also record the censoring result the `contained` policy produces.
    cens = []
    for policy in ("contained", "run_to_completion"):
        rng = np.random.default_rng(0)
        num = den = 0.0
        for s in sessions:
            w = calibration_windows(s, 45.0, 300.0, True, rng)
            m = calibrated_mean_duration(s, w, policy)
            if not m:
                continue
            true_mean = float(np.mean([x.duration for x in s.segments]))
            num += (true_mean / m - 1.0) * s.span
            den += s.span
        cens.append({"policy": policy, "mean_duration_bias": num / den})
    return {"schedules": out, "policies": cens}


def run_all(root: Path, csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_corpus(root)

    (out_dir / "corpus.json").write_text(
        json.dumps(corpus_report(root, csv_path), indent=2)
    )
    f = frontier(sessions, DEFAULT_KS)
    (out_dir / "frontier.json").write_text(json.dumps(f["rows"], indent=2, default=str))
    (out_dir / "discrimination.json").write_text(
        json.dumps(discrimination_report(root), indent=2)
    )
    (out_dir / "correction.json").write_text(
        json.dumps(correction_report(root), indent=2)
    )
    (out_dir / "spread.json").write_text(json.dumps(spread_report(root), indent=2))
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration_report(root), indent=2)
    )
