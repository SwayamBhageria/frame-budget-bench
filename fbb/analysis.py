"""The frontier: what each metric costs, and what each metric loses.

Four metrics a camera-based productivity system sells, measured at every frame
budget against the same ground truth:

  activity_share  what share of the shift the operator was working
  work_rate       completed work actions per minute
  action_recall   whether an individual action was seen at all
  ranking         which operator the dashboard puts at the bottom

A real deployment runs one sampling phase, not the average of all of them, so
per-phase spread is reported rather than only the mean. The mean is what a
vendor would quote; the spread is what a supervisor actually gets.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .corpus import NATIVE_FPS, Session
from .sampling import observe

# Every k divides the native 29.97 fps stream. k=1 is every frame.
DEFAULT_KS = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 300, 450, 600)


def per_session(sessions: list[Session], ks=DEFAULT_KS) -> dict:
    """Raw per-session, per-k observations, keyed by k then session id."""
    out: dict[int, dict[str, dict]] = {}
    for k in ks:
        row = {}
        for s in sessions:
            o = observe(s, k)
            af_err = np.abs(o.active_fraction_est - o.true_active_fraction)
            row[s.session_id] = {
                "participant": s.participant,
                "analysed_fps": o.analysed_fps,
                "n_analysed": o.n_analysed,
                "span": s.span,
                "n_true_segments": o.n_true_segments,
                "segments_seen_mean": float(o.segments_seen.mean()),
                "segment_recall_mean": float(o.segments_seen.mean() / o.n_true_segments),
                "segment_recall_min_phase": float(o.segments_seen.min() / o.n_true_segments),
                "true_active_fraction": o.true_active_fraction,
                "active_fraction_err_mean": float(af_err.mean()),
                "active_fraction_err_p90": float(np.quantile(af_err, 0.90)),
                "active_fraction_err_max": float(af_err.max()),
                "runs_reported_mean": float(o.runs_reported.mean()),
                "true_work_rate": 60.0 * o.n_true_segments / s.span,
                "obs_work_rate_mean": float(60.0 * o.runs_reported.mean() / s.span),
            }
        out[k] = row
    return out


def by_participant(rows: dict[str, dict]) -> dict[str, dict]:
    """Pool a k's sessions up to the participant, weighting by session length."""
    acc: dict[str, dict] = defaultdict(
        lambda: {"span": 0.0, "true_seg": 0, "seen": 0.0, "true_active": 0.0}
    )
    for r in rows.values():
        a = acc[r["participant"]]
        a["span"] += r["span"]
        a["true_seg"] += r["n_true_segments"]
        a["seen"] += r["segments_seen_mean"]
        a["true_active"] += r["true_active_fraction"] * r["span"]
    out = {}
    for p, a in acc.items():
        out[p] = {
            "span": a["span"],
            "true_work_rate": 60.0 * a["true_seg"] / a["span"],
            "obs_work_rate": 60.0 * a["seen"] / a["span"],
            "action_recall": a["seen"] / a["true_seg"],
            "true_activity_share": a["true_active"] / a["span"],
        }
    return out


def rank_damage(part: dict[str, dict], key_true: str, key_obs: str) -> dict:
    """How the dashboard's ordering of operators differs from the true one."""
    ps = sorted(part)
    true_v = np.array([part[p][key_true] for p in ps])
    obs_v = np.array([part[p][key_obs] for p in ps])
    true_order = [ps[i] for i in np.argsort(true_v)]
    obs_order = [ps[i] for i in np.argsort(obs_v)]

    inversions = 0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if np.sign(true_v[i] - true_v[j]) != np.sign(obs_v[i] - obs_v[j]):
                inversions += 1
    total_pairs = len(ps) * (len(ps) - 1) // 2
    return {
        "bottom_true": true_order[0],
        "bottom_obs": obs_order[0],
        "bottom_correct": true_order[0] == obs_order[0],
        "bottom3_overlap": len(set(true_order[:3]) & set(obs_order[:3])),
        "inversions": inversions,
        "pair_disagreement": inversions / total_pairs,
        "spearman": float(
            np.corrcoef(np.argsort(np.argsort(true_v)), np.argsort(np.argsort(obs_v)))[0, 1]
        ),
    }


def frontier(sessions: list[Session], ks=DEFAULT_KS) -> dict:
    """The whole table: one row per frame budget."""
    raw = per_session(sessions, ks)
    rows = []
    for k in ks:
        r = raw[k]
        part = by_participant(r)
        recalls = {p: v["action_recall"] for p, v in part.items()}
        worst_p = min(recalls, key=recalls.get)
        best_p = max(recalls, key=recalls.get)
        # Corpus-level action recall, weighted by how many actions each has.
        seen = sum(v["segments_seen_mean"] for v in r.values())
        true = sum(v["n_true_segments"] for v in r.values())
        af_p90 = float(np.mean([v["active_fraction_err_p90"] for v in r.values()]))
        af_max = float(np.max([v["active_fraction_err_max"] for v in r.values()]))
        rows.append(
            {
                "k": k,
                "analysed_fps": NATIVE_FPS / k,
                "cost_ratio": 1.0 / k,
                "action_recall": seen / true,
                "action_recall_worst_participant": recalls[worst_p],
                "action_recall_best_participant": recalls[best_p],
                "worst_participant": worst_p,
                "best_participant": best_p,
                "recall_spread": recalls[best_p] - recalls[worst_p],
                "activity_share_err_p90": af_p90,
                "activity_share_err_max": af_max,
                "rank_work_rate": rank_damage(part, "true_work_rate", "obs_work_rate"),
                "per_participant_recall": recalls,
            }
        )
    return {"rows": rows, "raw": raw}
