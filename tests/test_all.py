"""Tests that assert what the results claim, not merely that the code runs.

Run with `make test`. Every test that touches the corpus asserts a positive
expectation about it, so a silently empty or truncated download fails here
rather than producing a plausible wrong number downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fbb.analysis import by_participant, per_session
from fbb.corpus import NATIVE_FPS, fit_frame_rate, load_corpus
from fbb.estimators import (
    calibrated_mean_duration,
    calibration_windows,
    duty_cycle_cost_ratio,
    estimate_work_rate,
)
from fbb.sampling import observe, segment_visibility

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "inhard_gt"
CSV = ROOT / "data" / "InHARD_GT.csv"


@pytest.fixture(scope="module")
def sessions():
    s = load_corpus(DATA)
    assert len(s) == 38, "expected 38 sessions; the download is incomplete"
    return s


def test_corpus_is_the_size_it_should_be(sessions):
    assert len({x.participant for x in sessions}) == 16
    assert sum(len(x.segments) for x in sessions) == 4803
    assert 4.4 < sum(x.span for x in sessions) / 3600 < 4.6


def test_every_segment_is_well_formed(sessions):
    for s in sessions:
        for seg in s.segments:
            assert seg.end > seg.start
            assert seg.duration < 60, "no annotated action should run a minute"
        starts = [x.start for x in s.segments]
        assert starts == sorted(starts), f"{s.session_id} segments out of order"


def test_frame_rate_agrees_with_the_independent_annotation():
    """Two separately published copies of P01_R01 must describe one session."""
    fit = fit_frame_rate(DATA / "P01_R01.json", CSV)
    assert fit["segments"] == 109
    assert fit["label_mismatches"] == 0
    assert 118.5 < fit["fitted_hz"] < 119.5
    assert fit["max_residual_seconds"] < 0.2


def test_every_frame_analysed_loses_nothing(sessions):
    """k=1 is the control: the pipeline must reproduce the truth exactly."""
    for s in sessions[:5]:
        o = observe(s, 1)
        assert o.segments_seen.min() == o.n_true_segments
        assert abs(o.active_fraction_est.mean() - o.true_active_fraction) < 0.01


def test_recall_falls_monotonically_with_the_frame_budget(sessions):
    s = sessions[0]
    recalls = [observe(s, k).segments_seen.mean() / len(s.segments) for k in (1, 30, 60, 150, 300)]
    assert recalls == sorted(recalls, reverse=True)
    assert recalls[-1] < 0.5, "sampling 0.1 fps should lose most actions"


def test_short_actions_are_the_ones_that_go_missing(sessions):
    """The whole mechanism, stated exactly: visibility is min(1, d / interval).

    A plain correlation understates this because visibility saturates at 1 for
    every action longer than the sampling interval, so assert the law itself.
    """
    k = 60
    interval = k / NATIVE_FPS
    s = sessions[0]
    vis = segment_visibility(s, k)
    dur = np.array([x.duration for x in s.segments])

    long_ = dur >= interval
    assert long_.sum() > 10
    assert vis[long_].min() == 1.0, "an action longer than the gap cannot be missed"

    short = dur < interval
    assert short.sum() > 10
    assert np.abs(vis[short] - dur[short] / interval).max() < 0.05
    assert vis[short].mean() < 0.9


def test_activity_share_survives_what_counting_does_not(sessions):
    """The two metrics must not degrade together; that is the whole point."""
    rows = per_session(sessions, [60])[60]
    recall = np.mean([r["segment_recall_mean"] for r in rows.values()])
    share_err = np.mean([r["active_fraction_err_p90"] for r in rows.values()])
    assert recall < 0.90, "counting should already be hurt at 0.5 fps"
    assert share_err < 0.05, "time-in-work should still be within 5 points"


def test_the_loss_lands_on_the_faster_operators(sessions):
    from scipy import stats

    part = by_participant(per_session(sessions, [60])[60])
    ps = sorted(part)
    rec = [part[p]["action_recall"] for p in ps]
    rate = [part[p]["true_work_rate"] for p in ps]
    r = stats.spearmanr(rate, rec)
    assert r.statistic < -0.5, "expected faster operators to lose more actions"
    assert r.pvalue < 0.01


def test_jitter_removes_the_calibration_aliasing(sessions):
    def bias(jitter, seeds):
        out = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            num = den = 0.0
            for s in sessions:
                w = calibration_windows(s, 45.0, 300.0, jitter, rng)
                m = calibrated_mean_duration(s, w)
                if not m:
                    continue
                true_mean = float(np.mean([x.duration for x in s.segments]))
                num += (true_mean / m - 1.0) * s.span
                den += s.span
            out.append(num / den)
        return float(np.mean(out))

    fixed, jittered = bias(False, 1), bias(True, 30)
    assert fixed > 0.03, "a fixed schedule on cyclical work should alias"
    assert abs(jittered) < 0.01
    assert abs(jittered) < fixed / 4


def test_calibration_always_yields_some_actions(sessions):
    """A jittered window must never fall off the end of a session."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        for s in sessions:
            w = calibration_windows(s, 45.0, 300.0, True, rng)
            assert w, f"{s.session_id}: no calibration window"
            for a, b in w:
                assert b > a
                assert a >= s.segments[0].start - 1e-9
                assert b <= s.segments[-1].end + 1e-9
            assert calibrated_mean_duration(s, w) is not None


def test_the_correction_is_unbiased_and_indifferent_to_speed(sessions):
    from collections import defaultdict

    from scipy import stats

    biases = []
    for seed in range(10):
        agg = defaultdict(lambda: {"t": 0.0, "c": 0.0, "sp": 0.0})
        for s in sessions:
            r = estimate_work_rate(s, 60, seed=seed)
            a = agg[r["participant"]]
            a["t"] += r["true_work_rate"] * r["span"]
            a["c"] += r["corrected_work_rate"] * r["span"]
            a["sp"] += r["span"]
        ps = sorted(agg)
        tr = np.array([agg[p]["t"] / agg[p]["sp"] for p in ps])
        cw = np.array([agg[p]["c"] / agg[p]["sp"] for p in ps])
        biases.append((cw - tr) / tr)
    b = np.mean(biases, axis=0)
    tr_last = tr
    assert abs(b.mean()) < 0.02, "corrected estimator should be near unbiased"
    assert stats.spearmanr(tr_last, b).pvalue > 0.05, "bias must not track speed"


def test_correction_is_charged_for_its_calibration_frames():
    cheap = duty_cycle_cost_ratio(60, 0.0, 300.0)
    with_cal = duty_cycle_cost_ratio(60, 45.0, 300.0)
    assert with_cal > cheap
    assert duty_cycle_cost_ratio(1, 45.0, 300.0) == pytest.approx(1.0)


def test_published_results_match_a_fresh_run(sessions):
    """Guards against results/ drifting away from the code."""
    p = ROOT / "results" / "discrimination.json"
    if not p.exists():
        pytest.skip("run `make results` first")
    published = {r["k"]: r for r in json.loads(p.read_text())["rows"]}
    part = by_participant(per_session(sessions, [30])[30])
    gap = 100 * (
        max(v["action_recall"] for v in part.values())
        - min(v["action_recall"] for v in part.values())
    )
    assert gap == pytest.approx(published[30]["recall_gap_pp"], abs=1e-6)


def test_native_fps_is_the_rate_the_clips_actually_have():
    assert NATIVE_FPS == pytest.approx(29.97, abs=0.01)


def test_console_data_agrees_with_the_published_results(sessions):
    """The page and the repository must not be able to disagree.

    The console recomputes spread and cameras-per-accelerator in the browser.
    Both definitions drifted from the Python once already, so pin them.
    """
    web = ROOT / "web" / "data.json"
    if not web.exists():
        pytest.skip("run `make web` first")
    d = json.loads(web.read_text())
    spread = {r["k"]: r for r in json.loads((ROOT / "results" / "spread.json").read_text())["rows"]}
    bench = json.loads((ROOT / "results" / "bench.json").read_text())

    assert d["corpus"]["participants"] == 16
    assert d["corpus"]["sessions"] == 38
    assert d["bench"]["decode_floor_ms_per_video_second"] == pytest.approx(
        bench["decode_floor_ms_per_video_second"]
    )

    for row in d["rows"]:
        ops = sorted(row["operators"], key=lambda o: -o["true"])
        q = lambda f: (  # noqa: E731
            sum(f(o) for o in ops[:4]) / 4
        ) / (sum(f(o) for o in ops[-4:]) / 4)
        tr = q(lambda o: o["true"])
        for key, field in (("naive", "naive_retention"), ("corrected", "corrected_retention")):
            ret = (q(lambda o, k=key: o[k]) - 1) / (tr - 1)
            assert ret == pytest.approx(spread[row["k"]][field], abs=1e-9), (
                f"k={row['k']} {key}: console data disagrees with results/spread.json"
            )
