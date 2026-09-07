"""Substitute measured numbers into the prose.

Every figure in README.md and FINDINGS.md sits between an <!--AUTO:NAME--> and a
<!--/AUTO:NAME--> marker and is written by this script from results/*.json. Nothing
is typed by hand, so a number cannot survive a change to the code that produced it.

`make check` re-runs this and fails if anything moved, which is what stops the
documents drifting behind the analysis.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def pct(x: float, dp: int = 1) -> str:
    return f"{100 * x:.{dp}f}%"


def build() -> dict[str, str]:
    corpus = load("corpus")
    disc = {r["k"]: r for r in load("discrimination")["rows"]}
    corr = {r["k"]: r for r in load("correction")["rows"]}
    spread = load("spread")
    spread_by_k = {r["k"]: r for r in spread["rows"]}
    front = {r["k"]: r for r in load("frontier")}
    bench = load("bench")

    ms = {r["provider"]: r["ms_per_frame"] for r in bench["runs"]}
    fast = min(ms.values())
    cam = lambda fps: 3600.0 / (fps * 3600.0 * fast / 1000.0)  # noqa: E731

    d = corpus["duration_s"]
    fit = corpus["frame_rate_fit"]

    out = {}

    dep = {
        round(m["analysed_fps"], 2): m
        for r in bench["runs"]
        if r["ms_per_frame"] == fast
        for m in r["deployment"]
    }

    out["CORPUS"] = (
        f"{corpus['sessions']} recording sessions of {corpus['participants']} people "
        f"doing the same assembly job, {corpus['action_segments']:,} annotated work "
        f"actions over {corpus['total_span_h']:.2f} hours. Median action "
        f"{d['median']:.2f} s, a tenth of them under {d['p10']:.2f} s. The cameras run "
        f"at {corpus['native_fps']:.2f} fps."
    )

    out["FIT"] = (
        f"{fit['segments']} segments, {fit['label_mismatches']} label mismatches, "
        f"fitted rate {fit['fitted_hz']:.2f} Hz, worst residual "
        f"{fit['max_residual_seconds']:.3f} s across {fit['timing_pairs']} boundaries"
    )

    k30, k90 = disc[30], disc[90]
    out["HEADLINE"] = (
        "| frame budget | cost per camera | work actions seen | of the fastest operator's | of the slowest operator's |\n"
        "|---|---|---|---|---|\n"
        f"| every frame ({corpus['native_fps']:.1f} fps) | 1.00x | 100% | 100% | 100% |\n"
        f"| {front[5]['analysed_fps']:.1f} fps | {front[5]['cost_ratio']:.2f}x | "
        f"{pct(front[5]['action_recall'],2)} | {pct(disc[5]['fastest_recall'],2)} | "
        f"{pct(disc[5]['slowest_recall'],2)} |\n"
        f"| {k30['analysed_fps']:.1f} fps | {front[30]['cost_ratio']:.3f}x | "
        f"{pct(front[30]['action_recall'])} | **{pct(k30['fastest_recall'])}** | "
        f"{pct(k30['slowest_recall'])} |\n"
        f"| {k90['analysed_fps']:.2f} fps | {front[90]['cost_ratio']:.3f}x | "
        f"{pct(front[90]['action_recall'])} | **{pct(k90['fastest_recall'])}** | "
        f"{pct(k90['slowest_recall'])} |"
    )

    out["FREE"] = (
        f"Down to {front[3]['analysed_fps']:.0f} analysed frames a second - a "
        f"{1/front[3]['cost_ratio']:.0f}x cut - this corpus loses nothing at all: action "
        f"recall {pct(front[3]['action_recall'],2)}, and not one of the sixteen operators "
        f"below {pct(disc[3]['worst_recall'],2)}. At {front[5]['analysed_fps']:.0f} fps, a "
        f"{1/front[5]['cost_ratio']:.0f}x cut, recall is {pct(front[5]['action_recall'],2)} "
        f"and the worst-served operator is still at {pct(disc[5]['worst_recall'],2)}. "
        f"Measured on real footage with a real model, that {1/front[5]['cost_ratio']:.0f}x "
        f"is the difference between {dep[29.97]['cameras_per_device']:.1f} and "
        f"{dep[6.0]['cameras_per_device']:.0f} cameras on one accelerator."
    )

    out["DISCRIM"] = (
        "| analysed fps | correlation of recall with how fast the operator works | p |\n"
        "|---|---|---|\n"
        + "\n".join(
            f"| {r['analysed_fps']:.2f} | {r['recall_vs_work_rate_rho']:+.3f} | "
            f"{r['recall_vs_work_rate_p']:.2g} |"
            for r in load("discrimination")["rows"]
            if r["recall_vs_work_rate_rho"] is not None
        )
    )

    out["QUARTILE"] = (
        "| analysed fps | actions seen, four fastest operators | four slowest | shortfall ratio |\n"
        "|---|---|---|---|\n"
        + "\n".join(
            f"| {r['analysed_fps']:.2f} | {pct(r['fastest_quartile_recall'],1)} | "
            f"{pct(r['slowest_quartile_recall'],1)} | "
            + (f"{r['quartile_loss_ratio']:.2f}x |" if r["quartile_loss_ratio"] else "- |")
            for r in load("discrimination")["rows"]
            if r["k"] >= 15
        )
    )

    out["GAP"] = (
        f"At {k30['analysed_fps']:.1f} fps the busiest operator in the corpus "
        f"({k30['worst_participant']}) has {pct(1-k30['worst_recall'])} of their work "
        f"actions go unrecorded, against {pct(1-k30['best_recall'])} for the least busy "
        f"({k30['best_participant']}) - a gap of {k30['recall_gap_pp']:.1f} percentage "
        f"points. At {k90['analysed_fps']:.2f} fps the correlation reaches "
        f"{k90['recall_vs_work_rate_rho']:+.3f} (p = {k90['recall_vs_work_rate_p']:.1g})."
    )

    s30, s90 = spread_by_k[30], spread_by_k[90]
    out["SPREAD"] = (
        "| analysed fps | measured best/worst ratio | share of the real gap kept |\n"
        "|---|---|---|\n"
        + "\n".join(
            f"| {r['analysed_fps']:.2f} | {r['naive_ratio']:.3f} | {pct(r['naive_retention'])} |"
            for r in spread["rows"]
            if r["k"] in (1, 5, 15, 30, 45, 60, 90, 150, 300)
        )
        + f"\n\nThe true ratio is {s30['true_ratio']:.3f}."
    )

    c30, c90 = corr[30], corr[90]
    out["FIX"] = (
        "| analysed fps | counted, mean bias | counted, correlation with speed | corrected, mean bias | corrected, correlation |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['analysed_fps']:.2f} | {r['naive_bias_mean']:+.1%} | "
            f"{r['naive_rho']:+.2f} (p={r['naive_p']:.1g}) | {r['corrected_bias_mean']:+.2%} | "
            f"{r['corrected_rho']:+.2f} (p={r['corrected_p']:.2g}) |"
            for r in load("correction")["rows"]
        )
    )

    cal = load("calibration")
    sch = cal["schedules"][0]
    out["ALIAS"] = (
        f"A {sch['window_s']:.0f}-second calibration window every "
        f"{sch['period_s']:.0f} seconds, placed at the start of each period, biases the "
        f"calibrated mean action duration by {sch['fixed']['mean_duration_bias']:+.2%}. "
        f"Placing the same window at a random offset inside the period takes that to "
        f"{sch['jittered']['mean_duration_bias']:+.2%}, while collecting "
        f"{sch['jittered']['mean_calibration_actions']:.1f} calibration actions instead "
        f"of {sch['fixed']['mean_calibration_actions']:.1f} - fewer measurements, a "
        f"better answer."
    )

    # The last budget at which plain counting still beats the correction, and
    # the first at which it does not: picked from the data, not written in.
    above = [r for r in spread["rows"] if r["naive_retention"] >= r["corrected_retention"]]
    below = [r for r in spread["rows"] if r["naive_retention"] < r["corrected_retention"]]
    hi, lo = above[-1], below[0]
    out["CROSSOVER"] = (
        f"The correction is not free and is not always right. At "
        f"{hi['analysed_fps']:.2f} analysed fps plain counting still keeps more of the "
        f"real spread than it does, {pct(hi['naive_retention'])} against "
        f"{pct(hi['corrected_retention'])}, so counting is the better estimator and the "
        f"calibration frames are wasted. At {lo['analysed_fps']:.2f} fps that reverses: "
        f"{pct(lo['naive_retention'])} against {pct(lo['corrected_retention'])}. "
        f"**It earns its keep only below about {lo['analysed_fps']:.1f} fps**, and by "
        f"{spread_by_k[90]['analysed_fps']:.2f} fps counting is down to "
        f"{pct(spread_by_k[90]['naive_retention'])} while the correction has not moved."
    )

    out["BENCH"] = (
        "| execution provider | ms per frame | frames per second |\n|---|---|---|\n"
        + "\n".join(
            f"| {r['provider'].replace('ExecutionProvider','')} | {r['ms_per_frame']:.2f} | "
            f"{r['frames_per_second']:.1f} |"
            for r in bench["runs"]
        )
        + f"\n\n`{bench['runs'][0]['model']}`, {bench['runs'][0]['frames_timed']} real "
        f"frames from this corpus, median of {bench['runs'][0]['repeats']} passes, "
        "Apple M4."
    )

    out["CAMERAS"] = (
        "| frame budget | cameras on one accelerator | if decode were free | decode's share of the bill |\n"
        "|---|---|---|---|\n"
        + "\n".join(
            f"| {m['analysed_fps']:.2f} fps | **{m['cameras_per_device']:.1f}** | "
            f"{m['cameras_per_device_inference_only']:.1f} | {pct(m['decode_share'])} |"
            for m in sorted(dep.values(), key=lambda m: -m["analysed_fps"])
        )
        + f"\n\nAt {fast:.2f} ms per analysed frame and a decode floor of "
        f"{bench['decode_floor_ms_per_video_second']:.2f} ms per second of video."
    )

    dec = {d["stride"]: d["ms_per_video_second"] for d in bench["decode"]}
    out["DECODE"] = (
        "| frames converted | decode cost per second of video |\n|---|---|\n"
        + "\n".join(
            f"| {'every frame' if k == 1 else f'1 in {k}'} | {v:.2f} ms |"
            for k, v in sorted(dec.items())
        )
        + f"\n\nAnalysing {max(dec)}x fewer frames makes decoding only "
        f"{dec[1]/dec[max(dec)]:.1f}x cheaper, not {max(dec)}x."
    )

    a, b = dep[1.0], dep[0.33]
    out["FLOOR"] = (
        f"Going from {a['analysed_fps']:.0f} analysed frame a second down to "
        f"{b['analysed_fps']:.2f} looks like a "
        f"{b['cameras_per_device_inference_only']/a['cameras_per_device_inference_only']:.1f}x "
        f"win on inference alone - {a['cameras_per_device_inference_only']:.0f} cameras to "
        f"{b['cameras_per_device_inference_only']:.0f}. Once decode is charged it is "
        f"{b['cameras_per_device']/a['cameras_per_device']:.2f}x: "
        f"{a['cameras_per_device']:.0f} to {b['cameras_per_device']:.0f}. For that you give "
        f"up {pct(1-disc[90]['fastest_quartile_recall'])} of the fastest quartile's recorded "
        f"output instead of {pct(1-disc[30]['fastest_quartile_recall'])}. **The deep cuts do "
        f"not pay, and the reason is not statistical, it is that the bill stops falling.**"
    )

    six = dep[6.0]
    bend_row = next(r for r in spread["rows"] if r["naive_retention"] < 0.98)
    bend = bend_row["analysed_fps"]
    bend_disc = next(
        r
        for r in load("discrimination")["rows"]
        if abs(r["analysed_fps"] - bend) < 1e-6
    )
    unequal = next(
        r["analysed_fps"]
        for r in load("discrimination")["rows"]
        if r["recall_vs_work_rate_p"] is not None and r["recall_vs_work_rate_p"] < 0.05
    )
    out["RECOMMEND"] = (
        f"On this corpus and this hardware, **{six['analysed_fps']:.0f} analysed frames a "
        f"second**. Action recall {pct(front[5]['action_recall'],2)}, no operator below "
        f"{pct(disc[5]['worst_recall'],2)}, the full best-to-worst spread intact at "
        f"{pct(spread_by_k[5]['naive_retention'])}, and "
        f"{six['cameras_per_device']:.0f} cameras on the accelerator against "
        f"{dep[29.97]['cameras_per_device']:.1f} at native rate.\n\n"
        + (
            f"Both failure modes start at the same place. {bend:.0f} fps is the first "
            f"budget where the measured spread between operators drops below its true "
            f"value ({pct(bend_row['naive_retention'])}), and it is also the first where "
            f"the loss stops being spread evenly across people "
            f"(rho {bend_disc['recall_vs_work_rate_rho']:+.2f}, p = "
            f"{bend_disc['recall_vs_work_rate_p']:.2g}). Above it nothing is wrong; below "
            f"it both things go wrong together."
            if abs(bend - unequal) < 1e-6
            else f"The measurement bends at {bend:.2f} fps and operators start being "
            f"served unequally at {unequal:.2f} fps."
        )
    )

    return out


def apply(paths: list[Path], values: dict[str, str], check: bool = False) -> int:
    changed = []
    for p in paths:
        if not p.exists():
            continue
        src = p.read_text()
        out = src
        for name, val in values.items():
            pat = re.compile(
                rf"(<!--AUTO:{name}-->)(.*?)(<!--/AUTO:{name}-->)", re.S
            )
            if not pat.search(out):
                continue
            out = pat.sub(lambda m: f"{m.group(1)}\n{val}\n{m.group(3)}", out)
        for m in re.finditer(r"<!--AUTO:([A-Z_]+)-->", src):
            if m.group(1) not in values:
                print(f"{p.name}: marker {m.group(1)} has no value", file=sys.stderr)
                return 2
        if out != src:
            changed.append(p.name)
            if not check:
                p.write_text(out)
    if check and changed:
        print(f"stale: {', '.join(changed)} - run `make docs`", file=sys.stderr)
        return 1
    print("docs up to date" if not changed else f"updated {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(
        apply(
            [ROOT / "README.md", ROOT / "FINDINGS.md"],
            build(),
            check="--check" in sys.argv,
        )
    )
