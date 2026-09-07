# frame-budget-bench

**A camera watching a production line costs money per frame it looks at, so
every deployment looks at fewer. This measures what that costs you, and finds
that the bill is not split evenly: the faster an operator works, the more of
their work the cheap pipeline fails to see.**

The pipeline here is assumed to be **perfect**. Every frame it analyses gets its
true label. So nothing below is a claim about anyone's model accuracy, and none
of it can be answered by a more accurate model, because there is no accuracy
left to add. It is the error remaining after the vision problem is solved, and a
real classifier's own mistakes land on top of it.

**Live console: [frame-budget-bench.vercel.app](https://frame-budget-bench.vercel.app)**
— pull the frame budget down and watch sixteen real operators' measured output
drift away from their real output, then switch the estimator and watch it come
back.

---

## The corpus

<!--AUTO:CORPUS-->
38 recording sessions of 16 people doing the same assembly job, 4,803 annotated work actions over 4.50 hours. Median action 1.92 s, a tenth of them under 0.84 s. The cameras run at 29.97 fps.
<!--/AUTO:CORPUS-->

[InHARD](https://doi.org/10.5281/zenodo.4003541) (Dallel et al., IEEE ICHMS
2020, CC-BY-4.0) is industrial assembly work, annotated action by action on a
motion capture clock. The per-session annotations used here are the ones
published with [GSK-C2F](https://github.com/ToufikBenmessabih/GSK-C2F); the
video used for the timings is the [published
sample](https://github.com/Maeng-Su/InHARD-sample).

Two separately published copies of session `P01_R01` exist, one in mocap frames
and one in seconds. Regressing them against each other is how the frame rate
here was set rather than assumed, and is also the check that the two sources
describe the same recording:

<!--AUTO:FIT-->
109 segments, 0 label mismatches, fitted rate 119.00 Hz, worst residual 0.119 s across 217 boundaries
<!--/AUTO:FIT-->

## What a frame budget costs

<!--AUTO:HEADLINE-->
| frame budget | cost per camera | work actions seen | of the fastest operator's | of the slowest operator's |
|---|---|---|---|---|
| every frame (30.0 fps) | 1.00x | 100% | 100% | 100% |
| 6.0 fps | 0.20x | 99.98% | 99.83% | 100.00% |
| 1.0 fps | 0.033x | 94.4% | **92.8%** | 93.6% |
| 0.33 fps | 0.011x | 64.8% | **54.5%** | 73.1% |
<!--/AUTO:HEADLINE-->

## First, the part that is free

<!--AUTO:FREE-->
Down to 10 analysed frames a second - a 3x cut - this corpus loses nothing at all: action recall 100.00%, and not one of the sixteen operators below 100.00%. At 6 fps, a 5x cut, recall is 99.98% and the worst-served operator is still at 99.83%. Measured on real footage with a real model, that 5x is the difference between 6.2 and 28 cameras on one accelerator.
<!--/AUTO:FREE-->

<!--AUTO:BENCH-->
| execution provider | ms per frame | frames per second |
|---|---|---|
| CPU | 19.91 | 50.2 |
| CoreML | 5.30 | 188.5 |

`yolov8n-pose.onnx`, 120 real frames from this corpus, median of 3 passes, Apple M4.
<!--/AUTO:BENCH-->

<!--AUTO:CAMERAS-->
| frame budget | cameras on one accelerator | if decode were free | decode's share of the bill |
|---|---|---|---|
| 29.97 fps | **6.2** | 6.3 | 2.2% |
| 15.00 fps | **12.0** | 12.6 | 4.3% |
| 6.00 fps | **28.3** | 31.4 | 10.0% |
| 2.00 fps | **70.7** | 94.3 | 25.0% |
| 1.00 fps | **113.1** | 188.5 | 40.0% |
| 0.50 fps | **161.6** | 377.0 | 57.1% |
| 0.33 fps | **189.2** | 571.3 | 66.9% |

At 5.30 ms per analysed frame and a decode floor of 3.53 ms per second of video.
<!--/AUTO:CAMERAS-->

The milliseconds are a property of this laptop and this model and do not
transfer. The console takes your own per-frame time as an input.

### Inference is linear in the frame budget. Decode is not.

Analysing fewer frames does not mean decoding fewer. The stream still arrives in
real time and every frame still has to be walked past to stay in sync; only the
colour conversion and the resize are skipped.

<!--AUTO:DECODE-->
| frames converted | decode cost per second of video |
|---|---|
| every frame | 9.99 ms |
| 1 in 2 | 6.58 ms |
| 1 in 5 | 4.50 ms |
| 1 in 10 | 4.05 ms |
| 1 in 30 | 3.53 ms |

Analysing 30x fewer frames makes decoding only 2.8x cheaper, not 30x.
<!--/AUTO:DECODE-->

That floor is what stops the curve, and it makes the interesting part of this
question much narrower than it looks:

<!--AUTO:FLOOR-->
Going from 1 analysed frame a second down to 0.33 looks like a 3.0x win on inference alone - 189 cameras to 571. Once decode is charged it is 1.67x: 113 to 189. For that you give up 45.2% of the fastest quartile's recorded output instead of 8.2%. **The deep cuts do not pay, and the reason is not statistical, it is that the bill stops falling.**
<!--/AUTO:FLOOR-->

## Then, the part that is not

Below roughly two analysed frames a second, work actions start being missed
entirely. An action is seen only if an analysed frame lands inside it, so an
action shorter than the gap between frames is a coin toss. Measured exactly, the
chance of seeing an action of duration `d` at sampling interval `i` is
`min(1, d / i)`, which `tests/test_all.py` asserts holds on the real segments to
within 0.05.

That is not, by itself, interesting. This is: **fast work is made of short
actions**, so the loss lands on the people doing the most.

<!--AUTO:DISCRIM-->
| analysed fps | correlation of recall with how fast the operator works | p |
|---|---|---|
| 5.99 | -0.151 | 0.58 |
| 3.00 | -0.569 | 0.021 |
| 2.00 | -0.593 | 0.015 |
| 1.50 | -0.609 | 0.012 |
| 1.00 | -0.591 | 0.016 |
| 0.67 | -0.729 | 0.0013 |
| 0.50 | -0.844 | 3.9e-05 |
| 0.33 | -0.921 | 4.3e-07 |
| 0.25 | -0.941 | 5.6e-08 |
<!--/AUTO:DISCRIM-->

<!--AUTO:QUARTILE-->
| analysed fps | actions seen, four fastest operators | four slowest | shortfall ratio |
|---|---|---|---|
| 2.00 | 96.6% | 99.1% | 3.97x |
| 1.50 | 95.0% | 98.4% | 3.21x |
| 1.00 | 91.8% | 96.9% | 2.60x |
| 0.67 | 84.1% | 93.7% | 2.53x |
| 0.50 | 73.1% | 88.7% | 2.39x |
| 0.33 | 54.8% | 74.7% | 1.78x |
| 0.25 | 43.1% | 61.9% | 1.49x |
<!--/AUTO:QUARTILE-->

<!--AUTO:GAP-->
At 1.0 fps the busiest operator in the corpus (P08) has 11.5% of their work actions go unrecorded, against 0.6% for the least busy (P01) - a gap of 11.0 percentage points. At 0.33 fps the correlation reaches -0.921 (p = 4e-07).
<!--/AUTO:GAP-->

Time spent working does not have this problem. Uniformly sampled instants
estimate the share of a shift someone was working without bias at any rate,
because it is an average rather than a count. Counting events is what breaks.
Both metrics come off the same frames, so a system that reports both is right
about one and wrong about the other, and there is nothing in the output to say
which.

## Why this costs money and not only fairness

Separating operators is what the product is for. A dashboard where everyone
looks alike has stopped working even if each individual number is only a little
off. So the metric that matters commercially is how much of the real gap between
the best and worst operator survives:

<!--AUTO:SPREAD-->
| analysed fps | measured best/worst ratio | share of the real gap kept |
|---|---|---|
| 29.97 | 1.955 | 100.0% |
| 5.99 | 1.955 | 100.0% |
| 2.00 | 1.904 | 94.7% |
| 1.00 | 1.852 | 89.2% |
| 0.67 | 1.752 | 78.7% |
| 0.50 | 1.608 | 63.6% |
| 0.33 | 1.435 | 45.6% |
| 0.20 | 1.314 | 32.9% |
| 0.10 | 1.232 | 24.3% |

The true ratio is 1.955.
<!--/AUTO:SPREAD-->

## The fix, and where it stops being worth it

Estimate the count from the time instead of counting:

```
actions ~= time observed working / mean duration of one action
```

The mean duration is the one number the cheap stream cannot supply. Take it from
a duty cycle: run at full rate for a short window now and then, cheap the rest of
the time. `duty_cycle_cost_ratio` charges for those calibration frames, so the
correction is never credited with a saving it did not make.

<!--AUTO:FIX-->
| analysed fps | counted, mean bias | counted, correlation with speed | corrected, mean bias | corrected, correlation |
|---|---|---|---|---|
| 1.00 | -4.9% | -0.59 (p=0.02) | +0.03% | -0.29 (p=0.27) |
| 0.50 | -17.8% | -0.84 (p=4e-05) | +0.03% | -0.29 (p=0.27) |
| 0.33 | -33.9% | -0.92 (p=4e-07) | +0.03% | -0.29 (p=0.27) |
<!--/AUTO:FIX-->

The correlation with operator speed is gone. The mean bias is under a tenth of a
percent.

**Two things about the calibration were wrong the first time and are worth
stating, because neither shows up in review.**

Keeping only the actions that fit entirely inside a calibration window throws
away long actions preferentially, since a long action is likelier to straddle the
edge. That underestimates the mean duration and so overestimates the count. Keep
every action that *starts* inside the window and hold the frame rate until it
ends.

And the schedule itself has to be irregular:

<!--AUTO:ALIAS-->
A 45-second calibration window every 300 seconds, placed at the start of each period, biases the calibrated mean action duration by +5.22%. Placing the same window at a random offset inside the period takes that to -0.09%, while collecting 25.5 calibration actions instead of 28.5 - fewer measurements, a better answer.
<!--/AUTO:ALIAS-->

The job on a line is cyclical, so a calibration window on a fixed period lands on
the same phase of the work cycle every time and measures the same few actions
over and over. This was the larger of the two effects and it was invisible until
the numbers were run.

<!--AUTO:CROSSOVER-->
The correction is not free and is not always right. At 2.00 analysed fps plain counting still keeps more of the real spread than it does, 94.7% against 94.3%, so counting is the better estimator and the calibration frames are wasted. At 1.50 fps that reverses: 92.8% against 94.3%. **It earns its keep only below about 1.5 fps**, and by 0.33 fps counting is down to 45.6% while the correction has not moved.
<!--/AUTO:CROSSOVER-->

## Where to run

<!--AUTO:RECOMMEND-->
On this corpus and this hardware, **6 analysed frames a second**. Action recall 99.98%, no operator below 99.83%, the full best-to-worst spread intact at 100.0%, and 28 cameras on the accelerator against 6.2 at native rate.

Both failure modes start at the same place. 3 fps is the first budget where the measured spread between operators drops below its true value (97.7%), and it is also the first where the loss stops being spread evenly across people (rho -0.57, p = 0.021). Above it nothing is wrong; below it both things go wrong together.
<!--/AUTO:RECOMMEND-->

## What this is not

- **It is not a claim about anyone's accuracy.** The classifier is assumed
  perfect. Real per-frame error adds to everything here; none of it subtracts.
- **It is bench assembly, not a garment or shoe line.** The mechanism depends on
  the distribution of action durations, which is published in
  `results/corpus.json` so the transfer can be judged rather than asserted. Work
  built from shorter actions than this would be hit harder, not less hard, but
  this repository does not measure any other line and does not claim to.
- **Sampling here is uniform.** A motion-triggered or event-driven sampler is a
  different policy with a different bias, and is not measured here.
- **Sixteen people, one job, 4.5 hours.** The correlations are significant on
  that corpus and it is a small corpus.
- **The timings are from one laptop.** Cost per frame does not transfer; the
  linearity does.

## Running it

```
make setup      # venv and dependencies
make data       # fetch the corpus (~17 MB) and the model
make results    # write every published number to results/
make bench      # measure inference cost on the real clips
make docs       # substitute those numbers into README.md and FINDINGS.md
make test       # 14 tests, including that the docs match the results
make web        # rebuild the console's data file
```

Everything runs offline after `make data`, with no account, no API key and no
paid data.

## Licence

Code MIT. The InHARD annotations and clips are CC-BY-4.0 from their authors and
are fetched, not vendored; see `tools/fetch_data.py` and `CITATION.md`.

The benchmark model, `yolov8n-pose`, is **AGPL-3.0**. Its weights are downloaded
at run time and never redistributed here, no Ultralytics code is used, and it
contributes exactly one number to the whole repository: how many milliseconds one
frame takes. That number is an input you can replace with your own, so nothing in
the findings carries an AGPL obligation. `CITATION.md` sets this out in full.
