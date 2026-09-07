# Findings

Five results, in the order they were found. Every number is written into this
file by `tools/inject.py` from `results/*.json`; none is typed by hand.

---

## 1. Most of the frame budget is free

<!--AUTO:FREE-->
Down to 10 analysed frames a second - a 3x cut - this corpus loses nothing at all: action recall 100.00%, and not one of the sixteen operators below 100.00%. At 6 fps, a 5x cut, recall is 99.98% and the worst-served operator is still at 99.83%. Measured on real footage with a real model, that 5x is the difference between 6.2 and 28 cameras on one accelerator.
<!--/AUTO:FREE-->

There is a wide band where the measurement is untouched and the bill falls
linearly. Anyone running these cameras at the rate they arrive is paying for
nothing.

## 2. Below about 3 fps, the loss stops being fair

Whether an action is seen at all is a function of its length. The chance of a
sampled frame landing inside an action of duration `d`, at sampling interval
`i`, is `min(1, d / i)` — asserted against the real segments in
`tests/test_all.py` to within 0.05.

Fast work is made of short actions. So the undercount lands on the operators
doing the most:

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

**On the obvious objection.** Work rate is defined as actions per minute, so it
is fair to ask whether this is circular — of course counting short things is
harder. Two answers. The mechanism is meant to be plain, and stating it plainly
is the point; what is not obvious is the size, or that it holds with a perfect
classifier. And the correlation survives a measure that is not built out of
counting at all: recall also tracks each operator's *share of time spent
working*, which is estimated from sampled instants rather than counted events.
Both columns are in `results/discrimination.json`.

## 3. The metric that pays for the product degrades first

Telling operators apart is what a productivity dashboard is for. Measured as the
ratio of the fastest quartile's output to the slowest quartile's:

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

Time-in-work, by contrast, is barely affected at any budget: uniform sampling
estimates a *share* without bias, because it averages instants instead of
counting events. Both numbers come off the same frames, and nothing in the
output distinguishes the reliable one from the unreliable one.

## 4. The count can be recovered without buying the frames back

Estimate output as `time observed working / mean duration of one action`, taking
the mean duration from short full-rate calibration windows.

<!--AUTO:FIX-->
| analysed fps | counted, mean bias | counted, correlation with speed | corrected, mean bias | corrected, correlation |
|---|---|---|---|---|
| 1.00 | -4.9% | -0.59 (p=0.02) | +0.03% | -0.29 (p=0.27) |
| 0.50 | -17.8% | -0.84 (p=4e-05) | +0.03% | -0.29 (p=0.27) |
| 0.33 | -33.9% | -0.92 (p=4e-07) | +0.03% | -0.29 (p=0.27) |
<!--/AUTO:FIX-->

The systematic penalty on faster operators disappears. Two things about the
calibration were wrong on the first attempt, and neither would have surfaced in
review:

**Censoring.** Keeping only the actions that fit entirely inside a window drops
long actions preferentially, because a long action is likelier to straddle the
edge. Keep every action that *starts* inside the window and hold the rate until
it finishes.

**Aliasing.** The job is cyclical, so a calibration window on a fixed period
lands on the same phase of the work cycle every time.

<!--AUTO:ALIAS-->
A 45-second calibration window every 300 seconds, placed at the start of each period, biases the calibrated mean action duration by +5.22%. Placing the same window at a random offset inside the period takes that to -0.09%, while collecting 25.5 calibration actions instead of 28.5 - fewer measurements, a better answer.
<!--/AUTO:ALIAS-->

This was the larger of the two effects. The delta-method term for a noisy
denominator, `CV² / n`, predicts only about a third of it; randomising the offset
removes the rest, which is what identified the cause.

<!--AUTO:CROSSOVER-->
The correction is not free and is not always right. At 2.00 analysed fps plain counting still keeps more of the real spread than it does, 94.7% against 94.3%, so counting is the better estimator and the calibration frames are wasted. At 1.50 fps that reverses: 92.8% against 94.3%. **It earns its keep only below about 1.5 fps**, and by 0.33 fps counting is down to 45.6% while the correction has not moved.
<!--/AUTO:CROSSOVER-->

## 5. The deep cuts do not pay anyway

Inference cost is linear in analysed frames. Decode is not: the stream still
arrives in real time and every frame is still walked past to stay in sync.

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

<!--AUTO:FLOOR-->
Going from 1 analysed frame a second down to 0.33 looks like a 3.0x win on inference alone - 189 cameras to 571. Once decode is charged it is 1.67x: 113 to 189. For that you give up 45.2% of the fastest quartile's recorded output instead of 8.2%. **The deep cuts do not pay, and the reason is not statistical, it is that the bill stops falling.**
<!--/AUTO:FLOOR-->

So the argument closes on itself. The band worth having is the one where nothing
breaks, and past it you are trading a real and uneven measurement error for a
saving that has already mostly stopped.

<!--AUTO:RECOMMEND-->
On this corpus and this hardware, **6 analysed frames a second**. Action recall 99.98%, no operator below 99.83%, the full best-to-worst spread intact at 100.0%, and 28 cameras on the accelerator against 6.2 at native rate.

Both failure modes start at the same place. 3 fps is the first budget where the measured spread between operators drops below its true value (97.7%), and it is also the first where the loss stops being spread evenly across people (rho -0.57, p = 0.021). Above it nothing is wrong; below it both things go wrong together.
<!--/AUTO:RECOMMEND-->

---

## Corpus and provenance

<!--AUTO:CORPUS-->
38 recording sessions of 16 people doing the same assembly job, 4,803 annotated work actions over 4.50 hours. Median action 1.92 s, a tenth of them under 0.84 s. The cameras run at 29.97 fps.
<!--/AUTO:CORPUS-->

The frame rate was fitted, not assumed. Two independently published copies of
session `P01_R01` exist — one in motion capture frames, one in seconds — and
regressing them against each other both sets the rate and confirms the two
sources describe the same recording:

<!--AUTO:FIT-->
109 segments, 0 label mismatches, fitted rate 119.00 Hz, worst residual 0.119 s across 217 boundaries
<!--/AUTO:FIT-->

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

## Limits

- The classifier is assumed perfect. Making a real one more accurate cannot
  remove any of this, because there is no accuracy left to add. A real
  classifier's own errors land on top.
- Bench assembly, not a garment or shoe line. The mechanism depends on the
  distribution of action durations, published in `results/corpus.json` so the
  transfer can be judged rather than asserted.
- Sampling is uniform. A motion-triggered sampler is a different policy with a
  different bias and is not measured here.
- Sixteen people, one job, 4.5 hours.
- Timings are from one laptop and one model. Cost per frame does not transfer.
- The correction calibrates per session. In production that means per station,
  and a station whose work changes between calibration windows will drift.
