# Sources

This repository vendors no data. `tools/fetch_data.py` downloads everything it
uses from the publishers below, all of it openly licensed.

## InHARD

The corpus. Industrial assembly work, annotated action by action.

> Dallel, M., Havard, V., Baudry, D., Savatier, X. (2020). *InHARD - Industrial
> Human Action Recognition Dataset in the Context of Industrial Collaborative
> Robotics.* IEEE International Conference on Human-Machine Systems (ICHMS).
> Dataset: https://doi.org/10.5281/zenodo.4003541 — CC-BY-4.0

The full Zenodo record is roughly 50 GB of RGB and skeleton video across twelve
archive parts. This work needs only the temporal annotations, which are
published separately and are about a megabyte:

- **Per-session annotations** (`P01_R01` … `P16_R02`), from
  [ToufikBenmessabih/GSK-C2F](https://github.com/ToufikBenmessabih/GSK-C2F),
  `ground_truth/GT_bvh/`. Action start and end frames on the motion capture
  clock.
- **A second copy of `P01_R01` in seconds**, and the RGB clips used for the
  inference timings, from
  [Maeng-Su/InHARD-sample](https://github.com/Maeng-Su/InHARD-sample).

Having the same session from two independent publishers is what makes
`fit_frame_rate` possible: the frame rate used here is regressed from one
against the other rather than assumed, and the agreement of the two label
sequences is the check that they describe the same recording.

## Model, and a licensing note worth reading before you reuse this

[`Xenova/yolov8n-pose`](https://huggingface.co/Xenova/yolov8n-pose) — an ONNX
export of YOLOv8n-pose, used only as a realistic per-frame vision workload for
the cost measurement. Nothing in the analysis depends on its accuracy; the
classifier in the analysis is assumed perfect.

**That model is AGPL-3.0** (as is Ultralytics YOLOv8 upstream), while the code in
this repository is MIT. The two do not mix, and this repository is careful not to
mix them:

- **The weights are never redistributed.** `tools/fetch_data.py` downloads them
  from Hugging Face at run time. No model file is committed here.
- **No Ultralytics code is used, included or linked.** The file is loaded as a
  plain ONNX graph through `onnxruntime`.
- **Nothing in the findings depends on it.** It supplies exactly one number, the
  milliseconds one frame takes, and that number is an *input*. Swap in any other
  detector, or your own measured per-frame time, and every conclusion holds with
  the arithmetic redone. The console takes it as a field for that reason.

So if you are evaluating this for commercial use: the analysis carries no AGPL
obligation, and the only AGPL component is a benchmark stand-in you would replace
with your own model anyway.
