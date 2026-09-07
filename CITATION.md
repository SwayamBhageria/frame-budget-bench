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

## Model

[`Xenova/yolov8n-pose`](https://huggingface.co/Xenova/yolov8n-pose) — an ONNX
export of YOLOv8n-pose, used only as a realistic per-frame vision workload for
the cost measurement. Nothing in the analysis depends on its accuracy; the
classifier in the analysis is assumed perfect.
