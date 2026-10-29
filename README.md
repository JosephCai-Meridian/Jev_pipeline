# Jev video evaluation pipeline

Sample MP4 videos, stack RGB frames into PyTorch tensors, and classify gym
exercises using TypeSafe Jev with user-defined candidate options.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
cp .env.example .env
```

Set `TYPESAFE_API_KEY` in `.env`. `JEV_API` is also accepted. Do not commit keys.
API calls send sampled video frames to TypeSafe and may incur charges.

## Preprocessing

```bash
python video_preprocessing.py dataset/squat.mp4 --local
```

Selects frames 0, 4, 8, ... without padding. Returns a float32 RGB tensor in
`[0, 1]` with shape `[T, 3, H, W]`. Every fourth frame gives 8 FPS for a 32 FPS
source and 7.5 FPS for a 30 FPS source. Default image size is 224 x 224.

With `local=True` or `--local`, the tensor is also saved to
`parsed_dataset/<video_stem>.pt` in the current working directory.

```python
from video_preprocessing import preprocess_video

frames = preprocess_video("dataset/squat.mp4", local=True)
```

## Evaluate one video

```bash
python evaluation.py dataset/squat.mp4 \
  --options "squat" "bench press" "deadlift" \
  --label "squat" --local
```

Alternatively use `--options-file options.example.json`. The JSON must be a
list of exercise names. `--label` is optional ground truth for scoring.

Jev receives ordered JPEG frames at 280 x 280 and a Choice question, rather
than the PyTorch tensor itself. Ground truth and filenames are not sent to the
model. The default model is `jev-1.14.0`; images use TypeSafe's experimental
image interface. See https://docs.typesafe.ai/images.

## Evaluate a dataset

Edit `OPTION_SET` at the top of `evaluate_dataset.py`, create `dataset/`, and
add MP4 videos. Each clip should contain one main exercise.

```bash
python evaluate_dataset.py
```

The script recursively discovers MP4 files. Filenames become ground truth:
`squat.mp4` and `squat_001.mp4` map to `squat`, while
`bench_press_002.mp4` maps to `bench press`. Use `--label-regex` with a named
`label` group to customize parsing. All labels are checked against `OPTION_SET`
before any API calls.

Batch evaluation enables `local=True`. Results are written to:

- `parsed_dataset/`: saved tensors.
- `dataset_results/predictions.jsonl`: per-video predictions, probabilities,
  token usage, and errors.
- `dataset_results/metrics.json`: accuracy, macro/weighted F1, per-class
  one-vs-rest AUROC, macro AUROC, and coverage.

Metrics use successfully evaluated clips only. Macro F1 includes every
candidate class. AUROC is `null` where positive or negative examples are
missing; macro AUROC is `null` unless every candidate's AUROC is defined.
Long clips may exceed API request/context limits; use shorter clips.

Videos, tensors, results, credentials, and the local `test/` folder are ignored
by Git. Local checks, when present, run with
`python -m unittest discover -s test -v`.
