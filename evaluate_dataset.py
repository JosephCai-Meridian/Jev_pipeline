"""Evaluate all MP4s under dataset/ using their filenames as ground truth.

Edit OPTION_SET below, then run: python evaluate_dataset.py
Names such as squat.mp4 and squat_001.mp4 map to 'squat'; underscores and
hyphens become spaces. --label-regex overrides this rule (named group: label).
Ground truth and filenames are never included in Jev's state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from evaluation import evaluate_video, validate_options


# EDIT HERE: the same fixed candidate set is used for every video.
OPTION_SET = [
    "squat",
    "bench press",
    "deadlift",
    "biceps curl",
    "overhead press",
]

DEFAULT_LABEL_REGEX = r"(?P<label>.+?)(?:[_ -]\d+)?"


def normalize_label(text: str) -> str:
    return " ".join(re.sub(r"[_-]+", " ", text).casefold().split())


def collect_videos(dataset: Path, options: list[str], label_regex: str) -> list[tuple[Path, str]]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset}")
    pattern = re.compile(label_regex)
    if "label" not in pattern.groupindex:
        raise ValueError("label-regex must contain a named group: (?P<label>...)")
    label_map = {normalize_label(option): option for option in options}
    if len(label_map) != len(options):
        raise ValueError("OPTION_SET contains names that become identical after filename normalization")
    videos = sorted(p for p in dataset.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")
    if not videos:
        raise ValueError(f"No MP4 videos found in {dataset}")
    samples = []
    invalid = []
    for path in videos:
        match = pattern.fullmatch(path.stem)
        name = match.group("label") if match else None
        label = label_map.get(normalize_label(name)) if name else None
        if label is None:
            invalid.append(str(path))
        else:
            samples.append((path, label))
    if invalid:
        raise ValueError("Filename labels do not match OPTION_SET; fix names/options or --label-regex "
                         "before API calls:\n" + "\n".join(invalid))
    return samples


def validate_probabilities(probabilities: dict, options: list[str]) -> list[float]:
    if not isinstance(probabilities, dict) or set(probabilities) != set(options):
        raise ValueError("Jev must return a probability for every candidate")
    scores = [float(probabilities[option]) for option in options]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Invalid probability values")
    if not math.isclose(sum(scores), 1.0, abs_tol=1e-3):
        raise ValueError("Candidate probabilities must sum to 1")
    return scores


def compute_metrics(records: list[dict], options: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

    if not records:
        return {"accuracy": None, "f1_macro": None, "f1_weighted": None,
                "auroc_macro_ovr": None, "auroc_per_class": {},
                "auroc_note": "No successfully evaluated videos"}
    truth = [r["label"] for r in records]
    predictions = [r["prediction"] for r in records]
    scores = [validate_probabilities(r["probabilities"], options) for r in records]
    per_class_auc = {}
    for index, option in enumerate(options):
        binary_truth = [int(label == option) for label in truth]
        per_class_auc[option] = (
            float(roc_auc_score(binary_truth, [row[index] for row in scores]))
            if len(set(binary_truth)) == 2 else None
        )
    auc_defined = all(value is not None for value in per_class_auc.values())
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "f1_macro": float(f1_score(truth, predictions, labels=options, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(truth, predictions, labels=options, average="weighted", zero_division=0)),
        "auroc_macro_ovr": sum(per_class_auc.values()) / len(options) if auc_defined else None,
        "auroc_per_class": per_class_auc,
        "auroc_note": (None if auc_defined else
                       "Macro AUROC undefined: every candidate needs positive and negative examples. "
                       "Undefined classes are null; they are not silently omitted from the macro average."),
        "classification_report": classification_report(
            truth, predictions, labels=options, output_dict=True, zero_division=0),
    }


def run_dataset(
    dataset: Path, output_dir: Path, options: list[str], *,
    model: str = "jev-1.14.0", stride: int = 4, timeout: float = 120,
    label_regex: str = DEFAULT_LABEL_REGEX,
) -> dict:
    options = validate_options(options)
    if stride <= 0 or timeout <= 0:
        raise ValueError("stride and timeout must be positive")
    samples = collect_videos(dataset, options, label_regex)
    # Check metric dependency before any paid requests.
    from sklearn import metrics as _metrics  # noqa: F401

    output_dir.mkdir(parents=True, exist_ok=True)
    successful = []
    failures = []
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as log:
        for index, (video, label) in enumerate(samples, 1):
            print(f"[{index}/{len(samples)}] {video}", flush=True)
            try:
                # evaluate_video calls preprocess_video before sending the frame sequence.
                response = evaluate_video(video, options, model=model, label=label,
                                          stride=stride, timeout=timeout, local=True)
                answer = response["result"]
                validate_probabilities(answer["probabilities"], options)
                if answer["prediction"] not in options:
                    raise ValueError("Prediction is outside OPTION_SET")
                record = {"status": "ok", "video": str(video), "label": label,
                          "prediction": answer["prediction"],
                          "probabilities": answer["probabilities"],
                          "confidence": answer["confidence"], "model": response["model"],
                          "tensor_shape": response["tensor_shape"], "usage": response["usage"]}
                successful.append(record)
            except Exception as error:
                record = {"status": "error", "video": str(video), "label": label,
                          "error": f"{type(error).__name__}: {error}"}
                failures.append(record)
                print(f"  Failed: {record['error']}", flush=True)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
    summary = {
        "dataset": str(dataset), "options": options, "requested_model": model,
        "frame_stride": stride, "label_regex": label_regex,
        "total_videos": len(samples), "evaluated_videos": len(successful),
        "failed_videos": len(failures), "coverage": len(successful) / len(samples),
        "label_counts": dict(Counter(label for _, label in samples)),
        "metric_scope": "Successfully evaluated videos only; macro F1 includes every OPTION_SET class",
        "metrics": compute_metrics(successful, options),
        "failures": failures,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_results"))
    parser.add_argument("--model", default="jev-1.14.0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--label-regex", default=DEFAULT_LABEL_REGEX)
    args = parser.parse_args()
    summary = run_dataset(args.dataset, args.output_dir, OPTION_SET, model=args.model,
                          stride=args.stride, timeout=args.timeout, label_regex=args.label_regex)
    print(json.dumps(summary["metrics"], indent=2, allow_nan=False))
    print(f"Evaluated {summary['evaluated_videos']}/{summary['total_videos']} videos. "
          f"Results: {args.output_dir}")
    if summary["failed_videos"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
