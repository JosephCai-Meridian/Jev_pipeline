"""MP4 -> stride-4 PyTorch preprocessing -> TypeSafe Jev image evaluation.

API reference: https://docs.typesafe.ai/images
Each input clip should contain one main exercise. All sampled frames are sent
in temporal order, with one Choice question using the user's option list.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import uuid


def validate_options(options: list[str]) -> list[str]:
    if not isinstance(options, list) or not 1 <= len(options) <= 255:
        raise ValueError("options must be a JSON list containing 1 to 255 exercise names")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("Every option must be a non-empty string")
    options = [option.strip() for option in options]
    if len(set(option.casefold() for option in options)) != len(options):
        raise ValueError("Duplicate exercise options are not allowed")
    return options


def build_questions(options: list[str]) -> dict:
    return {
        "exercise": {
            "type": "choice",
            "instructions": (
                "Which gym exercise is the person performing in `frames`? "
                "The frames are ordered from earliest to latest."
            ),
            "criteria": {label: f"The person is performing {label}." for label in options},
        }
    }


def encode_frames(frames, stride: int) -> tuple[list[dict], dict]:
    """Encode [T, 3, H, W] RGB tensors as JPEG API assets, without resampling."""
    import cv2

    state_frames = []
    assets = {}
    for index, frame in enumerate(frames):
        rgb = frame.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError(f"JPEG encoding failed at sampled frame {index}")
        reference = str(uuid.uuid4())
        assets[reference] = {
            "type": "image", "format": "jpg",
            "data_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
        state_frames.append({"source_frame_index": index * stride, "image": reference})
    return state_frames, assets


def summarize_answers(response: dict, options: list[str], label: str | None) -> dict:
    answer = response["answers"]["exercise"]
    if answer.get("choice") not in options:
        raise ValueError("Jev returned a choice outside the user-provided options")
    eligible = label in options if label is not None else None
    return {
        "options": options,
        "prediction": answer["choice"],
        "confidence": answer.get("confidence"),
        "probabilities": answer.get("probabilities"),
        "ground_truth": label,
        "ground_truth_in_options": eligible,
        "correct": answer["choice"] == label if eligible else None,
    }


def evaluate_video(
    video_path: str | Path,
    options: list[str],
    *,
    model: str = "jev-1.14.0",
    label: str | None = None,
    stride: int = 4,
    timeout: float = 120,
    local: bool = False,
) -> dict:
    options = validate_options(options)
    import requests
    from dotenv import load_dotenv
    from video_preprocessing import preprocess_video

    load_dotenv(Path.cwd() / ".env", override=False)
    api_key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API")
    if not api_key:
        raise ValueError("Set TYPESAFE_API_KEY or JEV_API in the environment or .env before running evaluation")

    frames = preprocess_video(video_path, size=(280, 280), frame_stride=stride, local=local)
    state_frames, assets = encode_frames(frames, stride)
    payload = {
        "model": model,
        "state": {"frames": state_frames},
        "assets": assets,
        "questions": build_questions(options),
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(body) > 32 * 1024 * 1024:
        raise ValueError("Request exceeds Jev's 32 MiB limit; use a shorter video clip")
    response = requests.post(
        "https://api.typesafe.ai/v1/systemone",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        },
        data=body,
        timeout=timeout,
    )
    if not response.ok:
        # Do not include the request body, encoded frames, or credentials in errors.
        raise RuntimeError(
            f"Jev HTTP {response.status_code}. For 400/413, check image/context "
            "limits and try a shorter clip; for 401/403, check your API key/access."
        )
    raw = response.json()
    return {
        "video": str(video_path),
        "requested_model": model,
        "model": raw.get("model", model),
        "frame_stride": stride,
        "tensor_shape": list(frames.shape),
        "result": summarize_answers(raw, options, label),
        "usage": raw.get("usage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--options", nargs="+", help="Exercise names; quote names containing spaces")
    candidates.add_argument("--options-file", type=Path, help="JSON file containing a list of exercise names")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    parser.add_argument("--label", help="Optional ground truth, e.g. 'squat' or 'bench press'")
    parser.add_argument("--model", default="jev-1.14.0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--local", action="store_true", help="Save the parsed tensor under parsed_dataset/")
    args = parser.parse_args()
    if args.output.resolve() == args.video.resolve():
        parser.error("Output must differ from the input video")
    if args.stride <= 0 or args.timeout <= 0:
        parser.error("stride and timeout must be positive")
    if args.options_file and args.output.resolve() == args.options_file.resolve():
        parser.error("Output must differ from the options file")
    try:
        options = args.options
        if args.options_file:
            options = json.loads(args.options_file.read_text(encoding="utf-8"))
        options = validate_options(options)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    result = evaluate_video(args.video, options, model=args.model, label=args.label,
                            stride=args.stride, timeout=args.timeout, local=args.local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    answer = result["result"]
    print(f"{len(options)} options: {answer['prediction']} | confidence={answer['confidence']} "
          f"| correct={answer['correct']}")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
