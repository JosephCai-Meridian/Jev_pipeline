"""Take every fourth MP4 frame and stack RGB frames into a PyTorch tensor.

Install from this directory: pip install .
Run: python video_preprocessing.py input.mp4 --output frames.pt

Select frame indices 0, 4, 8, ... without padding or repeating frames.
This gives 8 fps for a 32 fps source, or 7.5 fps for a 30 fps source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch


def preprocess_video(
    video_path: str | Path,
    size: tuple[int, int] | None = (224, 224),
    frame_stride: int = 4,
    local: bool = False,
) -> torch.Tensor:
    """Return float32 RGB pixels in [0, 1].

    Args:
        video_path: Input MP4 file.
        size: Output (height, width); None preserves the source dimensions.
        frame_stride: Take one frame every this many source frames.
        local: Also save the returned tensor to parsed_dataset/<video_stem>.pt
            under the current working directory. An existing file is replaced.

    Returns [T, 3, H, W] with no padding. The entire output is held in CPU
    memory; T is the number of sampled frames.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Input must be an .mp4 file")
    if size is not None and (len(size) != 2 or any(n <= 0 for n in size)):
        raise ValueError("size must contain a positive height and width")
    if not isinstance(frame_stride, int) or frame_stride <= 0:
        raise ValueError("frame_stride must be a positive integer")

    cap = cv2.VideoCapture(str(path))
    frames: list[torch.Tensor] = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if size is not None:
                    rgb = cv2.resize(rgb, (size[1], size[0]))
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                frames.append(tensor)
            frame_index += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from: {path}")
    parsed_video = torch.stack(frames).to(torch.float32).div_(255.0)
    if local:
        output_dir = Path("parsed_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(parsed_video, output_dir / f"{path.stem}.pt")
    return parsed_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Output tensor path (default: frames.pt unless --local is used)")
    parser.add_argument("--local", action="store_true",
                        help="Save to parsed_dataset/<video_stem>.pt")
    parser.add_argument("--size", type=int, nargs=2, default=(224, 224),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--stride", type=int, default=4,
                        help="Take one frame every N source frames (default: 4)")
    args = parser.parse_args()
    if args.output is not None and args.output.resolve() == args.video.resolve():
        parser.error("Output must differ from the input video")
    result = preprocess_video(args.video, tuple(args.size), args.stride, local=args.local)
    if args.local:
        print(f"Saved {tuple(result.shape)} ({result.dtype}) to "
              f"{Path('parsed_dataset') / (args.video.stem + '.pt')}")
    output = args.output if args.output is not None else (None if args.local else Path("frames.pt"))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
        print(f"Saved {tuple(result.shape)} ({result.dtype}) to {output}")


if __name__ == "__main__":
    main()
