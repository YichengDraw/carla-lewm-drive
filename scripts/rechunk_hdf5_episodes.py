from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split long HDF5 episodes into shorter contiguous pseudo-episodes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=900)
    parser.add_argument("--min-frames", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_segments(ep_offset: np.ndarray, ep_len: np.ndarray, chunk_frames: int, min_frames: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for offset, length in zip(ep_offset.tolist(), ep_len.tolist(), strict=True):
        start = int(offset)
        end = start + int(length)
        pos = start
        while pos < end:
            nxt = min(end, pos + chunk_frames)
            if end - nxt < min_frames and segments and pos != start:
                prev_start, _ = segments.pop()
                segments.append((prev_start, end))
                break
            if nxt - pos >= min_frames:
                segments.append((pos, nxt))
            pos = nxt
    return segments


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as src:
        ep_len = src["ep_len"][:].astype(np.int64)
        ep_offset = src["ep_offset"][:].astype(np.int64)
        n = int(src["pixels"].shape[0])
        segments = make_segments(ep_offset, ep_len, int(args.chunk_frames), int(args.min_frames))
        new_len = np.asarray([end - start for start, end in segments], dtype=np.int32)
        new_offset = np.cumsum([0, *new_len[:-1]], dtype=np.int64).astype(np.int32)

        with h5py.File(args.output, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["rechunked_from"] = str(args.input)
            dst.attrs["chunk_frames"] = int(args.chunk_frames)
            dst.attrs["min_frames"] = int(args.min_frames)

            for key in src.keys():
                if key in {"ep_len", "ep_offset", "ep_idx", "step_idx"}:
                    continue
                data = src[key]
                if data.shape and int(data.shape[0]) == n:
                    pieces = [data[start:end] for start, end in segments]
                    arr = np.concatenate(pieces, axis=0)
                    chunks = (min(256, arr.shape[0]), *arr.shape[1:]) if key == "pixels" else None
                    dst.create_dataset(key, data=arr, chunks=chunks)
                else:
                    dst.create_dataset(key, data=data[:])

            ep_idx = np.concatenate(
                [np.full(int(length), idx, dtype=np.int32) for idx, length in enumerate(new_len.tolist())]
            )
            step_idx = np.concatenate([np.arange(int(length), dtype=np.int32) for length in new_len.tolist()])
            dst.create_dataset("ep_len", data=new_len)
            dst.create_dataset("ep_offset", data=new_offset)
            dst.create_dataset("ep_idx", data=ep_idx)
            dst.create_dataset("step_idx", data=step_idx)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "source_episodes": int(len(ep_len)),
        "episodes": int(len(new_len)),
        "frames": int(new_len.sum()),
        "min_ep_len": int(new_len.min()),
        "max_ep_len": int(new_len.max()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
