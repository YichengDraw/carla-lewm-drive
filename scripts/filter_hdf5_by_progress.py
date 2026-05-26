from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a CARLA HDF5 dataset by route_progress_m ranges.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-progress-m", type=float, default=None)
    parser.add_argument("--max-progress-m", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def contiguous_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    run_start = int(indices[0])
    prev = int(indices[0])
    for raw_idx in indices[1:].tolist():
        idx = int(raw_idx)
        if idx != prev + 1:
            segments.append((run_start, prev + 1))
            run_start = idx
        prev = idx
    segments.append((run_start, prev + 1))
    return segments


def filter_by_progress(
    input_path: Path,
    output_path: Path,
    *,
    min_progress_m: float | None = None,
    max_progress_m: float | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        progress = src["route_progress_m"][:].astype(np.float32)
        ep_len = src["ep_len"][:].astype(np.int64)
        ep_offset = src["ep_offset"][:].astype(np.int64)
        n = int(src["pixels"].shape[0])

        keep_mask = np.ones(n, dtype=bool)
        if min_progress_m is not None:
            keep_mask &= progress >= float(min_progress_m)
        if max_progress_m is not None:
            keep_mask &= progress <= float(max_progress_m)

        segments: list[tuple[int, int]] = []
        for offset, length in zip(ep_offset.tolist(), ep_len.tolist(), strict=True):
            start = int(offset)
            end = start + int(length)
            keep = np.flatnonzero(keep_mask[start:end]) + start
            segments.extend(contiguous_segments(keep))

        if not segments:
            raise ValueError(
                f"No frames found in {input_path} for progress range "
                f"[{min_progress_m}, {max_progress_m}]"
            )

        new_len = np.asarray([end - start for start, end in segments], dtype=np.int32)
        new_offset = np.cumsum([0, *new_len[:-1]], dtype=np.int64).astype(np.int32)

        with h5py.File(output_path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["filtered_from"] = str(input_path)
            if min_progress_m is not None:
                dst.attrs["min_progress_m"] = float(min_progress_m)
            if max_progress_m is not None:
                dst.attrs["max_progress_m"] = float(max_progress_m)

            for key in src.keys():
                if key in {"ep_len", "ep_offset", "ep_idx", "step_idx"}:
                    continue
                data = src[key]
                if data.shape and int(data.shape[0]) == n:
                    arr = np.concatenate([data[start:end] for start, end in segments], axis=0)
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

            lane = dst["lane_offset_m"][:] if "lane_offset_m" in dst else np.asarray([], dtype=np.float32)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "min_progress_m": min_progress_m,
        "max_progress_m": max_progress_m,
        "episodes": int(len(new_len)),
        "frames": int(new_len.sum()),
        "min_ep_len": int(new_len.min()),
        "max_ep_len": int(new_len.max()),
    }
    if lane.size:
        report.update(
            {
                "lane_min": float(lane.min()),
                "lane_max": float(lane.max()),
                "lane_abs_gt_0p15": float(np.mean(np.abs(lane) > 0.15)),
                "lane_abs_gt_0p20": float(np.mean(np.abs(lane) > 0.20)),
                "lane_abs_gt_0p25": float(np.mean(np.abs(lane) > 0.25)),
                "lane_abs_gt_0p35": float(np.mean(np.abs(lane) > 0.35)),
            }
        )
    return report


def main() -> None:
    args = parse_args()
    report = filter_by_progress(
        args.input,
        args.output,
        min_progress_m=args.min_progress_m,
        max_progress_m=args.max_progress_m,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
