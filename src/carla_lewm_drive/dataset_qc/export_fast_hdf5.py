from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CARLA HDF5 dataset optimized for random training reads.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def export_fast_hdf5(src: Path, dst: Path, chunk_frames: int = 256, overwrite: bool = False) -> dict:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        for key, value in fin.attrs.items():
            fout.attrs[key] = value
        for key in fin.keys():
            data = fin[key]
            if key == "pixels":
                chunks = (min(int(chunk_frames), int(data.shape[0])), *data.shape[1:])
                out = fout.create_dataset("pixels", shape=data.shape, dtype=data.dtype, chunks=chunks)
                for start in tqdm(range(0, data.shape[0], chunk_frames), desc="copy pixels"):
                    end = min(start + chunk_frames, data.shape[0])
                    out[start:end] = data[start:end]
            else:
                fout.create_dataset(key, data=data[()], dtype=data.dtype)
    tmp.replace(dst)
    with h5py.File(dst, "r") as f:
        frames = int(f["pixels"].shape[0])
    return {"src": str(src), "dst": str(dst), "frames": frames, "chunk_frames": int(chunk_frames)}


def main() -> None:
    args = parse_args()
    summary = export_fast_hdf5(args.src, args.dst, args.chunk_frames, args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
