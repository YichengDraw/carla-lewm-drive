from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact visual QC sheet for a CARLA HDF5 dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumb-size", type=int, default=168)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.98]
    with h5py.File(args.dataset, "r") as h5:
        pixels = h5["pixels"]
        ep_len = h5["ep_len"][:]
        ep_off = h5["ep_offset"][:]
        routes = h5["route_id"][:]
        actions = h5["action"][:]
        lanes = h5["lane_offset_m"][:]
        headings = h5["heading_error_rad"][:]

        thumbs: list[Image.Image] = []
        labels: list[str] = []
        for ep, (off, length) in enumerate(zip(ep_off, ep_len, strict=True)):
            for q in qs:
                idx = int(off + min(length - 1, max(0, round(q * (length - 1)))))
                thumbs.append(Image.fromarray(pixels[idx]).resize((args.thumb_size, args.thumb_size)))
                labels.append(
                    f"ep{ep} r{int(routes[idx])} {q:.0%} "
                    f"lane={float(lanes[idx]):+.2f} steer={float(actions[idx, 1]):+.2f}"
                )

        cols = len(qs)
        rows = len(ep_len)
        label_h = 26
        sheet = Image.new("RGB", (cols * args.thumb_size, rows * (args.thumb_size + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, thumb in enumerate(thumbs):
            x = (i % cols) * args.thumb_size
            y = (i // cols) * (args.thumb_size + label_h)
            sheet.paste(thumb, (x, y))
            draw.text((x + 4, y + args.thumb_size + 3), labels[i], fill=(0, 0, 0))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.output, quality=90)

        route_counts = {str(int(r)): int((routes == r).sum()) for r in sorted(set(routes.tolist()))}
        report = {
            "output": str(args.output),
            "frames": int(pixels.shape[0]),
            "ep_len": [int(x) for x in ep_len.tolist()],
            "routes": route_counts,
            "action_mean": actions.mean(axis=0).round(4).tolist(),
            "action_abs_p95": np.quantile(np.abs(actions), 0.95, axis=0).round(4).tolist(),
            "lane_abs_p99": float(np.quantile(np.abs(lanes), 0.99)),
            "heading_abs_p99": float(np.quantile(np.abs(headings), 0.99)),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
