from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from carla_lewm_drive.schema import REQUIRED_H5_KEYS


@dataclass
class QCIssue:
    severity: str
    key: str
    episode: int | None
    frame: int | None
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CARLA Driving-LeWM HDF5 data frame by frame.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/qc"))
    parser.add_argument("--sample-frames", type=int, default=64)
    parser.add_argument("--max-collision-frames", type=int, default=0)
    parser.add_argument("--max-offroad-frame-frac", type=float, default=0.01)
    parser.add_argument("--max-red-light-frame-frac", type=float, default=0.0)
    parser.add_argument("--max-blocked-frame-frac", type=float, default=0.10)
    parser.add_argument("--allow-infractions", action="store_true", help="Disable default D0-clean infraction thresholds.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when hard quality gates fail.")
    return parser.parse_args()


def validate_hdf5(
    dataset_path: Path,
    out_dir: Path,
    sample_frames: int = 64,
    *,
    max_collision_frames: int | None = None,
    max_offroad_frame_frac: float | None = None,
    max_red_light_frame_frac: float | None = None,
    max_blocked_frame_frac: float | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[QCIssue] = []
    frame_rows: list[dict] = []

    with h5py.File(dataset_path, "r") as f:
        for key in REQUIRED_H5_KEYS:
            if key not in f:
                issues.append(QCIssue("hard", key, None, None, f"missing required key {key}"))
        if issues:
            return write_report(dataset_path, out_dir, issues, frame_rows, {})

        n = int(f["pixels"].shape[0])
        ep_len = f["ep_len"][:].astype(np.int64)
        ep_offset = f["ep_offset"][:].astype(np.int64)
        expected_n = int(ep_len.sum())
        if expected_n != n:
            issues.append(QCIssue("hard", "ep_len", None, None, f"sum(ep_len)={expected_n} but pixels={n}"))
        if len(ep_offset) != len(ep_len):
            issues.append(QCIssue("hard", "ep_offset", None, None, "ep_len and ep_offset length mismatch"))
        if len(ep_offset) and int(ep_offset[0]) != 0:
            issues.append(QCIssue("hard", "ep_offset", 0, 0, "first offset must be 0"))
        if len(ep_offset) > 1 and not np.all(ep_offset[1:] == np.cumsum(ep_len)[:-1]):
            issues.append(QCIssue("hard", "ep_offset", None, None, "episode offsets are not contiguous"))

        for key in ("action", "state", "proprio", "ep_idx", "step_idx"):
            if int(f[key].shape[0]) != n:
                issues.append(QCIssue("hard", key, None, None, f"row count {f[key].shape[0]} does not match pixels {n}"))

        pixel_shape = f["pixels"].shape
        if len(pixel_shape) != 4 or pixel_shape[-1] != 3:
            issues.append(QCIssue("hard", "pixels", None, None, f"expected NHW3 pixels, got {pixel_shape}"))
        if f["pixels"].dtype != np.dtype("uint8"):
            issues.append(QCIssue("hard", "pixels", None, None, f"expected uint8 pixels, got {f['pixels'].dtype}"))

        action = f["action"][:]
        if not np.isfinite(action).all():
            bad = np.argwhere(~np.isfinite(action))[0]
            issues.append(QCIssue("hard", "action", int(f["ep_idx"][bad[0]]), int(f["step_idx"][bad[0]]), "non-finite action"))
        if action.shape[1] != 3:
            issues.append(QCIssue("hard", "action", None, None, f"expected throttle/steer/brake action dim 3, got {action.shape[1]}"))

        for ep, length in enumerate(ep_len):
            start = int(ep_offset[ep])
            end = start + int(length)
            step_idx = f["step_idx"][start:end]
            if not np.array_equal(step_idx, np.arange(length)):
                issues.append(QCIssue("hard", "step_idx", ep, None, "step_idx is not 0..ep_len-1"))
            route = f["route_progress_m"][start:end] if "route_progress_m" in f else None
            if route is not None and np.any(np.diff(route) < -1e-3):
                bad = int(np.where(np.diff(route) < -1e-3)[0][0])
                issues.append(QCIssue("soft", "route_progress_m", ep, bad, "route progress decreases"))

        sample_indices = np.linspace(0, max(n - 1, 0), min(sample_frames, n), dtype=np.int64)
        for idx in sample_indices:
            img = f["pixels"][idx]
            mean = float(img.mean())
            std = float(img.std())
            ep = int(f["ep_idx"][idx])
            step = int(f["step_idx"][idx])
            row = {
                "row": int(idx),
                "episode": ep,
                "step": step,
                "pixel_mean": round(mean, 4),
                "pixel_std": round(std, 4),
                "speed_mps": float(f["speed_mps"][idx]) if "speed_mps" in f else 0.0,
                "route_progress_m": float(f["route_progress_m"][idx]) if "route_progress_m" in f else 0.0,
                "collision": float(f["collision"][idx]) if "collision" in f else 0.0,
                "offroad": float(f["offroad"][idx]) if "offroad" in f else 0.0,
                "red_light": float(f["red_light"][idx]) if "red_light" in f else 0.0,
            }
            frame_rows.append(row)
            if mean < 2 or mean > 253 or std < 2:
                issues.append(QCIssue("hard", "pixels", ep, step, f"suspicious blank frame mean={mean:.2f} std={std:.2f}"))

        collision_frames = int(f["collision"][:].sum()) if "collision" in f else 0
        offroad_frames = int(f["offroad"][:].sum()) if "offroad" in f else 0
        red_light_frames = int(f["red_light"][:].sum()) if "red_light" in f else 0
        blocked_frames = int(f["blocked"][:].sum()) if "blocked" in f else 0
        summary = {
            "dataset": str(dataset_path),
            "frames": n,
            "episodes": int(len(ep_len)),
            "hard_issue_count": sum(1 for x in issues if x.severity == "hard"),
            "soft_issue_count": sum(1 for x in issues if x.severity == "soft"),
            "collision_frames": collision_frames,
            "offroad_frames": offroad_frames,
            "red_light_frames": red_light_frames,
            "blocked_frames": blocked_frames,
            "offroad_frame_frac": float(offroad_frames / max(n, 1)),
            "red_light_frame_frac": float(red_light_frames / max(n, 1)),
            "blocked_frame_frac": float(blocked_frames / max(n, 1)),
        }
        if max_collision_frames is not None and collision_frames > max_collision_frames:
            issues.append(
                QCIssue("hard", "collision", None, None, f"collision_frames={collision_frames} exceeds {max_collision_frames}")
            )
        if max_offroad_frame_frac is not None and summary["offroad_frame_frac"] > max_offroad_frame_frac:
            issues.append(
                QCIssue(
                    "hard",
                    "offroad",
                    None,
                    None,
                    f"offroad_frame_frac={summary['offroad_frame_frac']:.4f} exceeds {max_offroad_frame_frac}",
                )
            )
        if max_red_light_frame_frac is not None and summary["red_light_frame_frac"] > max_red_light_frame_frac:
            issues.append(
                QCIssue(
                    "hard",
                    "red_light",
                    None,
                    None,
                    f"red_light_frame_frac={summary['red_light_frame_frac']:.4f} exceeds {max_red_light_frame_frac}",
                )
            )
        if max_blocked_frame_frac is not None and summary["blocked_frame_frac"] > max_blocked_frame_frac:
            issues.append(
                QCIssue(
                    "hard",
                    "blocked",
                    None,
                    None,
                    f"blocked_frame_frac={summary['blocked_frame_frac']:.4f} exceeds {max_blocked_frame_frac}",
                )
            )
        summary["hard_issue_count"] = sum(1 for x in issues if x.severity == "hard")
        summary["soft_issue_count"] = sum(1 for x in issues if x.severity == "soft")
        write_contact_sheet(f["pixels"], sample_indices, out_dir / "contact_sheet.jpg")

    return write_report(dataset_path, out_dir, issues, frame_rows, summary)


def write_contact_sheet(pixels, indices: np.ndarray, out_path: Path) -> None:
    if len(indices) == 0:
        return
    thumbs = []
    for idx in indices[:64]:
        img = Image.fromarray(np.asarray(pixels[idx], dtype=np.uint8)).resize((112, 112))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 54, 16), fill=(0, 0, 0))
        draw.text((3, 2), str(int(idx)), fill=(255, 255, 255))
        thumbs.append(img)
    cols = 8
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 112, rows * 112), (255, 255, 255))
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * 112, (i // cols) * 112))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_report(dataset_path: Path, out_dir: Path, issues: list[QCIssue], frame_rows: list[dict], summary: dict) -> dict:
    report = {
        "dataset": str(dataset_path),
        "summary": summary,
        "issues": [asdict(issue) for issue in issues],
        "gate": "pass" if not any(issue.severity == "hard" for issue in issues) else "fail",
    }
    (out_dir / "qc_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "sampled_frames.csv").open("w", newline="", encoding="utf-8") as f:
        if frame_rows:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    return report


def main() -> None:
    args = parse_args()
    max_collision_frames = None if args.allow_infractions else args.max_collision_frames
    max_offroad_frame_frac = None if args.allow_infractions else args.max_offroad_frame_frac
    max_red_light_frame_frac = None if args.allow_infractions else args.max_red_light_frame_frac
    max_blocked_frame_frac = None if args.allow_infractions else args.max_blocked_frame_frac
    report = validate_hdf5(
        args.dataset,
        args.out_dir,
        args.sample_frames,
        max_collision_frames=max_collision_frames,
        max_offroad_frame_frac=max_offroad_frame_frac,
        max_red_light_frame_frac=max_red_light_frame_frac,
        max_blocked_frame_frac=max_blocked_frame_frac,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.strict and report["gate"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
