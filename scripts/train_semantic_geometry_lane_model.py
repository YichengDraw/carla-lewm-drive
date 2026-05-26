from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from carla_lewm_drive.semantic_geometry import (
    DEFAULT_SEMANTIC_GEOMETRY_ROWS,
    SemanticGeometryLaneModel,
    semantic_geometry_features_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small ridge model from semantic masks to lane geometry.")
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--dataset-weight", action="append", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="*", default=list(DEFAULT_SEMANTIC_GEOMETRY_ROWS))
    parser.add_argument("--band", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--model-type", choices=["ridge", "hist_gbr", "random_forest"], default="ridge")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-frames-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def _load_dataset(
    path: Path,
    *,
    rows: tuple[int, ...],
    band: int,
    image_size: int,
    batch_size: int,
    max_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        image_key = "pixels" if "pixels" in h5 else "images"
        if image_key not in h5:
            raise KeyError(f"{path} must contain a 'pixels' or 'images' dataset")
        images = h5[image_key]
        n = int(images.shape[0])
        if max_frames > 0 and n > max_frames:
            indices = np.sort(rng.choice(n, size=max_frames, replace=False))
        else:
            indices = np.arange(n)
        features = []
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            batch = images[idx]
            features.append(semantic_geometry_features_batch(batch, rows=rows, band=band, image_size=image_size))
        x = np.concatenate(features, axis=0)
        y = np.stack(
            [
                np.asarray(h5["lane_offset_m"])[indices],
                np.asarray(h5["heading_error_rad"])[indices],
            ],
            axis=1,
        ).astype(np.float32)
    return x, y


def _weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z = (x - mean.reshape(1, -1)) / scale.reshape(1, -1)
    z_aug = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float32)], axis=1)
    sw = np.sqrt(sample_weight.astype(np.float32)).reshape(-1, 1)
    zw = z_aug * sw
    yw = y * sw
    reg = np.eye(z_aug.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    params = np.linalg.solve(zw.T.astype(np.float64) @ zw.astype(np.float64) + reg, zw.T.astype(np.float64) @ yw)
    weights = params[:-1].astype(np.float32)
    bias = params[-1].astype(np.float32)
    return mean.astype(np.float32), scale.astype(np.float32), weights, bias


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    lane = pred[:, 0]
    heading = pred[:, 1]
    lane_t = target[:, 0]
    heading_t = target[:, 1]
    active = np.abs(lane_t) >= 0.1
    return {
        "lane_mae": float(np.mean(np.abs(lane - lane_t))),
        "heading_mae": float(np.mean(np.abs(heading - heading_t))),
        "lane_corr": float(np.corrcoef(lane, lane_t)[0, 1]) if len(lane) > 1 else float("nan"),
        "heading_corr": float(np.corrcoef(heading, heading_t)[0, 1]) if len(heading) > 1 else float("nan"),
        "lane_sign_acc_0p1": float(np.mean(np.sign(lane[active]) == np.sign(lane_t[active]))) if bool(active.any()) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    rows = tuple(int(value) for value in args.rows)
    weights = args.dataset_weight or [1.0] * len(args.dataset)
    if len(weights) != len(args.dataset):
        raise ValueError("--dataset-weight count must match --dataset count")
    rng = np.random.default_rng(int(args.seed))
    xs = []
    ys = []
    sample_weights = []
    per_dataset = {}
    for path, weight in zip(args.dataset, weights):
        x, y = _load_dataset(
            path,
            rows=rows,
            band=int(args.band),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            max_frames=int(args.max_frames_per_dataset),
            rng=rng,
        )
        xs.append(x)
        ys.append(y)
        sample_weights.append(np.full(x.shape[0], float(weight), dtype=np.float32))
        per_dataset[str(path)] = {"frames": int(x.shape[0]), "weight": float(weight)}
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    w_all = np.concatenate(sample_weights, axis=0)
    if args.model_type == "ridge":
        mean, scale, ridge_weights, bias = _weighted_ridge(x_all, y_all, w_all, ridge=float(args.ridge))
        model = SemanticGeometryLaneModel(
            rows=rows,
            band=int(args.band),
            image_size=int(args.image_size),
            feature_mean=mean,
            feature_scale=scale,
            weights=ridge_weights,
            bias=bias,
        )
        model.save(args.output)
        pred_all = model.predict_features(x_all)
    else:
        import joblib
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        if args.model_type == "hist_gbr":
            base = HistGradientBoostingRegressor(
                max_iter=int(args.max_iter),
                learning_rate=float(args.learning_rate),
                max_leaf_nodes=int(args.max_leaf_nodes),
                l2_regularization=float(args.ridge),
                random_state=int(args.seed),
            )
            estimator = MultiOutputRegressor(base)
        elif args.model_type == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=int(args.n_estimators),
                max_depth=int(args.max_depth),
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=int(args.seed),
            )
        else:
            raise ValueError(f"Unsupported model type {args.model_type!r}")
        estimator.fit(x_all, y_all, sample_weight=w_all)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": f"semantic_geometry_lane_{args.model_type}",
                "rows": list(rows),
                "band": int(args.band),
                "image_size": int(args.image_size),
                "estimator": estimator,
            },
            args.output,
        )
        pred_all = np.asarray(estimator.predict(x_all), dtype=np.float32)
    report = {
        "output": str(args.output),
        "model_type": str(args.model_type),
        "ridge": float(args.ridge),
        "rows": list(rows),
        "band": int(args.band),
        "image_size": int(args.image_size),
        "total_frames": int(x_all.shape[0]),
        "datasets": per_dataset,
        "train_metrics": _metrics(pred_all, y_all),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
