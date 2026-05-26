from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEMANTIC_GEOMETRY_ROWS = (96, 112, 128, 144, 160, 176, 192, 208)


def semantic_geometry_features(
    image: np.ndarray,
    *,
    rows: tuple[int, ...] | list[int] = DEFAULT_SEMANTIC_GEOMETRY_ROWS,
    band: int = 5,
    image_size: int = 224,
) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected semantic RGB image with shape [H,W,3], got {pixels.shape}")
    rgb = pixels.astype(np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    score_maps = (
        np.maximum(green - np.maximum(red, blue), 0.0),
        np.maximum(red - np.maximum(green, blue), 0.0),
        np.maximum(blue - np.maximum(red, green), 0.0),
    )
    h, w = pixels.shape[:2]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    out: list[float] = []
    half_w = w // 2
    for row in rows:
        scaled_row = int(round(float(row) * float(h) / max(1.0, float(image_size))))
        y0 = max(0, scaled_row - int(band))
        y1 = min(h, scaled_row + int(band) + 1)
        for score in score_maps:
            band_score = score[y0:y1].sum(axis=0)
            mass = float(max(band_score.sum(), 1e-6))
            centroid = float((band_score * xs).sum() / mass)
            left_mass = float(band_score[:half_w].sum())
            right_mass = float(band_score[half_w:].sum())
            balance = float((left_mass - right_mass) / mass)
            spread = float((band_score * np.square(xs - centroid)).sum() / mass)
            log_mass = float(np.log1p(mass))
            out.extend((centroid, balance, spread, log_mass))
    return np.asarray(out, dtype=np.float32)


def semantic_geometry_features_batch(
    images: np.ndarray,
    *,
    rows: tuple[int, ...] | list[int] = DEFAULT_SEMANTIC_GEOMETRY_ROWS,
    band: int = 5,
    image_size: int = 224,
) -> np.ndarray:
    pixels = np.asarray(images)
    if pixels.ndim == 3:
        return semantic_geometry_features(pixels, rows=rows, band=band, image_size=image_size)[None, :]
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected semantic RGB images with shape [N,H,W,3], got {pixels.shape}")
    return np.stack(
        [semantic_geometry_features(image, rows=rows, band=band, image_size=image_size) for image in pixels],
        axis=0,
    ).astype(np.float32)


@dataclass(frozen=True)
class SemanticGeometryLaneModel:
    rows: tuple[int, ...]
    band: int
    image_size: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        norm = (feats - self.feature_mean.reshape(1, -1)) / self.feature_scale.reshape(1, -1)
        return (norm @ self.weights + self.bias.reshape(1, -1)).astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        features = semantic_geometry_features(
            image,
            rows=self.rows,
            band=self.band,
            image_size=self.image_size,
        )
        return self.predict_features(features)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "semantic_geometry_lane_ridge",
            "rows": list(self.rows),
            "band": int(self.band),
            "image_size": int(self.image_size),
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_scale": self.feature_scale.astype(float).tolist(),
            "weights": self.weights.astype(float).tolist(),
            "bias": self.bias.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticGeometryLaneModel":
        return cls(
            rows=tuple(int(value) for value in payload["rows"]),
            band=int(payload["band"]),
            image_size=int(payload.get("image_size", 224)),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32),
            weights=np.asarray(payload["weights"], dtype=np.float32),
            bias=np.asarray(payload["bias"], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticGeometryLaneModel":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


@dataclass(frozen=True)
class SklearnSemanticGeometryLaneModel:
    rows: tuple[int, ...]
    band: int
    image_size: int
    estimator: Any

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        return np.asarray(self.estimator.predict(feats), dtype=np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        features = semantic_geometry_features(
            image,
            rows=self.rows,
            band=self.band,
            image_size=self.image_size,
        )
        return self.predict_features(features)[0]


def load_semantic_geometry_lane_model(path: str | Path) -> SemanticGeometryLaneModel | SklearnSemanticGeometryLaneModel:
    model_path = Path(path)
    if model_path.suffix.lower() == ".json":
        return SemanticGeometryLaneModel.load(model_path)
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required to load sklearn semantic geometry lane models") from exc
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "estimator" in payload:
        return SklearnSemanticGeometryLaneModel(
            rows=tuple(int(value) for value in payload.get("rows", DEFAULT_SEMANTIC_GEOMETRY_ROWS)),
            band=int(payload.get("band", 5)),
            image_size=int(payload.get("image_size", 224)),
            estimator=payload["estimator"],
        )
    if hasattr(payload, "predict"):
        return SklearnSemanticGeometryLaneModel(
            rows=DEFAULT_SEMANTIC_GEOMETRY_ROWS,
            band=5,
            image_size=224,
            estimator=payload,
        )
    raise ValueError(f"Unsupported semantic geometry lane model payload in {model_path}")
