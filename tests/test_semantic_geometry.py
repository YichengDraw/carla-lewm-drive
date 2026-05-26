import numpy as np

from carla_lewm_drive.semantic_geometry import SemanticGeometryLaneModel, semantic_geometry_features


def test_semantic_geometry_features_are_stable_for_simple_masks():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[10:12, 7:9, 1] = 255
    image[8:14, :, 0] = 255

    features = semantic_geometry_features(image, rows=[10], band=1, image_size=16)

    assert features.shape == (12,)
    assert np.isfinite(features).all()
    assert features[3] > 0.0
    assert features[7] > 0.0


def test_semantic_geometry_lane_model_roundtrip(tmp_path):
    feature_mean = np.zeros(12, dtype=np.float32)
    feature_scale = np.ones(12, dtype=np.float32)
    weights = np.zeros((12, 2), dtype=np.float32)
    weights[0, 0] = 2.0
    bias = np.asarray([0.1, -0.2], dtype=np.float32)
    model = SemanticGeometryLaneModel(
        rows=(10,),
        band=1,
        image_size=16,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        bias=bias,
    )
    path = tmp_path / "model.json"

    model.save(path)
    loaded = SemanticGeometryLaneModel.load(path)
    pred = loaded.predict_features(np.asarray([0.5] + [0.0] * 11, dtype=np.float32))

    assert pred.shape == (1, 2)
    assert np.isclose(pred[0, 0], 1.1)
    assert np.isclose(pred[0, 1], -0.2)
