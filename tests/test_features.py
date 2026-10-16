"""Unit test ekstraksi fitur statistik: set top12 vs all25 (Tabel 3.1 Hakim / Nour et al.)."""

import numpy as np
import pytest
import torch
from scipy import stats

from src.features.statistical import (
    extract_statistical_features,
    get_feature_names,
    infer_feature_set,
)
from src.models.bilstm import ECGMultiTaskBiLSTM


@pytest.fixture
def signals():
    rng = np.random.default_rng(0)
    return rng.random((4, 1250)).astype(np.float32)


def test_top12_default_shapes(signals):
    f = extract_statistical_features(signals)
    assert f["sbp"].shape == (4, 6)
    assert f["dbp"].shape == (4, 6)
    assert f["map"].shape == (4, 12)


def test_all25_shapes_and_prefix_matches_top12(signals):
    f12 = extract_statistical_features(signals, feature_set="top12")
    f25 = extract_statistical_features(signals, feature_set="all25")
    assert f25["sbp"].shape == (4, 13)
    assert f25["dbp"].shape == (4, 12)
    assert f25["map"].shape == (4, 25)
    assert np.isfinite(f25["map"]).all()
    # 12 fitur pertama tiap head identik (ablasi adil)
    np.testing.assert_allclose(f25["sbp"][:, :6], f12["sbp"])
    np.testing.assert_allclose(f25["dbp"][:, :6], f12["dbp"])


def test_names_match_dims():
    for fs, n in (("top12", 12), ("all25", 25)):
        names = get_feature_names(fs)
        assert len(names["map"]) == n
        assert infer_feature_set(n) == fs


def test_extra_features_match_reference_definitions(signals):
    f = extract_statistical_features(signals, feature_set="all25")
    names = get_feature_names("all25")
    sbp = dict(zip(names["sbp"], f["sbp"].T))
    dbp = dict(zip(names["dbp"], f["dbp"].T))
    x = signals.astype(np.float64)
    np.testing.assert_allclose(sbp["maximum"], x.max(-1), rtol=1e-5)
    np.testing.assert_allclose(sbp["median"], np.median(x, -1), rtol=1e-5)
    np.testing.assert_allclose(sbp["trimmed_mean_50"], stats.trim_mean(x, 0.25, axis=-1), rtol=1e-4)
    np.testing.assert_allclose(sbp["trimmed_mean_25"], stats.trim_mean(x, 0.125, axis=-1), rtol=1e-4)
    np.testing.assert_allclose(dbp["minimum"], x.min(-1), rtol=1e-5)
    np.testing.assert_allclose(dbp["iqr"], stats.iqr(x, axis=-1), rtol=1e-4)
    np.testing.assert_allclose(dbp["kurtosis"], stats.kurtosis(x, axis=-1, fisher=False), rtol=1e-4)
    np.testing.assert_allclose(dbp["standard_error"], x.std(-1, ddof=1) / np.sqrt(x.shape[-1]), rtol=1e-4)


def test_invalid_feature_set(signals):
    with pytest.raises(ValueError):
        extract_statistical_features(signals, feature_set="all30")


def test_model_forward_with_all25(signals):
    f = extract_statistical_features(signals, feature_set="all25")
    model = ECGMultiTaskBiLSTM(
        input_size=1, hidden_size=32, num_layers=1,
        sbp_feat_dim=13, dbp_feat_dim=12, map_feat_dim=25,
    )
    x = torch.from_numpy(signals).unsqueeze(-1)
    feats = {k: torch.from_numpy(v) for k, v in f.items()}
    sbp, dbp, mp = model(x, features=feats)
    assert sbp.shape[0] == dbp.shape[0] == mp.shape[0] == 4
