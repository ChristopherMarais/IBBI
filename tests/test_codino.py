"""Co-DINO arthropod detector: pure-PyTorch model, pre/post-processing and wrapper (offline, tiny random model)."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from ibbi.models._codino import CoDINO, _rope_tables, soft_nms_linear
from ibbi.models.codino import CoDINODetector

TINY = {"num_queries": 50, "window_block_indexes": [0], "embed_dim": 64, "depth": 2, "num_heads": 4}


@pytest.fixture(scope="module")
def tiny_codino(tmp_path_factory):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    model = CoDINO(num_classes=1, **TINY)
    d = tmp_path_factory.mktemp("codino")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    cfg = {
        "architecture": TINY,
        "preprocessing": {"scale": [128, 96], "size_divisor": 32, "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]},
        "postprocessing": {"soft_nms_iou": 0.8, "max_per_img": 1000},
        "operating_conf": 0.65,
        "inference_defaults": {"conf": 0.0, "max_det": 300},
    }
    (d / "config.json").write_text(json.dumps(cfg))
    return CoDINODetector(str(d / "model.safetensors"), cfg, device="cpu")


def test_soft_nms_linear():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9.5], [20, 20, 30, 30], [0, 0, 10, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    keep, new = soft_nms_linear(boxes, scores, iou_threshold=0.8)
    assert keep.tolist() == [0, 2, 3, 1]  # box 1 overlaps box 0 by 0.95 and drops to 0.8 * 0.05
    assert new[0] == pytest.approx(0.9) and new[1] == pytest.approx(0.7) and new[2] == pytest.approx(0.6)  # IoU 0.5 < 0.8: untouched
    assert new[3] == pytest.approx(0.8 * (1 - 0.95), rel=1e-4)
    # boxes whose score falls below min_score are dropped
    keep, _ = soft_nms_linear(boxes[:2], np.array([0.9, 0.01], dtype=np.float32), iou_threshold=0.8, min_score=1e-3)
    assert keep.tolist() == [0]


def test_rope_tables_match_eva02_window_buffer():
    # EVA-02 precomputes the 24 x 24 window table with pt_seq_len 16; get_rope(H, W) builds the same for any grid
    cos, sin = _rope_tables(24, 24)
    assert cos.shape == (576, 64) and torch.allclose(cos[0], torch.ones(64)) and torch.allclose(sin[0], torch.zeros(64))
    freqs = 1.0 / (10000 ** (torch.arange(0, 32, 2).float() / 32))
    assert torch.allclose(torch.acos(cos[24, 0].clamp(-1, 1)), (torch.tensor(1 / 24 * 16) * freqs[0]) % (2 * np.pi), atol=1e-5)


def test_preprocess_matches_mmdet_rescale(tiny_codino):
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    x, mask, (nh, nw), (sx, sy) = tiny_codino._preprocess(img)
    # fit inside (long 128, short 96): factor min(128/500, 96/300) = 0.256 -> 128 x 76.8 -> 77, padded to 96 x 128
    assert (nw, nh) == (128, 77) and tuple(x.shape) == (1, 3, 96, 128)
    assert mask[0, :77, :128].sum() == 0 and mask[0, 77:].all()
    assert sx == pytest.approx(128 / 500) and sy == pytest.approx(77 / 300)


def test_predict_outputs(tiny_codino):
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((90, 160, 3)) * 255).astype(np.uint8))
    r = tiny_codino.predict(img)
    assert set(r) >= {"boxes", "scores", "labels", "class_ids"}
    assert 0 < len(r["boxes"]) <= 50 and len(r["boxes"]) == len(r["scores"]) == len(r["labels"])
    b = np.asarray(r["boxes"])
    assert (b[:, 0] >= 0).all() and (b[:, 2] <= 160 + 1e-3).all() and (b[:, 3] <= 90 + 1e-3).all()
    assert r["labels"][0] == "arthropod" and r["scores"] == sorted(r["scores"], reverse=True)
    assert tiny_codino.predict(img, conf=1.01)["boxes"] == []
    assert len(tiny_codino.predict([img, img])) == 2
    assert tiny_codino.predict_proba([img]).shape == (1, 1)
    assert tuple(tiny_codino.extract_features(img).shape) == (1, 64)
    assert tiny_codino.get_classes() == ["arthropod"] and tiny_codino.operating_conf == 0.65


def test_weights_must_match(tmp_path, tiny_codino):
    from safetensors.torch import save_file

    save_file({"backbone.pos_embed": torch.zeros(1, 1025, 64)}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(RuntimeError):
        CoDINODetector(str(tmp_path / "bad.safetensors"), tiny_codino.config, device="cpu")


def test_registry_requires_pretrained():
    import ibbi

    with pytest.raises(ValueError):
        ibbi.create_model("codino_arthropod_detector", pretrained=False)
