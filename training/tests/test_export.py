"""The shipped ONNX model is the trained network: parity with PyTorch, metadata, size."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lumina_train.export import FIXTURE_PATH, PARITY_ATOL, PARITY_RTOL, SIZE_BUDGET
from lumina_train.model import LuminaNet
from lumina_train.model_contract import INPUT_NAME, MODEL_PATH, OUTPUT_NAME
from lumina_train.paths import MODEL_META_PATH
from lumina_train.placeholder import export_onnx


def _run(path, images_u8: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = images_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return np.stack(
        [session.run([OUTPUT_NAME], {INPUT_NAME: x[i : i + 1]})[0][0] for i in range(len(x))]
    )


def test_committed_model_matches_pytorch_outputs_recorded_at_export():
    fixture = np.load(FIXTURE_PATH)
    got = _run(MODEL_PATH, fixture["images"])
    np.testing.assert_allclose(got, fixture["expected"], atol=PARITY_ATOL, rtol=PARITY_RTOL)


def test_committed_model_is_trained_small_and_described():
    proto = onnx.load(str(MODEL_PATH))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta["lumina.placeholder"] == "false"
    assert MODEL_PATH.stat().st_size < SIZE_BUDGET
    info = json.loads(MODEL_META_PATH.read_text())
    assert info["sha256"] == hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    assert info["bytes"] == MODEL_PATH.stat().st_size
    assert info["version"] == meta["lumina.version"]


def test_export_path_preserves_the_network(tmp_path):
    torch.manual_seed(0)
    model = LuminaNet().eval()
    path = tmp_path / "net.onnx"
    export_onnx(model, path, placeholder=False)
    images = np.random.default_rng(1).integers(0, 256, (3, 96, 128, 3), dtype=np.uint8)
    with torch.no_grad():
        x = torch.from_numpy(images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        expected = model(x).numpy()
    np.testing.assert_allclose(_run(path, images), expected, atol=PARITY_ATOL, rtol=PARITY_RTOL)
