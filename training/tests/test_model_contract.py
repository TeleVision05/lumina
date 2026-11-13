"""The committed web/public/models/lumina-sh.onnx honours the contract (placeholder or trained)."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lumina_train.model_contract import (
    CONTRACT_VERSION,
    INPUT_NAME,
    INPUT_SHAPE,
    MIN_OPSET,
    MODEL_PATH,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
)
from lumina_train.placeholder import build_placeholder, export_onnx


def _dims(value_info) -> tuple[int, ...]:
    return tuple(d.dim_value for d in value_info.type.tensor_type.shape.dim)


def test_graph_signature_and_opset():
    proto = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(proto)
    opset = max(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))
    assert opset >= MIN_OPSET
    assert [i.name for i in proto.graph.input] == [INPUT_NAME]
    assert [o.name for o in proto.graph.output] == [OUTPUT_NAME]
    assert _dims(proto.graph.input[0]) == INPUT_SHAPE
    assert _dims(proto.graph.output[0]) == OUTPUT_SHAPE
    float32 = onnx.TensorProto.FLOAT
    assert proto.graph.input[0].type.tensor_type.elem_type == float32
    assert proto.graph.output[0].type.tensor_type.elem_type == float32
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta.get("lumina.contract") == CONTRACT_VERSION


def test_runs_in_onnxruntime_with_contract_shapes():
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    image = rng.random(INPUT_SHAPE, dtype=np.float32)
    (sh,) = session.run([OUTPUT_NAME], {INPUT_NAME: image})
    assert sh.shape == OUTPUT_SHAPE
    assert sh.dtype == np.float32
    assert np.all(np.isfinite(sh))


def test_placeholder_export_matches_torch(tmp_path):
    model = build_placeholder()
    path = tmp_path / "placeholder.onnx"
    export_onnx(model, path, placeholder=True)
    image = torch.rand(*INPUT_SHAPE, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model(image).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (actual,) = session.run([OUTPUT_NAME], {INPUT_NAME: image.numpy()})
    np.testing.assert_allclose(actual, expected, atol=1e-5)
