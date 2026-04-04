"""Export a trained IR-50/IR-18 face embedding model to ONNX format.

Usage::

    python scripts/export_onnx.py \\
        --weights training/outputs/checkpoint_epoch_028.pth \\
        --output models/custom_ir50.onnx \\
        --arch ir50 \\
        --input-size 112

After export, validate with::

    python scripts/export_onnx.py \\
        --weights training/outputs/checkpoint_epoch_028.pth \\
        --output models/custom_ir50.onnx \\
        --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from training.models.ir_resnet import build_model


def export_to_onnx(
    weights_path: str,
    output_path: str,
    arch: str = "ir50",
    embedding_dim: int = 512,
    input_size: int = 112,
    opset_version: int = 17,
) -> None:
    """Export a PyTorch face model to ONNX."""
    device = torch.device("cpu")

    model = build_model(arch, embedding_dim).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_p),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )
    print(f"✓ Exported ONNX model to {output_p}")
    print(f"  Architecture: {arch}")
    print(f"  Input: (batch, 3, {input_size}, {input_size})")
    print(f"  Output: (batch, {embedding_dim})")
    print(f"  Opset: {opset_version}")
    print(f"  File size: {output_p.stat().st_size / 1024 / 1024:.1f} MB")


def validate_onnx(
    weights_path: str,
    onnx_path: str,
    arch: str = "ir50",
    embedding_dim: int = 512,
    input_size: int = 112,
) -> None:
    """Validate ONNX model output matches PyTorch."""
    import onnxruntime as ort

    device = torch.device("cpu")
    model = build_model(arch, embedding_dim).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"\n✓ Validation:")
    print(f"  Max absolute diff: {max_diff:.8f}")
    if max_diff < 1e-4:
        print("  Status: PASS — outputs match within tolerance")
    else:
        print("  Status: WARNING — outputs differ significantly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export face model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default="models/custom_ir50.onnx", help="ONNX output path")
    parser.add_argument("--arch", default="ir50", choices=["ir18", "ir34", "ir50", "ir100"])
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--input-size", type=int, default=112)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--validate", action="store_true", help="Validate after export")
    args = parser.parse_args()

    export_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        arch=args.arch,
        embedding_dim=args.embedding_dim,
        input_size=args.input_size,
        opset_version=args.opset,
    )

    if args.validate:
        validate_onnx(
            weights_path=args.weights,
            onnx_path=args.output,
            arch=args.arch,
            embedding_dim=args.embedding_dim,
            input_size=args.input_size,
        )


if __name__ == "__main__":
    main()
