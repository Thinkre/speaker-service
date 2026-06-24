"""Export local ERES2NetV2 checkpoint to ONNX for onnxruntime inference.

Usage:
    uv run python scripts/export_onnx.py

Input:  models/iic/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.ckpt
Output: models/onnx/speaker_embedding_v2_local.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add scripts/ to path for the vendored 3dspeaker package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _3dspeaker.ERes2NetV2 import ERes2NetV2  # noqa: E402

CKPT_PATH = Path("models/iic/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.ckpt")
OUTPUT_PATH = Path("models/onnx/speaker_embedding_v2.onnx")


def main() -> None:
    if not CKPT_PATH.exists():
        print(f"Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    print(f"Loading checkpoint: {CKPT_PATH}")
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)

    print("Building ERes2NetV2 model (feat_dim=80, embedding_size=192)")
    model = ERes2NetV2(feat_dim=80, embedding_size=192)
    model.load_state_dict(state)
    model.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 345, 80)
    print(f"Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["feature"],
        output_names=["embedding"],
        dynamic_axes={
            "feature": {0: "batch_size", 1: "frame_num"},
            "embedding": {0: "batch_size"},
        },
    )
    print(f"Done: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
