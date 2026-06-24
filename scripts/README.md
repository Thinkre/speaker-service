# Scripts

## export_onnx.py

Export the local ERES2NetV2 PyTorch checkpoint to ONNX format for onnxruntime inference.

### Prerequisites

```bash
uv sync
uv pip install onnx onnxscript
```

### Usage

```bash
uv run python scripts/export_onnx.py
```

### Input / Output

| | Path |
|---|---|
| Checkpoint | `models/iic/speech_eres2netv2_sv_zh-cn_16k-common/pretrained_eres2netv2.ckpt` |
| ONNX output | `models/onnx/speaker_embedding_v2_local.onnx` |

### ONNX Model Spec

| Property | Value |
|---|---|
| Input | `feature` — shape `(batch, frames, 80)`, float32 |
| Output | `embedding` — shape `(batch, 192)`, float32 |
| Opset | 11 |

### Fbank Preprocessing

The ONNX model expects 80-dim log-mel filterbank features computed with the
same parameters as `torchaudio.compliance.kaldi.fbank`:

```python
import torch
import torchaudio.compliance.kaldi as Kaldi

audio = torch.from_numpy(pcm_float32).unsqueeze(0)  # (1, samples)
feat = Kaldi.fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=0)
feat = feat - feat.mean(0, keepdim=True)  # CMN (cepstral mean normalization)
```

Key fbank defaults (from `torchaudio.compliance.kaldi.fbank`):

| Parameter | Value |
|---|---|
| frame_length | 25ms |
| frame_shift | 10ms |
| window_type | povey |
| preemphasis_coefficient | 0.97 |
| remove_dc_offset | True |
| low_freq | 20Hz |
| high_freq | Nyquist (8000Hz at 16kHz) |
| dither | 0.0 |

## 3dspeaker/

Vendored model classes from [3D-Speaker](https://github.com/modelscope/3D-Speaker)
required for ONNX export. Contains `ERes2NetV2.py`, `pooling_layers.py`, `fusion.py`.
