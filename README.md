# speaker_service

gRPC Speaker Embedding microservice — ERes2NetV2 / CamPlus

## Quick Start

```bash
# Install dependencies
uv sync

# Generate protobuf stubs
bash scripts/gen_proto.sh

# Configure environment
cp .env.example .env

# Start server (defaults to 0.0.0.0:50052)
uv run python server.py
```

## Evaluation

The `eval_speaker.py` script runs an offline diarization evaluation using the
local FireRed VAD (not via gRPC) for speech segmentation, then calls the gRPC
service for speaker embeddings. Results are written to the `reports/` directory
and compared against the WeSpeaker baseline (DER/JER).

```bash
# Evaluate against local alimeeting_mini testset
uv run python eval_speaker.py \
    --testset ../data/testsets/alimeeting_mini \
    --scene near_2spk

# Evaluate against a remote gRPC endpoint
uv run python eval_speaker.py \
    --testset ../data/testsets/alimeeting_mini \
    --scene near_2spk \
    --host 10.0.0.5 \
    --port 50052
```

Output reports land in `reports/` as JSON files with per-scene DER, JER, Miss,
False-Alarm, and Confusion breakdown alongside the WeSpeaker offline baseline.

## API

| RPC | Request fields | Response fields |
|-----|---------------|-----------------|
| `ExtractEmbedding` | `pcm` (bytes, 16-bit LE mono 16kHz), `engine` (string: `eresnetv2` or `campplus`) | `embedding` (L2-normalized float32 array) |

## Models

| Engine | Local path | Environment variable |
|--------|-----------|----------------------|
| ERes2NetV2 | `models/iic/speech_eres2netv2_sv_zh-cn_16k-common` | `ERES2NET_MODEL_PATH` |
| CamPlus | `models/iic/speech_campplus_sv_zh-cn_16k-common` | `CAMPPLUS_MODEL_PATH` |

All model files are stored locally under `models/`. The `models/` directory is
excluded from git (see `.gitignore`). Do not set ModelScope or HuggingFace
model ID strings in code — always load from the local paths above.
