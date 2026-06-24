# Speaker Embedding Service

HTTP Speaker Embedding microservice — ERes2NetV2 / CamPlus

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env

# Start server (defaults to 0.0.0.0:8080)
uv run python api.py
```

## API

`POST /v1/audio/speaker/embedding` — extract L2-normalized speaker embedding from WAV audio.

See [docs/api.md](docs/api.md) for the full API reference.

### Quick example

```python
from client.speaker_client import SpeakerClient

with SpeakerClient(host="localhost", port=8080) as client:
    emb = client.extract_embedding(pcm_bytes, engine="eresnetv2")
    # emb.shape → (192,)
    # np.linalg.norm(emb) → 1.0
```

## Evaluation

```bash
# Speaker verification eval using RTTM ground truth
uv run python client/eval_speaker.py --data data --engine eresnetv2
```

## Models

| Engine | Local path | Environment variable |
|--------|-----------|----------------------|
| ERes2NetV2 | `models/iic/speech_eres2netv2_sv_zh-cn_16k-common` | `ERES2NET_MODEL_PATH` |
| CamPlus | `models/iic/speech_campplus_sv_zh-cn_16k-common` | `CAMPPLUS_MODEL_PATH` |

All model files are stored locally under `models/`. The `models/` directory is
excluded from git (see `.gitignore`).
