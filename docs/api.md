# Speaker Service API 文档

---

## HTTP 接口

### POST /v1/audio/speaker/embedding

从一段音频中提取 L2 归一化的说话人嵌入向量，同步返回结果。

**端口：** 8080（由 `HTTP_PORT` 环境变量控制）

#### 请求头

| 字段 | 值 |
|------|----|
| `Content-Type` | `application/json` |

#### 请求体

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `url` | string | 否 | WAV 音频 URL（与 base64 二选一） |
| `base64` | string | 否 | base64 编码的 WAV 文件字节（与 url 二选一） |
| `model` | string | 是 | `"eresnetv2"` \| `"campplus"` |
| `normalize` | bool | 否 | 是否 L2 归一化，默认 `true` |
| `user` | string | 是 | 调用方标识，用于日志追踪 |

音频要求：WAV 格式，16kHz，单声道，int16。非 16kHz 会自动重采样。

#### 响应体

| 字段 | 类型 | 说明 |
|------|------|------|
| `embeddings` | array | 提取结果，成功时含 1 个元素，失败时为空 |
| `error` | string | 错误信息，成功时为空字符串 |

**embeddings 元素：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 固定为 0 |
| `start` | float | 片段起始时间（秒），固定为 0.0 |
| `end` | float | 片段时长（秒） |
| `confidence` | float | 固定为 1.0 |
| `embedding` | array[float] | L2 归一化 float32 向量 |
| `dimensions` | int | 向量维度，eresnetv2=192，campplus=192 |

#### 示例

**请求：**
```json
{
  "base64": "<wav_base64>",
  "model": "eresnetv2",
  "user": "diarization"
}
```

**成功响应：**
```json
{
  "embeddings": [{
    "id": 0,
    "start": 0.0,
    "end": 1.0,
    "confidence": 1.0,
    "embedding": [0.026, 0.134, 0.107, "..."],
    "dimensions": 192
  }],
  "error": ""
}
```

**失败响应：**
```json
{
  "embeddings": [],
  "error": "Extraction failed (audio too short or model error)."
}
```

#### Python 调用示例

```python
import base64, io, wave
import httpx
import numpy as np

def pcm_to_wav_b64(pcm: bytes, sr: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()

async def extract_embedding(pcm: bytes, model: str = "eresnetv2") -> np.ndarray | None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "http://localhost:8080/v1/audio/speaker/embedding",
            json={
                "base64": pcm_to_wav_b64(pcm),
                "model": model,
                "user": "diarization",
            },
        )
        data = resp.json()
    if data["error"] or not data["embeddings"]:
        return None
    return np.array(data["embeddings"][0]["embedding"], dtype=np.float32)
```

---

## gRPC 接口

**端口：** 50052（由 `PORT` 环境变量控制）  
**Proto：** `proto/speaker.proto`

### ExtractEmbedding

```protobuf
rpc ExtractEmbedding (ExtractRequest) returns (ExtractResponse);

message ExtractRequest {
  bytes  pcm    = 1;  // int16 mono 16kHz PCM
  string engine = 2;  // "eresnetv2" | "campplus"，默认 eresnetv2
}

message ExtractResponse {
  repeated float embedding = 1;  // L2-normalized float32 向量
  int32          dim       = 2;  // 向量维度（192）
  bool           success   = 3;
}
```

#### Python 调用示例

```python
from client.speaker_client import SpeakerClient
import numpy as np

with SpeakerClient(host="localhost", port=50052) as client:
    emb = client.extract_embedding(pcm_bytes, engine="eresnetv2")
    if emb is not None:
        print(emb.shape)          # (192,)
        print(np.linalg.norm(emb)) # ≈ 1.0
```

---

## 模型对比

| 模型 | 参数 | EER | minDCF | 说明 |
|------|------|-----|--------|------|
| `eresnetv2` | ~200MB | **19.1%** | 0.0100 | 默认，推荐 |
| `campplus` | ~50MB | 21.8% | **0.0098** | 轻量，精度略低 |

> 评测数据集：alimeeting_mini（far+near），RTTM ground-truth 切段，collar=0.25s

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HTTP_PORT` | `8080` | HTTP 服务端口 |
| `PORT` | `50052` | gRPC 服务端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `ERES2NET_MODEL_PATH` | `./models/iic/speech_eres2netv2_sv_zh-cn_16k-common` | ERes2NetV2 模型路径 |
| `CAMPPLUS_MODEL_PATH` | `./models/iic/speech_campplus_sv_zh-cn_16k-common` | CamPlus 模型路径 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
