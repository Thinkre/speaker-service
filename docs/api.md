# Speaker Service API 文档

---

## HTTP 接口

### POST /v1/audio/speaker/embedding

从一段音频中提取 L2 归一化的说话人嵌入向量，同步返回结果。

**端口：** 8080（由 `HTTP_PORT` 环境变量控制）

#### 请求头

| 字段 | 值 | 必需 |
|------|----|------|
| `Content-Type` | `application/json` | 是 |
| `Authorization` | `Bearer <token>` | 是（`API_TOKEN` 未配置时跳过校验） |

#### 请求体

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `url` | string | 否 | WAV 音频 URL（与 base64 二选一） |
| `base64` | string | 否 | base64 编码的 WAV 文件字节（与 url 二选一） |
| `model` | string | 是 | 固定为 `"eres2netv2"` |
| `normalize` | bool | 否 | 是否 L2 归一化，默认 `true` |
| `user` | string | 是 | 调用方标识，用于日志追踪 |
| `sample_rate` | number | 否 | 采样率提示；省略时从 WAV 头自动读取 |

音频要求：WAV 格式，16kHz，单声道，int16。非 16kHz 会自动重采样。

#### 响应体

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | string | 任务类型，固定为 `"speaker_embedding"` |
| `task_id` | string | 请求级 UUID，用于追踪 |
| `duration` | number | 音频时长（秒） |
| `embeddings` | array[float] | 平铺的 L2 归一化 float32 向量 |
| `dimensions` | number | 向量维度，`eres2netv2` 为 192 |
| `error` | string | 错误信息，成功时为空字符串 |

#### 示例

**请求：**
```json
{
  "base64": "<wav_base64>",
  "model": "eres2netv2",
  "user": "my-app"
}
```

**成功响应：**
```json
{
  "task": "speaker_embedding",
  "task_id": "a1b2c3d4e5f67890",
  "duration": 1.0,
  "embeddings": [0.026, 0.134, 0.107, "..."],
  "dimensions": 192,
  "error": ""
}
```

**失败响应：**
```json
{
  "task": "speaker_embedding",
  "task_id": "a1b2c3d4e5f67890",
  "duration": 0.0,
  "embeddings": [],
  "dimensions": 0,
  "error": "Extraction failed (audio too short or model error)."
}
```

#### Python 调用示例

```python
import base64, io, os, wave
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

async def extract_embedding(pcm: bytes, model: str = "eres2netv2") -> np.ndarray | None:
    token = os.environ.get("API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "http://localhost:8080/v1/audio/speaker/embedding",
            json={
                "base64": pcm_to_wav_b64(pcm),
                "model": model,
                "user": "my-app",
            },
            headers=headers,
        )
        data = resp.json()
    if data["error"] or not data["embeddings"]:
        return None
    return np.array(data["embeddings"], dtype=np.float32)
```

---

## 模型

| 模型 | 参数 | EER | minDCF | 说明 |
|------|------|-----|--------|------|
| `eres2netv2` | ~200MB | **19.1%** | 0.0100 | 唯一公开模型名 |

> 评测方式：`python client/eval_speaker.py --data data --engine eres2netv2`

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HTTP_PORT` | `8080` | HTTP 服务端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `ERES2NET_MODEL_PATH` | `./models/iic/speech_eres2netv2_sv_zh-cn_16k-common` | ERes2NetV2 模型路径 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `API_TOKEN` | (空) | HTTP Bearer token；为空时跳过认证 |
| `SPEAKER_MODEL_POOL_SIZE` | `max(2, min(cpu//2, 8))` | 模型实例池大小 |
