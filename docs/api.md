# Speaker Service gRPC 接口文档

**Proto 文件：** `proto/speaker.proto`  
**默认端口：** 50052  
**协议：** gRPC (HTTP/2, Protobuf)

---

## Service: SpeakerService

### ExtractEmbedding

从 PCM 音频片段中提取 L2 归一化的说话人嵌入向量。

**Request: ExtractRequest**

| 字段 | 类型 | 说明 |
|------|------|------|
| `pcm` | `bytes` | int16 mono 16kHz PCM；建议时长 ≥ 500ms（≥ 16000 bytes） |
| `engine` | `string` | `"eresnetv2"` \| `"campplus"`，默认 `"eresnetv2"` |

**Response: ExtractResponse**

| 字段 | 类型 | 说明 |
|------|------|------|
| `embedding` | `repeated float` | L2 归一化 float32 向量，dim=192 |
| `dim` | `int32` | 向量维度（ERes2Net: 192，CamPlus: 192） |
| `success` | `bool` | 提取成功为 true；PCM 过短或模型错误时为 false |

---

## Python 客户端示例

```python
import numpy as np
from client.speaker_client import SpeakerClient

with SpeakerClient(host="localhost", port=50052) as client:
    # 提取单段 embedding
    with open("segment.pcm", "rb") as f:
        pcm = f.read()

    emb = client.extract_embedding(pcm, engine="eresnetv2")
    if emb is not None:
        print(f"Embedding shape: {emb.shape}")  # (192,)
        print(f"L2 norm: {np.linalg.norm(emb):.4f}")  # ≈ 1.0

    # 计算两段音频的余弦相似度
    emb1 = client.extract_embedding(pcm_spk1)
    emb2 = client.extract_embedding(pcm_spk2)
    similarity = float(np.dot(emb1, emb2))  # 两向量均已 L2 归一化
    print(f"Cosine similarity: {similarity:.4f}")
    # 同一说话人约 0.7+，不同说话人约 0.1–0.4
```

---

## 配合评估脚本使用

```bash
# 启动服务
uv run python server.py

# 离线聚类评估（推荐，DER≈19.4%）
uv run python client/eval_speaker.py \
  --testset /path/to/alimeeting_mini \
  --mode offline \
  --engine eresnetv2

# 在线聚类评估（实时场景，DER≈47.3%）
uv run python client/eval_speaker.py \
  --testset /path/to/alimeeting_mini \
  --mode online

# 同时跑两种模式对比
uv run python client/eval_speaker.py \
  --testset /path/to/alimeeting_mini \
  --mode both
```

报告输出到 `reports/eval_speaker_YYYYMMDD_HHMMSS.{json,md}`，包含 DER/JER 与 WeSpeaker 基准对比。

---

## 引擎对比

| 引擎 | 模型大小 | 模型路径 | 环境变量 | 说明 |
|------|---------|---------|---------|------|
| `eresnetv2` | ~200MB | `models/iic/speech_eres2netv2_sv_zh-cn_16k-common` | `ERES2NET_MODEL_PATH` | **默认，推荐**，ERes2NetV2 |
| `campplus` | ~50MB | `models/iic/speech_campplus_sv_zh-cn_16k-common` | `CAMPPLUS_MODEL_PATH` | 轻量，精度略低 |

---

## 服务特性

- **无状态**：每次 `ExtractEmbedding` 调用独立，无会话概念
- **懒加载**：引擎在第一次被调用时初始化，首次调用耗时约 5–10s
- **并发**：`ThreadPoolExecutor(max_workers=4)`，支持多路并发请求
- **向量格式**：float32，L2 归一化（可直接做点积计算余弦相似度）
