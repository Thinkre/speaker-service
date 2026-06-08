# Speaker Service 评估报告

**测试日期：** 2026-06-08  
**测试集：** AliMeeting-mini（4 clips，共约 400s，涵盖远场/近场）  
**评估指标：** DER / JER（collar=0.25s，业界标准）  
**VAD：** 本地 FireRed VAD（不走 gRPC），threshold=0.10  
**聚类：** 1s/0.5s 滑动窗口提取 embedding，spectral 聚类  

---

## 汇总结果

| 模式 | Avg DER% | Avg JER% | 说明 |
|------|--------:|--------:|------|
| **Offline**（全批聚类） | **19.40** | **23.51** | 先收集所有 embedding 再聚类，接近 WeSpeaker 基准 |
| Online（流式聚类） | 47.29 | 58.83 | 增量赋值，cold-start 问题明显 |
| WeSpeaker 基准 | 17.10 | — | ERes2NetV2 + FireRed + WeSpeaker 全批聚类 |

---

## 逐 Clip 明细

### Offline 模式

| Clip | DER% | JER% | Miss% | FA% | Conf% | Ref Spk | Hyp Spk | Baseline DER% |
|------|-----:|-----:|------:|----:|------:|--------:|--------:|--------------:|
| R8002_M8002_MS802 | 17.30 | 20.52 | 1218.20 | 164.70 | 34.90 | 2 | 2 | 16.26 |
| R8008_M8014_MS807 | 8.89 | 9.90 | 514.90 | 192.90 | 92.20 | 3 | 3 | 14.39 |
| R8009_M8021_MS810 | 24.81 | 23.73 | 1438.80 | 490.90 | 72.90 | 2 | 2 | 16.26 |
| R8002_M8002_N_SPK8005 | 26.61 | 39.88 | 1580.00 | 92.30 | 508.80 | 2 | 2 | 31.95 |
| **Average** | **19.40** | **23.51** | | | | | | **17.10** |

### Online 模式

| Clip | DER% | JER% | Baseline DER% |
|------|-----:|-----:|--------------:|
| R8002_M8002_MS802 | 45.96 | 55.09 | 16.26 |
| R8008_M8014_MS807 | 40.82 | 62.36 | 14.39 |
| R8009_M8021_MS810 | 37.74 | 39.74 | 16.26 |
| R8002_M8002_N_SPK8005 | 64.63 | 78.14 | 31.95 |
| **Average** | **47.29** | **58.83** | **17.10** |

---

## 分析

**Offline vs WeSpeaker 基准差距（2.3%）** 来源于：
- WeSpeaker 使用完整语音段（VAD 段内全部帧）提取 embedding
- 本服务使用 1s/0.5s 滑动窗口，边界帧利用率较低

**Online 模式差距大（30%）** 来源于：
- `SpectralClusterer` 增量赋值的 cold-start 阶段（前 16 帧为 provisional）
- 早期帧聚类不稳定导致说话人 ID 混乱，后续无法回溯修正
- 适合延迟敏感的实时场景，精度损失是已知代价

---

## 推荐配置

```bash
# 离线评估 / 高精度场景
uv run python client/eval_speaker.py --mode offline --engine eresnetv2

# 实时场景（接受精度损失）
uv run python client/eval_speaker.py --mode online --engine eresnetv2
```
