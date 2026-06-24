# Bella Web — Speaker Embedding 接口文档

> 来源：[落兵台接口管理平台](https://weapons.ke.com/project/18932/interface/api/1892409)  
> 项目：智能架构组 / openapi  
> Mock 地址：`https://weapons.ke.com/mock/18932/v1/audio/speaker/embedding`  
> 创建人：shenenqing001 · 最后更新：2025-08-07 · 状态：未完成

---

## 项目概览

Bella Web 项目（#18932）下 openapi 分类包含以下接口：

| 分类 | 接口 | 方法 |
|------|------|------|
| audio | speaker diarization | POST |
| audio | **speaker embedding** | POST |
| audio | vad（暂无） | POST |
| audio | flash asr | POST |
| audio | transcription | POST |
| audio | query transcription result | POST |

---

## POST /v1/audio/speaker/embedding

从一段音频中提取说话人嵌入向量。

### 请求头

| 参数名称 | 参数值 | 必需 |
|----------|--------|------|
| `Content-Type` | `application/json` | 是 |
| `Authorization` | `Bearer <token>` | 是 |

### 请求体（JSON）

| 名称 | 类型 | 必需 | 默认值 | 备注 |
|------|------|------|--------|------|
| `url` | string | 否 | — | 音频需要 WAV 格式，16k |
| `base64` | string | 否 | — | base64 编码的 WAV 音频 |
| `model` | string | 是 | — | 模型选择 |
| `normalize` | boolean | 否 | — | 是否 L2 归一化 |
| `user` | string | 是 | — | 调用方标识 |
| `sample_rate` | number | 否 | — | 采样率 |

### 响应体

| 名称 | 类型 | 必需 | 备注 |
|------|------|------|------|
| `task` | string | 否 | 任务类型 |
| `task_id` | string | 否 | 任务 ID |
| `duration` | number | 否 | 音频时长 |
| `embeddings` | array | 否 | 嵌入向量数组 |
| `dimensions` | number | 否 | 向量维度，192 / 512 维 |
| `error` | string | 否 | 错误信息，判断不为空即为失败 |

---

## 关联接口

同一项目下与 speaker embedding 相关的接口：

- [speaker diarization](https://weapons.ke.com/project/18932/interface/api/1735235) — 说话人分离
- [speaker embedding (当前)](https://weapons.ke.com/project/18932/interface/api/1892409) — 说话人嵌入
