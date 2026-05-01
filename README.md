# 模力方舟 L1 模型部署与应用入门认证实验手册

这是一个学生视角整理的开源实验手册，用来记录“模力方舟 / 沐曦 MetaX / L1 模型部署与应用入门认证”的实际操作过程、关键截图、代码和报错排查。

当前项目主要覆盖认证前置准备、GPU 实例租用、环境检查，以及任务 1～任务 7 的完成过程。README 只作为项目导航，具体命令、截图和排错过程请进入对应 Markdown 文档查看。

## 当前进度

| 任务 | 状态 | 说明 |
|---|---|---|
| 任务 1：学习沐曦的 smi 指令和专门的适配库 | 已通过 | 已记录环境检查、`mx-smi`、MetaX 相关包检查和检测通过截图 |
| 任务 2：部署文本生成模型 | 已通过 | 已记录 Qwen3-8B 文本推理、`/data/exam/text_inference.txt`、vLLM 服务部署和检测通过截图 |
| 任务 3：部署图像生成模型 | 已通过 | 已记录 Z-Image-Turbo 单次推理、FastAPI `/v1/images/generations` 和检测通过截图 |
| 任务 4：部署语音识别模型 | 已通过 | 已记录 Qwen3-ASR-1.7B 单次推理、FastAPI `/v1/audio/transcriptions` 和检测通过截图 |
| 任务 5：部署语音合成模型 TTS | 已通过 | 已记录 IndexTTS-2 单次推理、FastAPI `/v1/audio/speech`、multipart/form-data 修复和检测通过截图 |
| 任务 6：部署 OCR 模型 | 已通过 | 已记录 FireRed-OCR 单次推理、FastAPI `/v1/vision/ocr` 和检测通过截图 |
| 任务 7：部署向量化模型与向量数据库 | 已通过 | 已记录 Qwen3-Embedding-8B、Chroma、bge reranker、1024 维向量和检测通过截图 |

## 文档导航

| 文档 | 作用 |
|---|---|
| `01-certification-overview.md` | 认证任务概览，记录 L1 认证的整体目标、任务列表、通用要求和手册整理思路 |
| `02-gpu-rental.md` | GPU 实例租用流程，记录算力券、算力市场、镜像选择、实例配置和进入实验环境的方法 |
| `03-env-check.md` | 基础环境检查，记录终端、系统资源、`mx-smi`、`pip list` 等基础验证步骤 |
| `04-task1-mx-smi.md` | 任务 1 实验记录，重点记录沐曦 `mx-smi` 指令和 MetaX 适配库检查过程 |
| `05-task2-text-model.md` | 任务 2 实验记录，重点记录 Qwen3-8B 文本生成、输出文件保存、vLLM API 服务部署和相关报错排查 |
| `06-task3-image-model.md` | 任务 3 实验记录，重点记录 Z-Image-Turbo 图像生成、输出图片保存和 FastAPI 服务部署 |
| `07-task4-asr.md` | 任务 4 实验记录，重点记录 Qwen3-ASR-1.7B 语音识别、输出文本保存和 FastAPI 文件上传接口 |
| `08-task5-tts.md` | 任务 5 实验记录，重点记录 IndexTTS-2 语音合成、WAV 输出、JSON/form 请求兼容和 `b64_json` 修复 |
| `09-task6-ocr.md` | 任务 6 实验记录，重点记录 FireRed-OCR 单次推理、输出文本保存和 FastAPI 图片上传接口 |
| `10-task7-embedding-rerank.md` | 任务 7 实验记录，重点记录 vLLM Embedding API、Chroma 向量库、Top-10 检索和 rerank |
| `11-errors-and-fixes.md` | 任务排错记录，集中记录依赖、模型加载、接口格式、平台检测失败等问题与解决方法 |

## 目录说明

| 路径 | 作用 |
|---|---|
| `assets/` | 保存实验手册正文中使用的关键截图，文件名按文档编号和截图内容命名 |
| `assets/unused/` | 保存重复截图、临时截图、用途不明确或暂时未放入正文的截图，不删除原始信息 |
| `code/` | 保存实验中使用或整理出的脚本代码，例如任务 2～任务 7 的推理、FastAPI 服务和向量检索代码 |
| `recordings/` | 预留用于保存录屏或操作记录文件；当前项目中如未创建该目录，表示暂未整理录屏材料 |

## 使用方式

建议按文档编号顺序阅读：

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10
```

如果只关心已通过任务，可以重点阅读：

```text
04-task1-mx-smi.md
05-task2-text-model.md
06-task3-image-model.md
07-task4-asr.md
08-task5-tts.md
09-task6-ocr.md
10-task7-embedding-rerank.md
```

如果需要核对图片、路径和文档状态，可以查看：

```text
docs-check-report.md
assets/rename-report.md
```

## 说明

本仓库不是官方文档，而是一次认证实践过程的开源实验记录。文档中的步骤、截图和报错排查以当前实验环境为准，后续任务会在实际完成后继续补充。
