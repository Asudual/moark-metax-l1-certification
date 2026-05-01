# 12｜录屏大纲

## 1. 录屏目标

录屏不是重新跑完整任务，而是讲解如何根据本实验手册复现认证过程。

目标受众是后续同学，重点是：

1. 展示关键操作路径，而不是所有试错过程；
2. 解释每一步为什么这样做；
3. 让观看者可以对照手册独立完成认证。

---

## 2. 推荐讲解顺序

建议按文档编号顺序录制，每个任务一段短视频：

```text
算力租用 -> 环境检查 -> 任务 1 -> 任务 2 -> 任务 3 -> 任务 4 -> 任务 5 -> 任务 6 -> 任务 7 -> 错误排查总结 -> 最终交付
```

也可以合并为 3～4 段：

| 段落 | 覆盖内容 | 预计时长 |
|---|---|---|
| 第一段 | 算力租用、环境检查、任务 1 | 5～8 分钟 |
| 第二段 | 任务 2（文本生成 + vLLM） | 5～8 分钟 |
| 第三段 | 任务 3～6（图像、ASR、TTS、OCR） | 10～15 分钟 |
| 第四段 | 任务 7（Embedding + Rerank）、错误排查、交付 | 5～8 分钟 |

最终录屏链接：

| 段落 | 主题 | 链接 |
|---|---|---|
| 第 1 段 | 项目背景与仓库结构 | https://www.bilibili.com/video/BV1ehRTB2E8N/ |
| 第 2 段 | 任务1-2 环境检查与文本模型 | https://www.bilibili.com/video/BV1vhRTB2EwP/ |
| 第 3 段 | 任务3-5 图像ASR与TTS | https://www.bilibili.com/video/BV1ehRTB2Exc/ |
| 第 4 段 | 任务6-7 OCR与向量检索 | https://www.bilibili.com/video/BV1ehRTB2EUu/ |

---

## 3. 每个任务录屏要展示的内容

### 3.1 算力租用（对应 `02-gpu-rental.md`）

1. 打开模力方舟算力市场；
2. 展示 C500 实例规格和镜像选择；
3. 说明为什么先选 16GB 低成本实例；
4. 展示实例创建成功页面；
5. 展示 Lab 进入方式。

### 3.2 环境检查（对应 `03-env-check.md`）

1. 进入 Lab 终端；
2. 运行 `date`、`hostname`、`uname -a`；
3. 运行 `python --version`、`pip --version`；
4. 运行 `lscpu`、`free -h`、`df -h`；
5. 创建 `/data/exam` 目录；
6. 说明这些检查是为后续任务建立环境基线。

### 3.3 任务 1（对应 `04-task1-mx-smi.md`）

1. 打开认证页面，展示任务 1 要求；
2. 运行 `mx-smi`；
3. 运行 `pip list | grep -e maca -e metax`；
4. 回到认证页面，展示任务 1 已通过。

### 3.4 任务 2（对应 `05-task2-text-model.md`）

1. 展示 Qwen3-8B 模型目录；
2. 说明 PyTorch 镜像和 vLLM 镜像的区别；
3. 展示单次推理脚本和输出文件；
4. 展示 vLLM 服务启动命令和参数；
5. 使用 `curl` 测试 `/v1/models` 和 `/v1/chat/completions`；
6. 展示 `/data/exam/text_inference.txt`；
7. 展示任务 2 检测通过。

重点解释：
- 为什么需要切换到 vLLM 镜像；
- Qwen3 输出 `<think>` 的处理方式；
- 第一台 vLLM 实例失败后重建的判断过程。

### 3.5 任务 3（对应 `06-task3-image-model.md`）

1. 展示 Z-Image-Turbo 模型目录和 `model_index.json`；
2. 说明如何根据 `_class_name` 选择 pipeline；
3. 展示单次推理输出图片；
4. 展示 FastAPI 服务和 curl 测试；
5. 展示任务 3 检测通过。

### 3.6 任务 4（对应 `07-task4-asr.md`）

1. 展示 Qwen3-ASR-1.7B 模型目录；
2. 说明为什么不能用 Whisper 示例，需要用 `qwen-asr` 包；
3. 展示单次推理和输出文件；
4. 展示 FastAPI 服务和 multipart/form-data curl 测试；
5. 展示任务 4 检测通过。

### 3.7 任务 5（对应 `08-task5-tts.md`）

1. 展示 IndexTTS-2 模型目录和参考音频；
2. 说明依赖修复过程（transformers 回退、缓存目录迁移）；
3. 展示单次推理和 WAV 输出；
4. 说明平台 multipart/form-data 和 b64_json 的坑；
5. 展示任务 5 检测通过。

### 3.8 任务 6（对应 `09-task6-ocr.md`）

1. 展示 FireRed-OCR 模型目录；
2. 说明 config.json 中 qwen3_vl 架构的含义；
3. 展示单次推理和输出文件；
4. 说明 uvicorn `--app-dir` 启动方式；
5. 展示任务 6 检测通过。

### 3.9 任务 7（对应 `10-task7-embedding-rerank.md`）

1. 展示 Qwen3-Embedding-8B 和 bge-reranker-v2-m3 模型目录；
2. 展示 vLLM Embedding API 启动和 1024 维验证；
3. 展示文档切片、Chroma 入库和 rerank 脚本运行；
4. 展示 `/data/exam/reranking_results.json`；
5. 展示任务 7 检测通过。

### 3.10 错误排查总结

1. 快速过一遍 `11-errors-and-fixes.md` 中的关键问题；
2. 重点讲解：镜像选择、实例重建、依赖版本、平台请求格式；
3. 说明排错链的价值。

### 3.11 Git 标签与最终交付

1. 展示 `git log --oneline` 和 `git tag`；
2. 说明当前最终标签 `v0.7-task7-embedding-rerank`；
3. 展示项目目录结构；
4. 说明 `13-final-delivery-checklist.md` 的检查项。

---

## 4. 录屏文件建议命名

| 文件名 | 内容 |
|---|---|
| `recordings/01-setup-and-task1.mp4` | 算力租用、环境检查、任务 1 |
| `recordings/02-task2-text-model.mp4` | 任务 2 文本生成 |
| `recordings/03-task3-to-task6.mp4` | 任务 3～6 |
| `recordings/04-task7-and-delivery.mp4` | 任务 7、错误排查、交付 |

---

## 5. 录屏注意事项

1. 录屏前确认终端字体大小足够清晰；
2. 不需要展示所有试错过程，只展示关键路径；
3. 每个任务开头先展示认证页面的任务要求；
4. 每个任务结尾展示检测通过截图；
5. 遇到需要解释的地方可以暂停讲解，不需要一口气跑完；
6. 录屏目标是让后续同学可以对照视频确认关键步骤。
