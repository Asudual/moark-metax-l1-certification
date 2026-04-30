# 07 任务 4：Qwen3-ASR-1.7B 语音识别

## 1. 任务目标

任务 4 的目标是部署 ASR 语音识别模型，并提供文件上传形式的语音转写接口。

本任务完成内容：

1. 使用 `Qwen3-ASR-1.7B` 完成单次语音识别推理。
2. 使用 `qwen-asr` 加载 ASR 模型。
3. 使用 FastAPI 部署 HTTP 服务。
4. 提供接口：

```text
/v1/audio/transcriptions
```

5. 使用 `multipart/form-data` 上传音频文件。
6. 使用 `curl` 调用接口完成测试。
7. 在认证平台完成检测，并记录截图与 Git 提交。

任务详情截图：

![任务 4 详情](../assets/07-task4-detail.png)

说明：当前仓库没有 `screenshots/` 目录；已有截图统一保存在 `assets/` 目录中，本文只引用仓库中真实存在的截图文件。

## 2. 环境检查

本任务依赖 Python、PyTorch、FastAPI、python-multipart、qwen-asr 以及本地 ASR 模型文件。

环境检查示例命令：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import qwen_asr; print(qwen_asr)"
pip list | grep -E "qwen|fastapi|uvicorn|python-multipart|torch"
```

实际命令文本输出未在仓库中保存为日志文件，结果以截图记录为准：

![任务 4 包检查](../assets/07-task4-package-check.png)

依赖安装过程截图：

![任务 4 安装依赖](../assets/07-task4-install-deps.png)

`qwen-asr` 安装过程截图：

![安装 qwen-asr](../assets/07-task4-install-qwen-asr.png)

## 3. 模型路径

任务 4 使用的 ASR 模型为：

```text
/mnt/moark-models/Qwen3-ASR-1.7B
```

单次推理使用的测试音频为：

```text
/mnt/moark-models/L1_exam/asr_demo.wav
```

代码中的相关文件：

```text
code/task4_asr_inference.py
code/task4_asr_server.py
```

模型与音频路径检查截图：

![ASR 模型与音频路径检查](../assets/07-asr-model-audio-path-check.png)

模型配置检查截图：

![ASR 模型配置检查](../assets/07-asr-model-config-check.png)

模型配置细节检查截图：

![ASR 模型配置细节检查](../assets/07-asr-model-config-detail-check.png)

## 4. 单次推理

单次推理脚本：

```text
code/task4_asr_inference.py
```

该脚本会加载本地模型目录，识别测试音频，并保存识别文本到：

```text
/data/exam/asr_output.txt
```

单次推理示例命令：

```bash
python code/task4_asr_inference.py
```

实际命令文本输出未在仓库中保存为日志文件，结果以截图记录为准：

![ASR 单次推理输出](../assets/07-asr-inference-output.png)

输出文件检查截图：

![ASR 输出文件检查](../assets/07-asr-output-file-check.png)

## 5. FastAPI 服务

FastAPI 服务脚本：

```text
code/task4_asr_server.py
```

服务启动后监听：

```text
0.0.0.0:8188
```

核心接口：

```text
POST /v1/audio/transcriptions
```

接口接收 `multipart/form-data`，其中：

| 字段 | 说明 |
|---|---|
| `file` | 必填，上传音频文件 |
| `model` | 可选，默认 `Qwen3-ASR-1.7B` |
| `language` | 可选，默认 `zh`，服务内会归一化为 `Chinese` |

服务会在启动时加载：

```text
/mnt/moark-models/Qwen3-ASR-1.7B
```

识别文本会保存到：

```text
/data/exam/asr_output.txt
```

服务启动示例命令：

```bash
python code/task4_asr_server.py
```

服务启动截图：

![FastAPI ASR 服务启动](../assets/07-fastapi-asr-server-start.png)

## 6. curl 测试

接口测试示例命令：

```bash
curl -X POST "http://127.0.0.1:8188/v1/audio/transcriptions" \
  -F "file=@/mnt/moark-models/L1_exam/asr_demo.wav" \
  -F "model=Qwen3-ASR-1.7B" \
  -F "language=zh"
```

说明：以上为示例命令。实际响应内容没有以文本日志形式保存到仓库，不能在文档中补写不存在的完整输出。

curl 测试截图：

![FastAPI ASR curl 测试](../assets/07-fastapi-asr-curl-test.png)

接口调用后的输出文件检查截图：

![FastAPI ASR 输出文件检查](../assets/07-fastapi-asr-output-file-check.png)

## 7. 平台检测

完成单次推理、FastAPI 服务启动和 curl 调用后，在认证平台提交检测。

平台检测通过截图：

![任务 4 平台检测通过](../assets/07-task4-pass.png)

## 8. Git 提交

任务 4 相关 Git 记录来自当前仓库历史：

```text
e60d793 task4: complete asr deployment
```

对应版本标签记录在当前仓库历史中：

```text
v0.3-task4-asr
```

当前标签所在提交：

```text
7668913 chore: add missing task3 task4 scripts and ignore cache
```

## 9. 本任务总结

任务 4 已完成 `Qwen3-ASR-1.7B` 的单次语音识别推理和 FastAPI 服务部署，接口路径为 `/v1/audio/transcriptions`，上传方式为 `multipart/form-data`，服务端口为 `8188`。

仓库中已有任务详情、依赖安装、模型检查、单次推理、服务启动、curl 测试、输出文件检查和平台检测通过截图。由于没有保存完整命令文本输出，本文只记录示例命令，并将实际结果指向真实截图。
