# 06 任务 3：Z-Image-Turbo 图像生成

## 1. 任务目标

任务 3 的目标是部署图像生成模型，并提供 OpenAI 兼容风格的图像生成接口。

本任务完成内容：

1. 使用 `Z-Image-Turbo` 完成单次图像生成推理。
2. 使用 FastAPI 部署 HTTP 服务。
3. 提供接口：

```text
/v1/images/generations
```

4. 使用 `curl` 调用接口完成测试。
5. 在认证平台完成检测，并记录截图与 Git 提交。

任务详情截图：

![任务 3 详情](../assets/06-task3-detail.png)

说明：当前仓库没有 `screenshots/` 目录；已有截图统一保存在 `assets/` 目录中，本文只引用仓库中真实存在的截图文件。

## 2. 环境检查

本任务依赖 Python、PyTorch、diffusers、accelerate、transformers、sentencepiece、safetensors 等包，并要求当前实例可以访问 MetaX GPU 环境。

环境检查示例命令：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import diffusers; print(diffusers.__version__)"
pip list | grep -E "diffusers|accelerate|transformers|torch"
```

实际命令文本输出未在仓库中保存为日志文件，结果以截图记录为准：

![任务 3 包检查](../assets/06-task3-package-check.png)

安装或升级 diffusers / accelerate 的过程截图：

![安装 diffusers 和 accelerate](../assets/06-install-diffusers-accelerate.png)

## 3. 模型路径

任务 3 使用的主模型为：

```text
/mnt/moark-models/Z-Image-Turbo
```

代码中的相关文件：

```text
code/task3_image_inference.py
code/task3_image_server.py
```

模型路径检查截图：

![Z-Image-Turbo 模型路径检查](../assets/06-image-model-path-check.png)

## 4. 单次推理

单次推理脚本：

```text
code/task3_image_inference.py
```

该脚本会加载本地模型目录，生成一张图片，并保存到：

```text
/data/exam/image_output.png
```

单次推理示例命令：

```bash
python code/task3_image_inference.py
```

实际命令文本输出未在仓库中保存为日志文件，结果以截图记录为准：

![图像生成单次推理输出](../assets/06-image-inference-output.png)

输出文件检查截图：

![图像输出文件检查](../assets/06-image-output-file-check.png)

## 5. FastAPI 服务

FastAPI 服务脚本：

```text
code/task3_image_server.py
```

服务启动后监听：

```text
0.0.0.0:8188
```

核心接口：

```text
POST /v1/images/generations
```

服务会在启动时加载：

```text
/mnt/moark-models/Z-Image-Turbo
```

首张生成图片会保存到：

```text
/data/exam/image_output.png
```

服务启动示例命令：

```bash
python code/task3_image_server.py
```

服务启动截图：

![FastAPI 图像生成服务启动](../assets/06-fastapi-server-start.png)

## 6. curl 测试

接口测试示例命令：

```bash
curl -X POST "http://127.0.0.1:8188/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Z-Image-Turbo",
    "prompt": "A small white robot standing beside a glowing lake at sunrise, cinematic lighting, detailed, clean composition",
    "size": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "seed": 42,
    "n": 1
  }'
```

说明：以上为示例命令。实际响应内容没有以文本日志形式保存到仓库，不能在文档中补写不存在的完整输出。

curl 测试截图：

![FastAPI 图像生成 curl 测试](../assets/06-fastapi-curl-test.png)

接口生成后的输出文件检查截图：

![FastAPI 图像输出文件检查](../assets/06-fastapi-image-output-check.png)

## 7. 平台检测

完成单次推理、FastAPI 服务启动和 curl 调用后，在认证平台提交检测。

平台检测通过截图：

![任务 3 平台检测通过](../assets/06-task3-pass.png)

## 8. Git 提交

任务 3 相关 Git 记录来自当前仓库历史：

```text
727cb0e task3: complete image generation deployment
```

对应版本标签：

```text
v0.2-task3-image-model
```

后续补充脚本与缓存忽略规则的提交：

```text
7668913 chore: add missing task3 task4 scripts and ignore cache
```

## 9. 本任务总结

任务 3 已完成 `Z-Image-Turbo` 图像生成模型的本地推理和 FastAPI 服务部署，接口路径为 `/v1/images/generations`，服务端口为 `8188`。

仓库中已有任务详情、环境检查、单次推理、服务启动、curl 测试、输出文件检查和平台检测通过截图。由于没有保存完整命令文本输出，本文只记录示例命令，并将实际结果指向真实截图。
