# Assets Rename Report

整理日期：2026-04-30

## 处理原则

- 所有 `QQ2026*.png`、`QQ截图*.png` 等临时截图名已按图片真实内容处理。
- 未删除任何图片。
- 重复图、过渡截图、代码片段截图或用途较弱的图片已移动到 `assets/unused/`。
- 已有规范文件名但内容不匹配的图片已改名。

## 重命名与移动记录

| 原文件名 | 新文件名 | 截图内容说明 | 处理原因 | 是否移动到 unused |
|---|---|---|---|---|
| `assets/05-vllm-qwen3-0_6b-model-path.png` | `assets/05-qwen3-model-path.png` | `/mnt/moark-models/Qwen3-8B` 模型目录，能看到 `config.json`、`tokenizer`、`safetensors` 权重文件 | 原文件名写成 Qwen3-0.6B，但内容实际是 Qwen3-8B 模型路径 | 否 |
| `assets/05-vllm-generated-text-output.png` | `assets/05-text-inference-final-output.png` | 普通 Transformers 推理脚本 `python code/task2_text_inference.py` 成功生成文本并保存到 `/data/exam/text_inference.txt` | 内容不是 vLLM API 输出，原文件名误带 `vllm` | 否 |
| `assets/05-vllm-instance-package-check.png` | `assets/05-vllm-not-installed-check.png` | PyTorch 镜像中 `python -m vllm` / `pip` 检查显示 vLLM 未安装，同时列出 MetaX 相关包 | 内容重点是确认 vLLM 未安装，原命名不够准确 | 否 |
| `assets/05-text-inference-output-file.png` | `assets/05-vllm-api-generated-text-output.png` | Qwen3-8B vLLM API 结果写入 `/data/exam/text_inference.txt`，并用 `cat` / `wc -m` 验证字符数超过 100 | 内容是 vLLM API 生成并保存最终输出，改成更准确名称 | 否 |
| `assets/05-task2-before-submit.png` | `assets/unused/05-task2-before-submit-duplicate.png` | 与 `05-task2-detail.png` 完全相同的任务 2 详情页截图 | 完全重复，保留 `05-task2-detail.png`，重复图移入 unused | 是 |
| `assets/QQ20260430-205658.png` | `assets/unused/05-task2-torch-cuda-check-with-path-error.png` | 上方包含脚本路径缺失提示，主体是 PyTorch / CUDA / MetaX C500 检查 | 内容混杂，且已有更干净的 CUDA 检查截图 | 是 |
| `assets/QQ20260430-210004.png` | `assets/11-task2-missing-accelerate.png` | `device_map="auto"` 触发 `requires accelerate` 报错 | 报错内容清晰，对应任务 2 缺少 accelerate 的排查图 | 否 |
| `assets/QQ20260430-210140.png` | `assets/11-task2-missing-accelerate-output-file-missing.png` | 缺少 accelerate 报错后检查 `/data/exam/text_inference.txt` 不存在 | 内容不是单纯最终输出，而是 accelerate 缺失导致输出文件未生成 | 否 |
| `assets/QQ20260430-210541.png` | `assets/05-task2-torch-cuda-check.png` | PyTorch / CUDA / MetaX C500 可用性检查 | 临时截图名改为真实内容名称 | 否 |
| `assets/QQ20260430-210736.png` | `assets/unused/05-task2-device-map-auto-code-snippet.png` | `AutoModelForCausalLM.from_pretrained(... device_map="auto")` 代码片段 | 只是代码片段截图，不是核心实验过程证据 | 是 |
| `assets/QQ20260430-210903.png` | `assets/unused/05-text-inference-loading-warning-intermediate.png` | 文本推理脚本加载模型和 warning 的中间状态 | 未展示最终结果，作为过渡截图移入 unused | 是 |
| `assets/QQ20260430-212025.png` | `assets/unused/05-task2-prompt-requirements-snippet.png` | 中文 prompt 要求代码片段 | 只是 prompt 片段截图，不是核心运行证据 | 是 |
| `assets/QQ20260430-212329.png` | `assets/05-text-inference-final-output-wc-check.png` | `cat /data/exam/text_inference.txt` 和 `wc -m`，字符数为 147 | 内容可作为普通 Transformers 最终输出长度验证，使用更具体名称避免覆盖 | 否 |
| `assets/QQ20260430-213520.png` | `assets/unused/05-vllm-instance-list-2min.png` | vLLM 实例列表，运行约 2 分钟 | 与 Lab 入口截图内容接近，信息价值较低 | 是 |
| `assets/QQ20260430-213526.png` | `assets/unused/05-vllm-instance-spec-tooltip.png` | vLLM 实例列表及规格悬浮提示 | 辅助信息截图，非核心证据 | 是 |
| `assets/QQ20260430-213541.png` | `assets/05-vllm-lab-entry-or-instance-list.png` | vLLM 实例列表，红箭头标注 Lab / 工具入口 | 内容不是 release broken instance，改为实例列表 / Lab 入口含义 | 否 |
| `assets/QQ20260430-214832.png` | `assets/11-task2-metax-queue-segfault.png` | `mxkwCreateQueueBlock`、`mxc_queue_acquire failed`、`Segmentation fault` | 对应 MetaX queue / segfault 报错排查图 | 否 |
| `assets/QQ20260430-215222.png` | `assets/11-task2-text-inference-script-path-missing.png` | `/data/code/task2_text_inference.py` 和 `/code/task2_text_inference.py` 找不到，后面也包含 queue 报错 | 按主要内容保留为脚本路径缺失图；queue 报错已由单独截图记录 | 否 |
| `assets/QQ截图20260430213707.png` | `assets/05-vllm-new-instance-package-check.png` | 新 vLLM 实例中检查 Python、Transformers、Torch、vLLM、Qwen3-8B 模型路径，并安装 accelerate | 内容较杂，但能证明新 vLLM 实例环境和模型路径情况 | 否 |
| `assets/QQ截图20260430221239.png` | `assets/05-vllm-qwen3-0_6b-server-start.png` | Qwen3-0.6B 通过 `vllm serve` 启动，显示 vLLM API server 和模型参数 | 对应 Qwen3-0.6B vLLM 服务启动截图 | 否 |
| `assets/QQ截图20260430222637.png` | `assets/05-vllm-qwen3-8b-server-start.png` | Qwen3-8B 通过 `vllm serve` 启动，显示 vLLM API server、8188 端口和模型参数 | 对应 Qwen3-8B vLLM 服务启动截图 | 否 |

## 保持原名的任务 2 截图

| 文件名 | 内容说明 |
|---|---|
| `assets/05-task2-detail.png` | 任务 2 详情页，包含 Qwen3-8B、`text_inference.txt`、vLLM、8188、`/v1/chat/completions` 等要求 |
| `assets/05-task2-package-check.png` | PyTorch 镜像中检查 `python --version`、`pip show transformers`、`pip show torch`、`pip show vllm` |
| `assets/05-install-accelerate.png` | 安装并检查 `accelerate` |
| `assets/05-text-inference-thinking-output.png` | 第一次文本推理成功但输出包含 `<think>` |
| `assets/05-text-inference-short-output.png` | 去掉 `<think>` 后输出偏短 |
| `assets/05-vllm-qwen3-0_6b-curl-test.png` | Qwen3-0.6B 的 `/v1/models` 和 `/v1/chat/completions` curl 测试成功 |
| `assets/05-vllm-qwen3-8b-curl-test.png` | Qwen3-8B 的 `/v1/models` 和 `/v1/chat/completions` curl 测试成功 |
| `assets/05-task2-pass.png` | 任务 2 检测通过页面 |
| `assets/11-task2-vllm-qwen3-8b-engine-init-failed.png` | 第一台 vLLM 实例中 Qwen3-8B 启动失败，`Engine core initialization failed` |
| `assets/11-task2-vllm-use-v1-assertion.png` | 设置 `VLLM_USE_V1=0` 后出现 `AssertionError` |
| `assets/11-task2-vllm-qwen3-0_6b-engine-init-failed.png` | 第一台 vLLM 实例中 Qwen3-0.6B 启动失败，`Engine core initialization failed` |

## 缺失或无法匹配

| 目标文件名 | 状态 | 说明 |
|---|---|---|
| `assets/05-task2-learning-guide-vllm.png` | 未找到可匹配截图 | 当前图片中没有明确展示官方教程说明 vLLM 需要 vLLM / `vllm:0.11.0` / Python 3.10 / maca 3.3.x 专用镜像的截图 |
| `assets/05-vllm-new-instance-mx-smi.png` | 未找到可匹配截图 | 当前图片中没有新 vLLM 实例里的 `mx-smi` 截图；只有 PyTorch / CUDA 检查和新实例 package/model path 检查 |
| `assets/mkdir.png` | 未发现该文件 | 当前 `assets/` 目录中没有名为 `mkdir.png` 的文件 |

## 复核结果

- `assets/` 根目录下已无 `QQ2026*.png` 或 `QQ截图*.png` 临时命名图片。
- 仍存在一组重复内容：`assets/05-task2-detail.png` 与 `assets/unused/05-task2-before-submit-duplicate.png`。这是有意保留的重复备份，根目录只保留规范文件。
