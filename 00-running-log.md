# 运行日志

## 任务 1：学习沐曦的 smi 指令和专门的适配库

目标：确认沐曦 GPU 环境可用，为后续模型部署建立环境基线。

关键流水：

1. 租用沐曦 C500 / 16GB 实例，选择 PyTorch 2.6.0 / Python 3.10 / maca 3.2.1.3 镜像；
2. 通过 Lab 进入实例终端；
3. 执行基础环境检查：date、hostname、uname -a、python --version、pip --version；
4. 执行 lscpu、free -h、df -h 确认系统资源；
5. 创建 /data/exam 输出目录；
6. 运行 mx-smi，确认沐曦 GPU 可被识别；
7. 运行 pip list | grep -e maca -e metax，确认存在沐曦适配库；
8. 回到认证页面申请检测，任务 1 通过。

对应文档：`04-task1-mx-smi.md`

## 任务 2：部署文本生成模型

目标：
1. 使用 /mnt/moark-models/Qwen3-8B 做单次推理；
2. 生成 100 字以上短篇小说；
3. 保存到 /data/exam/text_inference.txt；
4. 使用 vLLM 在 8188 端口启动 OpenAI 兼容 API；
5. 使用 curl 测试 /v1/chat/completions；
6. 申请检测。

对应文档：`05-task2-text-model.md`

## 任务 3：部署图像生成模型

目标：使用 Z-Image-Turbo 完成图像生成，并部署 FastAPI 图像生成 API。

关键流水：

1. 检查并安装 diffusers、accelerate 等依赖；
2. 检查模型路径 /mnt/moark-models/Z-Image-Turbo；
3. 读取 model_index.json，确认 _class_name 为 ZImagePipeline；
4. 编写并运行单次推理脚本，生成 /data/exam/image_output.png（1024x1024 RGB PNG）；
5. 编写 FastAPI 服务，接口为 /v1/images/generations，端口 8188；
6. curl 测试返回 b64_json，服务端保存第一张图片到 /data/exam/image_output.png；
7. 平台检测任务 3 通过。

对应文档：`06-task3-image-model.md`

## 任务 4：部署语音识别模型

目标：使用 Qwen3-ASR-1.7B 完成语音识别，并部署 FastAPI 音频上传接口。

关键流水：

1. 检查模型路径 /mnt/moark-models/Qwen3-ASR-1.7B 和测试音频 /mnt/moark-models/L1_exam/asr_demo.wav；
2. 安装 accelerate、librosa、soundfile、fastapi、uvicorn、python-multipart 等依赖；
3. 第一次尝试用 AutoModelForSpeechSeq2Seq 加载失败，Transformers 不识别 qwen3_asr；
4. 检查 config.json 发现 auto_map 为 None，trust_remote_code 无法解决；
5. 按 README 改用 qwen-asr 包和 Qwen3ASRModel；
6. 处理 backend 参数错误和 language="zh" 不被支持的问题；
7. 完成单次推理，生成 /data/exam/asr_output.txt；
8. 编写 FastAPI 服务，接口为 /v1/audio/transcriptions，端口 8188，支持 multipart/form-data；
9. 平台检测任务 4 通过。

对应文档：`07-task4-asr.md`

## 任务 5：部署语音合成模型 TTS

关键流水：

1. 检查 IndexTTS 仓库、IndexTTS-2 模型目录、config.yaml 和参考音频路径；
2. 检查 torch 版本为 2.6.0+metax3.2.1.3，CUDA 可用；
3. 补齐 matplotlib、audiotools、tn 等依赖；
4. 将 transformers 回退到 4.52.1，将 tokenizers 回退到 0.21.0；
5. 确认 accelerate 版本为 1.8.1，IndexTTS2 import OK；
6. 处理 /mnt/moark-models 下 tagger_cache 只读问题，将缓存目录改到 /tmp；
7. 完成单次 TTS 推理，生成 /data/exam/tts_output.wav；
8. 检查 WAV 文件：单声道、16 bit、22050 Hz，时长约 5 秒；
9. 启动 FastAPI 服务，端口 8188，接口 /v1/audio/speech；
10. 本地 JSON curl 成功后，平台检测出现 HTTP 500；
11. 排查发现平台实际发送 multipart/form-data，补充 form 请求兼容；
12. 第二次平台检测提示缺少有效 b64_json 字段；
13. 修改响应格式，返回顶层 b64_json 和 data[0].b64_json；
14. 再次提交检测，任务 5 通过。

对应文档：`08-task5-tts.md`

## 任务 6：部署 OCR 模型

关键流水：

1. 检查任务要求，确认模型为 FireRed-OCR；
2. 检查模型路径 /mnt/moark-models/FireRed-OCR；
3. 检查测试图片 /mnt/moark-models/L1_exam/ocr_test_image.jpg；
4. 检查 config.json，确认 model_type 为 qwen3_vl，architectures 为 Qwen3VLForConditionalGeneration，auto_map 为 None；
5. 检查 torch、transformers、PIL、fastapi、uvicorn、python-multipart 等依赖；
6. 编写并运行单次 OCR 推理脚本，生成 /data/exam/ocr_output.txt；
7. 编写 FastAPI 服务，接口为 /v1/vision/ocr，端口 8188；
8. uvicorn 使用 code.task6_ocr_server:app 启动失败后，改用 --app-dir /data/code task6_ocr_server:app；
9. 使用 curl 上传 /mnt/moark-models/L1_exam/ocr_test_image.jpg 测试接口；
10. 平台检测任务 6 通过。

对应文档：`09-task6-ocr.md`

## 任务 7：部署向量化模型与向量数据库

关键流水：

1. 选择 vLLM 镜像，普通镜像中没有 vLLM；
2. 安装 `chromadb` 和 `requests`；
3. 检查 `/mnt/moark-models/Qwen3-Embedding-8B`、`/mnt/moark-models/bge-reranker-v2-m3` 和 `/mnt/moark-models/L1_exam/embedding_documents.txt`；
4. 使用 vLLM 启动 Qwen3-Embedding-8B，服务端口为 8188；
5. 验证 `/v1/embeddings` 可用；
6. 直接请求 embeddings 默认返回 4096 维，加入 `dimensions=1024` 后返回 1024 维；
7. 读取文档并切片，至少生成 10 个 chunk；
8. 使用 Qwen3-Embedding-8B 生成向量并写入 Chroma；
9. 从 Chroma 检索 Top-10 候选；
10. 使用 bge-reranker-v2-m3 对 Top-10 重排；
11. 生成 `/data/exam/reranking_results.json`；
12. Task 7 检测通过。

对应文档：`10-task7-embedding-rerank.md`
