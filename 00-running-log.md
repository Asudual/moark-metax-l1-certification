# 运行日志

## 任务 2：部署文本生成模型

目标：
1. 使用 /mnt/moark-models/Qwen3-8B 做单次推理；
2. 生成 100 字以上短篇小说；
3. 保存到 /data/exam/text_inference.txt；
4. 使用 vLLM 在 8188 端口启动 OpenAI 兼容 API；
5. 使用 curl 测试 /v1/chat/completions；
6. 申请检测。

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
