# 13｜最终交付检查清单

## 1. 任务通过状态

| 任务 | 状态 | 检测截图 |
|---|---|---|
| 任务 1：学习沐曦的 smi 指令和专门的适配库 | 已通过 | `assets/04-task1-pass.png` |
| 任务 2：部署文本生成模型 | 已通过 | `assets/05-task2-pass.png` |
| 任务 3：部署图像生成模型 | 已通过 | `assets/06-task3-pass.png` |
| 任务 4：部署语音识别模型 | 已通过 | `assets/07-task4-pass.png` |
| 任务 5：部署语音合成模型 TTS | 已通过 | `assets/08-task5-pass.png` |
| 任务 6：部署 OCR 模型 | 已通过 | `assets/09-task6-pass.png` |
| 任务 7：部署向量化模型与向量数据库 | 已通过 | `assets/10-task7-pass.png` |

---

## 2. 文档完整性

| 文档 | 状态 | 说明 |
|---|---|---|
| `README.md` | 完成 | 项目导航、进度表、目录说明 |
| `00-running-log.md` | 完成 | 任务 1～7 运行日志 |
| `01-certification-overview.md` | 完成 | 认证概览、任务列表、通用要求 |
| `02-gpu-rental.md` | 完成 | GPU 租用流程 |
| `03-env-check.md` | 完成 | 基础环境检查 |
| `04-task1-mx-smi.md` | 完成 | 任务 1 实验记录 |
| `05-task2-text-model.md` | 完成 | 任务 2 实验记录 |
| `06-task3-image-model.md` | 完成 | 任务 3 实验记录 |
| `07-task4-asr.md` | 完成 | 任务 4 实验记录 |
| `08-task5-tts.md` | 完成 | 任务 5 实验记录 |
| `09-task6-ocr.md` | 完成 | 任务 6 实验记录 |
| `10-task7-embedding-rerank.md` | 完成 | 任务 7 实验记录 |
| `11-errors-and-fixes.md` | 完成 | 任务 2～7 排错记录 |
| `12-recording-outline.md` | 完成 | 录屏大纲 |
| `13-final-delivery-checklist.md` | 完成 | 本文件 |
| `docs-check-report.md` | 完成 | 历史检查报告 |
| `assets/rename-report.md` | 完成 | 截图重命名记录 |

---

## 3. 代码完整性

| 脚本 | 对应任务 | 状态 |
|---|---|---|
| `code/task2_text_inference.py` | 任务 2 单次推理 | 存在 |
| `code/task2_text_inference(0).py` | 任务 2 历史版本 | 存在 |
| `code/task2_text_inference(1).py` | 任务 2 历史版本 | 存在 |
| `code/task3_image_inference.py` | 任务 3 单次推理 | 存在 |
| `code/task3_image_server.py` | 任务 3 FastAPI 服务 | 存在 |
| `code/task4_asr_inference.py` | 任务 4 单次推理 | 存在 |
| `code/task4_asr_server.py` | 任务 4 FastAPI 服务 | 存在 |
| `code/task5_tts_inference.py` | 任务 5 单次推理 | 存在 |
| `code/task5_tts_server.py` | 任务 5 FastAPI 服务 | 存在 |
| `code/task6_ocr_inference.py` | 任务 6 单次推理 | 存在 |
| `code/task6_ocr_server.py` | 任务 6 FastAPI 服务 | 存在 |
| `code/task7_embedding_rerank.py` | 任务 7 向量检索 | 存在 |

---

## 4. 截图完整性

- [ ] 每个任务的检测通过截图存在
- [ ] 每个任务的关键操作截图存在
- [ ] `assets/` 根目录无临时命名截图（`QQ2026*.png` 等）
- [ ] `assets/unused/` 中的重复截图已标记，不影响正文

---

## 5. Markdown 引用检查

- [ ] 所有图片引用 `![...](assets/...)` 对应的文件存在
- [ ] 所有代码路径引用 `code/...` 对应的文件存在
- [ ] 代码块围栏（` ``` `）全部闭合
- [ ] 无残留未完成占位符

---

## 6. Git 状态检查

- [ ] `git status` 干净（无未跟踪的临时文件）
- [ ] 无 `__pycache__/` 或 `*.pyc` 被 Git 跟踪
- [ ] `.gitignore` 包含 `__pycache__/` 和 `*.pyc`
- [ ] 标签 `v0.7-task7-embedding-rerank` 存在

---

## 7. 录屏状态

- [ ] 录屏大纲已完成（`12-recording-outline.md`）
- [x] 录屏链接已补充到 `README.md` 和 `12-recording-outline.md`
- [ ] 录屏文件已录制并保存到 `recordings/`

---

## 8. 最终交付前检查命令

```bash
# 检查 git 状态
git status --short

# 检查残留占位符
grep -rn -E "截图待[补]|已通过 0 [项]|Qwen3-TTS-12H[z]" -- *.md

# 检查 Markdown 图片引用是否全部存在
grep -roh 'assets/[^)]*' *.md | sort -u | while read f; do
  [ ! -f "$f" ] && echo "MISSING: $f"
done

# 检查是否有 Git 跟踪的 __pycache__ 或 *.pyc
git ls-files '*.pyc' '__pycache__'

# 检查标签
git tag -l 'v0.7*'

# 检查代码文件是否存在
ls -1 code/*.py
```

---

## 9. 交付判断

当以上所有检查项通过后，本实验手册具备交付条件：

1. 任务 1～7 全部通过平台检测；
2. 文档、代码、截图齐全；
3. Markdown 引用无断链；
4. Git 仓库状态干净；
5. 录屏大纲已完成。
