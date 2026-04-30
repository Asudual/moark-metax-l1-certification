# Docs Check Report

检查日期：2026-05-01

## 检查范围

- Markdown 文档：项目根目录下全部 `.md` 文件。
- 重点文档：`01-certification-overview.md` 至 `05-task2-text-model.md`。
- 图片目录：`assets/` 及 `assets/unused/`。

## 检查结果

| 检查项 | 结果 | 说明 |
|---|---|---|
| Markdown 图片引用是否存在 | 通过 | 全部图片引用均能在当前项目中找到对应文件 |
| `assets/` 根目录临时文件名 | 通过 | 未发现 `QQ2026*.png`、`QQ截图*.png`、`mkdir.png` |
| 01～05 文档编号连续性 | 通过 | `01`、`02`、`03`、`04`、`05` 均存在且编号连续 |
| 图片路径是否写错 | 通过 | 未发现断链图片路径 |
| 代码块是否闭合 | 通过 | 全部 `.md` 文件的 fenced code block 数量为偶数 |
| 明显重复图片 | 需注意 | 仅发现一组重复：`assets/05-task2-detail.png` 与 `assets/unused/05-task2-before-submit-duplicate.png`；重复图已位于 `unused`，不影响交付 |

## 图片引用明细

| 文档 | 行号 | 图片路径 | 状态 |
|---|---:|---|---|
| `01-certification-overview.md` | 21 | `assets/01-certification-detail-with-task-list.png` | 存在 |
| `02-gpu-rental.md` | 507 | `assets/04-task1-pass.png` | 存在 |
| `03-env-check.md` | 257 | `assets/03-mx-smi-output.png` | 存在 |
| `04-task1-mx-smi.md` | 224 | `assets/03-mx-smi-output.png` | 存在 |
| `04-task1-mx-smi.md` | 397 | `assets/04-task1-pass.png` | 存在 |
| `05-task2-text-model.md` | 145 | `assets/05-qwen3-model-path.png` | 存在 |
| `05-task2-text-model.md` | 321 | `assets/11-task2-missing-accelerate.png` | 存在 |
| `05-task2-text-model.md` | 342 | `assets/05-install-accelerate.png` | 存在 |
| `05-task2-text-model.md` | 350 | `assets/11-task2-missing-accelerate-output-file-missing.png` | 存在 |
| `05-task2-text-model.md` | 364 | `assets/05-text-inference-thinking-output.png` | 存在 |
| `05-task2-text-model.md` | 391 | `assets/05-text-inference-short-output.png` | 存在 |
| `05-task2-text-model.md` | 464 | `assets/05-vllm-not-installed-check.png` | 存在 |
| `05-task2-text-model.md` | 484 | `assets/05-task2-learning-guide-vllm.png` | 存在 |
| `05-task2-text-model.md` | 567 | `assets/11-task2-metax-queue-segfault.png` | 存在 |
| `05-task2-text-model.md` | 645 | `assets/11-task2-vllm-use-v1-assertion.png` | 存在 |
| `05-task2-text-model.md` | 694 | `assets/05-vllm-qwen3-0_6b-server-start.png` | 存在 |
| `05-task2-text-model.md` | 718 | `assets/05-vllm-qwen3-0_6b-curl-test.png` | 存在 |
| `05-task2-text-model.md` | 750 | `assets/05-vllm-qwen3-8b-server-start.png` | 存在 |
| `05-task2-text-model.md` | 789 | `assets/05-vllm-qwen3-8b-curl-test.png` | 存在 |
| `05-task2-text-model.md` | 908 | `assets/05-task2-pass.png` | 存在 |

## 代码块闭合检查

| 文档 | 围栏行数量 | 状态 |
|---|---:|---|
| `00-running-log.md` | 0 | 闭合 |
| `01-certification-overview.md` | 86 | 闭合 |
| `02-gpu-rental.md` | 70 | 闭合 |
| `03-env-check.md` | 34 | 闭合 |
| `04-task1-mx-smi.md` | 44 | 闭合 |
| `05-task2-text-model.md` | 130 | 闭合 |
| `06-task3-image-model.md` | 0 | 闭合 |
| `07-task4-asr.md` | 0 | 闭合 |
| `08-task5-tts.md` | 0 | 闭合 |
| `09-task6-ocr.md` | 0 | 闭合 |
| `10-task7-embedding-rerank.md` | 0 | 闭合 |
| `11-errors-and-fixes.md` | 0 | 闭合 |

## 已修复问题

本次检查未发现明确的图片路径错误，因此没有修改 01～05 正文内容，也没有移动图片文件。

## 仍需人工确认

- `02-gpu-rental.md` 第 507 行引用 `assets/04-task1-pass.png`。图片文件存在，但该截图属于任务 1 通过页，是否应放在 GPU 租用文档中需要人工确认上下文是否合适。
- `05-task2-text-model.md` 中存在多张已整理但未引用的任务 2 截图，例如 `05-vllm-api-generated-text-output.png`、`05-text-inference-final-output.png`、`05-text-inference-final-output-wc-check.png`、`05-vllm-new-instance-package-check.png`、`05-task2-torch-cuda-check.png`、`05-vllm-lab-entry-or-instance-list.png`。这不属于路径错误，但若目标是完整开源手册，可人工判断是否补充进正文。
- `assets/unused/05-task2-before-submit-duplicate.png` 与 `assets/05-task2-detail.png` 内容完全相同。重复图已经在 `unused` 中，保留不影响交付。

## 交付判断

从结构和资源完整性角度看，任务 1、任务 2 当前已具备可交付基础：

- 01～05 文档编号连续。
- 任务 1、任务 2 所引用图片均存在。
- 根目录 assets 截图命名已规范，无明显临时截图名残留。
- Markdown 代码块闭合。

建议在正式发布前做一次人工通读，重点确认正文叙事是否覆盖所有关键截图，以及 `02-gpu-rental.md` 中引用任务 1 通过页是否符合文档意图。
