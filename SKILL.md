---
name: local-style-writer
description: >-
  端侧个性化内容生成助手（local, on-device personalized writing assistant）。基于 QLoRA 个人风格学习，
  本地 Qwen3-8B INT4 模型经 OpenVINO 在 NPU/GPU/CPU 上异构推理，直接生成符合用户风格的原创内容，草稿 0 网络外联。
  当用户说"帮我写一篇""用我的风格生成""保持人味""团队风格统一""像我自己写的""保护未发表稿件""扩写/改写/续写"，
  或 write/draft/rewrite/expand in my style、keep it human、on-device/local/offline writing 时触发。
  也支持风格一致性分析（analyze / 风格检查）。适合未发表内容、企业机密文稿等隐私敏感场景。
  触发词包含中文（写/撰写/扩写/改写/生成/风格分析）与英文（write/draft/generate/rewrite/analyze style），
  以及 本地/离线/offline/OpenVINO/AIPC/NPU 等语境。
  Prefer this skill whenever the user wants locally-generated, style-consistent content that must not leave the device.
---

# Local Style Writer

端侧个性化内容生成助手 — 保护未发表成果，保留个人品牌语调。

## Usage

宿主只需调用唯一入口 `scripts\run.ps1`，它会自动完成硬件检测、环境安装、模型下载、服务拉起，再返回生成结果。

```
scripts\run.ps1 "<写作主题>" [--length 300字] [--tone <语气预设>] [--rag] [--analyze "<文本>"] [--user <用户ID>]

> 参数经 `run.ps1` 原样透传给 `client.py`（argparse），请使用 **POSIX 双横线**（如 `--length`）；`-Length` 这类 PowerShell 单横线写法无法被 argparse 识别。
```

Examples：

| 意图 | 命令 |
| --- | --- |
| 用我的风格写一篇短文 | `scripts\run.ps1 "帮我写一篇关于 AI PC 优势的短文" --length 300字` |
| 指定语气 | `scripts\run.ps1 "写一条产品推广文案" --tone marketing` |
| 参考历史文章（RAG） | `scripts\run.ps1 "写一篇同风格的新稿子" --rag` |
| 风格一致性分析 | `scripts\run.ps1 --analyze "待分析的文本"` |
| 模型下载中断后续传 | `scripts\run.ps1`（重新运行会再次触发 download.ps1 续下） |

### 输出解读

- **【生成内容】**：生成的原创正文；**【风格分析】** 给出风格评分与明细；
- 启用 RAG 时会标注"（已启用 RAG 参考个人历史文章）"。

Important：
- `scripts\run.ps1` 是唯一受支持的接口，请勿直接调用 `client.py` / `server.py` 等内部脚本。
- 首次调用会下载模型（约 5.6GB）；若超时中断（退出码 3），直接重新运行 `scripts\run.ps1` 即可继续下载。
- 在不支持的硬件（无 OpenVINO 运行时）上会打印平台错误并以退出码 `1` 结束。
- 纯本地运行，任何情况下都不回退到云端服务。

## What this skill does NOT do

- 不做模型训练与导出/量化（QLoRA 训练、INT4 转换属离线一次性流程，见 `FINETUNE_GUIDE.md`）。
- 不联网检索或调用任何云端大模型。

---

## 内部实现参考

> 仅供开发/调试参考，宿主无需了解。整套运行形态是 **命名管道（named pipe）单服务**，不是 HTTP 服务。

- `scripts\run.ps1`：固定入口。硬件检测 → `install-env.ps1` 建 venv → 确保模型 → 调用 `client.py`。
- `scripts\client.py`：短命客户端。确保 `server.py` 已作为后台常驻进程拉起，等待就绪后通过命名管道发送请求并读回结果。
- `scripts\server.py`：长驻服务，监听 `\\.\pipe\local-style-writer`，模型懒加载。提供 `status` / `request` / `shutdown` 三个操作；生成时自动注入用户风格画像（`profiles/<user_id>.json`）与可选 RAG 参考。
- `device_manager` 在启动时探测本机加速器（NPU/GPU/CPU）与内存，为不同模型拼出最优异构链（如 `AUTO:NPU,GPU,CPU`）；无 OpenVINO 时整体回退 CPU。

### 环境变量（可选，启动前设置）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PERSONAL_MODEL_PATH` | `models/qwen3_personal_int4` | Qwen3-8B 生成模型路径 |
| `ANALYSIS_MODEL_PATH` | `models/qwen2.5_int4` | Qwen2.5-0.5B 风格分析模型路径 |
| `BGE_MODEL_PATH` | `models/bge_int8` | BGE 嵌入模型路径 |

## 注意事项

- 模型自动分配到最优设备（NPU/GPU/CPU）。带 NPU 的 Intel AI PC 上 8B 生成可达 30+ tok/s，实时出稿；风格分析（0.5B）在各设备均秒级。纯 CPU（无 NPU/独显）环境 8B 生成极慢（约 0.2 字/秒，300 字需 20+ 分钟），仅适合短文或风格分析——如有 NPU/独显请在其上运行以获得最佳体验。
- 新用户无历史数据时不启用 RAG，使用默认风格模板兜底。
- 纯本地运行，内容不离开用户设备，无需担心隐私泄露。
- 并发：同一模型推理中时，新请求返回错误提示，稍后重试即可。
- 端到端测试见 `tests\test.ps1`。
