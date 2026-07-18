# 端侧个性化写作助手

> **local-style-writer** — 基于 QLoRA 个人风格学习的本地 AI 写作工具  
> **保护未发表成果 · 保留个人品牌语调**

---

## 目录

- [架构总览](#架构总览)
- [目录结构](#目录结构)
- [运行时调用链路](#运行时调用链路)
- [部署与日常使用](#部署与日常使用)
  - [1. 唯一入口 `scripts\run.ps1`](#1-唯一入口-scriptsrunps1)
  - [2. 参数说明](#2-参数说明)
  - [3. 环境变量（可选）](#3-环境变量可选)
- [SKILL.md 与 Qoder 意图匹配](#skillmd-与-qoder-意图匹配)
- [内部实现参考](#内部实现参考)
  - [命名管道协议](#命名管道协议)
  - [用户风格画像](#用户风格画像)
  - [模型加载与并发](#模型加载与并发)
  - [RAG 检索增强](#rag-检索增强)
  - [设备调度](#设备调度)
- [测试](#测试)
- [文档元信息](#文档元信息)

---

## 架构总览

本技能是一个 **纯本地、零网络外联** 的个性化写作助手：

- 宿主（Qoder / Marvis）只需调用唯一入口 `scripts\run.ps1 "<主题>"`；
- 运行形态是 **命名管道（named pipe）单服务**，不是 HTTP 服务，不在本机开任何端口；
- 长驻服务 `server.py` 监听 `\\.\pipe\local-style-writer`，模型**懒加载**，首次请求才把模型读进内存；
- 推理由 OpenVINO 在 NPU / GPU / CPU 上做异构加速，草稿全程不出电脑。

> 与早期版本的区别：早期用 FastAPI 起了 8000 / 8001 两个 HTTP 服务、并把 `meta_server` 单独拆了出来；现统一为命名管道单服务，画像加载逻辑也并入了 `server.py`。

---

## 目录结构

```
local-style-writer/
├── info.json                 # 技能元数据（venv 名、模型清单、所需内存）
├── meta.json                 # 技能展示元数据（名称 / 用途 / 版本）
├── SKILL.md                  # Qoder 技能注册表（触发词 + 唯一入口说明）
├── README.md                 # 本文件
├── requirements.txt          # 推理端依赖（OpenVINO / optimum-intel / psutil / pydantic ...，已锁版本）
├── training/                 # 训练 / 导出 / 量化脚本（开发工作区，见 FINETUNE_GUIDE.md）
│   ├── requirements-training.txt  # 训练端锁版本依赖（torch cu124 / LLaMA-Factory / OpenVINO 栈）
│   ├── train_qwen3_personal.sh
│   ├── train_qwen2.5_style.sh
│   ├── export_lora.sh
│   ├── merge_and_export.py
│   ├── convert_a_to_alpaca.py / convert_b_to_alpaca.py
│   ├── convert_bge_to_ov.sh / end_to_end.py / validate_data.py
│   └── *.yaml                # LLaMA-Factory 训练配置
├── FINETUNE_GUIDE.md         # 训练 / 导出 / 量化指南（离线一次性流程）
├── .gitignore
├── scripts/                  # 运行时代码（对外只暴露 run.ps1）
│   ├── run.ps1               #   固定入口：环境 → 模型 → client.py
│   ├── install-env.ps1       #   建 venv 并装依赖（uv 优先，回退 python -m venv）
│   ├── client.py             #   短命客户端：拉起服务、发请求、读回结果
│   ├── server.py             #   长驻服务：命名管道 + 模型懒加载
│   ├── device_manager.py     #   异构设备调度（CPU / GPU / NPU 自动探测）
│   ├── qoder_inference.py    #   推理引擎（Qwen3-8B + Qwen2.5-0.5B）
│   └── rag_engine.py         #   RAG 检索增强（bge 嵌入）
├── models/                   # 模型下载入口
│   ├── download.ps1          #   下载 3 个模型（~5.6 GB）
│   └── download.sh
├── profiles/                 # 用户风格画像（user_default.json 兜底）
│   └── user_default.json
├── data/
│   ├── dataset_info.json     # 数据集注册表（LLaMA-Factory 映射 4 个 dataset 键）
│   └── rag_index/            # RAG 向量索引（运行时生成）
├── logs/                     # 运行时日志（server.out 等；Python 业务日志见 ~/.openvino/logs/local-style-writer/）
└── saves/                    # 输出 / 中间产物
```

---

## 运行时调用链路

```
用户输入: "帮我写一篇 AI PC 优势的文章，用我的风格"
         │
         ▼
┌────────────────────────────────────┐
│          Qoder Agent                │
│  1. LLM 解析意图                    │
│  2. 匹配 SKILL.md 触发词 → 命中技能 │
│  3. 调用 scripts\run.ps1 "<prompt>"│
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│  scripts\run.ps1                    │
│  ─────────────                       │
│  1. install-env.ps1 建 venv + 装依赖 │
│  2. 检测 OpenVINO（无则退出码 1）    │
│  3. 模型缺失则 download.ps1         │
│     （未完成 → 退出码 3，提示重跑）  │
│  4. client.py @args                 │
└──────────────┬─────────────────────┘
               │ 命名管道 \\.\pipe\local-style-writer
┌──────────────▼─────────────────────┐
│  scripts\server.py（后台常驻）       │
│  1. load_profile(user_id) → 风格画像 │
│  2. get_model("personal") 懒加载     │
│     device_manager.pick → AUTO:...  │
│  3. [可选] RAG 检索注入 top-3 片段   │
│  4. generate_personal_style()       │
│  5. 返回 {ok, content} → client     │
└────────────────────────────────────┘
```

---

## 部署与日常使用

### 1. 唯一入口 `scripts\run.ps1`

宿主（Qoder）只调用这一个脚本，它会依次完成：建环境 → 检测 OpenVINO → 确保模型就绪 → 调起客户端并透传参数。

```powershell
# PowerShell，从技能根目录运行
scripts\run.ps1 "帮我写一篇关于 AI PC 优势的短文" --length 300字
```

首次使用会自动创建虚拟环境（位于 `%USERPROFILE%\.openvino\venv\local-style-writer`）并下载约 5.6 GB 模型。若下载超时中断（脚本退出码 3），**直接重新运行 `scripts\run.ps1` 即可继续**——它会再次触发 `download.ps1`，ModelScope 通常会断点续传已下载的部分。

### 2. 参数说明

`run.ps1` 把参数原样透传给 `client.py`（argparse），因此请使用 **POSIX 双横线参数**（如 `--length`），不要使用 PowerShell 单横线风格（`-Length` 无法被 argparse 识别）。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `topic`（位置参数） | **必填** 写作主题 | — |
| `--user` | 用户画像 id（对应 `profiles/<user_id>.json`） | `default` |
| `--length` | 目标长度，如 `300字` | 模型/画像默认 |
| `--tone` | 风格预设 | 画像首个 style_tag |
| `--key-points` | 要点列表（可多个） | 无 |
| `--rag` | 启用 RAG，参考个人历史文章 | 关 |
| `--analyze` | 改为分析给定文本的风格（值为待分析文本） | 关（生成模式） |
| `--max-new-tokens` | 最大生成 token 数 | 512 |
| `--temperature` | 随机性 | 0.7 |
| `--json` | 以 JSON 输出完整响应 | 关（人类可读） |

示例：

```powershell
# 用我的风格写一篇短文
scripts\run.ps1 "帮我写一篇关于 AI PC 优势的短文" --length 300字

# 指定语气
scripts\run.ps1 "写一条产品推广文案" --tone marketing

# 参考历史文章（RAG）
scripts\run.ps1 "写一篇同风格的新稿子" --rag

# 风格一致性分析（--analyze 后跟待分析文本）
scripts\run.ps1 --analyze "这是一段用于风格分析的测试文本。"

# 指定用户画像
scripts\run.ps1 "写一篇周报" --user alice
```

输出中：`【生成内容】` 为原创正文；`【风格分析】` 为风格评分与明细；启用 RAG 时会追加提示。

### 3. 环境变量（可选，启动前设置）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PERSONAL_MODEL_PATH` | `models/qwen3_personal_int4` | Qwen3-8B 生成模型路径 |
| `ANALYSIS_MODEL_PATH` | `models/qwen2.5_int4` | Qwen2.5-0.5B 风格分析模型路径 |
| `BGE_MODEL_PATH` | `models/bge_int8` | BGE 嵌入模型路径 |
| `SKILL_LOG_DIR` | （空） | 日志目录前缀；设置后业务日志写到 `<SKILL_LOG_DIR>/local-style-writer/`，否则默认 `~/.openvino/logs/local-style-writer`（绝对路径，避免 CWD 漂移） |

---

## SKILL.md 与 Qoder 意图匹配

`SKILL.md` 是 Qoder Agent 的技能注册表，关键是 `description` 里的触发词。**匹配基于语义理解**，不依赖关键字硬匹配——即使输入是"给我整一篇 AI PC 的稿子"也能命中。

命中后，宿主调用唯一入口 `scripts\run.ps1 "<prompt>"`，背后通过命名管道与 `server.py` 通信，宿主本身不需要知道管道地址或任何端口。

---

## 内部实现参考

> 仅供开发 / 调试参考，宿主无需了解。

### 命名管道协议

`server.py` 监听 `\\.\pipe\local-style-writer`，认证密钥为技能名。客户端（`client.py`）首次使用时若管道未就绪，会以**脱离控制台的后台进程**拉起 `server.py`，然后轮询 `status` 直到 `running`（最长等待 8 分钟，仍无模型则退出码 3）。

三个操作：

| op | 说明 |
|---|---|
| `status` | 返回 state / 已加载模型 / RAG 状态 / 设备分配 / 内存 |
| `request` | `args.action = generate` 生成正文；`= analyze` 分析风格 |
| `shutdown` | 置停止标志，服务在收完当前连接后退出 |

### 用户风格画像

`server.load_profile(user_id)` 读取 `profiles/<user_id>.json`，缺失时回退 `profiles/user_default.json`。画像提供风格标签、语气预设、默认 `max_new_tokens` / `temperature` 等，在生成时被注入 prompt，让输出"像本人写的"。（这部分逻辑在重构时从原 `meta_server` 并入 `server.py`。）

### 模型加载与并发

模型**懒加载**——服务启动时不加载，首次 `generate` / `analyze` 请求才加载，进程生命周期内只加载一次：

```
首次 request
   │
   ▼ get_model("personal")
   │
   ▼ device_manager.pick("personal")
   │   AUTO:NPU,GPU,CPU 或 AUTO:CPU（无 OpenVINO 加速时）
   ▼
   OVModelForCausalLM.from_pretrained("models/qwen3_personal_int4", device="AUTO:...")
   │
   ▼ 模型读入内存（首次约 10–30 秒），后续请求复用
```

加载采用 **加锁 + 失败缓存** 双重保护：

- 加锁：无论成功或异常都释放加载锁，避免后续请求因锁残留而永久阻塞；
- 失败缓存：加载出错（OOM、文件损坏等）后记录时间戳，60 秒内重试直接快速失败，避免每次请求都卡几十秒。

并发：OpenVINO 的 `InferRequest` 不支持并发调用，每个模型（`personal` / `analysis`）维护独立推理锁。模型正在推理时，新请求不阻塞，而是返回 `{"ok": false, "error": "Model 'personal' is busy, please retry later."}`，稍后重试即可。

### RAG 检索增强

`--rag` 是**请求级开关**，与索引数据是否存在无关：

| 情况 | `--rag` 开 | `--rag` 关 |
|---|---|---|
| 有索引文件 | 检索相似片段 → 注入 prompt | 不使用 RAG |
| 无索引文件 | 静默降级为普通生成（不报错） | 不使用 RAG |

- `data/rag_index/` 有内容 = "已有参考文章索引库"（数据）
- 建索引方式见 `rag_engine.py`（从 JSONL 构建 bge 嵌入索引）

### 设备调度

`device_manager.pick(model_type)` 按模型规格与本机可用加速器（NPU / GPU / CPU）和系统内存拼出最优 `AUTO:...` 链；内存不足时自动降级（如 ≥16GB 含 NPU，12–16GB 去掉 NPU，<12GB 仅 CPU），无 OpenVINO 时整体回退 CPU。用户零配置。

---

## 文档元信息

| 字段 | 值 |
|---|---|
| 项目名称 | `local-style-writer` |
| Skill 类型 | Model-as-Skill（Qoder 被动调用） |
| 运行模式 | 命名管道单服务（`\\.\pipe\local-style-writer`，无 HTTP / 无端口） |
| 核心模型 | Qwen3-8B（QLoRA 个性化）、Qwen2.5-0.5B（风格分析）、bge-small-zh（RAG） |
| 推理框架 | OpenVINO AUTO（INT4 + INT8，CPU/GPU/NPU 自动异构） |
| 隐私模型 | 完全本地，0 网络外联 |
| 唯一对外接口 | `scripts/run.ps1` |
| 内部脚本 | `scripts/client.py` / `scripts/server.py` / `scripts/install-env.ps1` |
| 文档版本 | v3.0（重构为命名管道单服务） |
| 最后更新 | 2026-07-17 |
