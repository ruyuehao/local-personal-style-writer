# personal-style-writer 二次微调指南

> **personal-style-writer** — 模型二次微调操作指南  
> 本文档指导用户从原始 .txt 文章出发，完成数据准备、云端环境配置、QLoRA 微调、模型导出到 INT4 的完整流程。

---

## 目录

- [Phase 0：数据准备（从 .txt 到训练数据）](#phase-0数据准备从-txt-到训练数据)
- [Phase 1：云端环境配置与微调](#phase-1云端环境配置与微调魔搭-notebook-gpu)
- [Phase 2：端侧模型准备](#phase-2端侧模型准备本地-ai-pc)

---

## Phase 0：数据准备（从 .txt 到训练数据）

**目标**：将原始 `.txt` 文章转换为符合微调格式的 JSONL 文件。

### 0.1 数据集 A 格式（QLoRA 个性化生成微调，Qwen3-8B）

**Alpaca 对象格式，每行一条 JSON：**

```json
{
  "instruction": "请根据以下主题和要点，生成符合{风格}风格的原创内容",
  "input": {
    "topic": "汉语功能块自动标注方法",
    "key_points": ["汉语组块分析", "条件随机场", "功能块标注"],
    "target_length": "300-500字",
    "tone_preset": "academic",
    "preserve_terms": ["条件随机场", "功能块"]
  },
  "output": "汉语组块分析是将汉语句子中的词首先组合成基本块..."
}
```

**字段约束：**

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `instruction` | string | ✅ | 固定模板，风格名替换为下表 5 选 1 |
| `input.topic` | string | ✅ | 10-100 字 |
| `input.key_points` | list[string] | ✅ | 1-5 个要点 |
| `input.target_length` | string | ⚙️ 可选 | "100-200字" / "300-500字" / "500-1000字" |
| `input.tone_preset` | string | ✅ | 5 选 1 |
| `input.preserve_terms` | list[string] | ⚙️ 可选 | 0-10 个术语 |
| `output` | string | ✅ | 100-1000 字，覆盖所有 key_points |

**5 种 tone_preset：**

| tone_preset | instruction 中的风格名 | 特征 |
|---|---|---|
| `academic` | 学术论述 | 长句 / 术语引用 / 无 emoji |
| `storytelling` | 故事叙事 | 细节描写 / 画面感 / 比喻 |
| `marketing` | 营销推广 | 感叹号 / emoji / 紧迫感 / 数字 |
| `professional` | 专业职场 | 正式书面 / 数据支撑 / 无 emoji |
| `casual_sarcastic` | 吐槽调侃 | 短句 / 反问 / 网络梗 / emoji 密集 |

### 0.2 数据集 B 格式（风格一致性分析微调，Qwen2.5-0.5B）

**回归格式，每行一条 JSON（即 `data/dataset_b_regression_train.jsonl` 的格式）：**

```json
{
  "text": "她说：“不喜欢，甚至可以说是讨厌。”\n我问：“那你真正喜欢的是什么？”\n她说：“不知道。”\n这，才是问题的症结所在——这个女生的【自我意识】，还没有真正的觉醒。",
  "tone_preset": "casual_sarcastic",
  "style_score": 57,
  "style_analysis": {
    "perplexity": 73.2,
    "length_variance": 100,
    "vocabulary_match": 0.56
  }
}
```

**字段约束（已与 `data/dataset_b_regression_train.jsonl` 实测 1786 条对齐）：**

| 字段 | 类型 | 必填 | 约束 / 实测范围 |
|---|---|---|---|
| `text` | string | ✅ | 81–499 字（实测均值≈420，多数 350–500 字） |
| `tone_preset` | string | ✅ | 5 选 1：academic / storytelling / marketing / professional / casual_sarcastic（实测各 351–363 条，分布均衡） |
| `style_score` | int | ✅ | 0–100（实测 20–95，均值≈60；覆盖 20–59 / 60–69 / 70–79 / 80–89 / 90–95 各区间） |
| `style_analysis.perplexity` | float | ✅ | 10–200（实测 39.3–107.7，均值≈72） |
| `style_analysis.length_variance` | float | ✅ | 0–100（实测 6.7–100；**95.7% 样本=100**，区分度低，作弱信号） |
| `style_analysis.vocabulary_match` | float | ✅ | 0–1（实测 0.18–0.95，均值≈0.56） |

**评分标准：** 90-100 高度一致 / 80-89 良好 / 70-79 一般 / 60-69 较弱 / <60 不一致

> **训练实际喂入的格式**：原始 jsonl 不会直接送入 SFT，而是经 `training/convert_b_to_alpaca.py` 转为 Alpaca 格式后再训练。转换规则：
> - `instruction` 固定为 `请分析以下文本的风格特征，并给出风格一致性评分。`
> - `input` = 原 `text`
> - `output` = JSON 字符串 `{"tone_preset":..., "style_score":..., "perplexity":..., "length_variance":..., "vocabulary_match":...}`（字段与上方表格严格一致）
>
> 因此**务必保证原始 jsonl 的字段名与上方表格完全一致**，否则转换脚本会抛 KeyError。推理端 `analyze_style` 的 prompt 也严格按此 `instruction` 模板对齐（见 `scripts/qoder_inference.py`）。

### 0.3 数据量要求

| 数据集 | 用途 | 最低条数 | 推荐条数 |
|---|---|---|---|
| 数据集 A（Alpaca JSONL） | QLoRA 个性化生成微调（Qwen3-8B） | 150（每种风格 30） | 250（每种风格 50） |
| 数据集 B（回归 JSONL） | 风格一致性分析微调（Qwen2.5-0.5B） | 500（每种风格 100） | 1000（每种风格 200） |

**分布建议：** 每种风格包含 5+ 不同主题、3+ 种长度；数据集 B 含高分/中分/低分三个区间。

### 0.4 输出文件命名

```
data/
├── dataset_a_alpaca_train.jsonl       # 数据集 A 训练集（~90%）
├── dataset_a_alpaca_val.jsonl         # 数据集 A 验证集（~10%）
├── dataset_b_regression_train.jsonl   # 数据集 B 训练集（~90%）
├── dataset_b_regression_val.jsonl     # 数据集 B 验证集（~10%）
```

### 0.5 数据质量验证

```bash
python training/validate_data.py dataset_a data/dataset_a_alpaca_train.jsonl
python training/validate_data.py dataset_a data/dataset_a_alpaca_val.jsonl
python training/validate_data.py dataset_b data/dataset_b_regression_train.jsonl
python training/validate_data.py dataset_b data/dataset_b_regression_val.jsonl
```

### 0.6 从 .txt 转换说明

用户自行编写转换脚本。输入为 `.txt` 文章目录，输出为上述 4 个 jsonl 文件。关键步骤：

1. **分段**：每段 100-500 字作为一个样本
2. **标风格**：每段标注一个 `tone_preset`（5 选 1）
3. **组装数据集 A**：写入 Alpaca 对象格式，`output` 字段为该段原文
4. **组装数据集 B**：计算或人工评分填入 `style_score`/`style_analysis`
5. **切分**：随机 9:1 划分为训练集和验证集
6. **验证**：运行 `validate_data.py` 确认格式正确

> 详细的风格画像规格、FAQ 见 [DATA_SPECS.md](./DATA_SPECS.md)。

---

## Phase 1：云端环境配置与微调（魔搭 Notebook GPU）

**目标**：生成个性化的 LoRA 适配器。

### 1.1 环境配置（从零开始）

> 以下步骤针对 **纯净环境（无 Python、无 Git、无 CUDA）** 的魔搭 Notebook（Ubuntu）或新装 GPU 服务器。  
> 已有环境的用户可跳过对应步骤。

#### 1.1.1 前置依赖安装

```bash
# ---------- 0. 配置国内镜像源（魔搭 Notebook / 国内服务器加速）----------
# pip 全局镜像（阿里云，后续所有 pip install 自动走国内）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# apt 换阿里源（如 apt install 慢）
sudo sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
sudo sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
sudo apt update

# ---------- 1. 系统更新 ----------
sudo apt upgrade -y

# ---------- 2. 安装 Python 3.10+ ----------
python3 --version  # 需要 >= 3.10
# 如未安装或版本过低：
sudo apt install -y python3 python3-pip python3-venv

# ---------- 3. 安装 Git ----------
git --version || sudo apt install -y git

# ---------- 4. 安装 NVIDIA 驱动 + CUDA（GPU 训练必需）----------
nvidia-smi  # 确认驱动和 CUDA 版本
# 魔搭 Notebook 预装 CUDA，跳过安装；如纯净环境按官方指引安装对应版本
```

#### 1.1.2 创建虚拟环境

```bash
# 统一在 local-style-writer 目录操作（本指南所有命令均为 training/... 相对路径）
cd /path/to/production-ai-skill/local-style-writer

# 创建虚拟环境（隔离依赖，避免系统污染）
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows（端侧推理用）

# 验证：确认 python 路径指向 venv
which python  # 应输出 .../venv/bin/python
```

#### 1.1.3 安装核心依赖

```bash
# 升级 pip 本身
pip install --upgrade pip

# ---------- PyTorch（GPU 版，CUDA 12.4，匹配魔搭 Notebook）----------
# 魔搭为阿里云基础设施，直接用阿里云 PyTorch wheel 镜像（不走官方 CDN）
# VPC 内网地址（魔搭 Notebook 自动访问，比公网快 10-100 倍）：
#   http://mirrors.cloud.aliyuncs.com/pytorch-wheels/cu124/
# 公网备用（阿里云镜像站）：
#   https://mirrors.aliyun.com/pytorch-wheels/cu124/
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  -f http://mirrors.cloud.aliyuncs.com/pytorch-wheels/cu124/

# 验证 GPU 可用
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU数:', torch.cuda.device_count())"
# 应输出: CUDA可用: True, GPU数: ≥1

# ---------- LLaMA-Factory（QLoRA 微调框架，国内镜像加速）----------
# 方式 A：GitHub 官方（锁定稳定 tag 可复现；训练实测为 0.9.6.dev0 main 快照，见下方验证记录）
# 国内访问慢可用镜像代理前缀：https://github.91chi.fun/https://github.com/hiyouga/LLaMA-Factory.git
git clone -b v0.9.5 --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# ---------- OpenVINO + NNCF（模型导出与量化）----------
pip install openvino optimum[openvino] nncf

```

> ⚠️ **已验证版本兼容性**（2026-07-13 实际运行）：
> - `transformers==5.0.0` + `LLaMA-Factory 0.9.6.dev0`（editable install）
> - `openvino==2026.2.1` + `optimum-intel==2.0.0` + `nncf==3.2.0`
> - **关键发现**：`optimum-cli` 命令为 `optimum-cli`（非 `python -m optimum.cli`）
> - 导出时**必须**加 `--task text-generation`（否则无法从本地目录推断任务类型）
> - 完整可复现依赖清单（含 torch cu124 / LLaMA-Factory tag / OpenVINO 栈精确版本）见 `training/requirements-training.txt`

#### 1.1.4 环境验证

```bash
# 逐项确认关键组件可用
echo "=== 环境验证 ==="
echo "Python: $(python --version)"
echo "Git:    $(git --version)"
echo "CUDA:   $(nvcc --version 2>&1 | grep 'release' | awk '{print $6}')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "LLaMA-Factory: $(python -c 'import llamafactory; print("OK")' 2>&1 || echo '请检查安装')"
echo "OpenVINO: $(python -c 'import openvino; print(openvino.__version__)')"
echo "NNCF:    $(python -c 'import nncf; print(nncf.__version__)')"
echo "=== 验证完成 ==="
```

### 1.2 模型与数据准备

```bash
# ---------- 下载微调基座模型（使用 modelscope，魔搭内网加速）----------

# Qwen3-8B（QLoRA 个性化生成微调基座，~GB）
modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B

# Qwen2.5-0.5B（风格一致性分析微调基座，~1GB）
modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ./models/Qwen2.5-0.5B
```

> 数据准备（格式定义 / 数据量要求 / 文件命名 / 验证）详见 [Phase 0](#phase-0数据准备从-txt-到训练数据)。

### 1.3 执行 QLoRA 微调

> ⚠️ **关键经验**：推荐使用 **YAML 配置文件**（`llamafactory-cli train xxx.yaml`）而非 CLI 参数列表。
>
> **原因**：LLaMA-Factory 0.9.6 的 CLI 模式`--save_only_model True` 等参数可能不生效，导致训练完成后输出目录仅有 JSON 日志、无模型权重。YAML 配置方式经过验证可正确保存。

数据集 A 和 B 均注册在 `data/dataset_info.json` 中。

**训练 Qwen3-8B（个性化生成）：**

```bash
bash training/train_qwen3_personal.sh
```

该脚本自动完成：`training/convert_a_to_alpaca.py` 数据集转换 → `llamafactory-cli train` 启动训练。

**训练 Qwen2.5-0.5B（风格分析）：**

```bash
bash training/train_qwen2.5_style.sh
```

该脚本自动完成：`training/convert_b_to_alpaca.py` 数据集转换 → `llamafactory-cli train` 启动训练。

**并行训练（可选）：**

```bash
bash training/train_all.sh
```

后台同时启动两个训练任务，日志分别输出至 `logs/train_qwen3.log` 和 `logs/train_qwen2.5.log`。

**产物验证：**

```bash
ls -la saves/qwen3_personal_lora/
# 应包含 adapter_config.json + adapter_model.safetensors
ls -la saves/qwen2.5_0.5b_style_lora/
# 同上
```

> **如果仍然为空**，回退到 `training/end_to_end.py`（纯 HF Trainer + peft，无需 LLaMA-Factory）：
> ```bash
> python training/end_to_end.py \
>   --model ./models/Qwen2.5-0.5B \
>   --train_file data/dataset_b_regression_train_alpaca.jsonl \
>   --val_file data/dataset_b_regression_val_alpaca.jsonl \
>   --output_dir saves/qwen2.5_0.5b_style_lora \
>   --lora_rank 8
> ```

**Qwen3 特殊注意事项**：必须设置 `template: qwen3_nothink`（YAML 中已内置），否则训练时提示"using reasoning template"并可能影响生成质量。

### 1.4 导出为 OpenVINO IR 格式（合并 LoRA + 重新量化）

> 将训好的 LoRA 合并进基座并量化，供端侧使用。

> #### ⚠️ 关键原则：基座选择
>
> **导出时的基座永远是原始 `Qwen3-8B` (FP16)，不是合并后的 INT4 模型。**
>
> | 选项 | 是否可行 | 原因 |
> |---|---|---|
> | ✅ `Qwen3-8B` (FP16) + 本次 LoRA | **推荐** | 干净基座，量化一次 |
> | ❌ 上一次合并的 INT4 模型 | **不可行** | INT4 权重无梯度，无法继续训练 |
> | ❌ 上一次合并的 FP16 模型 | **不推荐** | 两轮量化精度损失 + 风格漂移累积 |
>
> **结论：每次重训都从原始 FP16 基座开始，保证干净起点 + 可回滚 + 可对比。**

#### 1.4.1 导出方式选择

> **注意**：当前 `optimum-cli export openvino --adapter` 标志在 `optimum-intel==2.0.0` 中**不可用**（返回 unrecognized arguments）。
> 
> 替代方案：使用 `training/merge_and_export.py`（先 peft 合并，再调用 optimum-cli 量化）。

**方式 A：使用 `training/export_lora.sh`（自动化，推荐）**

```bash
bash training/export_lora.sh
```

该脚本自动完成：备份 LoRA → peft 合并 LoRA + 基座 → 保存合并后的 FP16 模型 → optimum-cli 导出 INT4。

**方式 B：分步执行：**

```bash
# 1) 备份 LoRA
cp -r saves/qwen3_personal_lora saves/qwen3_personal_lora_v1_$(date +%Y%m%d_%H%M%S)

# 2) peft 合并 + optimum-cli 导出（使用 merge_and_export.py）
python training/merge_and_export.py \
  --base_model ./models/Qwen3-8B \
  --adapter ./saves/qwen3_personal_lora \
  --output ./models/qwen3_8b_personal_v1_int4 \
  --weight-format int4

# 3) Qwen2.5 同理
python training/merge_and_export.py \
  --base_model ./models/Qwen2.5-0.5B \
  --adapter ./saves/qwen2.5_0.5b_style_lora \
  --output ./models/qwen2.5_int4 \
  --weight-format int4
```

> **关键参数**：必须加 `--task text-generation`（在 `merge_and_export.py` 中已内置），否则 optimum-cli 无法从本地目录推断任务类型，报错 `Cannot infer the task from a local directory yet`。

#### 1.4.2 产物清单

| 产物 | 体积 | 用途 |
|---|---|---|
| LoRA 适配器 (`adapter_model.safetensors`) | Qwen3: ~29 MB / Qwen2.5: ~2.1 MB | 可重新合并或复用 |
| INT4 OpenVINO 模型 (`openvino_model.bin`) | Qwen3: **4.6 GB** / Qwen2.5: **308 MB** | 端侧推理权重 |
| `openvino_model.xml` + tokenizer `*.bin/*.xml` | ~20 MB | 模型结构 + 分词器 |
| 训练日志 + 绘图 | < 100 KB | 监控用 |

**端侧部署时使用整个 `_int4/` 目录**，不是单独的 `.bin` 文件。

#### 1.4.3 Embedding 模型 OpenVINO 转换（bge-small-zh-1.5）

> bge-small-zh-v1.5 是 L2 RAG 检索的可选组件，用于将用户历史文章向量化后检索相似片段作为 Few-shot。
> ModelScope 上仅有 FP32 PyTorch 版本，需自行转换为 OpenVINO IR + INT8 供端侧使用。

**方式 A：一键转换（推荐）**

```bash
bash training/convert_bge_to_ov.sh
```

该脚本自动完成：下载 FP32 原始模型 → optimum-cli 导出 INT8 → 输出到 `models/bge_int8/`。

**方式 B：手动分步执行**

```bash
# 1) 下载原始 FP32 模型（如尚未下载）
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir ./models/bge_original

# 2) 转换为 OpenVINO IR + INT8 量化
optimum-cli export openvino \
  --model ./models/bge_original \
  --task feature-extraction \
  --weight-format int8 \
  ./models/bge_int8

# 3) 验证产物
ls -la ./models/bge_int8/
# 应包含 openvino_model.xml + openvino_model.bin
```

> **⚠️ 关键区别**：与 LLM 导出（`text-generation`）不同，embedding 模型的任务类型为 `feature-extraction`。此外 embedding 模型用 INT8 而非 INT4，以保证检索精度。

**产物清单**：

| 产物 | 体积 | 位置 |
|---|---|---|
| FP32 PyTorch 原始模型 | ~192 MB | `models/bge_original/` |
| INT8 OpenVINO IR 模型 | ~95 MB | `models/bge_int8/` |

> **清理建议**：确认 `bge_int8/` 可正常工作后，可删除 `bge_original/` 释放空间。

---

### 1.5 个性化模型重训机制（持续学习）

> 用户持续使用 Skill 写作时，本地会累积新文章。**为保持风格学习的时效性**，需要重训机制。

#### 1.5.1 重训触发条件（满足任一即触发）

| 触发条件 | 阈值 | 设计理由 |
|---|---|---|
| 累积新文章数 | **≥ 50 篇** | 数据量足够学出新风格差异 |
| 距离上次训练 | **≥ 90 天** | 防止长期不更新导致"风格失锚" |
| 用户主动触发 | **任意时刻** | 满足高阶用户控制感 |

**默认策略：宁缺毋滥**——不到阈值不重训，避免无效训练浪费算力。

#### 1.5.2 重训流程

```bash
# === Step 1: 累积全部历史数据 ===
DATA_DIR="./personal_style_data"
mkdir -p ${DATA_DIR}/v_$(date +%Y%m%d)

# 把最新一批文章归入新版本目录
cp ./new_articles/*.jsonl ${DATA_DIR}/v_$(date +%Y%m%d)/

# 合并所有历史版本，覆盖 data/ 下的训练集文件
cat ${DATA_DIR}/v_*/*.jsonl > ./data/dataset_a_alpaca_train.jsonl
echo "总样本数: $(wc -l < ./data/dataset_a_alpaca_train.jsonl)"

# === Step 2: 重新训练 LoRA（基座仍然是原始 FP16 Qwen3-8B）===
#     使用 YAML 配置文件（修改 output_dir 指向新版本目录）
sed "s|output_dir: ./saves/qwen3_personal_lora|output_dir: ./saves/qwen3_personal_lora_v2_$(date +%Y%m%d)|" \
  training/qwen3_lora_sft.yaml > training/qwen3_lora_sft_v2.yaml
llamafactory-cli train training/qwen3_lora_sft_v2.yaml

# === Step 3: 合并 + 重新量化（基座仍是原始 FP16）===
NEW_LORA="./saves/qwen3_personal_lora_v2_$(date +%Y%m%d)"
NEW_MERGED="./models/qwen3_8b_personal_v2_$(date +%Y%m%d)_int4"

python training/merge_and_export.py \
  --base_model ./models/Qwen3-8B \    # ⚠️ 永远是原始 FP16
  --adapter ${NEW_LORA} \
  --output ${NEW_MERGED} \
  --weight-format int4

# === Step 4: 更新元数据 ===
cat > ./saves/.last_train_meta.json <<EOF
{
  "trained_at": "$(date -Iseconds)",
  "lora_dir": "${NEW_LORA}",
  "merged_dir": "${NEW_MERGED}",
  "base_model": "Qwen/Qwen3-8B",
  "data_samples": $(wc -l < ./data/dataset_a_alpaca_train.jsonl),
  "lora_rank": 16,
  "previous_version": "v1"
}
EOF

# === Step 5: 端侧拉取新版本 ===
# 客户端检测到新 merged_dir 后自动下载并切换
```

#### 1.5.3 版本管理策略

**原则：永不覆盖，永不删除。**

```
saves/                                    # 云端
├── qwen3_personal_lora_v1_20260710/      # 第 1 版
├── qwen3_personal_lora_v2_20260810/      # 第 2 版（50 篇新文后）
├── qwen3_personal_lora_v3_20260915/      # 第 3 版（用户主动触发）
└── .last_train_meta.json                 # 指向当前最新版本

models/                                   # 云端 → 端侧
├── qwen3_8b_personal_v1_int4/            # 5.2 GB
├── qwen3_8b_personal_v2_int4/            # 5.2 GB
└── qwen3_8b_personal_v3_int4/            # 5.2 GB
```

**为什么不删旧版**：
- ✅ 用户可回滚到任何历史版本
- ✅ A/B 测试对比效果
- ✅ 新版出错时快速回退
- ✅ 数据完整性可追溯

#### 1.5.4 灾难性陷阱清单

| ❌ 不要做 | 后果 |
|---|---|
| 用合并后的 INT4 模型当基座 | 无法训练，破坏端侧架构 |
| 用合并后的 FP16 模型当基座 | 两轮量化精度损失 + 风格漂移 |
| 在旧 LoRA 上"接着练" | 灾难性遗忘，新数据覆盖旧风格 |
| 每次用户写文章都重训 | 算力浪费，用户无感 |
| 删掉旧版 LoRA | 出错无法回滚 |
| 删掉旧数据 | 无法复现 / 无法重训 |

#### 1.5.5 进阶（团队版，待 OpenVINO 动态 LoRA 支持后启用）

| 特性 | 说明 | 前置条件 |
|---|---|---|
| 多人共用基础模型 | 一份 5.2GB INT4，多人共享 | OpenVINO 2026.3+ |
| 一人一 LoRA 小文件 | 每个用户 50-200MB | ov::genai::Adapter |
| 动态切换 | 推理时按需加载不同 LoRA | LoRA 训练时按 1.4.1 流程 |
| A/B 测试 | 同时加载 2 个 LoRA，对比效果 | ov::genai::AdapterConfig |

---

## Phase 2：端侧模型准备（本地 AI PC）

**目标**：将云端训好的 INT4 模型拷贝到本地。

### 2.1 模型下载

将云端训好的 INT4 模型目录（`models/qwen3_8b_personal_*_int4/` 和 `models/qwen2.5_int4/`）打包下载到本地 `D:\models\` 下。可用 tar + scp 或直接从 ModelScope 下载已上传的模型。

### 2.2 端侧目录结构

```
D:\models\
├── qwen3_personal_int4\      # 4.6 GB — Qwen3-8B 个人风格 INT4
│   ├── openvino_model.bin
│   ├── openvino_model.xml
│   ├── openvino_tokenizer.bin
│   ├── openvino_detokenizer.bin
│   └── *.json
└── qwen2.5_int4\       # 308 MB — Qwen2.5-0.5B 风格分析 INT4
    └── ...
```

### 2.3 端侧依赖

```bash
pip install optimum[openvino]
```

## 后续步骤

模型导出完成后，端侧部署与运行请参考 `README.md`。

---
