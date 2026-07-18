#!/bin/bash
set -e

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$MODEL_DIR"

echo "[1/3] 下载 Qwen3-8B 个人化 INT4 (5.2GB)..."
modelscope download \
  --model ruyuehao/qwen3-personal-writting-style-int4 \
  --local_dir "$MODEL_DIR/qwen3_personal_int4"

echo "[2/3] 下载 Qwen2.5-0.5B 风格分析 INT4 (308MB)..."
modelscope download \
  --model ruyuehao/qwen2.5-0.5b-style-int4 \
  --local_dir "$MODEL_DIR/qwen2.5_int4"

echo "[3/3] 下载 bge-small-zh 嵌入 INT8 (95MB)..."
modelscope download \
  --model ruyuehao/bge-small-zh-v1.5-int8-ov \
  --local_dir "$MODEL_DIR/bge_int8"

echo ""
echo "✅ 所有模型下载完成"
echo "   总占用: ~5.6GB"
du -sh "$MODEL_DIR"/*/ 2>/dev/null || ls -d "$MODEL_DIR"/*/
