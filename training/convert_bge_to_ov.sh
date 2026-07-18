#!/bin/bash
# training/convert_bge_to_ov.sh - bge-small-zh → OpenVINO IR + INT8 转换
set -e

MODEL_DIR="./models"
ORIGINAL_DIR="${MODEL_DIR}/bge_original"
OUTPUT_DIR="${MODEL_DIR}/bge_int8"

echo "=========================================="
echo "  bge-small-zh → OpenVINO IR + INT8"
echo "=========================================="

# 如果 INT8 产物已存在，跳过
if [ -f "${OUTPUT_DIR}/openvino_model.xml" ] && [ -f "${OUTPUT_DIR}/openvino_model.bin" ]; then
    echo "✅ INT8 模型已存在: ${OUTPUT_DIR}"
    echo "   如需重新转换，请删除 ${OUTPUT_DIR} 后重试"
    exit 0
fi

# 检查原始 FP32 是否已下载
if [ ! -f "${ORIGINAL_DIR}/model.safetensors" ] && [ ! -f "${ORIGINAL_DIR}/pytorch_model.bin" ]; then
    echo "[1/2] 下载 bge-small-zh FP32 原始模型..."
    mkdir -p ${ORIGINAL_DIR}
    modelscope download \
        --model BAAI/bge-small-zh-v1.5 \
        --local_dir ${ORIGINAL_DIR}
else
    echo "[1/2] 原始模型已存在: ${ORIGINAL_DIR}"
fi

echo "[2/2] 转换为 OpenVINO IR + INT8 量化..."
optimum-cli export openvino \
    --model ${ORIGINAL_DIR} \
    --task feature-extraction \
    --weight-format int8 \
    ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "  ✅ 转换完成"
echo "=========================================="
echo "  FP32 原始:  ${ORIGINAL_DIR}  (~192 MB)"
echo "  INT8 产物:  ${OUTPUT_DIR}    (~95 MB)"
echo ""
echo "  端侧 RAG 请加载: ${OUTPUT_DIR}"
echo ""
echo "  清理建议：确认 ${OUTPUT_DIR} 正常后，可删除 ${ORIGINAL_DIR} 释放空间"
echo "    rm -rf ${ORIGINAL_DIR}"
echo "=========================================="
