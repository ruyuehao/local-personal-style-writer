#!/bin/bash
set -e

echo "=========================================="
echo "  Export LoRA → OpenVINO IR + INT4"
echo "=========================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ========== 1. Qwen3-8B Personal Style ==========
echo ""
echo "--- Qwen3-8B Personal Style ---"

LORA_TAG="v1_${TIMESTAMP}"
LORA_DIR="./saves/qwen3_personal_lora"
LORA_BAK="./saves/qwen3_personal_lora_${LORA_TAG}"
MERGED_DIR="./models/qwen3_8b_personal_${LORA_TAG}_int4"

echo "[1/4] Backing up LoRA → ${LORA_BAK}"
cp -r ${LORA_DIR} ${LORA_BAK}

echo "[2/4] Merging LoRA + exporting INT4 → ${MERGED_DIR}"
python training/merge_and_export.py \
  --base_model ./models/Qwen3-8B \
  --adapter ${LORA_DIR} \
  --output ${MERGED_DIR} \
  --weight-format int4

echo "[3/4] Writing training metadata"
cat > ./saves/.last_train_meta.json <<EOF
{
  "trained_at": "$(date -Iseconds)",
  "lora_dir": "${LORA_BAK}",
  "merged_dir": "${MERGED_DIR}",
  "base_model": "Qwen/Qwen3-8B",
  "data_samples": $(wc -l < ./data/dataset_a_alpaca_train_str.jsonl),
  "lora_rank": 16,
  "model": "qwen3_personal"
}
EOF

echo "[4/4] Qwen3-8B export complete"
ls -lh ${MERGED_DIR}/openvino_model.bin 2>/dev/null || true

# ========== 2. Qwen2.5-0.5B Style Analysis ==========
echo ""
echo "--- Qwen2.5-0.5B Style Analysis ---"

STYLE_MERGED="./models/qwen2.5_0.5b_style_int4"

echo "[1/2] Merging LoRA + exporting INT4 → ${STYLE_MERGED}"
python training/merge_and_export.py \
  --base_model ./models/Qwen2.5-0.5B \
  --adapter ./saves/qwen2.5_0.5b_style_lora \
  --output ${STYLE_MERGED} \
  --weight-format int4

echo "[2/2] Qwen2.5-0.5B export complete"
ls -lh ${STYLE_MERGED}/openvino_model.bin 2>/dev/null || true

# ========== Summary ==========
echo ""
echo "=========================================="
echo "  Export Summary"
echo "=========================================="
echo ""
echo "Qwen3-8B Personal:"
echo "  LoRA backup:  ${LORA_BAK}"
echo "  INT4 model:   ${MERGED_DIR}"
echo ""
echo "Qwen2.5-0.5B Style:"
echo "  INT4 model:   ${STYLE_MERGED}"
echo ""
echo "Next: upload INT4 models to end device"
