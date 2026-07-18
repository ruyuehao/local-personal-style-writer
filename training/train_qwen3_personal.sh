#!/bin/bash
set -e

echo "=========================================="
echo "  Training Qwen3-8B Personal Style LoRA"
echo "=========================================="

echo "[1/2] Converting dataset A to string format..."
python training/convert_a_to_alpaca.py \
  --input data/dataset_a_alpaca_train.jsonl \
  --output data/dataset_a_alpaca_train_str.jsonl
python training/convert_a_to_alpaca.py \
  --input data/dataset_a_alpaca_val.jsonl \
  --output data/dataset_a_alpaca_val_str.jsonl

echo "[2/2] Training with YAML config..."
llamafactory-cli train training/qwen3_lora_sft.yaml
