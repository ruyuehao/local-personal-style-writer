#!/bin/bash
set -e

echo "=========================================="
echo "  Training Qwen2.5-0.5B Style Analysis"
echo "=========================================="

echo "[1/2] Converting dataset B to Alpaca format..."
python training/convert_b_to_alpaca.py \
  --input data/dataset_b_regression_train.jsonl \
  --output data/dataset_b_regression_train_alpaca.jsonl
python training/convert_b_to_alpaca.py \
  --input data/dataset_b_regression_val.jsonl \
  --output data/dataset_b_regression_val_alpaca.jsonl

echo "[2/2] Training with YAML config..."
llamafactory-cli train training/qwen2.5_lora_sft.yaml
