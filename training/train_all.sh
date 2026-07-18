#!/bin/bash
# Run both trainings in parallel
set -e

echo "=========================================="
echo "  Starting all training jobs"
echo "=========================================="
echo ""
echo "  Qwen3-8B Personal Style  →  background (PID log)"
echo "  Qwen2.5-0.5B Style       →  background (PID log)"
echo ""

nohup bash training/train_qwen3_personal.sh > logs/train_qwen3.log 2>&1 &
echo "Qwen3-8B PID: $!"

nohup bash training/train_qwen2.5_style.sh > logs/train_qwen2.5.log 2>&1 &
echo "Qwen2.5-0.5B PID: $!"

echo ""
echo "Monitor:"
echo "  tail -f logs/train_qwen3.log"
echo "  tail -f logs/train_qwen2.5.log"
