import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_export(base_model_path, adapter_path, output_path, weight_format="int4", keep_merged=False):
    base_path = Path(base_model_path)
    out_path = Path(output_path)
    merged_path = out_path.parent / f"{out_path.name}_merged_fp16"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/4] Loading and merging LoRA: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print(f"[3/4] Saving merged FP16 model to {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_path)

    print(f"[4/4] Exporting to OpenVINO IR ({weight_format}): {out_path}")
    result = subprocess.run(
        [
            "optimum-cli", "export", "openvino",
            "--model", str(merged_path),
            "--task", "text-generation",
            "--weight-format", weight_format,
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        return False

    if not keep_merged:
        shutil.rmtree(merged_path, ignore_errors=True)

    print(f"\nDone: {out_path}")
    for f in sorted(out_path.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            unit = "MB" if size > 1024 * 1024 else "KB"
            val = size / (1024 * 1024) if unit == "MB" else size / 1024
            print(f"  {f.relative_to(out_path)}: {val:.1f} {unit}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and export to OpenVINO IR")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weight-format", default="int4", choices=["int4", "int8", "fp16"])
    parser.add_argument("--keep_merged", action="store_true")
    args = parser.parse_args()

    success = merge_and_export(
        args.base_model, args.adapter, args.output,
        args.weight_format, args.keep_merged,
    )
    sys.exit(0 if success else 1)
