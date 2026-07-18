"""直接测试 analyze_style（绕过命名管道 server），用于隔离模型层与通信层问题。"""
import sys, os, time

sys.path.insert(0, r"D:\ai-skill\production-ai-skill\local-style-writer\scripts")
from qoder_inference import load_model, analyze_style

MODEL = os.environ.get(
    "ANALYSIS_MODEL_PATH",
    r"D:\ai-skill\production-ai-skill\local-style-writer\models\qwen2.5_int4",
)

print("=== [1] 加载 analysis 模型 ===", flush=True)
t0 = time.time()
model, tok = load_model(MODEL, model_type="analysis", use_cache=False)
print(f"    加载完成：{time.time()-t0:.1f}s", flush=True)

print("=== [2] 调用 analyze_style ===", flush=True)
t0 = time.time()
res = analyze_style(model, tok, "今天天气不错，我去公园散步，看到很多花开了，心情很放松。")
print(f"    分析完成：{time.time()-t0:.1f}s", flush=True)
print("=== [3] 结果 ===", flush=True)
print(res)
