import sys, types

# 强制桩掉重型依赖（沙箱无 OpenVINO/模型）
for name in ["optimum.intel", "optimum", "transformers", "psutil",
             "device_manager", "rag_engine"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["optimum.intel"].OVModelForCausalLM = object
sys.modules["transformers"].AutoTokenizer = object
sys.modules["transformers"].AutoModelForCausalLM = object
sys.modules["device_manager"].device_manager = object()
sys.modules["rag_engine"].engine = object()

import importlib.util
spec = importlib.util.spec_from_file_location(
    "qoder_inference_desc",
    r"D:\ai-skill\production-ai-skill\local-style-writer\scripts\qoder_inference.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 用例1：用户之前的真实模型输出（基座模型写的散文）
raw = ("这段文本的风格特征比较符合新闻报道的风格，尤其是新闻报道的报道格式和"
      "用词，这表明它被设计为新闻报道的文本。因此，这个文本的风格特征评分较高，达到90分。")
print("[1] 模型散文清洗:")
print("   ", repr(m._clean_description(raw)))

# 用例2：模型只吐了数字/JSON（无描述文本）
print("[2] 纯数字回复清洗(应为空):")
print("   ", repr(m._clean_description('{"style_score":82}')))

# 用例3：规则兜底（模型无描述时）
fields = {"style_score": 100, "perplexity": 0.54, "length_variance": 0.0,
          "vocabulary_match": 1.0}
print("[3] 规则兜底描述:")
print("   ", m._rule_based_description(fields))

# 用例4：_build_description 组合（有散文优先用散文）
print("[4] _build_description(有散文):")
print("   ", m._build_description(raw, fields))
print("[5] _build_description(无散文->兜底):")
print("   ", m._build_description('{"style_score":82}', fields))
print("\nALL OK")
