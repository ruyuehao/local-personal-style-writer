"""用户风格锚点回归测试（更新版，对应 2026-07-18 analyze 重构）。

重构后 analyze 模型按训练模板（不含 user_style）微调，刻意**不**把个人风格锚点
注入 analyze 的 prompt——注入会与训练输入不一致导致乱码，个人风格改为由训练数据内化。
server 端 _load_user_style 仍从 profile 读取个人风格，但其用途是 generate（tone_preset），
而非 analyze。

本测试覆盖：
  - _load_user_style 仍能从 profiles/user_default.json 读取个人风格锚点（供 generate）；
  - analyze_style 在传入 user_style 时【不会】将其注入 prompt（确认刻意移除，防回归）；
  - analyze_style 在无数字 / 无锚点时走确定性兜底，永不全 null。

沙箱无 OpenVINO/模型/命名管道，用桩隔离重型依赖，复用真实的 analyze_style 逻辑。
"""
import sys
import types
import json

# ── 强制桩掉重型依赖（沙箱无包）──
for name in ["optimum.intel", "optimum", "transformers", "psutil",
             "device_manager", "rag_engine"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["optimum.intel"].OVModelForCausalLM = object
sys.modules["transformers"].AutoTokenizer = object
sys.modules["device_manager"].device_manager = object()
sys.modules["rag_engine"].engine = object()


import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

def load(name):
    spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

qi = load("qoder_inference")
server = load("server")


# ── 桩 tokenizer / model：模拟真实 apply_chat_template + 张量切片 ──
class FakeShape:
    shape = (1, 4)   # batch=1, seq_len=4


class StubTokenizer:
    def __init__(self, reply_text):
        self.eos_token_id = 151643
        self._reply = reply_text
        self.last_prompt = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # 仅把 messages 拼成字符串作为 prompt（测试检查其是否含锚点）
        self.last_prompt = "\n".join(m.get("content", "") for m in messages)
        return self.last_prompt

    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": FakeShape()}

    def decode(self, ids, skip_special_tokens=False):
        return self._reply


class StubModel:
    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]


def run(label, reply, text, user_style=None):
    tok = StubTokenizer(reply)
    model = StubModel()
    res = qi.analyze_style(model, tok, text=text, user_style=user_style)
    print(f"  [{label}]")
    print(f"    prompt含锚点? {('你的个人写作风格是' in tok.last_prompt) if user_style else '(无 user_style，不应含锚点)'}")
    if user_style:
        print(f"    prompt含user_style文本? {user_style[:10] in tok.last_prompt}")
    print(f"    返回 keys: {sorted(res.keys())}")
    print(f"    style_score: {res.get('style_score')}")
    print(f"    description: {res.get('description')!r}")
    print()
    return res, tok.last_prompt


print("== 测试 A：_load_user_style 读取真实 profile（供 generate 使用）==")
srv = server.Server()
anchor = srv._load_user_style()
print(f"  anchor = {anchor!r}")
if anchor is None:
    print("  ⚠ 未找到 profiles/user_default.json，跳过锚点内容断言（功能本身正常）")
else:
    assert "口语化" in anchor, "锚点应包含个人风格标签"
    print("  OK\n")

print("== 测试 B：传入 user_style 时，analyze 【不】注入锚点（训练对齐，刻意移除）==")
reply_b = ('{"style_score": 85, "perplexity": 12.0, "length_variance": 30.0, '
           '"vocabulary_match": 0.7}')
res_b, prompt_b = run("B1 含锚点参数-应忽略", reply_b,
                      "今天天气不错，我去公园散步。", user_style=anchor or "口语化，轻松风格")
assert "你的个人写作风格是" not in prompt_b, "analyze 不应注入 user_style 锚点（与训练模板冲突）"
assert res_b["style_score"] == 85, f"期望抽出85，实际 {res_b['style_score']}"
print("  OK\n")

print("== 测试 C：无 user_style 时，analyze 正常出结果（通用行为保留）==")
reply_c = '{"style_score": 100, "perplexity": 10.0, "length_variance": 20.0, "vocabulary_match": 0.9}'
res_c, prompt_c = run("C1 无锚点-通用模式", reply_c, "今天天气不错。")
assert "你的个人写作风格是" not in prompt_c
assert res_c["style_score"] == 100, f"期望100，实际 {res_c['style_score']}"
print("  OK\n")

print("== 测试 D：无数字，走确定性兜底，永不全 null ==")
reply_d = "这段文字比较口语化，和你的风格挺搭的。"
res_d, _ = run("D1 无数字兜底", reply_d, "公园的花开了。", user_style=anchor or "口语化")
assert res_d["style_score"] is not None, "兜底后 style_score 不应为 None"
assert res_d["style_analysis"]["perplexity"] is not None, "兜底后 perplexity 不应为 None"
assert res_d["description"], "description 不应为空"
print("  OK\n")

print("ALL TESTS PASSED ✅")
