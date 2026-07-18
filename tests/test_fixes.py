#!/usr/bin/env python3
# test_fixes.py — 逻辑层回归测试（无需 OpenVINO / 模型 / Intel 硬件即可运行）
#
# 覆盖本次两个修复点：
#   1) analyze_style 路径1 解析：基座模型中文输出也能抽出真实数字，永不全 null
#   2) server 并发修复：慢请求(8B 生成)不再阻塞其它请求(0.5B 分析)
#
# 运行：python local-style-writer/tests/test_fixes.py
#
# 实现说明：通过 stub 掉 optimum / transformers / psutil / device_manager / rag_engine
# 等重型依赖，使本测试在缺少这些包的环境（CI / Linux 沙箱）也能跑纯逻辑；
# 在已安装依赖的 Windows 环境会因 setdefault 不覆盖已存在模块而改用真实模块，同样可跑。
# 并发测试用 TCP 监听器替代 Windows 命名管道（沙箱无命名管道），但复用 server 模块
# 真实的 _serve_connection 与新 main() 同款"每连接一线程"循环。

import sys
import types
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # local-style-writer/
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _stub(name, **attrs):
    # 强制覆盖：本测试进程不依赖真实重型包，统一用桩隔离。
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---- 桩掉重型依赖（缺包环境需要；覆盖式，确保导入链不触真实包）----
opt_intel = _stub("optimum.intel", OVModelForCausalLM=object())
opt = _stub("optimum", intel=opt_intel)
_stub("transformers", AutoTokenizer=object())
_stub("psutil")
_stub("device_manager", device_manager=object())
_stub("rag_engine", engine=object())

import qoder_inference
import server
from multiprocessing.connection import Listener, Client

AUTHKEY = server.AUTHKEY


# ============================================================
# 测试 1：analyze_style 解析（路径1 修复）
# ============================================================
class FakeShape:
    """模拟 tokenizer(prompt) 返回的张量：input_ids.shape[1] 给出输入长度。

    analyze_style 重构后用 tokenizer.apply_chat_template 拼 prompt，并取
    outputs[0][input_len:] 切片，故桩需提供 .shape。
    """
    shape = (1, 4)  # batch=1, seq_len=4


class FakeTok:
    def __init__(self, text):
        self._t = text
        self.eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # 仅把 messages 拼成字符串作为 prompt（测试不检查其排版）
        return "\n".join(m.get("content", "") for m in messages)

    def __call__(self, *a, **k):
        return {"input_ids": FakeShape()}

    def decode(self, *a, **k):
        return self._t


class FakeModel:
    def generate(self, **k):
        return [[]]


def test_analyze_chinese():
    # 基座模型中文散文：期望抽出 style_score=90，其余确定性兜底非 null
    out = "这段文本的风格特征比较符合新闻报道的风格……这个文本的风格特征评分较高，达到90分。"
    r = qoder_inference.analyze_style(FakeModel(), FakeTok(out), text="今天天气不错")
    assert r["style_score"] == 90, r
    sa = r["style_analysis"]
    assert sa["perplexity"] is not None
    assert sa["length_variance"] is not None
    assert sa["vocabulary_match"] is not None
    print("  [1a] 中文散文抽取 style_score=90, 兜底非 null ... OK")


def test_analyze_json():
    out = '{"style_score": 82, "perplexity": 0.35, "length_variance": 0.42, "vocabulary_match": 0.7}'
    r = qoder_inference.analyze_style(FakeModel(), FakeTok(out), text="x")
    assert r["style_score"] == 82, r
    assert abs(r["style_analysis"]["perplexity"] - 0.35) < 1e-6
    print("  [1b] 标准 JSON 解析 style_score=82 ... OK")


def test_analyze_never_all_null():
    out = "纯闲聊，没有任何数字或评分。"
    r = qoder_inference.analyze_style(FakeModel(), FakeTok(out), text="今天天气不错，我去公园散步")
    assert r["style_score"] is not None
    assert None not in r["style_analysis"].values()
    print("  [1c] 无任何数字也永不全 null（确定性兜底）... OK")


# ============================================================
# 测试 2：server 并发（本次修复）
#   模拟：一个慢 generate（sleep 3s）在跑时，并发的快 analyze 应立即返回。
# ============================================================
def _fake_handle(msg):
    op = msg.get("op")
    args = msg.get("args", {})
    if op == "request" and args.get("action") == "generate":
        time.sleep(3.0)  # 模拟 8B 在纯 CPU 上慢慢磨
        return {"ok": True, "content": "x", "model": "qwen3-8b"}
    if op == "request" and args.get("action") == "analyze":
        return {"ok": True, "result": {"style_score": 90}, "model": "qwen2.5-0.5b"}
    return {"ok": True}


def test_concurrent_server():
    srv = server.Server()
    srv.handle = _fake_handle  # 用模拟 handler，避免触发真实模型加载

    with Listener(("127.0.0.1", 0), authkey=AUTHKEY) as listener:
        port = listener.address[1]
        stop = threading.Event()

        def accept_loop():
            # 与新 main() 同款的"每连接一个线程"并发循环
            while not stop.is_set():
                try:
                    conn = listener.accept()
                except Exception:
                    break
                threading.Thread(
                    target=server._serve_connection, args=(srv, conn), daemon=True
                ).start()

        threading.Thread(target=accept_loop, daemon=True).start()

        results = {}

        def client_send(payload, key):
            t0 = time.time()
            with Client(("127.0.0.1", port), authkey=AUTHKEY) as c:
                c.send(payload)
                results[key] = (c.recv(), time.time() - t0)

        th_gen = threading.Thread(
            target=client_send, args=({"op": "request", "args": {"action": "generate"}}, "gen")
        )
        th_an = threading.Thread(
            target=client_send, args=({"op": "request", "args": {"action": "analyze"}}, "an")
        )
        th_gen.start()
        time.sleep(0.2)
        th_an.start()
        th_gen.join()
        th_an.join()
        stop.set()

        gen_ok, gen_dt = results["gen"]
        an_ok, an_dt = results["an"]
        assert an_ok.get("ok") is True, an_ok
        assert gen_ok.get("ok") is True, gen_ok
        # 关键断言：analyze 在 generate 完成之前就返回了（未被慢请求阻塞）
        assert an_dt < gen_dt - 1.0, (an_dt, gen_dt)
        print(
            f"  [2] 并发验证：analyze 在 {an_dt:.2f}s 返回，generate 在 {gen_dt:.2f}s 返回"
            f"（analyze 未被阻塞）... OK"
        )


if __name__ == "__main__":
    print("== 测试 1：analyze_style 路径1 解析修复 ==")
    test_analyze_chinese()
    test_analyze_json()
    test_analyze_never_all_null()
    print("== 测试 2：server 并发修复 ==")
    test_concurrent_server()
    print("\nALL TESTS PASSED ✅")
