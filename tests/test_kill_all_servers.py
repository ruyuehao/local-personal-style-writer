"""验证 _kill_all_server_processes 的命令行匹配逻辑：

只杀「跑本项目 server.py 的 python 进程」，绝不误杀：
  - client.py 进程（命令行是 client.py）
  - 无关项目的 server.py（无 local-style-writer / ROOT 标记）
  - 自身进程
  - 非 python 进程

用假 psutil（注入 sys.modules）控制进程列表，纯逻辑层测试。
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
CLIENT = str(SCRIPTS / "client.py")

killed = []


class FakeProc:
    def __init__(self, info):
        self.info = info

    def kill(self):
        killed.append(self.info["pid"])


fake_psutil = types.ModuleType("psutil")
fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

_me = os.getpid()
_records = [
    # 111: venv 路径含 local-style-writer 标记 + server.py → 应杀
    {"pid": 111, "name": "python.exe",
     "cmdline": [r"C:\Users\86158\.openvino\venv\local-style-writer\Scripts\python.exe",
                 r"D:\ai-skill\production-ai-skill\local-style-writer\scripts\server.py"]},
    # 222: 项目 venv + server.py（ROOT 标记）→ 应杀
    {"pid": 222, "name": "python.exe",
     "cmdline": [r"D:\ai-skill\production-ai-skill\venv\Scripts\python.exe",
                 r"D:\ai-skill\production-ai-skill\local-style-writer\scripts\server.py"]},
    # 333: client.py（不是 server.py）→ 不杀
    {"pid": 333, "name": "python.exe",
     "cmdline": [r"D:\ai-skill\production-ai-skill\venv\Scripts\python.exe",
                 r"D:\ai-skill\production-ai-skill\local-style-writer\scripts\client.py"]},
    # 444: 裸 server.py 但无项目标记 → 不杀（无关项目）
    {"pid": 444, "name": "python.exe", "cmdline": ["python.exe", "server.py"]},
    # 自身 pid：即便命令行匹配也必须跳过
    {"pid": _me, "name": "python.exe",
     "cmdline": [r"D:\ai-skill\production-ai-skill\venv\Scripts\python.exe",
                 r"D:\ai-skill\production-ai-skill\local-style-writer\scripts\server.py"]},
    # 555: 非 python → 不杀
    {"pid": 555, "name": "explorer.exe", "cmdline": ["explorer.exe"]},
    # 666: cmdline 为 None（某些系统进程）→ 不崩
    {"pid": 666, "name": "python.exe", "cmdline": None},
]


def _iter(fields):
    for r in _records:
        yield FakeProc({k: r.get(k) for k in fields})


fake_psutil.process_iter = _iter
sys.modules["psutil"] = fake_psutil

spec = importlib.util.spec_from_file_location("lsw_client", CLIENT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

killed.clear()
n = client._kill_all_server_processes()

assert n == 2, f"期望杀 2 个，实际 {n}: {killed}"
assert set(killed) == {111, 222}, f"杀错对象: {killed}"
assert _me not in killed, "误杀自身!"
assert 333 not in killed, "误杀 client.py!"
assert 444 not in killed, "误杀无关 server.py!"
assert 555 not in killed, "误杀非 python 进程!"
print("ALL TESTS PASSED ✅  正确杀掉的 server.py pid:", sorted(killed))
