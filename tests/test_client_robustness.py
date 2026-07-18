"""client 健壮性回归：验证 _send 不再永久卡死，_ensure_server(force) 能重建。"""
import sys, time
from pathlib import Path
from unittest import mock

sys.path.insert(0, "D:/ai-skill/production-ai-skill/local-style-writer/scripts")
import client

print("== 测试 1：无 server 时 _send 快速失败（不永久卡）==")
t0 = time.time()
try:
    client._send({"op": "status"}, timeout=3.0)
    print("  FAIL: _send 居然没抛异常")
except Exception as e:
    dt = time.time() - t0
    print(f"  OK: _send 在 {dt:.1f}s 抛 {type(e).__name__}（≤3s，未永久卡死）")
    assert dt < 5.0, "超时回收应远小于 5s"

print("\n== 测试 2：无 server 时 _ping_server 返回 False ==")
print(f"  _ping_server() = {client._ping_server()}")
assert client._ping_server() is False
print("  OK")

print("\n== 测试 3：_read_pid_file 能安全读取（残留文件或不存在都不崩）==")
pid = client._read_pid_file()
print(f"  _read_pid_file() = {pid}")
# 残留 PID 应被正确识别为「非存活」或不抛异常；本机清理后通常为 None
print(f"  _is_process_alive({pid}) = {client._is_process_alive(pid) if pid else 'N/A'}")
print("  OK（读取不崩）")

print("\n== 测试 4：_ensure_server(force=True) 无残留时触发一次 Popen 拉起 ==")
with mock.patch.object(client.subprocess, "Popen", return_value=None) as m:
    # 清掉可能存在的 PID 文件干扰（沙箱无此文件，仅防御）
    client.PID_FILE = Path("D:/ai-skill/production-ai-skill/local-style-writer/.tmp_test_pid")
    if client.PID_FILE.exists():
        client.PID_FILE.unlink()
    client._ensure_server(force=True)
    print(f"  调用 Popen 次数 = {m.call_count}")
    assert m.call_count == 1, "期望拉起一次新 server"
    # 验证传入的脚本路径是 server.py
    args, _ = m.call_args
    called_cmd = " ".join(str(a) for a in args[0])
    print(f"  拉起命令含 server.py: {'server.py' in called_cmd}")
    assert "server.py" in called_cmd
print("  OK")

print("\nALL CLIENT ROBUSTNESS TESTS PASSED ✅")
