"""client.py — short-lived CLI client for the local-style-writer skill.

Ensures the named-pipe server is running (spawning it as a detached
background process on first use), waits until it is ready, sends the
request, and prints the result. This is what `run.ps1` invokes with the
user's natural-language prompt.
"""
import argparse
import json
import os
import sys
import time
import subprocess
import threading
import re
from pathlib import Path
from multiprocessing.connection import Client

SKILL_NAME = "local-style-writer"
PIPE_ADDRESS = rf"\\.\pipe\{SKILL_NAME}"
AUTHKEY = SKILL_NAME.encode("utf-8")

DOWNLOAD_WAIT_TIMEOUT = 8 * 60  # 模型未就绪时的最长等待（秒）

ROOT = Path(__file__).resolve().parent.parent
PID_FILE = ROOT / ".server.pid"

# 默认复用已就绪的健康 server（保留预热模型、省重启开销）；仅在无健康 server 时
# 才清场重建。若需每次强制重启（调试用），设 LSW_FORCE_RESTART=1。
_FORCE_RESTART = os.environ.get("LSW_FORCE_RESTART", "0") in ("1", "true", "True", "")


def _configure_stream_encoding(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


_configure_stream_encoding(sys.stdout)
_configure_stream_encoding(sys.stderr)


import logging
import logging.handlers


def _setup_logging() -> logging.Logger:
    env_dir = os.environ.get("SKILL_LOG_DIR")
    base = Path(env_dir) if env_dir else (Path.home() / ".openvino" / "logs")
    log_dir = base / SKILL_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(name)s pid=%(process)d] %(levelname)s %(message)s")
    h = logging.handlers.RotatingFileHandler(
        str(log_dir / "client.log"), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[h])
    return logging.getLogger("skill-client")


logger = _setup_logging()


def _resolve_python() -> str:
    """与 run.ps1 / install-env.ps1 解析同样的 venv python。"""
    info_path = ROOT / "info.json"
    venv_name = "local-style-writer"
    if info_path.exists():
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            venv_name = data.get("venv_name", venv_name)
        except Exception:
            pass
    cand = Path.home() / ".openvino" / "venv" / venv_name / "Scripts" / "python.exe"
    if cand.exists():
        return str(cand)
    return sys.executable


def _send(payload: dict, timeout: float = 60.0) -> dict:
    with Client(PIPE_ADDRESS, authkey=AUTHKEY) as conn:
        conn.send(payload)
        if conn.poll(timeout):
            return conn.recv()
        raise TimeoutError(f"server did not respond within {timeout}s")


def _length_to_tokens(length: str | None) -> int | None:
    """把 --length "300字" 粗略换算成 max_new_tokens，避免无意义生成到 512。

    中文约 1 token/字，留 ~60% 余量给标点/重复表达，封顶 512。无 NPU 的纯
    CPU 机器上 8B 模型推理约 0.2 字/秒，缩短上限能省下无谓的等待时间。
    """
    if not length:
        return None
    m = re.search(r"\d+", length)
    if not m:
        return None
    n = int(m.group())
    return max(128, min(int(n * 1.6), 512))


# ---- 单实例 / 健康检查：确保任意时刻只有一个健康 server 在跑 ----
def _read_pid_file() -> int | None:
    try:
        if PID_FILE.exists():
            return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return None


def _is_process_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _kill_process(pid: int) -> None:
    try:
        import psutil
        psutil.Process(pid).kill()
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _ping_server(timeout: float = 3.0) -> bool:
    """连上管道并验证 server 真能应答 status（不只是管道存在）。"""
    box: dict = {}

    def _worker() -> None:
        try:
            with Client(PIPE_ADDRESS, authkey=AUTHKEY) as conn:
                conn.send({"op": "status"})
                if conn.poll(timeout):
                    resp = conn.recv()
                    box["ok"] = bool(resp.get("ok"))
                else:
                    box["ok"] = False
        except Exception:
            box["ok"] = False

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout + 0.5)
    return box.get("ok", False)


def _request_shutdown_existing(timeout: float = 3.0) -> bool:
    """向管道上（可能）存活的旧 server 发送 shutdown；成功返回 True。

    用于强制重启前优雅释放被旧实例占用的命名管道，避免新 server 绑定管道时
    触发 [WinError 5] 拒绝访问（Access Denied）。
    """
    box: dict = {}

    def _w() -> None:
        try:
            with Client(PIPE_ADDRESS, authkey=AUTHKEY) as conn:
                conn.send({"op": "shutdown"})
                if conn.poll(timeout):
                    box["ok"] = bool(conn.recv().get("ok"))
                else:
                    box["ok"] = False
        except Exception:
            box["ok"] = False

    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(timeout + 0.5)
    return box.get("ok", False)


def _pipe_held(timeout: float = 2.0) -> bool:
    """探测命名管道当前是否仍被某个 server 实例占用（connect 探测，仅作兜底）。"""
    box: dict = {}

    def _w() -> None:
        try:
            with Client(PIPE_ADDRESS, authkey=AUTHKEY):
                box["held"] = True
        except Exception:
            box["held"] = False

    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(timeout + 0.5)
    return box.get("held", False)


def _pipe_exists() -> bool:
    """非阻塞判断命名管道是否仍被某个进程持有（句柄未关闭）。

    用 ``\\\\.\\pipe\\`` 目录列举：只要任一进程握着该管道句柄，名字就出现在列表里；
    最后一个句柄关闭（进程退出）后立即消失。比 connect 探测可靠——不会因
    server 正忙不 accept 连接而误判为「已释放」（这正是 [WinError 5] 的根因）。
    """
    try:
        return SKILL_NAME in os.listdir(r"\\.\pipe\\")
    except Exception:
        # 列举失败（非 Windows 等）退回 connect 探测（仅作兜底）
        return _pipe_held()


def _kill_all_server_processes() -> int:
    """杀掉所有正在运行本 skill 的 server.py 的 python 进程（除自身外）。

    用于清理僵死/占用命名管道的旧 server——这些进程的 PID 往往已不在
    ``.server.pid`` 里（被后续绑定失败的 server 覆盖/删除），只能靠命令行匹配定位。
    半死实例不 accept 连接、shutdown 到不了它，但进程仍在、cmdline 可读 → 强杀。
    """
    me = os.getpid()
    killed = 0
    try:
        import psutil
    except Exception:
        return 0
    root_marker = str(ROOT)
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            name = (p.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = p.info.get("cmdline") or []
            joined = " ".join(cmdline)
            # 命令行里跑的是本项目的 server.py：绝对路径或裸名都覆盖，
            # 并用 'local-style-writer' 或 ROOT 路径锁定本项目，避免误杀无关 server.py
            if any(str(arg).endswith("server.py") for arg in cmdline) and (
                "local-style-writer" in joined or root_marker in joined
            ):
                logger.warning(f"killing stale server process pid={p.info['pid']} by cmdline match")
                p.kill()
                killed += 1
        except Exception:
            continue
    return killed


def _wait_pipe_released(timeout: float = 12.0) -> bool:
    """轮询直到命名管道被 OS 释放（旧 server 进程已退出、句柄已关闭）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pipe_exists():
            return True
        time.sleep(0.5)
    return False


def _ensure_server(force: bool = False) -> None:
    """确保只有一个健康 server：彻底清掉残留/僵死实例后再拉起。

    无论 force 与否，只要需要拉新 server，就按「优雅 shutdown → PID 文件杀 →
    命令行匹配杀全部 server.py 进程 → 等管道真正释放」四步清场，根除
    「旧实例仍占用管道（[WinError 5] 拒绝访问）」与「半死实例（能回 status
    但请求卡死）」两类顽疾。force=True 时跳过「复用预热 server」直接清场重建。
    """
    # 0) 非强制且已有健康 server → 直接复用（保留预热模型，省去重启开销）
    if not force and _ping_server():
        return

    # 1) 优雅 shutdown 管道上可能存活的旧实例（仅对能 accept 的实例有效）
    _request_shutdown_existing()

    # 2) 按 PID 文件杀（命中 .server.pid 仍准确的实例）
    pid = _read_pid_file()
    if pid and _is_process_alive(pid):
        logger.warning(f"killing stale server pid={pid}")
        _kill_process(pid)

    # 3) 关键：命令行匹配杀掉所有跑 server.py 的 python 进程。
    #    PID 文件常被后续绑定失败的 server 覆盖/删除，找不到真正占用管道的旧进程；
    #    半死实例不 accept 连接、shutdown 到不了它，只能靠 cmdline 定位强杀。
    n = _kill_all_server_processes()
    if n:
        logger.warning(f"killed {n} stale server.py process(es) by cmdline match")
    time.sleep(1.0)

    # 4) 等命名管道被 OS 真正释放（用 \\.\pipe\ 列举，非阻塞且对忙碌 server 可靠）
    _wait_pipe_released(timeout=12.0)

    # 5) 拉起新 server：清残留 PID 文件
    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass
    time.sleep(1.0)
    python = _resolve_python()
    server_py = ROOT / "scripts" / "server.py"
    log_out = ROOT / "logs" / "server.out"
    log_out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("spawning server.py as detached background process")
    # DETACHED_PROCESS：与当前控制台脱钩，避免客户端退出时带走服务
    subprocess.Popen(
        [str(python), str(server_py)],
        cwd=str(ROOT),
        creationflags=0x00000008,
        stdout=open(str(log_out), "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )


def _wait_ready() -> None:
    """轮询直到 server 就绪（state=running）。

    关键：新 server 冷启动要 import openvino/transformers/psutil 等（约 10-20s）
    之后才绑定命名管道，期间 status 必然连不上。必须给足宽限时间，绝不能在
    import 期间就强制重启——否则会陷入「杀掉正在 import 的新 server → 再起一个 →
    又被杀」的死循环（即「杀进程太快」现象）。仅在长时间（>60s）完全无响应时
    才判定为真卡死并强制重建一次。
    """
    deadline = time.time() + DOWNLOAD_WAIT_TIMEOUT
    spawn_time = time.time()
    last_force = 0.0
    while time.time() < deadline:
        try:
            st = _send({"op": "status"}, timeout=8.0)
        except Exception as exc:
            now = time.time()
            # 自本次拉起/上次强制重启起，给 60s 宽限让 server 完成 import+绑管道；
            # 超过 60s 仍连不上才视为真卡死，强制重建（且两次重建间隔 ≥60s）。
            if (now - spawn_time) > 60.0 and (now - last_force) > 60.0:
                logger.warning(f"server unresponsive >60s ({exc}); force-restart once")
                _ensure_server(force=True)
                spawn_time = time.time()
                last_force = spawn_time
            time.sleep(2.0)
            continue
        state = st.get("state")
        if state == "running":
            return
        if state == "error":
            print(f"服务初始化失败: {st.get('error')}")
            sys.exit(1)
        time.sleep(1.0)
    print("模型/服务启动超时，请检查日志或重新运行 'scripts\\run.ps1' 继续")
    sys.exit(3)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="local-style-writer client")
    parser.add_argument("topic", nargs="?", help="写作主题")
    parser.add_argument("--user", default="default", help="用户画像 id")
    parser.add_argument("--length", default=None, help="目标长度，如 300字")
    parser.add_argument("--tone", default=None, help="风格预设")
    parser.add_argument("--key-points", nargs="*", default=None, help="要点列表")
    parser.add_argument("--rag", action="store_true", help="启用 RAG 参考个人历史文章")
    parser.add_argument("--analyze", default=None, help="改为分析给定文本的风格")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    _ensure_server(force=_FORCE_RESTART)
    _wait_ready()

    if args.analyze is not None:
        payload = {
            "op": "request",
            "args": {
                "action": "analyze",
                "text": args.analyze,
                # 128 已足够短风格描述；CPU 推理较慢，过大易撞 90s 超时
                "max_new_tokens": args.max_new_tokens or 128,
            },
        }
    else:
        if not args.topic:
            print("错误: 请提供写作主题 (topic) 或 --analyze 文本")
            return 2
        payload = {
            "op": "request",
            "args": {
                "action": "generate",
                "topic": args.topic,
                "user_id": args.user,
                "target_length": args.length,
                "tone_preset": args.tone,
                "key_points": args.key_points,
                "load_rag": args.rag,
                "max_new_tokens": args.max_new_tokens or _length_to_tokens(args.length),
                "temperature": args.temperature,
            },
        }

    # 超时策略区分两种场景：
    # - analyze 超时通常是「半死实例占用管道」，强制重启 server 并重试一次；
    # - generate 超时则是 8B 模型在纯 CPU 上生成本就慢（实测 ~0.2 字/秒），
    #   此时 server 仍在正常推理，重启只会丢弃已加载的 8B 模型重新来过，
    #   纯属浪费，因此不重试、直接给出硬件限制提示。
    is_generate = payload["args"]["action"] == "generate"
    req_timeout = 1800.0 if is_generate else 90.0
    try:
        resp = _send(payload, timeout=req_timeout)
    except TimeoutError:
        if is_generate:
            print("错误: 写作生成超时（已等待约 {:.0f} 分钟）。".format(req_timeout / 60))
            print("      本机为纯 CPU（无 NPU / 独显），8B 写作模型生成速度约 0.2 字/秒，")
            print("      长文（如 300 字）通常需要 20+ 分钟才能跑完，并非卡死。")
            print("      建议：")
            print("        1) 缩短 --length（如 100字，约 5 分钟）做功能验证；")
            print("        2) 在带 NPU / 独显的 Intel AI PC 上运行以获得实时速度。")
            return 1
        logger.warning("analyze timed out; forcing server restart and retrying once")
        _ensure_server(force=True)
        _wait_ready()
        resp = _send(payload, timeout=req_timeout)

    if not resp.get("ok"):
        print(f"错误: {resp.get('error')}")
        return 1

    if args.json:
        print(json.dumps(resp, ensure_ascii=False))
    elif "content" in resp:
        print("【生成内容】")
        print(resp["content"])
        if resp.get("rag_used"):
            print("\n(已启用 RAG 参考个人历史文章)")
    elif "result" in resp:
        print("【风格分析】")
        result = resp["result"]
        desc = result.pop("description", None) if isinstance(result, dict) else None
        if desc:
            print(desc)
            print("")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
