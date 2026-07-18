r"""server.py — named-pipe model server for the local-style-writer skill.

Listens on \\.\pipe\local-style-writer, loads models lazily, and serves
three ops: status / request / shutdown. It reuses the project's inference,
RAG and device-selection logic, and folds in the former meta_server profile
loading so a user's style profile is injected at generation time.

This is the only long-lived process; the host (or client.py) talks to it over
the named pipe, never over HTTP.
"""
import json
import os
import sys
import time
import traceback
import threading
import logging
import logging.handlers
import hashlib
from pathlib import Path
from multiprocessing.connection import Listener

# 让本目录下的模块可被导入（device_manager / qoder_inference / rag_engine 同目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psutil
from device_manager import device_manager
from qoder_inference import load_model, generate_personal_style, analyze_style
from rag_engine import engine as rag_engine

SKILL_NAME = "local-style-writer"
PIPE_ADDRESS = rf"\\.\pipe\{SKILL_NAME}"
AUTHKEY = SKILL_NAME.encode("utf-8")

def _resolve_log_dir() -> Path:
    # 优先使用宿主（如 Marvis）通过环境变量指定的日志目录；
    # 否则落到用户目录下的统一位置（绝对路径，避免 CWD 漂移）。
    env_dir = os.environ.get("SKILL_LOG_DIR")
    base = Path(env_dir) if env_dir else (Path.home() / ".openvino" / "logs")
    log_dir = base / SKILL_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


LOG_DIR = _resolve_log_dir()
_log_fmt = logging.Formatter(
    "%(asctime)s [%(name)s pid=%(process)d] %(levelname)s %(message)s"
)
_file_handler = logging.handlers.RotatingFileHandler(
    str(LOG_DIR / "skill.log"), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_file_handler])
logger = logging.getLogger("skill-server")

# ---- hot-reload watchdog（源码变化时主动退出，由宿主/run.ps1 重新拉起）----
def _source_signature() -> str:
    h = hashlib.md5()
    base = Path(__file__).resolve().parent
    for name in ("server.py", "qoder_inference.py", "rag_engine.py", "device_manager.py"):
        p = base / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


_START_SIGNATURE = _source_signature()


def _cleanup_pid_file() -> None:
    """退出时清理 PID 文件，避免残留 PID 误导 client 的单实例判断。"""
    try:
        pid_file = Path(__file__).resolve().parent.parent / ".server.pid"
        if pid_file.exists():
            pid_file.unlink()
    except Exception:
        pass


def _hot_reload_watchdog(interval: float = 30.0) -> None:
    while True:
        time.sleep(interval)
        if _source_signature() != _START_SIGNATURE:
            logger.info("source files changed, exiting for hot-restart")
            for h in logging.root.handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            _cleanup_pid_file()
            os._exit(0)


def _configure_stream_encoding(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


_configure_stream_encoding(sys.stdout)
_configure_stream_encoding(sys.stderr)


MODEL_DIRS = {
    "personal": Path(os.environ.get("PERSONAL_MODEL_PATH", "models/qwen3_personal_int4")),
    "analysis": Path(os.environ.get("ANALYSIS_MODEL_PATH", "models/qwen2.5_int4")),
}

PROFILES_DIR = Path("profiles")
PROFILES_DIR.mkdir(exist_ok=True)


class Server:
    def __init__(self) -> None:
        self.state = "starting"
        self.error = ""
        self.started_at = time.time()
        self.models = {}
        self._model_loading = {}
        self._model_failures = {}
        self._model_lock = threading.Lock()
        self._infer_locks = {}
        self._stop = False

    # ---- model management (lazy load + lock + failure cache) ----
    def get_model(self, key: str):
        if key in self._model_failures:
            elapsed = time.time() - self._model_failures[key]
            if elapsed < 60:
                raise RuntimeError(
                    f"Model {key} failed to load, retry in {60 - elapsed:.0f}s"
                )
            del self._model_failures[key]

        if key not in self.models:
            with self._model_lock:
                if key not in self.models:
                    model_dir = MODEL_DIRS.get(key)
                    if not model_dir or not model_dir.exists():
                        self._model_failures[key] = time.time()
                        raise RuntimeError(f"Model directory not found: {model_dir}")
                    self._model_loading[key] = True
                    logger.info(f"Loading {key} model from {model_dir}...")
                    t0 = time.time()
                    try:
                        self.models[key] = load_model(
                            str(model_dir), model_type=key, use_cache=False
                        )
                    except Exception:
                        self._model_failures[key] = time.time()
                        logger.exception(f"Failed to load {key} model")
                        raise
                    finally:
                        self._model_loading[key] = False
                    logger.info(f"{key} model loaded in {time.time() - t0:.1f}s")
        return self.models[key]

    def _acquire_infer(self, key: str) -> bool:
        lock = self._infer_locks.setdefault(key, threading.Lock())
        return lock.acquire(blocking=False)

    def _release_infer(self, key: str) -> None:
        lock = self._infer_locks.get(key)
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                pass

    # ---- profile (folded from the former meta_server) ----
    def load_profile(self, user_id: str) -> dict:
        path = PROFILES_DIR / f"{user_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        default_path = PROFILES_DIR / "user_default.json"
        if default_path.exists():
            return json.loads(default_path.read_text(encoding="utf-8"))
        return {"user_id": user_id, "style_tags": [], "tone_presets": {}, "preferences": {}}

    # ---- dispatch ----
    def handle(self, msg: dict) -> dict:
        op = msg.get("op")
        if op == "status":
            return self._status()
        if op == "request":
            return self._request(msg.get("args", {}))
        if op == "shutdown":
            self._stop = True
            return {"ok": True, "state": "shutting_down"}
        return {"ok": False, "error": f"unknown op: {op}"}

    def _status(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "ok": True,
            "state": self.state,
            "pid": os.getpid(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "error": self.error,
            "models_loaded": list(self.models.keys()),
            "models_loading": [k for k, v in self._model_loading.items() if v],
            "models_available": list(MODEL_DIRS.keys()),
            "device_allocation": device_manager.summary,
            "rag": rag_engine.status,
            "memory": {
                "total_gb": round(mem.total / 1024**3, 1),
                "available_gb": round(mem.available / 1024**3, 1),
                "percent_used": mem.percent,
            },
        }

    def _request(self, args: dict) -> dict:
        action = args.get("action", "generate")
        try:
            if action == "analyze":
                return self._do_analyze(args)
            return self._do_generate(args)
        except Exception as exc:
            logger.exception("request failed")
            return {"ok": False, "error": str(exc)}

    def _do_generate(self, args: dict) -> dict:
        key = "personal"
        if not self._acquire_infer(key):
            return {"ok": False, "error": "Model 'personal' is busy, please retry later."}
        try:
            t0 = time.time()
            model, tokenizer = self.get_model(key)
            profile = self.load_profile(args.get("user_id", "default"))
            tone = args.get("tone_preset") or (profile.get("style_tags") or [None])[0]
            max_tokens = args.get("max_new_tokens") or profile.get(
                "preferences", {}
            ).get("max_new_tokens", 512)
            temp = (
                args.get("temperature")
                if args.get("temperature") is not None
                else profile.get("preferences", {}).get("temperature", 0.7)
            )

            load_rag = bool(args.get("load_rag", False))
            if load_rag:
                query_parts = [f"主题：{args.get('topic', '')}"]
                if args.get("key_points"):
                    query_parts.append(f"要点：{'；'.join(args['key_points'])}")
                if args.get("target_length"):
                    query_parts.append(f"目标长度：{args['target_length']}")
                if tone:
                    query_parts.append(f"风格：{tone}")
                base_query = "\n".join(query_parts)
                rag_prompt = rag_engine.augment_prompt(
                    query=base_query,
                    user_prompt=f"<|im_start|>user\n请根据以下要求写一篇文章。\n{base_query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                )
                content = generate_personal_style(
                    model, tokenizer,
                    topic=args.get("topic", ""),
                    prompt_override=rag_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                )
            else:
                content = generate_personal_style(
                    model, tokenizer,
                    topic=args.get("topic", ""),
                    key_points=args.get("key_points"),
                    target_length=args.get("target_length"),
                    tone_preset=tone,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                )
            elapsed = round(time.time() - t0, 2)
            logger.info(f"Generated {len(content)} chars in {elapsed}s (rag={load_rag})")
            return {
                "ok": True,
                "content": content,
                "model": "qwen3-8b-personal",
                "elapsed_s": elapsed,
                "rag_used": load_rag,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            self._release_infer(key)

    def _load_user_style(self) -> str | None:
        """从 profiles/user_default.json 读取个人风格，拼成可读描述，供 analyze 对比锚点。"""
        profile_path = Path(__file__).resolve().parent.parent / "profiles" / "user_default.json"
        if not profile_path.exists():
            return None
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        tags = data.get("style_tags") or []
        tone = data.get("tone_presets") or {}
        default_tone_desc = ""
        for t in tags:
            if t in tone:
                default_tone_desc = tone[t]
                break
        parts = []
        if tags:
            parts.append("、".join(tags))
        if default_tone_desc:
            parts.append(default_tone_desc)
        return "；".join(parts) if parts else None

    def _do_analyze(self, args: dict) -> dict:
        key = "analysis"
        if not self._acquire_infer(key):
            return {"ok": False, "error": "Model 'analysis' is busy, please retry later."}
        try:
            t0 = time.time()
            model, tokenizer = self.get_model(key)
            # 注意：分析模型按训练模板（不含 user_style）微调，故此处不注入个人风格锚点，
            # 个人风格已由训练数据内化；注入会与训练输入不一致导致乱码。
            result = analyze_style(
                model, tokenizer,
                text=args.get("text", ""),
                max_new_tokens=args.get("max_new_tokens", 256),
            )
            elapsed = round(time.time() - t0, 2)
            return {
                "ok": True,
                "result": result,
                "model": "qwen2.5-0.5b-style",
                "elapsed_s": elapsed,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            self._release_infer(key)

    def init_async(self) -> None:
        threading.Thread(target=self._init, daemon=True).start()

    def _init(self) -> None:
        try:
            self.state = "loading"
            rag_engine.load_index()
            self.state = "running"
            logger.info("Server ready (RAG index loaded; models load on first request).")
        except Exception:
            self.error = traceback.format_exc()
            self.state = "error"
            logger.exception("init failed")


def _serve_connection(srv: "Server", conn) -> None:
    """在每个连接自己的线程里收消息、分发、回包，使慢请求（如 8B 生成）不会阻塞其它请求（如 0.5B 分析）。"""
    try:
        try:
            msg = conn.recv()
        except Exception:
            return
        try:
            resp = srv.handle(msg)
            conn.send(resp)
        except Exception as exc:
            logger.warning(f"connection handling failed: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> int:
    srv = Server()
    srv.init_async()
    # 写 PID 文件，供 client.py 做单实例/健康检查
    try:
        pid_file = Path(__file__).resolve().parent.parent / ".server.pid"
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    threading.Thread(target=_hot_reload_watchdog, daemon=True).start()
    logger.info(f"Listening on {PIPE_ADDRESS}")
    # 绑定管道时重试：杀掉旧 server 后其管道实例可能尚未释放，短暂等待即可避免绑定失败
    listener = None
    for attempt in range(10):
        try:
            listener = Listener(PIPE_ADDRESS, authkey=AUTHKEY)
            break
        except Exception as exc:
            logger.warning(f"pipe bind failed (attempt {attempt + 1}/10): {exc}")
            time.sleep(1.0)
    if listener is None:
        logger.error("failed to bind named pipe after retries")
        _cleanup_pid_file()
        return 1
    with listener:
        while not srv._stop:
            try:
                conn = listener.accept()
            except Exception:
                continue
            threading.Thread(target=_serve_connection, args=(srv, conn), daemon=True).start()
    _cleanup_pid_file()
    return 0


if __name__ == "__main__":
    sys.exit(main())
