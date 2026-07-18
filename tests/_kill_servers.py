import os
import sys
import time

try:
    import psutil
except ImportError:
    print("psutil not available; aborting")
    sys.exit(2)

ROOT_MARKERS = ("local-style-writer",)

killed = []
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        if not p.info["cmdline"]:
            continue
        cmd = " ".join(p.info["cmdline"])
        if p.info["name"] and "python" in p.info["name"].lower() and cmd.rstrip().endswith("server.py") and any(m in cmd for m in ROOT_MARKERS):
            if p.pid == os.getpid():
                continue
            p.kill()
            killed.append((p.pid, cmd))
    except Exception as exc:
        print("skip pid err:", exc)

if killed:
    print(f"killed {len(killed)} server process(es):")
    for pid, cmd in killed:
        print(f"  pid={pid} {cmd}")
else:
    print("no server.py processes found to kill")

# wait for pipe release
for _ in range(10):
    try:
        names = os.listdir(r"\\.\pipe\\")
        if "local-style-writer" not in names:
            print("pipe released: OK")
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    print("pipe still held after wait")
