"""Start TriageCare locally and open it in the browser. Works on macOS, Linux and Windows.

    python scripts/run.py            # port 8000, or the next free one
    python scripts/run.py --port 8010

- Runs `setup` first if the trained model or the database is missing.
- If TriageCare is already running on the port, just opens the browser.
- If something else is using the port, picks the next free one.
"""
import argparse
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"


def triagecare_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/v1/health", timeout=2) as r:
            return json.load(r).get("status") == "ok"
    except Exception:
        return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) != 0


def needs_setup() -> bool:
    sys.path.insert(0, str(ROOT))
    from triagecare.config import get_settings

    settings = get_settings()
    if not settings.model_path.exists():
        return True
    db_file = ROOT / "triagecare.db"
    return settings.database_url.startswith("sqlite") and (not db_file.exists() or db_file.stat().st_size == 0)


def open_when_ready(port: int) -> None:
    for _ in range(60):
        if triagecare_running(port):
            webbrowser.open(f"http://localhost:{port}")
            return
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if needs_setup():
        print("Model or database missing, running setup first (about a minute)...", flush=True)
        subprocess.run([sys.executable, "-m", "triagecare.cli", "setup"], cwd=ROOT, check=True)

    port = args.port
    if triagecare_running(port):
        print(f"TriageCare is already running at http://localhost:{port}")
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{port}")
        return
    while not port_free(port):
        print(f"Port {port} is in use by another program, trying {port + 1}")
        port += 1

    print(f"\nTriageCare: http://localhost:{port}   (sign in: admin / triage123, stop with Ctrl+C)\n", flush=True)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(port,), daemon=True).start()
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "triagecare.main:app", "--reload", "--reload-dir", "triagecare", "--reload-dir", "src", "--host", HOST, "--port", str(port)],
            cwd=ROOT,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
