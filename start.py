#!/usr/bin/env python3
"""Install missing dependencies and start the CashGap backend and frontend."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "backend" / "requirements.txt"
REQUIREMENTS_STAMP = VENV / ".cashgap-requirements"
SUPPORTED_PYTHONS = {(3, 11), (3, 12), (3, 13)}


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def use_supported_python() -> None:
    if sys.version_info[:2] in SUPPORTED_PYTHONS:
        return

    candidates = [
        VENV_PYTHON,
        *(shutil.which(name) for name in ("python3.13", "python3.12", "python3.11")),
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        version = subprocess.check_output(
            [str(candidate), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
        ).strip()
        if tuple(map(int, version.split("."))) in SUPPORTED_PYTHONS:
            os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])

    raise SystemExit("Нужен Python 3.11, 3.12 или 3.13.")


def require_runtime() -> str:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    node = shutil.which("node")
    if not npm or not node:
        raise SystemExit("Нужны Node.js 18+ и npm.")
    version = subprocess.check_output([node, "-p", "process.versions.node.split('.')[0]"], text=True)
    if int(version.strip()) < 18:
        raise SystemExit("Нужен Node.js 18 или новее.")
    return npm


def install_dependencies(npm: str) -> None:
    if not VENV_PYTHON.exists():
        print(f"Создаю .venv на Python {sys.version_info.major}.{sys.version_info.minor}...", flush=True)
        run([sys.executable, "-m", "venv", str(VENV)])

    requirements_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    installed_hash = REQUIREMENTS_STAMP.read_text().strip() if REQUIREMENTS_STAMP.exists() else ""
    if installed_hash != requirements_hash:
        print("Устанавливаю Python-зависимости...", flush=True)
        run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        REQUIREMENTS_STAMP.write_text(requirements_hash)

    if not (ROOT / "frontend" / "node_modules").is_dir():
        print("Устанавливаю Node.js-зависимости...", flush=True)
        run([npm, "install"], cwd=ROOT / "frontend")


def free_port(requested: int) -> int:
    port = requested
    while True:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    port += 1
                    continue
                return port
        return port


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    use_supported_python()
    npm = require_runtime()
    install_dependencies(npm)

    api_port = free_port(int(os.environ.get("CASHGAP_API_PORT", "8000")))
    ui_port = free_port(int(os.environ.get("CASHGAP_UI_PORT", "5173")))
    frontend_env = os.environ.copy()
    frontend_env["CASHGAP_API_PORT"] = str(api_port)

    backend = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ],
        cwd=ROOT / "backend",
    )
    frontend = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(ui_port), "--strictPort"],
        cwd=ROOT / "frontend",
        env=frontend_env,
    )

    print(f"API: http://127.0.0.1:{api_port}/docs", flush=True)
    print(f"UI:  http://127.0.0.1:{ui_port}", flush=True)
    print("Для остановки нажми Ctrl+C.", flush=True)
    try:
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        stop(frontend)
        stop(backend)

    return backend.returncode or frontend.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
