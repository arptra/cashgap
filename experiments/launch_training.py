#!/usr/bin/env python3
"""Launch cash-gap training outside Jupyter and inspect its durable status.

Examples:
  python experiments/launch_training.py benchmark --outflow ... --inflow ... \
    --output-dir artifacts/torch_10 --test-periods 10
  python experiments/launch_training.py full-benchmark --outflow ... --inflow ... \
    --output-dir artifacts/all_models_10 --test-periods 10
  python experiments/launch_training.py autotune --outflow ... --inflow ... \
    --output-dir artifacts/tuning --trials 30
  python experiments/launch_training.py status --output-dir artifacts/torch_10

The launcher imports no pandas, PyArrow, Optuna or Torch and returns as soon as
the detached supervisor is alive.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SCRIPT_BY_MODE = {
    "benchmark": "benchmark_torch_sequential.py",
    "full-benchmark": "benchmark_monthly_isolated.py",
    "autotune": "autotune_torch_stable.py",
}


def option_value(arguments: Sequence[str], option: str) -> Optional[str]:
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = option + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def pid_alive(pid: object) -> bool:
    try:
        number = int(pid)
        if number < 1:
            return False
        os.kill(number, 0)
        return True
    except (TypeError, ValueError, OSError, ProcessLookupError):
        return False


def read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def last_lines(path: Path, count: int) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return list(deque(stream, maxlen=max(1, count)))
    except OSError:
        return []


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def launch_script(
    mode: str, experiment_script: Path, forwarded: Sequence[str],
) -> Dict[str, object]:
    output_value = option_value(forwarded, "--output-dir")
    if not output_value:
        raise ValueError("Для внешнего запуска обязательно укажите --output-dir.")
    output = Path(output_value).expanduser().resolve()
    supervisor_root = output / "_supervisor"
    supervisor_root.mkdir(parents=True, exist_ok=True)
    status_file = supervisor_root / "status.json"
    previous = read_json(status_file)
    if previous and previous.get("state") in {"LAUNCHING", "STARTING", "RUNNING"}:
        supervisor_pid = previous.get("supervisor_pid")
        driver_pid = previous.get("driver_pid")
        if pid_alive(supervisor_pid) or pid_alive(driver_pid):
            raise RuntimeError(
                "Для этого --output-dir уже работает supervisor PID {} / driver PID {}. "
                "Сначала проверьте status.".format(supervisor_pid, driver_pid)
            )

    run_id = "{}-{}".format(
        dt.datetime.now().strftime("%Y%m%d-%H%M%S"), secrets.token_hex(3)
    )
    run_dir = supervisor_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    experiment_script = Path(experiment_script).resolve()
    supervisor_script = Path(__file__).with_name("training_supervisor.py").resolve()
    command = [sys.executable, "-X", "faulthandler", "-u", str(experiment_script)]
    command.extend(str(value) for value in forwarded)
    supervisor_command = [
        sys.executable,
        "-X", "faulthandler", "-u",
        str(supervisor_script),
        "--run-dir", str(run_dir),
        "--status-file", str(status_file),
        "--workdir", str(Path.cwd()),
        "--",
    ] + command
    boot_log = run_dir / "supervisor_boot.log"
    initial: Dict[str, object] = {
        "state": "LAUNCHING",
        "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "run_dir": str(run_dir),
        "driver_log": str(run_dir / "driver.log"),
        "heartbeat_log": str(run_dir / "heartbeat.log"),
        "supervisor_log": str(run_dir / "supervisor.log"),
        "supervisor_boot_log": str(run_dir / "supervisor_boot.log"),
        "command": command,
    }
    atomic_json(status_file, initial)
    with boot_log.open("ab", buffering=0) as output_stream:
        supervisor = subprocess.Popen(
            supervisor_command,
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            stdout=output_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    initial["supervisor_pid"] = supervisor.pid
    time_limit = dt.datetime.now().timestamp() + 10.0
    current = initial
    while dt.datetime.now().timestamp() < time_limit:
        candidate = read_json(status_file)
        if candidate:
            current = candidate
        if current.get("state") in {"RUNNING", "FAILED_TO_START", "FAILED"}:
            break
        if supervisor.poll() is not None:
            break
        import time
        time.sleep(0.1)
    print("\n=== ОБУЧЕНИЕ ВЫНЕСЕНО ИЗ JUPYTER KERNEL ===")
    print("Состояние: {}".format(current.get("state")))
    print("Supervisor PID: {}".format(current.get("supervisor_pid", supervisor.pid)))
    print("Driver PID: {}".format(current.get("driver_pid", "ещё запускается")))
    print("Полный лог: {}".format(current.get("driver_log", run_dir / "driver.log")))
    print("Статус: {}".format(status_file))
    print("Эту ячейку можно закрыть: обучение продолжит работать отдельно.")
    print("Проверка: %run experiments/launch_training.py status --output-dir {}".format(
        json.dumps(str(output), ensure_ascii=False)
    ))
    return current


def launch_mode(mode: str, forwarded: Sequence[str]) -> Dict[str, object]:
    if mode not in SCRIPT_BY_MODE:
        raise ValueError("Неизвестный режим {!r}: {}".format(mode, sorted(SCRIPT_BY_MODE)))
    experiment_script = Path(__file__).with_name(SCRIPT_BY_MODE[mode]).resolve()
    return launch_script(mode, experiment_script, forwarded)


def show_status(arguments: Sequence[str]) -> Dict[str, object]:
    output_value = option_value(arguments, "--output-dir")
    if not output_value:
        raise ValueError("Для status укажите --output-dir.")
    lines_value = option_value(arguments, "--lines") or "40"
    output = Path(output_value).expanduser().resolve()
    status_file = output / "_supervisor" / "status.json"
    status = read_json(status_file)
    if status is None:
        raise FileNotFoundError("Статус не найден: {}".format(status_file))
    state = str(status.get("state"))
    supervisor_is_alive = pid_alive(status.get("supervisor_pid"))
    if state in {"STARTING", "RUNNING", "LAUNCHING"} and not supervisor_is_alive:
        state = "SUPERVISOR_DIED"
    print("\n=== СТАТУС ВНЕШНЕГО ОБУЧЕНИЯ ===")
    print("Состояние: {}".format(state))
    print("Supervisor PID: {} | жив: {}".format(
        status.get("supervisor_pid"), supervisor_is_alive
    ))
    print("Driver PID: {} | жив: {}".format(
        status.get("driver_pid"), pid_alive(status.get("driver_pid"))
    ))
    print("Последний heartbeat: {}".format(status.get("last_heartbeat")))
    if "exit_code" in status:
        print("Exit code: {} | сигнал: {}".format(
            status.get("exit_code"), status.get("signal")
        ))
    print("RAM driver: {}".format(status.get("driver_memory", {})))
    tree = status.get("process_tree", {})
    print("RAM дерева driver+workers: {} KiB".format(
        tree.get("rss_total_kib") if isinstance(tree, dict) else "нет данных"
    ))
    print("Cgroup memory/лимит/OOM: {}".format(status.get("cgroup_memory", {})))
    print("RAM системы: {}".format(status.get("system_memory", {})))
    print("GPU: {}".format(status.get("gpu", "нет heartbeat")))
    driver_log = Path(str(status.get("driver_log", "")))
    print("Полный лог: {}".format(driver_log))
    tail = last_lines(driver_log, int(lines_value))
    print("\n--- ПОСЛЕДНИЕ {} СТРОК DRIVER LOG ---".format(len(tail)))
    print("".join(tail), end="")
    if not tail or state == "SUPERVISOR_DIED":
        for label, key in (
            ("SUPERVISOR LOG", "supervisor_log"),
            ("SUPERVISOR BOOT LOG", "supervisor_boot_log"),
        ):
            diagnostic = Path(str(status.get(key, "")))
            diagnostic_tail = last_lines(diagnostic, min(int(lines_value), 40))
            if diagnostic_tail:
                print("\n--- {} ---".format(label))
                print("".join(diagnostic_tail), end="")
    return status


def detach_current_script(
    mode: str, arguments: Sequence[str], script_path: Optional[Path] = None,
) -> bool:
    """Detach a `%run` invocation before any heavy native imports occur."""
    if __name__ == "__main__":
        return False
    if os.environ.get("CASHGAP_EXTERNAL_DRIVER") == "1":
        return False
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
    except ImportError:
        return False
    if script_path is None:
        launch_mode(mode, arguments)
    else:
        launch_script(mode, script_path, arguments)
    return True


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        print("Режимы: benchmark, full-benchmark, autotune, status")
        return 0
    mode = sys.argv[1]
    arguments = sys.argv[2:]
    if mode == "status":
        show_status(arguments)
        return 0
    launch_mode(mode, arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
