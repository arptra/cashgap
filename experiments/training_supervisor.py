#!/usr/bin/env python3
"""Internal detached supervisor for cash-gap experiment drivers.

This module intentionally imports only the Python standard library.  It keeps
running outside Jupyter, captures every driver byte on disk, records heartbeats
and preserves the native exit signal even when the driver cannot print a Python
traceback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def append_line(path: Path, value: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(value.rstrip() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_memory(pid: int) -> Dict[str, str]:
    status = Path("/proc") / str(pid) / "status"
    wanted = {"VmRSS", "VmPeak", "VmSize", "Threads"}
    result: Dict[str, str] = {}
    try:
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in wanted:
                result[key] = value.strip()
    except OSError:
        pass
    return result


def system_memory() -> Dict[str, str]:
    source = Path("/proc/meminfo")
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    result: Dict[str, str] = {}
    try:
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in wanted:
                result[key] = value.strip()
    except OSError:
        pass
    return result


def process_tree(root_pid: int) -> Dict[str, object]:
    """Return Linux descendant RSS so child workers cannot hide from diagnostics."""
    proc = Path("/proc")
    if not proc.exists():
        return {"processes": [], "rss_total_kib": None}
    parents: Dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            suffix = stat[stat.rfind(")") + 2:].split()
            parents[int(entry.name)] = int(suffix[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    rows: List[Dict[str, object]] = []
    total_kib = 0
    for pid in sorted(selected):
        memory = process_memory(pid)
        rss_text = memory.get("VmRSS", "0 kB").split()[0]
        try:
            total_kib += int(rss_text)
        except ValueError:
            pass
        try:
            raw_command = (proc / str(pid) / "cmdline").read_bytes()
            command = raw_command.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            command = ""
        rows.append({
            "pid": pid,
            "ppid": parents.get(pid),
            "memory": memory,
            "command": command[:500],
        })
    return {"processes": rows, "rss_total_kib": total_kib}


def cgroup_memory(pid: int) -> Dict[str, str]:
    """Expose the real container/Jupyter memory limit and OOM counters on Linux."""
    try:
        entries = (Path("/proc") / str(pid) / "cgroup").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return {}
    relative = None
    version = "v2"
    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            relative = parts[2].lstrip("/")
            break
    if relative is None:
        version = "v1"
        for entry in entries:
            parts = entry.split(":", 2)
            if len(parts) == 3 and "memory" in parts[1].split(","):
                relative = parts[2].lstrip("/")
                break
    if relative is None:
        return {}
    root = (
        Path("/sys/fs/cgroup") / relative
        if version == "v2"
        else Path("/sys/fs/cgroup/memory") / relative
    )
    result: Dict[str, str] = {"version": version, "path": str(root)}
    names = (
        ("memory.current", "memory.max", "memory.events", "memory.oom.group")
        if version == "v2"
        else (
            "memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes",
            "memory.failcnt", "memory.oom_control",
        )
    )
    for name in names:
        try:
            result[name] = (root / name).read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return result


def gpu_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=15, check=False
        )
        return (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "nvidia-smi недоступен: {}".format(error)


def signal_name(return_code: int) -> Optional[str]:
    if return_code >= 0:
        return None
    try:
        return signal.Signals(-return_code).name
    except ValueError:
        return "SIGNAL_{}".format(-return_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("после -- нужна команда driver")
    return args


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    status_file = Path(args.status_file).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = run_dir / "supervisor.log"
    driver_log = run_dir / "driver.log"
    heartbeat_log = run_dir / "heartbeat.log"
    command_file = run_dir / "command.json"
    command_file.write_text(
        json.dumps(args.command, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_status: Dict[str, object] = {
        "state": "STARTING",
        "started_at": utc_now(),
        "supervisor_pid": os.getpid(),
        "driver_pid": None,
        "workdir": str(Path(args.workdir).resolve()),
        "run_dir": str(run_dir),
        "driver_log": str(driver_log),
        "heartbeat_log": str(heartbeat_log),
        "supervisor_log": str(supervisor_log),
        "supervisor_boot_log": str(run_dir / "supervisor_boot.log"),
        "command_file": str(command_file),
        "command": list(args.command),
    }
    atomic_json(status_file, base_status)
    append_line(
        supervisor_log,
        "{} supervisor START pid={} Python={} OS={}".format(
            utc_now(), os.getpid(), sys.version.split()[0], platform.platform()
        ),
    )

    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "CASHGAP_EXTERNAL_DRIVER": "1",
    })
    with driver_log.open("w", encoding="utf-8", buffering=1) as output:
        output.write("CASHGAP EXTERNAL DRIVER\n")
        output.write("started_at: {}\n".format(base_status["started_at"]))
        output.write("workdir: {}\n".format(base_status["workdir"]))
        output.write("command: {}\n\n".format(json.dumps(args.command, ensure_ascii=False)))
        output.flush()
        os.fsync(output.fileno())
        try:
            driver = subprocess.Popen(
                args.command,
                cwd=args.workdir,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except BaseException as error:
            failed = dict(base_status)
            failed.update({
                "state": "FAILED_TO_START",
                "finished_at": utc_now(),
                "error": repr(error),
            })
            output.write("FAILED_TO_START: {!r}\n".format(error))
            output.flush()
            os.fsync(output.fileno())
            atomic_json(status_file, failed)
            append_line(supervisor_log, "{} FAILED_TO_START {!r}".format(utc_now(), error))
            return 127

        status = dict(base_status)
        status.update({"state": "RUNNING", "driver_pid": driver.pid, "last_heartbeat": utc_now()})
        atomic_json(status_file, status)
        append_line(supervisor_log, "{} driver START pid={}".format(utc_now(), driver.pid))

        stop_signal: List[Optional[int]] = [None]

        def request_stop(number, _frame) -> None:
            stop_signal[0] = int(number)

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        next_heartbeat = 0.0
        while driver.poll() is None:
            if stop_signal[0] is not None:
                append_line(
                    supervisor_log,
                    "{} supervisor received {}; stopping driver".format(
                        utc_now(), signal_name(-int(stop_signal[0]))
                    ),
                )
                try:
                    os.killpg(driver.pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    driver.terminate()
                try:
                    driver.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(driver.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        driver.kill()
                    driver.wait()
                break
            now = time.monotonic()
            if now >= next_heartbeat:
                heartbeat = {
                    "at": utc_now(),
                    "driver_pid": driver.pid,
                    "driver_memory": process_memory(driver.pid),
                    "process_tree": process_tree(driver.pid),
                    "cgroup_memory": cgroup_memory(driver.pid),
                    "system_memory": system_memory(),
                    "driver_log_bytes": driver_log.stat().st_size if driver_log.exists() else 0,
                    "gpu": gpu_snapshot(),
                }
                append_line(heartbeat_log, json.dumps(heartbeat, ensure_ascii=False))
                status.update({
                    "last_heartbeat": heartbeat["at"],
                    "driver_memory": heartbeat["driver_memory"],
                    "process_tree": heartbeat["process_tree"],
                    "cgroup_memory": heartbeat["cgroup_memory"],
                    "system_memory": heartbeat["system_memory"],
                    "gpu": heartbeat["gpu"],
                })
                atomic_json(status_file, status)
                next_heartbeat = now + 30.0
            time.sleep(2)

        return_code = int(driver.wait())
        output.write("\nDRIVER EXIT | code={} | signal={} | at={}\n".format(
            return_code, signal_name(return_code), utc_now()
        ))
        output.flush()
        os.fsync(output.fileno())

    finished = dict(status)
    finished.update({
        "state": "COMPLETED" if return_code == 0 else "FAILED",
        "finished_at": utc_now(),
        "exit_code": return_code,
        "signal": signal_name(return_code),
    })
    if stop_signal[0] is not None:
        finished["state"] = "STOPPED"
        finished["requested_signal"] = signal_name(-int(stop_signal[0]))
    atomic_json(status_file, finished)
    append_line(
        supervisor_log,
        "{} driver EXIT code={} signal={}".format(
            utc_now(), return_code, signal_name(return_code)
        ),
    )
    return 0 if return_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
