"""Status monitor for a running underfit dashboard.

Polls /api/runs, /api/datasets, /api/gradio and renders a compact summary —
either as an in-place HTML block (in IPython / Colab) or as a tailing
text stream (in a terminal).

Importable:
    from underfit.monitor import start_monitor
    start_monitor(interval=10)              # block, refresh every 10 s

CLI:
    python -m underfit.monitor              # auto-detect output mode
    python -m underfit.monitor --interval 30 --mode text
"""
from __future__ import annotations

import html as _html
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


def _get(url: str, fallback: Any) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            TimeoutError, OSError):
        return fallback


_PEAKS: dict[str, float] = {}  # session peaks (RAM in GB, per-GPU VRAM in MB)


def _tail_lines(path: str, n: int = 3) -> list[str]:
    """Return the last `n` non-blank lines of `path` (best-effort).

    Reads at most 16 KB from the end so it stays cheap even on multi-MB logs.
    On Colab the log lives on Drive (FUSE) — small tail reads are fine.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return lines[-n:]


def _disk_usage(path: str) -> dict | None:
    """Return {used_gb, total_gb, pct, path} or None if path can't be statfs'd.
    Catches FUSE/timeout/permission errors so a stale Drive mount can't
    break the whole monitor."""
    import shutil
    try:
        u = shutil.disk_usage(path)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if u.total <= 0:
        return None
    return {
        "path": path,
        "used_gb": round(u.used / 1e9, 1),
        "total_gb": round(u.total / 1e9, 1),
        "pct": round(100 * u.used / u.total, 1),
    }


def _disks() -> list[dict]:
    """Disk usage for the local filesystem and (if mounted) Google Drive.
    Each entry: {label, path, used_gb, total_gb, pct, level}.
    level ∈ {'ok', 'warn', 'critical'} based on pct."""
    out = []
    # Local: prefer /content (Colab convention), fall back to root.
    local_path = "/content" if os.path.isdir("/content") else "/"
    local = _disk_usage(local_path)
    if local is not None:
        local["label"] = "Local"
        out.append(local)
    # Google Drive — only when actually mounted. A bare /content/drive folder
    # without MyDrive means Drive didn't mount.
    drive_path = "/content/drive/MyDrive"
    if os.path.isdir(drive_path):
        drive = _disk_usage(drive_path)
        if drive is not None:
            drive["label"] = "Drive"
            out.append(drive)
    for d in out:
        d["level"] = "critical" if d["pct"] >= 90 else "warn" if d["pct"] >= 80 else "ok"
    return out


def _latest_run_log(runs: list[dict]) -> dict | None:
    """Pick the run whose log file mtime is most recent. None if no run has
    a readable log on disk. Carries kill_hint when the dashboard set one
    (run died fast — likely OOM/CUDA error/shell-fail)."""
    best = None
    best_run = None
    for r in runs:
        lp = r.get("log_path")
        if not lp:
            continue
        try:
            mt = os.path.getmtime(lp)
        except OSError:
            continue
        if best is None or mt > best["mtime"]:
            best = {
                "name": r.get("name") or r.get("id") or "?",
                "path": lp,
                "mtime": mt,
                "status": r.get("status", "?"),
            }
            best_run = r
    if best is not None and best_run is not None:
        if best_run.get("kill_hint"):
            best["kill_hint"] = best_run["kill_hint"]
        if best_run.get("error"):
            best["error"] = best_run["error"]
    return best


def fetch_status(base_url: str = "http://localhost:8787") -> dict[str, Any]:
    """Snapshot of dashboard state + local system RAM + GPU VRAM (via dashboard).
    Returns `{'error': msg}` if dashboard is unreachable."""
    runs_resp = _get(f"{base_url}/api/runs", None)
    if runs_resp is None:
        return {"error": f"can't reach {base_url}"}
    runs = runs_resp if isinstance(runs_resp, list) else runs_resp.get("runs", [])

    datasets_resp = _get(f"{base_url}/api/datasets", {})
    if isinstance(datasets_resp, dict):
        datasets = datasets_resp.get("datasets", [])
    else:
        datasets = datasets_resp or []

    gradios_resp = _get(f"{base_url}/api/gradio", {})
    if isinstance(gradios_resp, dict):
        gradios = gradios_resp.get("instances", gradios_resp.get("gradios", []))
    else:
        gradios = gradios_resp or []

    status_counts: dict[str, int] = {}
    for r in runs:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    active_gradios = sum(1 for g in gradios if g.get("status") in ("ready", "starting"))

    # System RAM via psutil (we share a VM with the dashboard, so local read is fine)
    ram = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / 1e9
        ram = {"used_gb": round(used_gb, 2), "total_gb": round(vm.total / 1e9, 2)}
        _PEAKS["ram_gb"] = max(_PEAKS.get("ram_gb", 0.0), used_gb)
        ram["peak_gb"] = round(_PEAKS["ram_gb"], 2)
    except Exception:
        pass

    # GPU VRAM via dashboard's /api/gpu — list of {gpu, used_mb, total_mb, util_pct, ...}
    gpu_resp = _get(f"{base_url}/api/gpu", {})
    if isinstance(gpu_resp, dict):
        gpus = gpu_resp.get("gpus", [])
    else:
        gpus = gpu_resp or []
    for g in gpus:
        idx = g.get("gpu")
        if idx is None:
            continue
        key = f"gpu_{idx}_mb"
        _PEAKS[key] = max(_PEAKS.get(key, 0.0), g.get("used_mb", 0))
        g["peak_mb"] = int(_PEAKS[key])

    # Tail of the most-recently-modified run log
    latest = _latest_run_log(runs)
    if latest is not None:
        latest["tail"] = _tail_lines(latest["path"], 3)

    return {
        "runs": runs,
        "datasets": datasets,
        "gradios": gradios,
        "runs_total": len(runs),
        "dataset_count": len(datasets),
        "active_gradios": active_gradios,
        "status_counts": status_counts,
        "ram": ram,
        "gpus": gpus,
        "disks": _disks(),
        "latest_run_log": latest,
    }


def format_text(data: dict[str, Any]) -> str:
    """Compact one-liner for terminal use."""
    ts = datetime.now().strftime("%H:%M:%S")
    if "error" in data:
        return f"[{ts}] dashboard unreachable ({data['error']})"
    sc = data["status_counts"]
    runs_str = f"{data['runs_total']} total"
    if sc:
        runs_str += f" ({', '.join(f'{n} {s}' for s, n in sorted(sc.items()))})"
    parts = [
        f"[{ts}] runs: {runs_str}",
        f"datasets: {data['dataset_count']}",
        f"gradio: {data['active_gradios']} active",
    ]
    if data.get("ram"):
        r = data["ram"]
        parts.append(f"RAM: {r['used_gb']:.1f}/{r['total_gb']:.1f} GB (peak {r['peak_gb']:.1f})")
    for g in data.get("gpus", []):
        gu = g["used_mb"] / 1024
        gt = g["total_mb"] / 1024
        gp = g.get("peak_mb", 0) / 1024
        parts.append(f"GPU{g['gpu']}: {gu:.1f}/{gt:.1f} GB (peak {gp:.1f}, {g.get('util_pct', 0)}%)")
    for d in data.get("disks", []):
        warn = " ⚠" if d["level"] == "critical" else ""
        parts.append(f"{d['label']}: {d['used_gb']:.1f}/{d['total_gb']:.1f} GB ({d['pct']:.0f}%){warn}")
    out = " | ".join(parts)
    log = data.get("latest_run_log")
    if log:
        status = log.get("status", "")
        header = f"\n  📜 {log['name']} [{status}] ({os.path.basename(log['path'])})"
        if log.get("kill_hint"):
            header += f"\n     ⚠ {log['kill_hint']}"
        if log.get("error"):
            header += f"\n     error: {log['error']}"
        out += header
        for line in log.get("tail", []) or []:
            out += f"\n     {line}"
    return out


def format_html(data: dict[str, Any]) -> str:
    """Multi-line HTML block for IPython.display.HTML."""
    ts = datetime.now().strftime("%H:%M:%S")
    if "error" in data:
        return (f"<pre style='margin:0;font-family:monospace;color:#a00'>"
                f"[{ts}] dashboard unreachable ({data['error']})</pre>")
    sc = ", ".join(f"{n} {s}" for s, n in sorted(data["status_counts"].items())) or "none"
    lines = [
        f"<b>📊 underfit @ {ts}</b>",
        f"  Runs:     {data['runs_total']} total ({sc})",
        f"  Datasets: {data['dataset_count']}",
        f"  Gradios:  {data['active_gradios']} active",
    ]
    if data.get("ram"):
        r = data["ram"]
        pct = 100 * r["used_gb"] / max(r["total_gb"], 0.001)
        lines.append(f"  RAM:      {r['used_gb']:5.1f} / {r['total_gb']:5.1f} GB  "
                     f"({pct:4.1f}%, peak {r['peak_gb']:5.1f} GB)")
    for g in data.get("gpus", []):
        gu = g["used_mb"] / 1024
        gt = g["total_mb"] / 1024
        gp = g.get("peak_mb", 0) / 1024
        pct = 100 * gu / max(gt, 0.001)
        lines.append(f"  GPU {g['gpu']:>2}:    {gu:5.1f} / {gt:5.1f} GB  "
                     f"({pct:4.1f}%, peak {gp:5.1f} GB, util {g.get('util_pct', 0):>3}%)")
    for d in data.get("disks", []):
        color = {"critical": "#e44", "warn": "#dc4", "ok": ""}[d["level"]]
        warn = " ⚠ near full" if d["level"] == "critical" else \
               " (running low)" if d["level"] == "warn" else ""
        style = f" style='color:{color}'" if color else ""
        label = (d["label"] + ":").ljust(8)
        lines.append(
            f"  <span{style}>{label}  {d['used_gb']:5.1f} / {d['total_gb']:5.1f} GB  "
            f"({d['pct']:4.1f}%){warn}</span>"
        )
    log = data.get("latest_run_log")
    if log:
        lines.append("")
        status = log.get("status", "")
        lines.append(
            f"<span style='color:#888'>📜 <b>{_html.escape(str(log['name']))}</b> "
            f"<span style='color:#aaa'>[{_html.escape(status)}]</span> "
            f"<span style='color:#666'>({_html.escape(os.path.basename(log['path']))})"
            f"</span></span>"
        )
        if log.get("kill_hint"):
            lines.append(
                f"   <span style='color:#fa6'>⚠ {_html.escape(str(log['kill_hint']))}</span>"
            )
        if log.get("error"):
            lines.append(
                f"   <span style='color:#e66'>error: {_html.escape(str(log['error']))}</span>"
            )
        for line in log.get("tail", []) or []:
            lines.append(f"   <span style='color:#aaa'>{_html.escape(line)}</span>")
    return ("<pre style='margin:0;font-family:monospace;line-height:1.4;"
            "white-space:pre-wrap;word-break:break-word'>"
            + "\n".join(lines) + "</pre>")


def _detect_mode() -> str:
    """'html' if running inside IPython/Jupyter/Colab, else 'text'."""
    try:
        from IPython import get_ipython
        return "html" if get_ipython() is not None else "text"
    except ImportError:
        return "text"


def dashboard_button(url: str | None = None,
                     port: int = 8787,
                     label: str = "🚀 Open underfit Dashboard") -> str | None:
    """Render a big clickable button in the notebook output that opens the dashboard.

    If `url` is provided (e.g. an ngrok public URL), it's used directly.
    Otherwise in Colab, we derive the proxied URL for `port` via
    `google.colab.kernel.proxyPort`. Returns the URL it rendered, or None if
    rendering wasn't possible (e.g. outside IPython).
    """
    try:
        from IPython.display import display, HTML
    except ImportError:
        return None

    if url is None:
        try:
            from google.colab.output import eval_js
            url = eval_js(f"google.colab.kernel.proxyPort({port})")
        except ImportError:
            url = f"http://localhost:{port}"

    display(HTML(f"""
<div style="margin: 24px 0; text-align: center;">
  <a href="{url}" target="_blank" rel="noopener" style="
    display: inline-block;
    background: linear-gradient(135deg, #ff6b35 0%, #ff9248 100%);
    color: white;
    padding: 18px 52px;
    border-radius: 12px;
    font-size: 19px;
    font-weight: 700;
    text-decoration: none;
    box-shadow: 0 6px 22px rgba(255, 107, 53, 0.35);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    letter-spacing: 0.3px;
  ">
    {label}
  </a>
  <div style="margin-top: 10px; font-size: 12px; color: #888; font-family: monospace;">
    {url}
  </div>
</div>
"""))
    return url


def restart_dashboard_button(port: int = 8787) -> None:
    """Render a note telling the user to restart the dashboard by re-running
    this cell.

    Previously a button, but Colab queues button callbacks behind any
    currently-running cell (e.g. the Step 5 monitor), so the button often
    appeared dead until the user stopped the monitor first. Re-running this
    cell calls launch_dashboard_subprocess again, which kills any existing
    process on `port` and starts a fresh one. Training runs are unaffected —
    they're separate detached processes managed by RunsRegistry.

    `port` is accepted for API compatibility with the previous button-version
    callers but is unused — the actual restart now happens via cell re-run.
    """
    del port  # unused; kept for back-compat
    try:
        from IPython.display import display, HTML
    except ImportError:
        return None

    display(HTML(
        "<div style='text-align:center;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",system-ui,sans-serif;"
        "margin:8px 0 18px 0;'>"
        "<div style='font-size:16px;font-weight:600;color:var(--colab-primary-text-color,#444)'>"
        "Restart dashboard by restarting this cell."
        "</div>"
        "<div style='font-size:12px;color:var(--colab-secondary-text-color,#888);margin-top:4px'>"
        "* doesn't affect training runs. Do this if dashboard freezes."
        "</div>"
        "</div>"
    ))


# ── Drive state-file sync ───────────────────────────────────────────────────
# State JSON files (runs.json, datasets.json, gradio_*.json) live on local SSD
# for fast dashboard reads/writes. This thread mirrors them to Drive every ~10s
# so a Colab session reset doesn't lose dashboard state.
#
# Atomicity: server.py uses _atomic_write_json (tmp + os.replace). Sync reads
# whole files via shutil.copy2 — between server's tmp-write and rename, the
# real file is unchanged, so we never read torn JSON.

_state_sync_stop: "threading.Event | None" = None  # type: ignore[name-defined]
_state_sync_thread = None  # threading.Thread

# State files we sync between local SSD and Drive. Anything not in this list
# stays where it is (e.g. seed_loras/, runs/, datasets/, audio/).
_STATE_FILE_NAMES = (
    "runs.json",
    "datasets.json",
    "gradio_instances.json",
    "gradio_vram_estimate.json",
)


def _atomic_copy(src, dst):
    """shutil.copy2 → tmp + os.replace, so a reader on `dst` never sees a
    half-copied file."""
    import shutil
    from pathlib import Path
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".sync.tmp")
    try:
        shutil.copy2(str(src), str(tmp))
        os.replace(str(tmp), str(dst))
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sync_state_files(src_dir, dst_dir) -> int:
    """Copy each state file from src to dst when src is newer. Returns count
    of files actually copied this pass."""
    from pathlib import Path
    n = 0
    for name in _STATE_FILE_NAMES:
        src = Path(src_dir) / name
        if not src.exists():
            continue
        dst = Path(dst_dir) / name
        try:
            smt = src.stat().st_mtime
            dmt = dst.stat().st_mtime if dst.exists() else 0.0
        except OSError:
            continue
        if smt > dmt + 0.5:
            try:
                Path(dst_dir).mkdir(parents=True, exist_ok=True)
                _atomic_copy(src, dst)
                n += 1
            except OSError:
                pass
    return n


def start_drive_state_sync(
    local_dir: str = "/content/underfit-state",
    drive_dir: str = "/content/drive/MyDrive/underfit-state",
    interval: int = 10,
) -> None:
    """Mirror state JSON files between SSD and Drive.

    SYNCHRONOUS cold-start seed runs BEFORE this function returns — call this
    in Cell 4 *before* `launch_dashboard_subprocess` so the dashboard's
    RunsRegistry/_load can read a populated runs.json on startup.

    After seeding, a daemon thread does Drive→local once (in case Drive has
    newer state from another session) and then local→Drive every `interval`
    seconds for the rest of the kernel's life.

    Calling again stops the prior sync thread and starts fresh — composes
    cleanly with `launch_dashboard_subprocess`'s restart pattern.
    """
    import threading
    global _state_sync_stop, _state_sync_thread

    # Stop any prior sync thread first
    if _state_sync_stop is not None:
        _state_sync_stop.set()
        if _state_sync_thread is not None:
            _state_sync_thread.join(timeout=5)
        _state_sync_stop = None
        _state_sync_thread = None

    from pathlib import Path
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # --- SYNCHRONOUS COLD-START SEED ---
    # Copy Drive → local for any state file where Drive is newer (or local is
    # missing). Returns after seed completes — guarantees runs.json etc. are
    # readable by the dashboard process when it starts up.
    seeded = _sync_state_files(drive_dir, local_dir)
    if seeded:
        print(f"[state-sync] seeded {seeded} state file(s) from Drive → local SSD",
              flush=True)

    stop = threading.Event()

    def _sync_loop():
        while not stop.wait(interval):
            try:
                _sync_state_files(local_dir, drive_dir)
            except Exception as e:
                print(f"[state-sync] error: {type(e).__name__}: {e}", flush=True)

    _state_sync_stop = stop
    _state_sync_thread = threading.Thread(
        target=_sync_loop, daemon=True, name="underfit-state-sync"
    )
    _state_sync_thread.start()
    print(
        f"[state-sync] running every {interval}s: {local_dir} ⇄ {drive_dir}",
        flush=True,
    )


# ── Drive log sync ──────────────────────────────────────────────────────────
# Training logs live on local SSD for fast dashboard reads. A background thread
# in the notebook kernel periodically copies them to Drive for durability across
# session resets. Single-singleton pattern: calling start_drive_log_sync again
# stops the prior thread and starts a new one (lifecycle tied to dashboard
# restart, which happens in the same launch cell).

_log_sync_stop: "threading.Event | None" = None  # type: ignore[name-defined]
_log_sync_thread = None  # threading.Thread


def _sync_logs_local_to_drive(local_dir, drive_dir) -> int:
    """One pass: copy each *.log* in local_dir to drive_dir when local is
    newer. Returns the number of files actually copied."""
    import shutil
    from pathlib import Path
    n = 0
    if not Path(local_dir).exists() or not Path(drive_dir).exists():
        return 0
    for local_file in Path(local_dir).glob("*.log*"):
        drive_file = Path(drive_dir) / local_file.name
        try:
            lmt = local_file.stat().st_mtime
            dmt = drive_file.stat().st_mtime if drive_file.exists() else 0.0
        except OSError:
            continue
        if lmt > dmt + 0.5:  # 0.5s slack to avoid clock-skew false-positives
            try:
                shutil.copy2(str(local_file), str(drive_file))
                n += 1
            except OSError:
                pass
    return n


def _seed_local_logs_from_drive(local_dir, drive_dir) -> int:
    """Cold-start: copy each *.log* on Drive to local SSD if local is missing
    or older. Lets the dashboard browse old-session logs at SSD speed and
    keeps runs.json log_path references resolvable when paths from a prior
    session still point at the Drive copy."""
    import shutil
    from pathlib import Path
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    if not Path(drive_dir).exists():
        return 0
    n = 0
    for drive_file in Path(drive_dir).glob("*.log*"):
        local_file = Path(local_dir) / drive_file.name
        try:
            dmt = drive_file.stat().st_mtime
            lmt = local_file.stat().st_mtime if local_file.exists() else 0.0
        except OSError:
            continue
        if dmt > lmt + 0.5:
            try:
                shutil.copy2(str(drive_file), str(local_file))
                n += 1
            except OSError:
                pass
    return n


def start_drive_log_sync(
    local_dir: str = "/content/underfit-logs",
    drive_dir: str = "/content/drive/MyDrive/underfit-state/runs",
    interval: int = 60,
) -> None:
    """Start a background thread that periodically copies live training logs
    from `local_dir` (fast SSD writes) to `drive_dir` (durable across Colab
    session resets).

    Lifecycle: calling this function stops any prior sync thread first, so it
    composes cleanly with `launch_dashboard_subprocess` — call it once after
    every dashboard launch/restart.

    On the very first call this session, also does a one-time Drive → local
    seed so `runs.json` log_path fields written in prior sessions still
    resolve when the dashboard reads them.

    Runs in the *notebook* kernel (not the dashboard subprocess), so it
    survives dashboard restarts and is independent of the Step 5 diagnostic
    monitor (which the user can stop without affecting Drive backups).
    """
    import threading
    global _log_sync_stop, _log_sync_thread

    # Stop any prior sync first
    if _log_sync_stop is not None:
        _log_sync_stop.set()
        if _log_sync_thread is not None:
            _log_sync_thread.join(timeout=5)
        _log_sync_stop = None
        _log_sync_thread = None

    # Cold-start seed (fast no-op if local already has the files)
    seeded = _seed_local_logs_from_drive(local_dir, drive_dir)
    if seeded:
        print(f"[log-sync] seeded {seeded} log file(s) from Drive → {local_dir}", flush=True)

    stop = threading.Event()

    def _sync_loop():
        # First sync runs after `interval` seconds — fresh local writes
        # haven't accumulated yet, no rush.
        while not stop.wait(interval):
            try:
                _sync_logs_local_to_drive(local_dir, drive_dir)
            except Exception as e:
                print(f"[log-sync] error: {type(e).__name__}: {e}", flush=True)

    _log_sync_stop = stop
    _log_sync_thread = threading.Thread(
        target=_sync_loop, daemon=True, name="underfit-log-sync"
    )
    _log_sync_thread.start()
    print(
        f"[log-sync] running every {interval}s: {local_dir} → {drive_dir}",
        flush=True,
    )


def launch_dashboard_subprocess(*, port: int = 8787,
                                server_script: str = "dashboard/server.py",
                                wait_for_ready: bool = True,
                                kill_existing: bool = True,
                                quiet: bool = False):
    """Launch dashboard/server.py as a detached background subprocess.

    - If `kill_existing`, runs `fuser -k <port>/tcp` first to clear any stale
      process bound to the port.
    - `start_new_session=True` keeps the subprocess alive when the notebook
      cell that started it is interrupted (Ctrl+C / ⏹).
    - If `wait_for_ready`, drains stdout until the "Dashboard running on …"
      ready-marker so callers know the HTTP server is up.
    - If `quiet`, drains stdout silently rather than echoing it — useful when
      restarting from inside an existing cell so the log doesn't duplicate.

    Returns the `subprocess.Popen` handle. Poll `proc.returncode` to detect
    crashes; it's None while alive.
    """
    import subprocess

    if kill_existing:
        subprocess.run(["fuser", "-k", f"{port}/tcp"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)

    proc = subprocess.Popen(
        ["uv", "run", "python", "-u", server_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,
    )

    if wait_for_ready:
        for line in iter(proc.stdout.readline, ""):
            if not quiet:
                print(line, end="")
            if "Dashboard running on" in line:
                break

    return proc


class MonitorHandle:
    """Returned by `start_monitor(background=True)`. Call `.stop()` to halt
    the background polling thread. `.is_alive()` reports whether it's still
    running. Re-entrant: `.stop()` on an already-stopped handle is a no-op."""
    def __init__(self, thread: threading.Thread, stop_event: threading.Event):
        self._thread = thread
        self._stop = stop_event

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def __repr__(self) -> str:
        state = "alive" if self.is_alive() else "stopped"
        return f"<MonitorHandle {state}>"


def _monitor_loop(base_url, interval, mode, on_unreachable, failure_threshold,
                  stop_event):
    """Shared body of start_monitor (foreground + background). `stop_event`
    is None for foreground (use KeyboardInterrupt) and a threading.Event for
    background (set by MonitorHandle.stop)."""
    display_id = "underfit-status"
    if mode == "html":
        from IPython.display import display, update_display, HTML
        display(HTML(format_html(fetch_status(base_url))), display_id=display_id)
    else:
        print(format_text(fetch_status(base_url)), flush=True)

    failures = 0
    try:
        while True:
            if stop_event is not None:
                # wait() with timeout: returns True when set (→ stop),
                # False on timeout (→ continue polling)
                if stop_event.wait(timeout=interval):
                    break
            else:
                time.sleep(interval)
            data = fetch_status(base_url)
            if "error" in data:
                failures += 1
                if on_unreachable is not None and failures >= failure_threshold:
                    print(f"\n⚠️  Dashboard unreachable for {failures} consecutive polls "
                          f"— invoking on_unreachable handler …", flush=True)
                    try:
                        on_unreachable()
                    except Exception as e:
                        print(f"on_unreachable handler raised {type(e).__name__}: {e}",
                              flush=True)
                    failures = 0
            else:
                failures = 0

            if mode == "html":
                from IPython.display import update_display, HTML
                update_display(HTML(format_html(data)), display_id=display_id)
            else:
                print(format_text(data), flush=True)
    except KeyboardInterrupt:
        print("\nStatus monitor stopped.")


def start_monitor(base_url: str = "http://localhost:8787",
                  interval: int = 10,
                  mode: str = "auto",
                  on_unreachable=None,
                  failure_threshold: int = 3,
                  background: bool = False):
    """Poll the dashboard and refresh a status block.

    mode="auto" (default) renders HTML in IPython and text in a terminal.
    Force a specific mode with mode="html" or mode="text".

    on_unreachable: optional callable() invoked when the dashboard has been
    unreachable for `failure_threshold` consecutive polls (default 3 — so ~30s
    of failures at the default 10s interval before triggering). Use it to
    auto-restart the dashboard. Exceptions raised by the callable are caught
    and printed; the monitor keeps running either way.

    background=False (default) blocks the calling cell/process until
    KeyboardInterrupt — old behavior. background=True runs the polling loop
    in a daemon thread and returns a MonitorHandle immediately, leaving the
    kernel idle so other Colab cells (and ipywidgets buttons like Restart
    Dashboard) can fire while polling continues. Call `.stop()` on the
    handle to halt polling.
    """
    if mode == "auto":
        mode = _detect_mode()

    if background:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_monitor_loop,
            args=(base_url, interval, mode, on_unreachable, failure_threshold, stop_event),
            daemon=True,
            name="underfit-monitor",
        )
        thread.start()
        return MonitorHandle(thread, stop_event)

    _monitor_loop(base_url, interval, mode, on_unreachable, failure_threshold, None)
    return None


def debug_info(state_dir: str = "/content/drive/MyDrive/underfit-state",
               repo_dir: str = "/content/underfit",
               n_runs: int = 1) -> None:
    """Print a comprehensive diagnostic snapshot for the most recent run(s).

    Designed for Colab: drop `from underfit.monitor import debug_info; debug_info()`
    in a cell and paste the output when reporting a bug.

    Includes:
      - git commit + dirty state of the underfit checkout
      - python interpreter, torch+CUDA build, supported archs
      - all GPUs (nvidia-smi: name, compute_cap, driver, memory)
      - dashboard process PID + start time
      - disk usage (local + Drive when mounted)
      - the N most recent runs: sibling files (log + sidecars), contents
        of each non-JSON sidecar, files inside the run dir, runs.json
        record

    Failures in any section are caught and printed inline — the rest of the
    report still runs.
    """
    import glob
    import shutil
    import subprocess

    def _section(title):
        print(f"\n{'='*4} {title} {'='*4}")

    def _safe_run(cmd, **kw):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, **kw).strip()
        except subprocess.CalledProcessError as e:
            return f"<{cmd[0]} failed: {e.output.strip()[:200]}>"
        except FileNotFoundError:
            return f"<{cmd[0]} not on PATH>"
        except Exception as e:
            return f"<{type(e).__name__}: {e}>"

    # ── repo ────────────────────────────────────────────────────────────
    _section("underfit repo")
    print(f"  path: {repo_dir}")
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        head = _safe_run(["git", "-C", repo_dir, "log", "-1", "--oneline"])
        print(f"  HEAD: {head}")
        status = _safe_run(["git", "-C", repo_dir, "status", "--porcelain"])
        print(f"  dirty files: {len(status.splitlines()) if status else 0}")
    else:
        print(f"  (not a git checkout)")

    # ── python / torch / CUDA ───────────────────────────────────────────
    _section("python + torch")
    print(f"  python: {sys.executable}")
    print(f"  version: {sys.version.split()[0]}")
    try:
        import torch
        print(f"  torch:  {torch.__version__}  (CUDA {torch.version.cuda})")
        print(f"  archs:  {torch.cuda.get_arch_list()}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        print(f"  mps available:  {torch.backends.mps.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                maj, mnr = torch.cuda.get_device_capability(i)
                print(f"  GPU {i}: {name} (sm{maj}{mnr})")
    except Exception as e:
        print(f"  torch import failed: {type(e).__name__}: {e}")

    # ── nvidia-smi (independent of torch) ──────────────────────────────
    _section("nvidia-smi")
    smi = _safe_run([
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    print(f"  {smi}" if "\n" not in smi else smi)

    # ── dashboard process ───────────────────────────────────────────────
    _section("dashboard process")
    # Tight match: only real python interpreters running dashboard/server.py.
    # Loose pgrep would also catch shell wrappers or unrelated processes.
    ps = _safe_run(["pgrep", "-af", r"python[^[:space:]]* .*dashboard/server\.py"])
    print(ps or "  (no dashboard/server.py process found)")

    # ── disk usage ──────────────────────────────────────────────────────
    _section("disk usage")
    for label, path in (("Local /content", "/content"),
                        ("Local /",        "/"),
                        ("Drive MyDrive",  "/content/drive/MyDrive")):
        if os.path.isdir(path):
            try:
                u = shutil.disk_usage(path)
                pct = 100 * u.used / max(u.total, 1)
                print(f"  {label:<20s} {u.used/1e9:7.1f} / {u.total/1e9:7.1f} GB  ({pct:5.1f}%)")
            except OSError as e:
                print(f"  {label:<20s} statfs failed: {e}")

    # ── recent runs ─────────────────────────────────────────────────────
    _section(f"latest {n_runs} run(s)")
    runs_root = os.path.join(state_dir, "runs")
    if not os.path.isdir(runs_root):
        print(f"  no runs dir at {runs_root}")
    else:
        dirs = sorted(
            [p for p in glob.glob(runs_root + "/*") if os.path.isdir(p)],
            key=os.path.getmtime,
            reverse=True,
        )
        for rd in dirs[:n_runs]:
            run_id = os.path.basename(rd)
            print(f"\n  --- {run_id} ---")
            siblings = sorted(glob.glob(os.path.join(runs_root, run_id + "*")))
            for p in siblings:
                sz = os.path.getsize(p) if os.path.isfile(p) else "<dir>"
                print(f"    {sz:>10}  {os.path.basename(p)}")
            for p in siblings:
                if not os.path.isfile(p):
                    continue
                name = os.path.basename(p)
                if name.endswith(".json"):
                    continue  # skip the big model/dataset config dumps
                print(f"\n    === {name} ({os.path.getsize(p)} bytes) ===")
                try:
                    txt = open(p, errors="replace").read()
                    if not txt.strip():
                        print("    (empty)")
                    else:
                        for line in txt[:6000].splitlines():
                            print(f"    {line}")
                except OSError as e:
                    print(f"    read failed: {e}")
            print(f"\n    files inside {rd}:")
            try:
                for f in sorted(os.listdir(rd)):
                    sub = os.path.join(rd, f)
                    sz = os.path.getsize(sub) if os.path.isfile(sub) else "<dir>"
                    print(f"      {sz:>10}  {f}")
            except OSError as e:
                print(f"      listdir failed: {e}")
            # runs.json
            try:
                rj_path = os.path.join(state_dir, "runs.json")
                rj = json.load(open(rj_path))
                rj_list = rj if isinstance(rj, list) else rj.get("runs", [])
                match = [r for r in rj_list if r.get("id") == run_id]
                if match:
                    r = match[0]
                    print("\n    runs.json record:")
                    for k in ("status", "pid", "gpu", "log_path", "kill_hint",
                              "error", "created_at", "restart_count", "seed_lora"):
                        if k in r:
                            print(f"      {k}: {r[k]}")
            except (OSError, json.JSONDecodeError):
                pass

    # ── module imports (catch broken installs early) ───────────────────
    _section("import sanity")
    for mod in ("underfit", "underfit.training.loop", "underfit.utils.lora_validate",
                "safetensors", "huggingface_hub"):
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except Exception as e:
            print(f"  ✗ {mod}: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tail underfit dashboard status")
    p.add_argument("--url", default="http://localhost:8787",
                   help="dashboard base URL (default: http://localhost:8787)")
    p.add_argument("--interval", type=int, default=10,
                   help="seconds between refreshes (default: 10)")
    p.add_argument("--mode", default="auto", choices=["auto", "html", "text"],
                   help="output format (default: auto-detect)")
    args = p.parse_args()
    start_monitor(args.url, args.interval, args.mode)
