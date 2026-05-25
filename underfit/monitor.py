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
    """Render a 'Restart Dashboard' button + note, sized to sit under the Open button.

    Clicking it kills any process bound to `port` and re-launches dashboard/server.py.
    Training runs are unaffected — they're separate detached processes managed by
    RunsRegistry, and the fresh dashboard re-reads their state from runs.json on startup.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        return None

    btn = widgets.Button(
        description="🔄 Restart Dashboard",
        button_style="warning",
        layout=widgets.Layout(width="240px", height="40px"),
    )
    out = widgets.Output()

    def _on_click(b):
        b.disabled = True
        b.description = "Restarting…"
        with out:
            try:
                launch_dashboard_subprocess(port=port, quiet=True)
                print(f"✓ Dashboard restarted on port {port} "
                      f"(training runs unaffected).", flush=True)
            except Exception as e:
                print(f"✗ Restart failed: {type(e).__name__}: {e}", flush=True)
            finally:
                b.disabled = False
                b.description = "🔄 Restart Dashboard"

    btn.on_click(_on_click)

    display(widgets.HBox([btn], layout=widgets.Layout(justify_content="center")))
    display(HTML(
        "<div style='text-align:center; font-size:12px; color:#888; "
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",system-ui,sans-serif; "
        "margin: 2px 0 18px 0;'>"
        "* doesn't affect training runs. Do this if dashboard freezes."
        "</div>"
    ))
    display(out)


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


def start_monitor(base_url: str = "http://localhost:8787",
                  interval: int = 10,
                  mode: str = "auto",
                  on_unreachable=None,
                  failure_threshold: int = 3) -> None:
    """Poll the dashboard and refresh a status block until KeyboardInterrupt.

    mode="auto" (default) renders HTML in IPython and text in a terminal.
    Force a specific mode with mode="html" or mode="text".

    on_unreachable: optional callable() invoked when the dashboard has been
    unreachable for `failure_threshold` consecutive polls (default 3 — so ~30s
    of failures at the default 10s interval before triggering). Use it to
    auto-restart the dashboard. Exceptions raised by the callable are caught
    and printed; the monitor keeps running either way.
    """
    if mode == "auto":
        mode = _detect_mode()

    display_id = "underfit-status"
    if mode == "html":
        from IPython.display import display, update_display, HTML
        display(HTML(format_html(fetch_status(base_url))), display_id=display_id)
    else:
        print(format_text(fetch_status(base_url)), flush=True)

    failures = 0
    try:
        while True:
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
                update_display(HTML(format_html(data)), display_id=display_id)
            else:
                print(format_text(data), flush=True)
    except KeyboardInterrupt:
        print("\nStatus monitor stopped.")


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
