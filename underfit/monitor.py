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

import json
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


def fetch_status(base_url: str = "http://localhost:8787") -> dict[str, Any]:
    """Snapshot of dashboard state. Returns `{'error': msg}` if unreachable."""
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

    return {
        "runs": runs,
        "datasets": datasets,
        "gradios": gradios,
        "runs_total": len(runs),
        "dataset_count": len(datasets),
        "active_gradios": active_gradios,
        "status_counts": status_counts,
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
    return (f"[{ts}] runs: {runs_str}"
            f" | datasets: {data['dataset_count']}"
            f" | gradio: {data['active_gradios']} active")


def format_html(data: dict[str, Any]) -> str:
    """Multi-line HTML block for IPython.display.HTML."""
    ts = datetime.now().strftime("%H:%M:%S")
    if "error" in data:
        return (f"<pre style='margin:0;font-family:monospace;color:#a00'>"
                f"[{ts}] dashboard unreachable ({data['error']})</pre>")
    sc = ", ".join(f"{n} {s}" for s, n in sorted(data["status_counts"].items())) or "none"
    lines = [
        f"<b>📊 underfit @ {ts}</b>",
        f"  Runs: {data['runs_total']} total ({sc})",
        f"  Datasets: {data['dataset_count']}",
        f"  Gradio instances: {data['active_gradios']} active",
    ]
    return ("<pre style='margin:0;font-family:monospace;line-height:1.4'>"
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
