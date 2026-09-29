"""Explain why the MLX trainer died, instead of just printing its exit code.

run_mlx_training (mlx_engine) calls diagnose() when the trainer exits non-zero.
The report combines everything left behind by the dead process:

  * the exit code / signal
  * the trainer's stderr tail (captured while it streamed to the run log) —
    where MLX's Metal errors, libc++abi aborts and Python tracebacks land
  * the run log's tail — what the trainer was doing when it died (a demo at
    step N, a training step, a download)
  * the demo config — long demos are the usual reason one GPU op is too big
  * system memory / swap
  * the macOS crash report for exactly this pid, if ReportCrash wrote one

Where one symptom has several possible causes (a Metal GPU timeout can be an
oversized op, memory pressure, or another app hogging the GPU), the report
lists them ranked, each with the evidence found for it.

Torch-free and best-effort throughout: a diagnosis failure must never mask the
original exit code.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096

# Matched against stderr + log tail, most specific first. Each maps to a key in
# _CAUSES below.
_PATTERNS = [
    ("gpu_timeout", re.compile(r"GPU Timeout Error|kIOGPUCommandBufferCallbackErrorTimeout")),
    ("gpu_oom", re.compile(
        r"Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|"
        r"\[metal::malloc\]|Resource limit \(\d+\) exceeded|Attempting to allocate \d+ bytes")),
    ("gpu_error", re.compile(r"\[METAL\] Command buffer execution failed")),
    ("download", re.compile(
        r"HfHubHTTPError|LocalEntryNotFoundError|RepositoryNotFoundError|"
        r"GatedRepoError|ConnectionError|ReadTimeout|401 Client Error|403 Client Error")),
]


def _signal_name(code):
    if code >= 0:
        return None
    try:
        return signal.Signals(-code).name
    except ValueError:
        return f"signal {-code}"


_SIGNAL_MEANING = {
    "SIGABRT": "the trainer aborted itself: an uncaught C++ exception (MLX/Metal) or a failed assertion",
    "SIGKILL": "killed from outside: on macOS usually the kernel's memory killer (jetsam), or someone stopped the run",
    "SIGSEGV": "native crash (bad memory access) inside MLX, numpy or another C extension",
    "SIGBUS": "native crash (bus error): often a memory-mapped file that shrank or vanished, or a native library bug",
    "SIGTERM": "asked to stop: the run was stopped from the dashboard or the process was terminated",
    "SIGINT": "interrupted (Ctrl-C)",
}


def _read_tail(path, nbytes=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _lines(text):
    """Split on \\n and \\r (tqdm redraws with \\r) and drop blanks."""
    return [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]


def _last_activity(log_lines):
    """What the trainer was doing when it died: the last demo it started, or the
    last training step, whichever is later in the log."""
    demo_re = re.compile(r"▸ (ARC )?demo (\d+): (\d+) latents \(~(\d+)s audio\)")
    step_re = re.compile(r"Step (\d+), Epoch (\d+)")
    for ln in reversed(log_lines):
        m = demo_re.search(ln)
        if m:
            kind = "ARC demo" if m.group(1) else "demo"
            return {"phase": "demo", "demo_index": int(m.group(2)),
                    "latents": int(m.group(3)), "seconds": int(m.group(4)),
                    "text": f"rendering {kind} {m.group(2)} "
                            f"({m.group(3)} latents, ~{m.group(4)} s of audio)"}
        m = step_re.search(ln)
        if m:
            return {"phase": "training", "step": int(m.group(1)),
                    "text": f"training at step {m.group(1)} (epoch {m.group(2)})"}
        if "ARC model" in ln and ("LoRA merged" in ln or " ready" in ln):
            return {"phase": "demo", "text": "preparing ARC demos (ARC model loaded)"}
        if ln.startswith("loading ARC model"):
            return {"phase": "arc_load",
                    "text": "loading the ARC model and merging the LoRA into it (for ARC demos)"}
        if re.match(r"demo decoder \S+ loaded", ln):
            return {"phase": "demo_setup",
                    "text": "setting up demos (decoder loaded; next: ARC model load or first demo)"}
        if ln.startswith("training: max"):
            return {"phase": "startup_demos",
                    "text": "setting up the step-0 baseline demos (before the first training step)"}
        if "downloading" in ln:
            return {"phase": "download", "text": ln}
    return {"phase": "unknown", "text": log_lines[-1][:200] if log_lines else "no output"}


def _demo_sizes(cmd):
    """(crop_latents, [(index, latents, seconds, arc)]) from the trainer's argv."""
    def arg(flag):
        try:
            return cmd[cmd.index(flag) + 1]
        except (ValueError, IndexError):
            return None
    crop = arg("--latent-crop-length")
    crop = int(crop) if crop and crop.isdigit() else None
    demos = []
    path = arg("--demo-config")
    if path:
        try:
            entries = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            entries = []
        for i, e in enumerate(entries):
            dur = e.get("duration")
            if dur is not None:
                lat = max(2, int(round(float(dur) * SAMPLE_RATE / SAMPLES_PER_LATENT)))
            else:
                lat = crop
            if lat:
                demos.append((i, lat, lat * SAMPLES_PER_LATENT / SAMPLE_RATE, bool(e.get("arc"))))
    return crop, demos


def _memory_info():
    """{'ram_gb', 'swap_used_gb', 'swap_total_gb'} on macOS; {} elsewhere."""
    if sys.platform != "darwin":
        return {}
    info = {}
    try:
        info["ram_gb"] = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=5)) / 2**30
        swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], timeout=5).decode()
        m_total = re.search(r"total = ([\d.]+)M", swap)
        m_used = re.search(r"used = ([\d.]+)M", swap)
        if m_total and m_used:
            info["swap_total_gb"] = float(m_total.group(1)) / 1024
            info["swap_used_gb"] = float(m_used.group(1)) / 1024
    except Exception:
        pass
    return info


def _crash_report(pid, since, wait_s=5.0):
    """Summary of the macOS crash report for `pid`, or None. ReportCrash writes
    it a moment after the process dies, so poll briefly."""
    if sys.platform != "darwin" or pid is None:
        return None
    d = Path.home() / "Library" / "Logs" / "DiagnosticReports"
    deadline = time.time() + wait_s
    while True:
        try:
            cands = [p for p in d.glob("Python-*.ips") if p.stat().st_mtime >= since - 1]
        except OSError:
            cands = []
        for p in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                _, body = p.read_text(errors="replace").split("\n", 1)
                j = json.loads(body)
            except (OSError, ValueError):
                continue
            if j.get("pid") != pid:
                continue
            exc = j.get("exception") or {}
            images = j.get("usedImages") or []
            frames = []
            try:
                th = j["threads"][j["faultingThread"]]
                for fr in th.get("frames", []):
                    img = images[fr["imageIndex"]].get("name", "?") if "imageIndex" in fr else "?"
                    sym = fr.get("symbol", "")
                    # The interesting part is where the failure surfaced, not
                    # libc's abort machinery.
                    if img.startswith(("libsystem_", "libc++abi", "libobjc")):
                        continue
                    frames.append(f"{img}: {sym}" if sym else img)
                    if len(frames) >= 6:
                        break
            except (KeyError, IndexError, TypeError):
                pass
            return {"path": str(p), "type": exc.get("type"), "signal": exc.get("signal"),
                    "frames": frames}
        if time.time() >= deadline:
            return None
        time.sleep(0.5)


def _python_error(text):
    """Last Python exception line ('ValueError: …') after a Traceback, if any."""
    idx = text.rfind("Traceback (most recent call last)")
    if idx < 0:
        return None
    for ln in reversed(text[idx:].splitlines()):
        ln = ln.strip()
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b.*", ln):
            return ln[:300]
    return None


def _uncaught_cpp(text):
    m = None
    for m in re.finditer(r"terminating due to uncaught exception of type ([\w:]+): (.+)", text):
        pass
    return (m.group(1), m.group(2).strip()[:300]) if m else None


def _causes(kind, activity, crop, demos, mem, sig):
    """Ranked [(cause, evidence)] for a diagnosed error kind."""
    out = []
    long_demos = [d for d in demos if crop and d[1] > crop * 4]
    swap_heavy = (mem.get("swap_used_gb") or 0) > 0.5 * (mem.get("swap_total_gb") or 1e9) \
        or (mem.get("swap_used_gb") or 0) > 4
    swap_txt = (f"swap {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB used"
                if "swap_used_gb" in mem else None)
    ram_txt = f"{mem['ram_gb']:.0f} GB RAM" if "ram_gb" in mem else None

    def demo_evidence():
        if activity.get("phase") == "demo" and "latents" in activity:
            return (f"it died {activity['text']}"
                    + (f", {activity['latents'] // crop}× the training crop of {crop}"
                       if crop else ""))
        if long_demos:
            i, lat, sec, _ = max(long_demos, key=lambda d: d[1])
            return (f"demo {i} is {lat} latents (~{sec:.0f} s), "
                    f"{lat // crop}× the training crop of {crop}")
        return None

    if kind in ("gpu_timeout", "gpu_oom", "gpu_error"):
        ev = demo_evidence()
        big_op = ("One GPU operation was too large for this machine: usually a long demo "
                  "(full-length demos attend over thousands of latents at once), "
                  "or a large crop/batch during training")
        mem_cause = ("Memory pressure: the models + activations don't fit in unified memory, "
                     "so the GPU stalls on swapped-out memory until the watchdog fires")
        other = ("Another app is using the GPU heavily (browser video, games, another "
                 "ML job), so this job's work queues behind it")
        mem_ev = ", ".join(x for x in (swap_txt, ram_txt) if x) or None
        mem_ev = mem_ev if swap_heavy else (mem_ev and mem_ev + " (not obviously full)")
        if activity.get("phase") in ("arc_load", "demo_setup"):
            # No demo had started, so demo length can't be the trigger yet; the
            # ARC load holds a second DiT and merges the LoRA in fp32.
            arc_mem = ("Memory: ARC demos load a second full DiT next to the training model "
                       "and merge the LoRA into it in fp32; together they overflow unified "
                       "memory, so the GPU runs out of memory or stalls on swap until "
                       "macOS's watchdog fires")
            ev_arc = f"it died {activity['text']}" + (f"; {mem_ev}" if mem_ev else "")
            ranked = [(arc_mem, ev_arc), (other, None)]
            if long_demos:
                ranked.append(("(Not reached yet, but next in line) " + big_op, demo_evidence()))
        else:
            ranked = [(big_op, ev), (mem_cause, mem_ev), (other, None)]
            if kind == "gpu_oom":
                ranked[0], ranked[1] = ranked[1], ranked[0]
        # Causes with evidence first; keep the rest as possibilities.
        ranked.sort(key=lambda c: c[1] is None)
        out = ranked
        if activity.get("phase") in ("arc_load", "demo_setup"):
            out.append(("Fix to try: drop the ARC demos (or use RF demos) so only one DiT is "
                        "loaded, use a smaller model (sm-music), or free memory (close apps)",
                        None))
        elif activity.get("phase") in ("demo", "startup_demos"):
            out.append(("Fix to try: shorten the long demo(s) to the crop length (or 30–60 s), "
                        "or pick a preset with fewer/shorter demos; a smaller model "
                        "(sm-music) leaves much more headroom", None))
        elif activity.get("phase") == "training":
            out.append(("Fix to try: lower the latent crop length or batch size, "
                        "or close other GPU-heavy apps", None))
    elif kind == "download":
        out = [("Weight download from HuggingFace failed (network, auth, or a gated repo)",
                None),
               ("Fix to try: run `huggingface-cli login` if the repo is gated, check the "
                "network, then restart the run", None)]
    elif sig == "SIGKILL":
        out = [("macOS killed it for using too much memory (jetsam)",
                ", ".join(x for x in (swap_txt, ram_txt) if x) or None),
               ("The run was stopped from the dashboard or with kill -9", None)]
        in_long_demo = (activity.get("phase") == "demo" and crop
                        and activity.get("latents", 0) > crop * 4)
        if long_demos or in_long_demo:
            out.insert(1, ("A long demo pushed memory over the edge", demo_evidence()))
    return out


def diagnose(code, cmd, pid=None, launched_at=None, stderr_tail="", log_path=None):
    """Return (summary, report): a one-line summary (the dashboard shows it as
    the run's kill_hint) and the full multi-line report."""
    sig = _signal_name(code)
    log_text = _read_tail(log_path) if log_path else ""
    log_lines = _lines(log_text)
    activity = _last_activity(log_lines)
    crop, demos = _demo_sizes(cmd)
    combined = stderr_tail + "\n" + log_text[-16384:]

    kind = next((k for k, rx in _PATTERNS if rx.search(combined)), None)
    cpp = _uncaught_cpp(combined)
    pyerr = _python_error(combined)
    mem = _memory_info()
    crash = _crash_report(pid, launched_at or time.time() - 3600) if code < 0 else None

    headline = {
        "gpu_timeout": "Metal GPU timeout: a GPU command ran past macOS's watchdog",
        "gpu_oom": "Metal out of memory",
        "gpu_error": "Metal command buffer failed",
        "download": "weight download failed",
    }.get(kind)
    if not headline:
        if pyerr:
            headline = pyerr
        elif cpp:
            headline = f"uncaught {cpp[0]}: {cpp[1]}"
        elif sig:
            headline = f"killed by {sig}"
        else:
            headline = f"exited with code {code}"
    summary = f"MLX trainer died ({sig or f'exit {code}'}) while {activity['text']}: {headline}"

    out = ["", "=" * 78, "MLX trainer failure report", "=" * 78, summary, ""]
    out.append(f"exit code   : {code}" + (f"  ({sig}: {_SIGNAL_MEANING.get(sig, 'terminated by a signal')})"
                                           if sig else ""))
    out.append(f"last action : {activity['text']}")
    if cpp:
        out.append(f"error       : uncaught {cpp[0]}: {cpp[1]}")
    elif pyerr:
        out.append(f"error       : {pyerr}")
    if crop:
        out.append(f"crop length : {crop} latents (~{crop * SAMPLES_PER_LATENT / SAMPLE_RATE:.1f} s)")
    if demos:
        out.append("demos       : " + ", ".join(
            f"{i}{'(ARC)' if arc else ''}={lat} lat/~{sec:.0f}s" for i, lat, sec, arc in demos))
    if mem:
        parts = []
        if "ram_gb" in mem:
            parts.append(f"{mem['ram_gb']:.0f} GB RAM")
        if "swap_used_gb" in mem:
            parts.append(f"swap {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB used (now)")
        out.append("memory      : " + ", ".join(parts))
    if crash:
        out.append(f"crash report: {crash['path']}")
        out.append(f"              {crash['type']} / {crash['signal']}; failing thread:")
        out.extend(f"                {f}" for f in crash["frames"])

    causes = _causes(kind, activity, crop, demos, mem, sig)
    if causes:
        out.append("")
        out.append("likely causes (most likely first):")
        n = 0
        for cause, ev in causes:
            if cause.startswith("Fix to try"):
                out.append(f"  → {cause}")
                continue
            n += 1
            out.append(f"  {n}. {cause}")
            if ev:
                out.append(f"     evidence: {ev}")

    err_lines = [ln for ln in _lines(stderr_tail)
                 if not re.search(r"\d+%\|", ln)][-12:]  # drop progress bars
    if err_lines:
        out.append("")
        out.append("last trainer stderr:")
        out.extend(f"  {ln[:300]}" for ln in err_lines)
    out.append("=" * 78)
    return summary, "\n".join(out)
