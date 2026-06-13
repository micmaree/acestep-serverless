"""RunPod serverless handler for ACE-Step XL turbo.

Strategy: spawn acestep-api as a child process at WORKER START (module load),
wait for /health, then handler() proxies each job through /release_task →
/query_result and returns the generated audio as base64 (RunPod has no
public file storage on serverless — caller decodes b64 → file).

Cold start: ~60-90s (acestep-api boot + DiT lazy load on first request).
Hot start: ~85-100s/gen for batch=2 + thinking=True on A40.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
import runpod

# ──────── boot the sidecar once at worker start ────────
SIDECAR_PORT = 8007
SIDECAR_URL = f"http://127.0.0.1:{SIDECAR_PORT}"
CACHE_DIR = Path(os.environ.get(
    "ACESTEP_CACHE_DIR",
    "/app/.cache/acestep/tmp/api_audio"
))


def _start_sidecar() -> subprocess.Popen:
    """Launch acestep-api in the venv and return the Popen handle.
    Inherits env from the container (ACESTEP_CONFIG_PATH, etc.)."""
    print("[handler] spawning acestep-api on :{}".format(SIDECAR_PORT), flush=True)
    return subprocess.Popen(
        ["/app/.venv/bin/acestep-api",
         "--host", "127.0.0.1",
         "--port", str(SIDECAR_PORT)],
        env=os.environ.copy(),
        stdout=None,  # inherit — RunPod captures these as worker logs
        stderr=None,
    )


def _wait_ready(timeout_sec: int = 600) -> None:
    """Block until /health returns 200. Cold start can be slow because the
    LM tokenizer takes ~30-90s to load. Caller times out at job level."""
    print("[handler] waiting for sidecar /health...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get(f"{SIDECAR_URL}/health", timeout=3)
            if r.status_code == 200:
                print(f"[handler] sidecar ready after {time.time()-t0:.1f}s",
                      flush=True)
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"sidecar did not come up in {timeout_sec}s")


def _dump_volume_layout() -> None:
    """Log what's actually mounted on this worker so we can debug volume
    mount issues. Prints to stdout (captured by RunPod worker logs)."""
    import os
    print("[handler] === VOLUME LAYOUT DEBUG ===", flush=True)
    print(f"[handler] ACESTEP_CHECKPOINT_DIR={os.environ.get('ACESTEP_CHECKPOINT_DIR','NOT_SET')}",
          flush=True)
    for path in ("/runpod-volume", "/workspace", "/runpod-volume/models",
                  "/runpod-volume/models/acestep",
                  "/workspace/models/acestep"):
        try:
            if os.path.exists(path):
                entries = sorted(os.listdir(path))[:20]
                print(f"[handler] {path}/ ({len(entries)} entries): {entries}",
                      flush=True)
            else:
                print(f"[handler] {path}/ — DOES NOT EXIST", flush=True)
        except Exception as exc:
            print(f"[handler] {path}/ — ERROR: {exc}", flush=True)
    print("[handler] === END DEBUG ===", flush=True)


# Boot at module load — RunPod imports this module once per worker process,
# so the sidecar lives across many handler() invocations.
_dump_volume_layout()
_SIDECAR = _start_sidecar()
_wait_ready()


# ──────── handler ────────
def _to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless entry. Input shape mirrors ACE-Step's /release_task
    payload directly (forwarded as-is)."""
    payload = event.get("input") or {}
    if not isinstance(payload, dict) or not payload.get("prompt"):
        return {"error": "input must include at least 'prompt'"}

    # Submit
    try:
        r = requests.post(f"{SIDECAR_URL}/release_task",
                          json=payload, timeout=600)
        r.raise_for_status()
        d = r.json()
        task_id = (d.get("data") or {}).get("task_id") or d.get("task_id")
        if not task_id:
            return {"error": "no task_id in response", "raw": d}
    except Exception as exc:
        return {"error": f"submit failed: {exc}"}

    # Poll
    deadline = time.time() + 1800  # 30 min ceiling
    while time.time() < deadline:
        time.sleep(2)
        try:
            r = requests.post(f"{SIDECAR_URL}/query_result",
                              json={"task_id": task_id}, timeout=10)
            d = r.json()
        except Exception:
            continue
        entries = d.get("data") or []
        if not entries:
            continue
        e = entries[0]
        status = e.get("status", 0)
        if status == 1:  # completed
            songs = e.get("result") or "[]"
            if isinstance(songs, str):
                songs = json.loads(songs)
            done = [s for s in songs if s.get("status") in (1, "1") and s.get("file")]
            if not done:
                return {"error": "no completed songs", "raw": e}
            # Read each generated mp3 and return as base64
            results = []
            for s in done[:4]:
                fpath = s.get("file", "")
                # ace-step returns absolute path inside the container
                p = Path(fpath if fpath.startswith("/") else f"/app/{fpath}")
                if not p.exists():
                    # Sometimes path is relative to ACESTEP_CACHE_DIR
                    candidate = CACHE_DIR / Path(fpath).name
                    if candidate.exists():
                        p = candidate
                if p.exists():
                    results.append({
                        "audio_b64": _to_b64(p),
                        "seed": s.get("seed_value", ""),
                        "filename": p.name,
                        "size_bytes": p.stat().st_size,
                    })
            return {"songs": results, "task_id": task_id}
        if status not in (0, 2):
            return {"error": f"sidecar status {status}", "raw": e}
    return {"error": "timeout after 30 min"}


# ──────── start serverless event loop ────────
runpod.serverless.start({"handler": handler})
