#!/usr/bin/env python3
"""Canonical 1-Click System Launcher for AIC 2026 System Tai.

Usage:
    python start_system.py [--port 8000] [--host 127.0.0.1] [--config configs/production.yaml] [--no-browser]
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def check_environment() -> dict[str, str | bool]:
    """Pre-flight check of local runtime capabilities."""
    report: dict[str, str | bool] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_valid": sys.version_info >= (3, 11),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "tesseract_available": shutil.which("tesseract") is not None,
        "cuda_available": False,
        "device": "cpu",
    }
    try:
        import torch
        if torch.cuda.is_available():
            report["cuda_available"] = True
            report["device"] = f"cuda ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass
    return report


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIC 2026 System Tai - Master 1-Click Launcher"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "production.yaml",
        help="Path to production configuration YAML profile",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional path to dataset root directory (with keyframes, mapping, clip)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open default browser",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("[+] AIC 2026 - SYSTEM TAI CANONICAL PRODUCTION LAUNCHER")
    print("=" * 80)

    # 1. Preflight Check
    env = check_environment()
    print(f"[*] Python Version:      {env['python_version']} {'[OK]' if env['python_valid'] else '[FAIL]'}")
    print(f"[*] Compute Device:      {env['device']}")
    print(f"[*] FFmpeg Available:    {'[YES]' if env['ffmpeg_available'] else '[NO - Audio fallback]'}")
    print(f"[*] Tesseract OCR:       {'[YES]' if env['tesseract_available'] else '[NO - OCR fallback]'}")
    print(f"[*] Profile Config:      {args.config}")
    print("-" * 80)

    # 2. Check Port Availability
    if is_port_in_use(args.port, args.host):
        print(f"[!] Warning: Port {args.port} is currently in use. Attempting to use port {args.port + 1}...")
        args.port += 1

    # 3. Mount UI static build if available
    from fastapi.staticfiles import StaticFiles
    from system_tai.server.app import create_app

    app = create_app(engine=None)

    ui_dist = SCRIPT_DIR / "ui" / "dist"
    if ui_dist.exists() and (ui_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")
        print(f"[*] Web UI Static Assets: Mounted from {ui_dist}")
    else:
        print("[*] Web UI Static Assets: Running API Gateway mode (/docs for Swagger UI)")

    server_url = f"http://{args.host}:{args.port}"
    api_docs_url = f"http://{args.host}:{args.port}/docs"

    print("\n" + "=" * 80)
    print(f"[*] SYSTEM TAI READY & OPERATIONAL!")
    print(f"[*] Application Web UI:    {server_url}")
    print(f"[*] Interactive API Docs:  {api_docs_url}")
    print("=" * 80 + "\n")

    if not args.no_browser:
        time.sleep(1.0)
        try:
            webbrowser.open(server_url)
        except Exception:
            pass

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
