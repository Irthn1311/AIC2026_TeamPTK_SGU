"""CLI Entry point to launch the system_tai FastAPI REST Gateway."""

from __future__ import annotations

import argparse

import uvicorn

from system_tai.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the system_tai FastAPI server.")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    app = create_app()
    print(f"Starting system_tai FastAPI Gateway on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
