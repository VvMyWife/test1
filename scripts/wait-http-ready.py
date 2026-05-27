#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--name", default="service")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None
    started = time.perf_counter()
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=3) as response:
                body = response.read().decode("utf-8")
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = body[:200]
                print(
                    f"{args.name} ready: {args.url} status={response.status} "
                    f"startup_seconds={time.perf_counter() - started:.3f} payload={payload}"
                )
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    raise SystemExit(f"{args.name} not ready: {last_error!r}")


if __name__ == "__main__":
    main()
