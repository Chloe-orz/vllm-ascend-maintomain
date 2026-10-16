# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Wait for a local vLLM server, then exercise its existing Worker transport."""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def run_client(url: str, plan: dict[str, object], timeout: float) -> list[dict[str, str | int]]:
    url = url.rstrip("/")
    deadline = time.monotonic() + timeout
    print(f"[LWD][wire-test] waiting for {url}/health", flush=True)
    while True:
        try:
            with urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("vLLM server did not become healthy before the deadline")
        time.sleep(1)
    print("[LWD][wire-test] server ready; requesting Worker transfers", flush=True)
    request = Request(
        url + "/collective_rpc",
        data=json.dumps({"method": "lwd_run_probe", "args": [json.dumps(plan)], "timeout": timeout}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout + 10) as response:
            results = json.load(response).get("results")
    except HTTPError as exc:
        raise RuntimeError(f"Worker RPC failed ({exc.code}): {exc.read().decode()}") from exc
    if (
        not isinstance(results, list)
        or not results
        or any(not isinstance(result, dict) or result.get("status") != "PASS" for result in results)
    ):
        raise RuntimeError(f"Not all local Workers passed: {results!r}")
    print(f"[LWD][wire-test] all {len(results)} local Workers PASS: {json.dumps(results)}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--channel", choices=("up", "down"), default="up")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.batches <= 0 or args.timeout <= 0:
        parser.error("batches and timeout must be positive")
    run_client(args.url, {"channel": args.channel, "batches": args.batches}, args.timeout)


if __name__ == "__main__":
    main()
