"""Upload a TFLite model as the raw body of POST /model on an ESP32."""

import argparse
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


def upload_model(model_path, url, timeout=30.0):
    """Upload a model and return the response body; HTTP/network errors propagate."""
    address = urlsplit(url)
    if address.scheme not in ("http", "https") or not address.hostname:
        raise ValueError("Provide an HTTP URL such as http://192.168.1.123/")
    if timeout <= 0:
        raise ValueError("Timeout must be positive")

    data = Path(model_path).read_bytes()
    if not data:
        raise ValueError("The model file is empty")
    if len(data) > 1024 * 1024:
        raise ValueError("The model exceeds the endpoint's 1 MiB body limit")

    endpoint = urljoin(url, "/model")
    request = Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(data)),
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the TFLite model file")
    parser.add_argument("--url", required=True, help="ESP32 base URL, e.g. http://192.168.1.123/")
    parser.add_argument("--timeout", type=float, default=30.0, help="Socket timeout in seconds (default: 30)")
    args = parser.parse_args()

    try:
        response = upload_model(args.model, args.url, args.timeout)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        print(f"Upload failed: HTTP {error.code} {error.reason}. {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        return 1

    print(f"Uploaded {args.model} to {urljoin(args.url, '/model')}")
    if response:
        print(response.decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
