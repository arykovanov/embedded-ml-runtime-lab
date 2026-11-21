#!/usr/bin/env bash
set -euo pipefail

if ! command -v idf.py >/dev/null 2>&1; then
    printf '%s\n' 'ESP-IDF is not active. Run: source /path/to/esp-idf/export.sh' >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec idf.py qemu \
    --qemu-extra-args="-m 8M -nic user,model=open_eth,hostfwd=tcp:127.0.0.1:8081-:80" \
    "$@" monitor
