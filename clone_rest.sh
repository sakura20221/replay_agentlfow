#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The old script cloned only two repositories,
# followed branch tips, used a host-specific path, and deleted failed targets.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/scripts/bootstrap_upstreams.py" "$@"
