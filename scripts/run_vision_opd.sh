#!/usr/bin/env bash
# Compatibility alias for the unified Ascend entrypoint.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/start_vision_opd_ascend.sh" "$@"
