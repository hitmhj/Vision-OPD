#!/usr/bin/env bash

# Backward-compatible wrapper. The public entry now owns the whole lifecycle;
# this name merely forces dependency installation before handing off to it.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export VOPD_INSTALL_MODE=always
exec bash "$PROJECT_ROOT/scripts/start_vision_opd_ascend.sh" "$@"
