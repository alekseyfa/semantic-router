#!/usr/bin/env bash
# Launch the router for the bench. Selects binary + library paths by AI_BINDING.
#
# Usage:
#   AI_BINDING=openvino CONFIG=config/bench-openvino.yaml bench/router-backend/run_router.sh
#   AI_BINDING=candle   CONFIG=config/bench-candle.yaml   bench/router-backend/run_router.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

AI_BINDING=${AI_BINDING:-candle}
CONFIG=${CONFIG:-config/bench-${AI_BINDING}.yaml}

case "$AI_BINDING" in
  openvino) BIN="$ROOT/bin/router-openvino" ;;
  candle)   BIN="$ROOT/bin/router-candle" ;;
  *) echo "Unknown AI_BINDING=$AI_BINDING"; exit 1 ;;
esac

[[ -x "$BIN" ]] || { echo "Binary not found: $BIN"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG"; exit 1; }

# Both backends need candle/ml/nlp shared libs (router still links them all).
# OV backend additionally needs the openvino-binding lib + openvino runtime.
export LD_LIBRARY_PATH="$ROOT/candle-binding/target/release:$ROOT/ml-binding/target/release:$ROOT/nlp-binding/target/release:$ROOT/openvino-binding/build:$ROOT/.venv/lib/python3.13/site-packages/openvino/libs:$ROOT/.venv/lib/python3.13/site-packages/openvino_tokenizers/lib"
export OPENVINO_TOKENIZERS_LIB="$ROOT/.venv/lib/python3.13/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so"

# Pin to first 16 cores for reproducibility across runs.
TASKSET="${TASKSET:-taskset -c 0-15}"

echo "[run_router] AI_BINDING=$AI_BINDING binary=$BIN config=$CONFIG"
exec $TASKSET "$BIN" -config="$CONFIG" --enable-system-prompt-api=true
