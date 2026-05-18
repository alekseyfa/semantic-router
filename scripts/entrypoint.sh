#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Config: respect CONFIG_FILE env, otherwise auto-detect
CONFIG_FILE_PATH=${CONFIG_FILE:-""}
if [[ -z "$CONFIG_FILE_PATH" ]]; then
  if [[ -f /app/config/config.yaml ]]; then
    CONFIG_FILE_PATH=/app/config/config.yaml
  else
    CONFIG_FILE_PATH="$PROJECT_ROOT/config/config.yaml"
  fi
fi

if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
  echo "[entrypoint] Config file not found at $CONFIG_FILE_PATH" >&2
  exit 1
fi

AI_BINDING=${AI_BINDING:-candle}

# Resolve binary path: check /app/ first (Docker), then bin/ (local)
resolve_binary() {
  local name=$1
  if [[ -f "/app/$name" ]]; then
    echo "/app/$name"
  elif [[ -f "$PROJECT_ROOT/bin/$name" ]]; then
    echo "$PROJECT_ROOT/bin/$name"
  else
    echo ""
  fi
}

case "$AI_BINDING" in
  onnx)
    BINARY=$(resolve_binary "router-onnx")
    ;;
  openvino)
    BINARY=$(resolve_binary "router-openvino")
    ;;
  candle|"")
    BINARY=$(resolve_binary "router-candle")
    ;;
  *)
    echo "[entrypoint] Unknown AI_BINDING='$AI_BINDING'. Valid values: candle (default), onnx, openvino" >&2
    exit 1
    ;;
esac

if [[ -z "$BINARY" ]]; then
  echo "[entrypoint] Binary not found for AI_BINDING=$AI_BINDING" >&2
  echo "[entrypoint] Falling back to candle binding..." >&2
  BINARY=$(resolve_binary "router-candle")
  AI_BINDING=candle
  if [[ -z "$BINARY" ]]; then
    echo "[entrypoint] Fallback binary also not found. Build with:" >&2
    echo "  cd src/semantic-router && go build -tags=$AI_BINDING -o ../../bin/router-$AI_BINDING ./cmd/main.go" >&2
    exit 1
  fi
fi

# Set up LD_LIBRARY_PATH for native bindings (local mode)
if [[ -d "$PROJECT_ROOT/openvino-binding/build" ]]; then
  export LD_LIBRARY_PATH="${PROJECT_ROOT}/candle-binding/target/release:${PROJECT_ROOT}/openvino-binding/build:${PROJECT_ROOT}/nlp-binding/target/release:${PROJECT_ROOT}/ml-binding/target/release:${LD_LIBRARY_PATH:-}"
fi

# Set OpenVINO tokenizers path if not already set
if [[ "$AI_BINDING" == "openvino" && -z "${OPENVINO_TOKENIZERS_LIB:-}" ]]; then
  OV_TOK="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so"
  if [[ -f "$OV_TOK" ]]; then
    export OPENVINO_TOKENIZERS_LIB="$OV_TOK"
  fi
fi

echo "[entrypoint] Starting semantic-router with AI_BINDING=$AI_BINDING"
echo "[entrypoint] Binary: $BINARY"
echo "[entrypoint] Config: $CONFIG_FILE_PATH"
[[ $# -gt 0 ]] && echo "[entrypoint] Additional args: $*"
exec "$BINARY" --config "$CONFIG_FILE_PATH" "$@"
