#!/bin/bash
# Start Envoy via func-e for local development
# Envoy listens on port 8801 (LLM API proxy) and 19000 (admin)
#
# Proxy settings (https_proxy, http_proxy, no_proxy) are inherited from the
# environment — set them in your shell before running this script if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../bin/func-e" run \
  --config-path "${SCRIPT_DIR}/../config/envoy.yaml" \
  --component-log-level "ext_proc:info,router:info,http:warn" \
  >> /tmp/envoy.log 2>&1
