#!/usr/bin/env python3
"""Read Envoy JSON access-log lines from stdin and print the FIRST chat-completion entry as upstream_cluster<TAB>upstream_host<TAB>response_code."""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line.startswith('{'):
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if d.get('request_path') == '/v1/chat/completions':
        print(f"{d.get('upstream_cluster','')}\t{d.get('upstream_host','')}\t{d.get('response_code','')}")
        break
