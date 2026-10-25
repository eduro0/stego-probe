#!/usr/bin/env bash
# M0 setup: create the vLLM API key (Modal secret + local .env) and deploy the Qwen3-8B server.
# Idempotent: reuses the key in .env if present.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! grep -q '^STEGO_VLLM_API_KEY=' .env 2>/dev/null; then
  key=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  echo "STEGO_VLLM_API_KEY=$key" >> .env
  chmod 600 .env
fi
key=$(grep '^STEGO_VLLM_API_KEY=' .env | cut -d= -f2-)
modal secret create --force stego-vllm-key "VLLM_API_KEY=$key" >/dev/null
echo "secret stego-vllm-key set"

modal deploy serve_vllm.py
