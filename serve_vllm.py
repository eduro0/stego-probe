"""Qwen3-8B behind an OpenAI-compatible vLLM server on Modal (generation for sender / overseer / receiver).

Qwen3-8B was deprecated on OpenRouter (Oct 9 2026), so we self-host the exact bf16 checkpoint that the
activation replay (modal_app.py) will load — generation and replay now see identical weights.

    modal deploy serve_vllm.py          # once; URL is stable, container scales to zero when idle
    modal app stop stego-vllm           # spend guard: stop when done for the day

Auth: vLLM's --api-key, read from the Modal secret `stego-vllm-key` (created by scripts/setup_m0.sh, the
same key is written to stego-probe/.env for the local client).
"""
import os
import subprocess

import modal

MODEL = "Qwen/Qwen3-8B"
VLLM_VERSION = "0.30.0"
PORT = 8000
GPU = "L40S"  # 48 GB: 16 GB bf16 weights + plenty of KV cache for B=32 concurrent episodes

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub>=0.34")
    .env({
        "HF_HOME": "/vol/hf",
        "VLLM_CACHE_ROOT": "/vol/vllm-cache",
        # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which debian_slim lacks
        # ("Could not find nvcc"); the PyTorch sampler needs no toolkit.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)
vol = modal.Volume.from_name("stego-probe", create_if_missing=True)
app = modal.App("stego-vllm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("stego-vllm-key")],
    timeout=30 * 60,  # per request
    scaledown_window=10 * 60,  # idle minutes before the GPU is released
    max_containers=1,  # spend guard
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--enable-prefix-caching",
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(cmd)
