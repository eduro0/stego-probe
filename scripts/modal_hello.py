"""M0 check: `modal run scripts/modal_hello.py` runs one cheap CPU container and returns."""
import modal

app = modal.App("stego-hello")


@app.function(timeout=60, max_containers=1)
def hello(x: int) -> str:
    import platform

    return f"hello from modal ({platform.python_version()}), x*2={x * 2}"


@app.local_entrypoint()
def main():
    print(hello.remote(21))
