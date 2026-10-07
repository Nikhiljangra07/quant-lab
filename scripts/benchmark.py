"""Serve one arm, verify identity/generation, then collect fixed-length repeated cells."""
import argparse
import json
import signal
import socket
import subprocess
import time
import urllib.request
from common import PROTOCOL as P, ROOT, RUN, now, validate_bench, verify_freeze, write_json

def get_json(url, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90 if data else 3) as r:
        return json.load(r)

def ready(base, model):
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            if response.status != 200:
                return False
        return model in [m["id"] for m in get_json(base + "/v1/models")["data"]]
    except (OSError, ValueError, KeyError):
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["BF16", "W4A16", "NVFP4"])
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    verify_freeze()
    prepared = json.loads((RUN / "prepared.json").read_text())
    path = prepared["base_path"] if a.arm == "BF16" else str(RUN / "models" / a.arm)
    dest = RUN / "bench" / a.arm
    dest.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    name = f"quant-lab-{a.arm}-{port}"
    s, b = P["serving"], P["bench"]
    vllm = str(ROOT / "venv_serve/bin/vllm")
    command = [vllm, "serve", path, "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", name, "--dtype", s["dtype"], "--max-model-len", str(s["max_model_len"]),
        "--gpu-memory-utilization", str(s["gpu_memory_utilization"]), "--seed", str(P["seed"]),
        "--no-enable-prefix-caching", "--generation-config", "vllm"]
    if a.arm == "NVFP4":
        command += ["--linear-backend", s["nvfp4_linear_backend"]]
    write_json(dest / "server-command.json", {"command": command, "started": now()})
    # Server shares the runner's stage group, so the global deadline also kills workers.
    server = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Stage interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    with (dest / "server.log").open("a") as log:
        try:
            server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while not ready(base, name):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {dest}/server.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server failed its readiness deadline")
                time.sleep(2)
            smoke = get_json(base + "/v1/chat/completions", {"model": name,
                "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
                "temperature": 0, "max_tokens": 32, "seed": P["seed"]})
            if not smoke.get("choices") or not smoke["choices"][0]["message"].get("content"):
                raise RuntimeError("Smoke generation returned no content")
            write_json(dest / "smoke.json", smoke)
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                                           "--format=csv"], text=True)
            (dest / "server-memory.csv").write_text(gpu)
            if a.smoke_only:
                return
            for c in b["concurrency"]:
                for rep in range(1, b["repetitions"] + 1):
                    count = b["prompts"][str(c)]
                    filename = f"c{c}-r{rep}.json"
                    cmd = [vllm, "bench", "serve", "--backend", "vllm", "--base-url", base,
                        "--model", name, "--tokenizer", path, "--dataset-name", "random",
                        "--random-input-len", str(b["input_tokens"]), "--random-output-len", str(b["output_tokens"]),
                        "--random-range-ratio", "0.0",
                        "--num-prompts", str(count), "--max-concurrency", str(c), "--seed", str(P["seed"]),
                        "--temperature", "0", "--ignore-eos", "--num-warmups", str(b["warmups"]),
                        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
                        "--save-result", "--save-detailed", "--result-dir", str(dest), "--result-filename", filename]
                    with (dest / f"c{c}-r{rep}.log").open("w") as cell_log:
                        subprocess.run(cmd, cwd=ROOT, stdout=cell_log, stderr=subprocess.STDOUT, check=True)
                    errors = validate_bench(json.loads((dest / filename).read_text()), count, b["output_tokens"], b["input_tokens"])
                    if errors:
                        raise RuntimeError(f"Invalid benchmark {filename}: {errors}")
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)

if __name__ == "__main__":
    main()
