"""Supervised pipeline with a rental-time deadline and independent run directories."""
import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from common import PROTOCOL as P, ROOT, budget_deadline, now, source_hashes, write_json

def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Kill leftover server/GPU workers even if their stage parent already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()

class Runner:
    def __init__(self, directory, deadline, env=None):
        self.directory, self.deadline = Path(directory), deadline
        self.env = dict(os.environ if env is None else env)
        self.env.update(QUANT_RUN_DIR=str(self.directory), PYTHONUNBUFFERED="1",
                        HF_HOME=str(ROOT / "cache" / "huggingface"), VLLM_NO_USAGE_STATS="1",
                        HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", PIP_NO_CACHE_DIR="1",
                        # Amendment A1 (2026-10-07): the pod image ships nvcc 12.8; FlashInfer's JIT needs
                        # CUDA >= 12.9 to target SM120, so its sampler warmup fails. Use vLLM's native sampler.
                        VLLM_USE_FLASHINFER_SAMPLER="0")
        self.events = []

    def stage(self, name, command, required=True, max_seconds=None):
        limit = self.deadline
        if max_seconds is not None:
            limit = min(limit, time.time() + max_seconds)
        event = {"stage": name, "command": command, "required": required, "started": now(), "status": "RUNNING"}
        self.events.append(event)
        self.save()
        print(f"{name}: starting", flush=True)
        process = None
        try:
            if time.time() >= limit:
                raise TimeoutError("Rental runtime allowance exhausted")
            with (self.directory / f"{name}.log").open("w") as output:
                process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=max(0.01, limit - time.time()))
                except subprocess.TimeoutExpired as e:
                    raise TimeoutError("Stage or rental runtime deadline exceeded") from e
            if code:
                raise RuntimeError(f"Exit code {code}; see {name}.log")
            event["status"] = "PASS"
            return True
        except (Exception, KeyboardInterrupt) as error:
            event.update(status="NO-RESULT", reason=f"{type(error).__name__}: {error}")
            if required or isinstance(error, (TimeoutError, KeyboardInterrupt)):
                raise
            return False
        finally:
            if process is not None:
                kill_group(process)
            event["ended"] = now()
            self.save()
            print(f"{name}: {event['status']}", flush=True)

    def save(self):
        write_json(self.directory / "stages.json", self.events)
        lines = ["# Run events", "", "Immutable source hashes are in freeze.json. Full stdout/stderr is in each named stage log.", ""]
        for event in self.events:
            lines.append(f"- {event.get('started', '')} {event['stage']}: {event['status']} {event.get('reason', '')}")
        (self.directory / "LOG.md").write_text("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True, help="actual all-in GPU hourly price in USD")
    ap.add_argument("--pod-started", required=True, help="original rental start ISO timestamp with timezone")
    ap.add_argument("--supervised", action="store_true", required=True,
                    help="operator will stop the billable pod; this program only stops compute")
    ap.add_argument("--with-nvfp4", action="store_true", help="attempt optional arm after BF16/W4A16")
    a = ap.parse_args()
    b = P["budget"]
    deadline = budget_deadline(a.pod_started, a.rate, b["cap_usd"], b["reserve_usd"])
    if deadline - time.time() < 300:
        raise SystemExit("Less than five minutes remain in rental allowance; stop the pod.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    directory = ROOT / "results" / "runs" / run_id
    directory.mkdir(parents=True)
    (ROOT / "results" / "LATEST").write_text(run_id + "\n")
    hashes = source_hashes()
    for relative in hashes:
        dest = directory / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    write_json(directory / "freeze.json", {"created": now(), "source_sha256": hashes,
        "protocol": P, "rate_usd_hour": a.rate, "pod_started": a.pod_started,
        "compute_deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        "billing": "SUPERVISED: process deadline does not stop Runpod rental/storage billing",
        "optional_nvfp4": a.with_nvfp4})
    runner = Runner(directory, deadline)
    qp, sp = str(ROOT / "venv_quant/bin/python"), str(ROOT / "venv_serve/bin/python")
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Runner interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    status, reason = "COMPLETE", None
    try:
        runner.stage("setup", ["bash", str(ROOT / "scripts/setup.sh")], max_seconds=b["setup_minutes"] * 60)
        runner.stage("prepare", [qp, str(ROOT / "scripts/prepare.py")])
        runner.stage("smoke-BF16", [sp, str(ROOT / "scripts/benchmark.py"), "BF16", "--smoke-only"])
        runner.stage("quant-W4A16", [qp, str(ROOT / "scripts/quantize.py"), "--scheme", "W4A16"])
        runner.stage("smoke-W4A16", [sp, str(ROOT / "scripts/benchmark.py"), "W4A16", "--smoke-only"])
        for arm in ("BF16", "W4A16"):
            runner.stage(f"eval-{arm}", [sp, str(ROOT / "scripts/evaluate.py"), arm])
            runner.stage(f"bench-{arm}", [sp, str(ROOT / "scripts/benchmark.py"), arm])
        if a.with_nvfp4:
            if deadline - time.time() < b["optional_min_remaining_minutes"] * 60:
                runner.events.append({"stage": "NVFP4", "status": "SKIPPED", "reason": "Insufficient remaining time for optional arm"})
                runner.save()
            elif runner.stage("quant-NVFP4", [qp, str(ROOT / "scripts/quantize.py"), "--scheme", "NVFP4"], required=False):
                if runner.stage("smoke-NVFP4", [sp, str(ROOT / "scripts/benchmark.py"), "NVFP4", "--smoke-only"], required=False):
                    runner.stage("eval-NVFP4", [sp, str(ROOT / "scripts/evaluate.py"), "NVFP4"], required=False)
                    runner.stage("bench-NVFP4", [sp, str(ROOT / "scripts/benchmark.py"), "NVFP4"], required=False)
    except (Exception, KeyboardInterrupt) as error:
        status, reason = "INCOMPLETE", f"{type(error).__name__}: {error}"
    finally:
        write_json(directory / "run-status.json", {"status": status, "reason": reason, "ended": now()})
        env = runner.env
        subprocess.run([sys.executable, str(ROOT / "scripts/collect.py")], cwd=ROOT, env=env, timeout=60, check=False)
        print(f"Run {status}. Results: {directory}\nCOPY RESULTS, THEN STOP THE RUNPOD POD IN THE CONSOLE. Rental billing continues until stopped.", flush=True)
    return 0 if status == "COMPLETE" else 1

if __name__ == "__main__":
    sys.exit(main())
