"""Pinned harness evaluation with sample scores and generation termination telemetry."""
import argparse
import hashlib
import json
import time
from common import PROTOCOL as P, ROOT, RUN, verify_freeze, write_json

def serializable(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unserializable result: {type(value).__name__}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["BF16", "W4A16", "NVFP4"])
    a = ap.parse_args()
    verify_freeze()
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.models.vllm_causallms import VLLM
    import lm_eval.models.vllm_causallms as backend
    # The standard backend can silently left-truncate prompts; reject that here.
    original_truncate = backend.maybe_truncate
    def checked_truncate(*args, **kwargs):
        original = kwargs.get("context", kwargs.get("tokens", args[0] if args else None))
        if original is None:
            raise RuntimeError("Cannot inspect harness truncation API; update with a logged amendment.")
        original_length = len(original)
        result = original_truncate(*args, **kwargs)
        if len(result[0]) < original_length:
            raise RuntimeError("Prompt truncation would occur. No valid result for this protocol.")
        return result
    backend.maybe_truncate = checked_truncate
    telemetry = []
    class MeasuredVLLM(VLLM):
        def _model_generate(self, *args, **kwargs):
            outputs = super()._model_generate(*args, **kwargs)
            for request in outputs:
                for out in request.outputs:
                    telemetry.append({"request_id": request.request_id,
                        "prompt_tokens": len(request.prompt_token_ids), "output_tokens": len(out.token_ids),
                        "finish_reason": out.finish_reason, "stop_reason": out.stop_reason})
            return outputs
    prep = json.loads((RUN / "prepared.json").read_text())
    model_path = prep["base_path"] if a.arm == "BF16" else str(RUN / "models" / a.arm)
    s = P["serving"]
    model_args = {"pretrained": model_path, "dtype": s["dtype"], "max_model_len": s["max_model_len"],
                  "seed": P["seed"], "gpu_memory_utilization": s["gpu_memory_utilization"],
                  "enable_prefix_caching": s["enable_prefix_caching"], "max_gen_toks": 1024}
    if a.arm == "NVFP4":
        model_args["linear_backend"] = s["nvfp4_linear_backend"]
    lm = MeasuredVLLM(**model_args)
    tm = TaskManager(include_path=str(RUN / "tasks"))
    dest = RUN / "eval" / a.arm
    dest.mkdir(parents=True)
    for name, spec in P["evals"].items():
        telemetry.clear()
        started = time.monotonic()
        result = simple_evaluate(model=lm, tasks=[f"quant_{name}"], task_manager=tm,
            batch_size="auto", apply_chat_template=True, fewshot_as_multiturn=True,
            random_seed=P["seed"], numpy_random_seed=P["seed"], torch_random_seed=P["seed"],
            fewshot_random_seed=P["seed"], bootstrap_iters=1000, log_samples=True)
        if result is None:
            raise RuntimeError("Harness returned no results")
        task = f"quant_{name}"
        samples = result.pop("samples")[task]
        (dest / f"samples_{name}.jsonl").write_text("".join(json.dumps(x, default=serializable) + "\n" for x in samples))
        (dest / f"results_{name}.json").write_text(json.dumps(result, indent=2, default=serializable) + "\n")
        write_json(dest / f"generation_{name}.json", {"seconds": time.monotonic() - started,
            "generation_count": len(telemetry), "length_stops": sum(t["finish_reason"] == "length" for t in telemetry),
            "prompt_truncations": 0, "requests": telemetry})
        if len(telemetry) != prep["eval_counts"][name]:
            raise RuntimeError(f"Incomplete generation telemetry for {name}")
        from collect import load_eval
        load_eval(RUN, a.arm, name, prep["eval_counts"][name])
    verify_freeze()

if __name__ == "__main__":
    main()
