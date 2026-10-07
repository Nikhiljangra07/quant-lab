"""Check hardware and resolved environments before model work."""
import importlib.metadata as metadata
import sys
import torch
from common import RUN, write_json

def main():
    environment = sys.argv[1]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one visible CUDA GPU.")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0) or props.total_memory < 90 * 1024**3:
        raise RuntimeError(f"Expected 96GB RTX PRO 6000 Blackwell SM120; got {props.name}, SM{props.major}{props.minor}.")
    a = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    if (a @ a).float().mean().item() != 128:
        raise RuntimeError("BF16 CUDA smoke failed")
    versions = {p: metadata.version(p) for p in ("torch", "transformers", "compressed-tensors", "datasets")}
    if environment == "quant":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.gptq import GPTQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        versions["llmcompressor"] = metadata.version("llmcompressor")
        GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], dampening_frac=0.01)
        QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
    else:
        import vllm
        import lm_eval
        versions.update({p: metadata.version(p) for p in ("vllm", "lm_eval", "flashinfer-python")})
    write_json(RUN / "env" / f"{environment}-hardware.json", {
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory, "torch_cuda": torch.version.cuda,
        "packages": versions, "bf16_smoke": "PASS"})

if __name__ == "__main__":
    main()
