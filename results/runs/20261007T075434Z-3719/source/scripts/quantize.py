"""GPTQ W4A16 or NVFP4 PTQ using a single frozen base/calibration selection."""
import argparse
import json
import time
from common import PROTOCOL as P, RUN, sha256, verify_freeze, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', choices=['W4A16', 'NVFP4'], required=True)
    a = ap.parse_args()
    verify_freeze()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    torch.manual_seed(P['seed'])
    torch.cuda.manual_seed_all(P['seed'])
    prepared = json.loads((RUN / 'prepared.json').read_text())
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(prepared['base_path'], dtype=torch.bfloat16,
                                               device_map='cuda', local_files_only=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(prepared['base_path'], local_files_only=True)
    rows = [json.loads(line) for line in (RUN / 'calibration.jsonl').read_text().splitlines()]
    ds = Dataset.from_list([{k: r[k] for k in ('input_ids', 'attention_mask')} for r in rows])
    recipe = (GPTQModifier(targets='Linear', scheme='W4A16', ignore=['lm_head'], dampening_frac=0.01)
              if a.scheme == 'W4A16' else QuantizationModifier(targets='Linear', scheme='NVFP4', ignore=['lm_head']))
    torch.cuda.reset_peak_memory_stats()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=P['calibration']['max_tokens'],
            num_calibration_samples=len(rows))
    out = RUN / 'models' / a.scheme
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    qc = json.loads((out / 'config.json').read_text()).get('quantization_config', {})
    groups = qc.get('config_groups', {})
    if not groups or not any(g.get('weights', {}).get('num_bits') == 4 for g in groups.values()):
        raise RuntimeError('Saved checkpoint did not contain a 4-bit quantization configuration.')
    if a.scheme == 'W4A16' and not all(g['weights'].get('group_size') == 128 and g['weights'].get('symmetric')
                                     for g in groups.values()):
        raise RuntimeError('Saved GPTQ checkpoint violates g=128 symmetric protocol.')
    if a.scheme == 'NVFP4' and not all(g['weights'].get('type') == 'float' and
            g['weights'].get('group_size') == 16 and g.get('input_activations', {}).get('num_bits') == 4
            for g in groups.values()):
        raise RuntimeError('Saved NVFP4 checkpoint does not contain the specified FP4 weights/activations.')
    if model.lm_head.weight.dtype != torch.bfloat16:
        raise RuntimeError('lm_head did not remain BF16.')
    files = list(out.glob('*.safetensors'))
    if not files:
        raise RuntimeError('No quantized weights saved.')
    write_json(RUN / f'quant_{a.scheme}.json', {
        'scheme': a.scheme, 'base': prepared['base'], 'calibration_sha256': sha256(RUN / 'calibration.jsonl'),
        'n_calib': len(rows), 'quantization_config': qc, 'size_bytes': sum(p.stat().st_size for p in files),
        'safetensors_sha256': {p.name: sha256(p) for p in files},
        'seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
