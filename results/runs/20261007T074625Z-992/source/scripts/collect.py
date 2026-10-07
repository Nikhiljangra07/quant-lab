"""Produce defensible summaries; missing or mismatched evidence is always NO-RESULT."""
import json
import math
import statistics
from pathlib import Path
from common import PROTOCOL as P, RUN, valid_score, validate_bench, verify_freeze, write_json

ARMS = ('BF16', 'W4A16', 'NVFP4')


def load_eval(directory, arm, task, expected):
    folder = Path(directory) / 'eval' / arm
    name = f'quant_{task}'
    metric, filter_name = P['evals'][task]['metric'].split(',')
    results = json.loads((folder / f'results_{task}.json').read_text())
    score = results['results'][name][P['evals'][task]['metric']]
    if not valid_score(score):
        raise ValueError('Metric must be a finite fraction in [0,1]')
    entries = [json.loads(line) for line in (folder / f'samples_{task}.jsonl').read_text().splitlines() if line.strip()]
    selected = [x for x in entries if x.get('filter') == filter_name]
    if len(selected) != expected:
        raise ValueError(f'Expected {expected} scored examples, found {len(selected)}')
    samples = {}
    for entry in selected:
        key = str(entry['doc_id'])
        value = entry[metric]
        if key in samples or value not in (True, False, 0, 1):
            raise ValueError('Duplicate example ID or non-binary sample metric')
        if not entry.get('doc_hash') or not entry.get('prompt_hash'):
            raise ValueError('Missing example or prompt provenance')
        samples[key] = {'value': float(value), 'doc_hash': entry['doc_hash'], 'prompt_hash': entry['prompt_hash']}
    if abs(statistics.mean(x['value'] for x in samples.values()) - score) > 1e-8:
        raise ValueError('Aggregate score disagrees with sample scores')
    gen = json.loads((folder / f'generation_{task}.json').read_text())
    if gen['generation_count'] != expected or gen['prompt_truncations'] != 0:
        raise ValueError('Incomplete generation telemetry or truncated input prompts')
    return {'score': score, 'samples': samples, 'length_stops': gen['length_stops']}


def paired_delta(reference, candidate):
    left, right = reference['samples'], candidate['samples']
    if not left or left.keys() != right.keys():
        raise ValueError('Unpaired example IDs')
    for key in left:
        if any(left[key][field] != right[key][field] for field in ('doc_hash', 'prompt_hash')):
            raise ValueError('Example/prompt hashes differ across arms')
    values = [right[k]['value'] - left[k]['value'] for k in left]
    mean = statistics.mean(values)
    # Descriptive normal interval for paired per-example differences, not an equivalence test.
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 1.0
    return {'delta_pp': 100 * mean, 'normal_95_interval_pp': [100 * max(-1, mean-half), 100 * min(1, mean+half)],
            'n': len(values), 'quant_only_passes': sum(x == 1 for x in values), 'bf16_only_passes': sum(x == -1 for x in values)}


def summarize(directory):
    directory = Path(directory)
    summary = {'arms': {}, 'warnings': [], 'benchmark': {}}
    try:
        prep = json.loads((directory / 'prepared.json').read_text())
        counts = prep['eval_counts']
    except (OSError, ValueError, KeyError) as error:
        summary['warnings'].append(f'No prepared dataset manifest: {error}')
        counts = {}
        prep = {}
    evidence = {}
    for arm in ARMS:
        errors, tasks = [], {}
        for task in P['evals']:
            try:
                tasks[task] = load_eval(directory, arm, task, counts[task])
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f'{task}: {type(error).__name__}: {error}')
        if arm != 'BF16':
            try:
                quant = json.loads((directory / f'quant_{arm}.json').read_text())
                if quant['base'] != P['base'] or quant['n_calib'] != P['calibration']['samples'] or not quant['safetensors_sha256']:
                    raise ValueError('Quantization provenance does not match protocol')
                from common import sha256
                if quant['calibration_sha256'] != sha256(directory / 'calibration.jsonl'):
                    raise ValueError('Calibration differs from the saved selection')
                size = quant['size_bytes']
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f'quantization: {error}')
                size = None
        else:
            size = prep.get('base_size_bytes')
        evidence[arm] = tasks
        summary['arms'][arm] = {'verdict': 'NO-RESULT' if errors else 'REFERENCE' if arm == 'BF16' else 'PENDING',
            'errors': errors, 'scores': {t: e['score'] for t, e in tasks.items()},
            'length_stops': {t: e['length_stops'] for t, e in tasks.items()}, 'size_bytes': size}
    for arm in ('W4A16', 'NVFP4'):
        row = summary['arms'][arm]
        if row['errors'] or summary['arms']['BF16']['errors']:
            row['verdict'] = 'NO-RESULT'
            if summary['arms']['BF16']['errors']:
                row['errors'].append('BF16 reference incomplete')
            continue
        try:
            comparisons = {t: paired_delta(evidence['BF16'][t], evidence[arm][t]) for t in P['evals']}
            row['comparisons'] = comparisons
            row['verdict'] = 'GO' if all(c['delta_pp'] >= -P['acceptance_margin_pp'] - 1e-9 for c in comparisons.values()) else 'NO-GO'
        except (ValueError, KeyError, TypeError) as error:
            row['verdict'] = 'NO-RESULT'
            row['errors'].append(f'paired comparison: {error}')
    for arm in ARMS:
        cells = {}
        for c in P['bench']['concurrency']:
            values, errors = [], []
            for rep in range(1, P['bench']['repetitions']+1):
                path = directory / 'bench' / arm / f'c{c}-r{rep}.json'
                try:
                    data = json.loads(path.read_text())
                    invalid = validate_bench(data, P['bench']['prompts'][str(c)], P['bench']['output_tokens'], P['bench']['input_tokens'])
                    if invalid:
                        raise ValueError('; '.join(invalid))
                    values.append(data)
                except (OSError, ValueError, TypeError) as error:
                    errors.append(f'{path.name}: {error}')
            cells[str(c)] = {'status': 'NO-RESULT' if errors else 'VALID', 'errors': errors}
            if not errors:
                for key in ('output_throughput', 'mean_ttft_ms', 'median_tpot_ms'):
                    vals = [v[key] for v in values]
                    cells[str(c)][key] = {'median': statistics.median(vals), 'min': min(vals), 'max': max(vals)}
        summary['benchmark'][arm] = cells
    return summary


def main():
    summary = summarize(RUN)
    try:
        verify_freeze()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        summary['warnings'].append(f'Freeze verification failed: {error}')
        for arm, row in summary['arms'].items():
            row['verdict'] = 'NO-RESULT'
            row['errors'].append('Source/calibration freeze not verifiable')
    lines = ['# Quantization results', '', f'Run: `{RUN.name}`', '',
             'Quality GO is a point-estimate rule (no more than 2 percentage points worse on either task). It is not proof of equivalence.', '',
             '| arm | GSM8K strict | IFEval prompt strict | quality verdict | weight GB |', '|---|---:|---:|---|---:|']
    for arm, row in summary['arms'].items():
        scores = [f"{row['scores'][t]*100:.2f}%" if t in row['scores'] else '—' for t in P['evals']]
        size = f"{row['size_bytes']/1e9:.3f}" if row['size_bytes'] is not None else '—'
        lines.append(f"| {arm} | {scores[0]} | {scores[1]} | {row['verdict']} | {size} |")
    lines += ['', '## Paired differences and generation limits', '']
    for arm, row in summary['arms'].items():
        for task, comp in row.get('comparisons', {}).items():
            lo, hi = comp['normal_95_interval_pp']
            lines.append(f"- {arm} {task}: {comp['delta_pp']:+.2f} pp; descriptive paired normal 95% interval [{lo:+.2f}, {hi:+.2f}], n={comp['n']}.")
        if row['length_stops']:
            lines.append(f"- {arm} generation-cap stops: {row['length_stops']}. These remain scored in the full-set evaluation.")
    lines += ['', '## Serving results', '', 'Median and range across three repetitions; fixed 1024-token inputs/outputs, prefix caching disabled.', '',
              '| arm | concurrency | output tok/s, median [min,max] | TTFT ms, median | TPOT ms, median |', '|---|---:|---|---:|---:|']
    for arm, cells in summary['benchmark'].items():
        for c, cell in cells.items():
            if cell['status'] != 'VALID':
                lines.append(f'| {arm} | {c} | NO-RESULT | — | — |')
            else:
                t = cell['output_throughput']
                lines.append(f"| {arm} | {c} | {t['median']:.1f} [{t['min']:.1f},{t['max']:.1f}] | {cell['mean_ttft_ms']['median']:.1f} | {cell['median_tpot_ms']['median']:.1f} |")
    lines += ['', '## Missing or invalid evidence', '']
    lines.extend('- ' + w for w in summary['warnings'])
    for arm, row in summary['arms'].items():
        lines.extend(f'- {arm}: {err}' for err in row['errors'])
    lines += ['', 'Raw stage logs, sample scores, termination telemetry and benchmark JSON are retained alongside this report. Long-context evaluation and MTP preservation are outside this protocol.']
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(RUN / 'summary.json', summary)
    (RUN / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved {RUN}/RESULTS.md')
    for arm,row in summary['arms'].items():
        print(f"{arm}: {row['verdict']}")


if __name__ == '__main__':
    main()
