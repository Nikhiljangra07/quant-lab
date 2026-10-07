"""CPU-only regression checks for costly failure modes and invalid scientific claims."""
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import PROTOCOL as P, budget_deadline, sha256, shingles, validate_bench
from collect import paired_delta, summarize
from run import Runner
from benchmark import ready

def make_eval(root, arm, task, scores, hashes=None):
    folder = root / "eval" / arm
    folder.mkdir(parents=True, exist_ok=True)
    metric, filter_name = P['evals'][task]['metric'].split(',')
    result = {'results': {f'quant_{task}': {P['evals'][task]['metric']: sum(scores)/len(scores)}}}
    (folder / f'results_{task}.json').write_text(json.dumps(result))
    samples = [{'doc_id': i, 'doc_hash': (hashes[i] if hashes else f'doc-{i}'), 'prompt_hash': f'prompt-{i}',
                'filter': filter_name, metric: bool(score)} for i,score in enumerate(scores)]
    # GSM8K logs both extraction filters. Only strict-match should be counted.
    if task == 'gsm8k':
        samples.extend({**s,'filter':'flexible-extract'} for s in list(samples))
    (folder / f'samples_{task}.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    (folder / f'generation_{task}.json').write_text(json.dumps({'generation_count':len(scores),
        'length_stops':1, 'prompt_truncations':0}))

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'prepared.json').write_text(json.dumps({'eval_counts':{'gsm8k':4,'ifeval':4}}))
        (self.root/'calibration.jsonl').write_text('{}\n')
        for arm in ('BF16','W4A16'):
            for task in P['evals']:
                make_eval(self.root,arm,task,[1,0,1,1])
        (self.root/'quant_W4A16.json').write_text(json.dumps({'base':P['base'], 'n_calib':512,
            'calibration_sha256':sha256(self.root/'calibration.jsonl'), 'safetensors_sha256':{'weights':'fixture'},
            'size_bytes':2048}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_paired_evidence_passes(self):
        summary = summarize(self.root)
        self.assertEqual(summary['arms']['W4A16']['verdict'],'GO')
        self.assertEqual(summary['arms']['NVFP4']['verdict'],'NO-RESULT')
        self.assertEqual(summary['arms']['W4A16']['comparisons']['gsm8k']['n'],4)

    def test_missing_baseline_is_no_result(self):
        (self.root/'eval/BF16/results_gsm8k.json').unlink()
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

    def test_missing_ifeval_is_no_result(self):
        (self.root/'eval/W4A16/results_ifeval.json').write_text('{"results":{}}')
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

    def test_nan_metric_is_no_result(self):
        f=self.root/'eval/W4A16/results_gsm8k.json'
        f.write_text(json.dumps({'results':{'quant_gsm8k':{'exact_match,strict-match':float('nan')}}}))
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

    def test_mismatched_holdout_is_no_result(self):
        make_eval(self.root,'W4A16','gsm8k',[1,0,1,1],hashes=['changed','doc-1','doc-2','doc-3'])
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

    def test_incomplete_sample_set_is_no_result(self):
        make_eval(self.root,'W4A16','ifeval',[1,1,1])
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

    def test_score_degradation_is_no_go(self):
        make_eval(self.root,'W4A16','gsm8k',[0,0,0,0])
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-GO')

    def test_changed_calibration_is_no_result(self):
        (self.root/'calibration.jsonl').write_text('{"changed":true}\n')
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

    def test_generation_truncation_is_no_result(self):
        (self.root/'eval/W4A16/generation_ifeval.json').write_text(json.dumps({'generation_count':4,
            'length_stops':0, 'prompt_truncations':1}))
        self.assertEqual(summarize(self.root)['arms']['W4A16']['verdict'],'NO-RESULT')

class RuntimeTests(unittest.TestCase):
    def test_budget_accounts_for_time_before_runner(self):
        started=datetime.fromtimestamp(time.time()-3600,timezone.utc).isoformat()
        deadline=budget_deadline(started,2,8,1)
        self.assertAlmostEqual(deadline-time.time(),2.5*3600,delta=2)

    def test_invalid_budget_inputs_rejected(self):
        for rate in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                budget_deadline(datetime.now(timezone.utc).isoformat(),rate,8,1)
        with self.assertRaises(ValueError):
            budget_deadline('2026-10-07T12:00:00',2,8,1)

    def test_timeout_kills_stage_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Runner(tmp,time.time()+60)
            started=time.monotonic()
            with self.assertRaises(TimeoutError):
                r.stage('hang',[sys.executable,'-c','import time; time.sleep(30)'],max_seconds=.15)
            self.assertLess(time.monotonic()-started,3)
            self.assertEqual(r.events[0]['status'],'NO-RESULT')

    def test_required_failure_stops_following_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Runner(tmp,time.time()+60)
            with self.assertRaises(RuntimeError):
                r.stage('fail',[sys.executable,'-c','raise SystemExit(5)'])
                r.stage('should-not-run',[sys.executable,'-c','print("wrong")'])
            self.assertEqual(len(r.events),1)

    def test_optional_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Runner(tmp,time.time()+60)
            self.assertFalse(r.stage('optional',[sys.executable,'-c','raise SystemExit(5)'],required=False))

    def test_http_errors_are_not_healthy(self):
        with patch('benchmark.urllib.request.urlopen',side_effect=OSError('HTTP 500')):
            self.assertFalse(ready('http://localhost:1','expected'))

    def test_other_model_is_not_healthy(self):
        class Health:
            status=200
            def __enter__(self): return self
            def __exit__(self,*args): pass
        with patch('benchmark.urllib.request.urlopen',return_value=Health()), patch('benchmark.get_json',return_value={'data':[{'id':'other'}]}):
            self.assertFalse(ready('http://localhost:1','expected'))

    def test_benchmark_rejects_eos_shortening_and_failures(self):
        valid={'completed':2,'failed':0,'output_lens':[1024,1024],'total_output_tokens':2048,
            'input_lens':[1024,1024],'total_input_tokens':2048,
            'duration':10,'output_throughput':204.8,'mean_ttft_ms':20,'median_tpot_ms':2}
        self.assertEqual(validate_bench(valid,2,1024),[])
        for change in ({'failed':1},{'output_lens':[1024,12]},{'completed':1},{'duration':0}, {'input_lens':[1024,1000]}):
            self.assertTrue(validate_bench({**valid,**change},2,1024))

    def test_normalized_shingle_overlap(self):
        self.assertTrue(shingles('One two three four five six seven eight!') & shingles('ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT'))

if __name__ == '__main__':
    unittest.main()
