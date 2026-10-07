"""Exercise the full benchmark wrapper with a local fake serving process, no GPU."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark
from common import PROTOCOL

FAKE = r'''
import json,sys
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
def value(flag): return sys.argv[sys.argv.index(flag)+1]
if sys.argv[1]=='serve':
    assert '--no-enable-prefix-caching' in sys.argv
    name=value('--served-model-name')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            data={'data':[{'id':name}]} if self.path=='/v1/models' else {}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'message':{'content':'4'}}]}).encode())
    HTTPServer(('127.0.0.1',int(value('--port'))),Handler).serve_forever()
else:
    assert '--save-detailed' in sys.argv and '--ignore-eos' in sys.argv
    assert value('--temperature')=='0' and value('--random-range-ratio')=='0.0'
    n=int(value('--num-prompts'));i=int(value('--random-input-len'));o=int(value('--random-output-len'))
    data={'completed':n,'failed':0,'input_lens':[i]*n,'output_lens':[o]*n,
          'total_input_tokens':n*i,'total_output_tokens':n*o,'duration':10,
          'output_throughput':n*o/10,'mean_ttft_ms':1,'median_tpot_ms':1}
    (Path(value('--result-dir'))/value('--result-filename')).write_text(json.dumps(data))
'''

class LifecycleTests(unittest.TestCase):
    def test_server_smoke_bench_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            executable=root/'venv_serve/bin/vllm'
            executable.parent.mkdir(parents=True)
            executable.write_text('#!'+sys.executable+'\n'+FAKE)
            executable.chmod(0o755)
            run=root/'results';run.mkdir()
            (run/'prepared.json').write_text(json.dumps({'base_path':'fixture-model'}))
            protocol=copy.deepcopy(PROTOCOL)
            protocol['bench'].update(concurrency=[1],prompts={'1':2},repetitions=2,warmups=1)
            # Hardware snapshot uses a stub executable rather than an NVIDIA driver.
            nvidia=root/'nvidia-smi'
            nvidia.write_text('#!/bin/sh\necho fake-gpu-memory\n');nvidia.chmod(0o755)
            with patch.object(benchmark,'ROOT',root), patch.object(benchmark,'RUN',run), \
                 patch.object(benchmark,'P',protocol), patch.object(benchmark,'verify_freeze'), \
                 patch.object(sys,'argv',['benchmark.py','BF16']), \
                 patch.dict(os.environ,{'PATH':str(root)+os.pathsep+os.environ['PATH']}):
                benchmark.main()
            folder=run/'bench/BF16'
            self.assertTrue((folder/'smoke.json').exists())
            self.assertTrue((folder/'c1-r2.json').exists())
            command=json.loads((folder/'server-command.json').read_text())['command']
            port=command[command.index('--port')+1]
            self.assertFalse(benchmark.ready('http://127.0.0.1:'+port,'anything'))

if __name__=='__main__':
    unittest.main()
