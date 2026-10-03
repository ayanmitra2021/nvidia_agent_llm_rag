"""Run the real frontend evaluator and task grader against a running notebook server.

Use the built frontend image, an isolated copy of /notebooks, and the same results
volume at /results and /dli/assessment_results. Mount assessment/tasktest.py at
/tasktest.py. The server at lab:9012 must serve the original assessment or its
separate solution key. Live model misses use the configured provider.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, '/web')
from frontend_block import rag_eval

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--expect', choices=('pass', 'retriever-failure', 'document-failure'), required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

# A failed rerun must revoke an earlier passing result.
Path('/results/PASSED').touch()
started = time.monotonic()
history = []
for history in rag_eval(history, 'RAG'):
    pass
grade = json.loads(subprocess.check_output([sys.executable, '/tasktest.py'], text=True))
result = {'expected': args.expect, 'score': grade['score'],
          'seconds': round(time.monotonic() - started, 2), 'history': history}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'history'}), flush=True)
expected_score = 100 if args.expect == 'pass' else 0
assert grade['score'] == expected_score, result
if args.expect != 'pass':
    message = ' '.join(str(row.get('content', '')) for row in history)
    required = 'Retriever check failed' if args.expect == 'retriever-failure' else 'Unreadable document text'
    assert required in message, message
else:
    assert sum('Preference probe:' in str(row.get('content', '')) for row in history) == 8
