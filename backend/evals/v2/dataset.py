"""Data contracts and integrity checks. Gold labels never come from app outputs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = 'fitzen-scenarios-2.1.0'
CATEGORIES = {'planning': 61, 'extraction': 50, 'authorization': 60,
              'rag': 100, 'memory': 40, 'reliability': 60, 'transport': 30}
KINDS = {'plan', 'extract_workout', 'extract_food', 'write', 'urgent', 'tool_denied',
         'rag', 'memory', 'reliability', 'transport'}
SPLITS = {'development', 'calibration', 'test'}


def load_cases(path=None):
    path = Path(path) if path else ROOT / 'cases.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(cases, corpus=None, *, require_counts=True):
    """Fail on family leakage, missing gold, unsafe fixtures, or invented evidence."""
    problems, ids, families, fingerprints = [], set(), defaultdict(set), set()
    for c in cases:
        cid = c.get('id', '?')
        missing = {'id', 'version', 'category', 'kind', 'family', 'split', 'input',
                   'expected', 'rationale', 'provenance', 'severity', 'core'} - c.keys()
        if missing:
            problems.append(f'{cid}: missing {sorted(missing)}')
            continue
        if cid in ids:
            problems.append(f'{cid}: duplicate id')
        ids.add(cid)
        if c['version'] != VERSION or c['kind'] not in KINDS or c['split'] not in SPLITS:
            problems.append(f'{cid}: unsupported version/kind/split')
        if c['category'] not in CATEGORIES or not c['expected'] or not c['rationale']:
            problems.append(f'{cid}: missing category/gold/rationale')
        if not c['input'].get('message') or not c['input'].get('now') or c['input'].get('timezone') != 'Asia/Shanghai':
            problems.append(f'{cid}: incomplete input context')
        families[c['family']].add(c['split'])
        fingerprint = json.dumps({k: c.get(k) for k in ('kind', 'input', 'conditions', 'expected')}, sort_keys=True, ensure_ascii=False)
        if fingerprint in fingerprints:
            problems.append(f'{cid}: identical scenario (IDs do not make a new case)')
        fingerprints.add(fingerprint)
        if c['kind'] == 'rag':
            required = {'answerable', 'evidence', 'claims', 'forbidden_claims'}
            if not required.issubset(c['expected']):
                problems.append(f'{cid}: incomplete RAG gold')
            if c['expected'].get('answerable') and not c['expected'].get('evidence'):
                problems.append(f'{cid}: answerable without evidence')
            for e in c['expected'].get('evidence', []):
                if not e.get('snippet') or not e.get('source'):
                    problems.append(f'{cid}: empty evidence')
                elif corpus is not None and (e['source'] not in corpus or e['snippet'] not in corpus[e['source']]['content']):
                    problems.append(f'{cid}: evidence not in frozen corpus')
        if c['category'] != 'rag' and c['split'] == 'calibration':
            problems.append(f'{cid}: only RAG uses threshold calibration')
    for family, splits in families.items():
        if len(splits) != 1:
            problems.append(f'{family}: split leakage {sorted(splits)}')
    counts = Counter(c.get('category') for c in cases)
    if require_counts and dict(counts) != CATEGORIES:
        problems.append(f'category counts {dict(counts)} != {CATEGORIES}')
    if require_counts and sum(c.get('core', False) for c in cases) != 80:
        problems.append('core must contain exactly 80 scenarios')
    return problems


def split_for(family, *, rag=False):
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 10
    if rag:
        return 'development' if bucket < 2 else 'calibration' if bucket < 5 else 'test'
    return 'development' if bucket < 3 else 'test'
