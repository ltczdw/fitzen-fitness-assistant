"""FitZen v2 scenarios: provider-free, network-denied, isolated SQLite evaluation.

Run from backend: python scripts/eval_scenarios.py [--core] [--split test]
Exit 0=all selected scenarios passed, 1=observed failures, 2=invalid/unsafe run.
Application behavior failures are deliberately preserved, not fixed by labels.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

BACKEND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
DISABLED={'DATABASE_URL':'','LLM_API_KEY':'','LLM_BASE_URL':'','RETRIEVAL_EMBEDDING_API_KEY':'',
          'RETRIEVAL_EMBEDDING_BASE_URL':'','RERANK_API_KEY':'','RERANK_BASE_URL':'',
          'LANGSMITH_API_KEY':'','MCP_URL':'','REDIS_ENABLED':'false',
          'LANGSMITH_TRACING':'false','RERANK_LEGACY_EMBEDDING_CONFIG':'false',
          'LANGCHAIN_TRACING_V2':'false','LANGCHAIN_API_KEY':'',
          'PLANNER_SHADOW_ENABLED':'false','RAG_QUERY_REWRITE_ENABLED':'false',
          'MEMORY_SUMMARY_ENABLED':'false','MEMORY_BACKGROUND_ENABLED':'false'}
# Do this before any app import, in Python so empty strings survive PowerShell versions.
os.environ.update(DISABLED)
os.environ.pop('FITZEN_DB_TESTS',None)
os.environ.pop('FITZEN_QUALITY_DB',None)




def metadata():
    def git(*args):
        return subprocess.check_output(['git','-C',str(BACKEND.parent),*args],stderr=subprocess.DEVNULL).decode('utf-8').strip()
    try:head,dirty=git('rev-parse','HEAD'),bool(git('status','--porcelain'))
    except subprocess.CalledProcessError:head,dirty=None,None
    # Includes tracked and untracked application Python files but no .env/data.
    files=sorted((BACKEND/'app').rglob('*.py'))
    fingerprint=hashlib.sha256()
    for path in files:
        fingerprint.update(path.relative_to(BACKEND).as_posix().encode())
        fingerprint.update(path.read_bytes())
    evaluator=hashlib.sha256()
    for path in sorted((BACKEND/'evals/v2').glob('*.py'))+[Path(__file__).resolve()]:
        evaluator.update(path.relative_to(BACKEND).as_posix().encode())
        evaluator.update(path.read_bytes())
    return {'head':head,'dirty_worktree':dirty,'application_source_sha256':fingerprint.hexdigest(),
            'evaluation_source_sha256':evaluator.hexdigest(),
            'python':sys.version.split()[0],'database':'temporary SQLite per scenario',
            'mode':'deterministic_mock_and_lexical_fallback','real_provider_tested':False,
            'network':'socket connections prohibited','generation_evaluated':False}


def save_report(destination, payload):
    destination.mkdir(parents=True,exist_ok=True)
    name=datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S-%fZ')
    jsonpath=destination/(name+'.json')
    jsonpath.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    s=payload['summary']
    lines=['# FitZen v2 scenario report','',f'- Dataset: {payload["dataset_version"]}',
           f'- Selected: {s["scenarios"]} scenarios / {s["families"]} families',
           f'- Counts: {s["counts"]}', f'- Release gate: {"PASS" if s["release_gate_passed"] else "FAIL or incomplete coverage"}',
           f'- Source hash: `{payload["environment"]["application_source_sha256"]}`',
           '- All provider credentials disabled; no real DeepSeek test; no generated-answer faithfulness score.',
           '- SQLite/mock measurements are not PostgreSQL locking or real-model capacity.',
           '', '| Category | Passed | Total | Rate |','|---|---:|---:|---:|']
    for category,row in s['categories'].items():lines.append(f'| {category} | {row["passed"]} | {row["total"]} | {row["pass_rate"]:.3f} |')
    lines+=['','## Default gate and evidence metrics by split','',
            '```json',json.dumps(s['rag_default_by_split'],ensure_ascii=False,indent=2),'```',
            '', '## Calibration suggestions (calibration rows only; not applied)','',
            '```json',json.dumps(s['calibration'],ensure_ascii=False,indent=2),'```',
            '', '## Failed scenarios','']
    for row in payload['results']:
        if row['status']!='passed':
            lines+= [f'- **{row["id"]}** ({row["split"]}, {row["severity"]}): '+ '; '.join(row['failures'])]
    if payload.get('gpt6_luna_simulation'):
        sim=payload['gpt6_luna_simulation']
        lines+=['','## GPT-6-luna simulation (not DeepSeek)','',
                f'- {sim["passed"]}/{sim["count"]} independent contract checks passed.',
                '- Raw outputs are preserved. Free-form/adversarial samples and normal probes are labelled separately.',
                '- This is validation/routing and selected SQLite write testing; no provider accuracy or generated-answer score.']
        for row in sim['results']:
            if row['status']!='passed':lines.append(f'- {row["id"]}: '+ '; '.join(row['failures']))
    lines+=['','Full expected/actual and timing are in the adjacent JSON. Inputs are synthetic/public only.','']
    mdpath=destination/(name+'.md')
    mdpath.write_text('\n'.join(lines),encoding='utf-8')
    (destination/'LATEST.md').write_text(f'Latest report: [{mdpath.name}]({mdpath.name})\n',encoding='utf-8')
    return mdpath


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--with-simulations',action='store_true',help='score stored GPT-6-luna outputs against independent gold')
    parser.add_argument('--category',choices=('planning','extraction','authorization','rag','memory','reliability','transport'))
    parser.add_argument('--split',choices=('development','calibration','test'))
    parser.add_argument('--output',type=Path,default=BACKEND/'evals/v2/reports')
    args=parser.parse_args()
    from evals.v2.dataset import load_cases,validate,ROOT,VERSION,digest
    allcases=load_cases()
    corpus=json.loads((ROOT/'corpus.json').read_text(encoding='utf-8'))
    from evals.v2.build import corpus_snapshot
    if corpus_snapshot()!=corpus:
        print('Frozen public corpus differs from current seed. Review and version the dataset before evaluation.');return 2
    problems=validate(allcases,corpus)
    if problems:
        print(json.dumps({'integrity_errors':problems},ensure_ascii=False));return 2
    if args.validate_only:
        print(json.dumps({'valid':True,'scenarios':len(allcases),'families':len({c['family'] for c in allcases}),
                          'core':sum(c['core'] for c in allcases),'split_counts':dict(Counter(c['split'] for c in allcases))}));return 0
    from app.core.config import settings
    probe={name:bool(getattr(settings,field)) for name,field in {
        'llm':'llm_api_key','embedding':'retrieval_embedding_api_key','rerank':'rerank_api_key',
        'langsmith':'langsmith_api_key','mcp':'mcp_url'}.items()}
    print('effective_provider_configuration='+json.dumps(probe))
    if any(probe.values()) or settings.database_url or settings.langsmith_tracing:return 2
    cases=[c for c in allcases if (not args.core or c['core']) and
           (not args.category or c['category']==args.category) and (not args.split or c['split']==args.split)]
    if not cases:print('No scenarios selected');return 2
    environment=metadata()
    rows=[]
    from evals.v2.harness import execute,CheckFailure
    from evals.v2.scoring import summarize
    # Expected fault injections are recorded per case; do not flood stdout
    # with application logger tracebacks. Unexpected exceptions stay in JSON.
    logging.disable(logging.CRITICAL)
    from evals.v2.safety import deny_network
    with deny_network() as network_audit:
        for case in cases:
            started=time.perf_counter()
            row={key:case[key] for key in ('id','category','kind','family','split','severity')}
            row.update(expected=case['expected'],input=case['input'],failures=[])
            try:row.update(status='passed',actual=execute(case))
            except CheckFailure as exc:row.update(status='failed',actual=exc.actual,failures=exc.failures)
            except Exception as exc:row.update(status='error',actual={},failures=['harness/application exception: '+type(exc).__name__+': '+str(exc)[:300]])
            row['latency_ms']=round((time.perf_counter()-started)*1000,2)
            rows.append(row)
            if len(rows)%20==0:print(f'evaluated={len(rows)}/{len(cases)}',flush=True)
    environment['network_audit']=network_audit
    payload={'dataset_version':VERSION,'dataset_sha256':digest(ROOT/'cases.jsonl'),
             'corpus_sha256':digest(ROOT/'corpus.json'),'environment':environment,
             'selection':{'core':args.core,'category':args.category,'split':args.split},
             'summary':summarize(rows),'results':rows}
    if args.with_simulations:
        from evals.v2.simulation import evaluate_simulations
        payload['gpt6_luna_simulation']=evaluate_simulations()
    report=save_report(args.output,payload)
    print(json.dumps({'counts':payload['summary']['counts'],'release_gate_passed':payload['summary']['release_gate_passed'],
                      'simulation_passed':payload.get('gpt6_luna_simulation',{}).get('passed'),
                      'simulation_total':payload.get('gpt6_luna_simulation',{}).get('count'),
                      'report':str(report)},ensure_ascii=False))
    if network_audit['blocked_connections']:return 2
    simulations=payload.get('gpt6_luna_simulation')
    if simulations and simulations['network_audit']['blocked_connections']:return 2
    return 0 if all(r['status']=='passed' for r in rows) and (not simulations or simulations['passed']==simulations['count']) else 1


if __name__=='__main__':raise SystemExit(main())
