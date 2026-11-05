"""Four-strategy diagnostic adapted from MEWHELP/eval_ch04.py.

Offline: lexical results, other strategies explicitly unavailable.
Proxy: hash vectors and deterministic lexical rerank, NOT real model quality.
Both modes deny network and database access; no API keys required or accepted.
"""
import argparse,asyncio,json,os,sys,time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

BACKEND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND))
for key in ('DATABASE_URL','LLM_API_KEY','LLM_BASE_URL','RETRIEVAL_EMBEDDING_API_KEY',
            'RETRIEVAL_EMBEDDING_BASE_URL','RERANK_API_KEY','RERANK_BASE_URL',
            'MCP_URL','LANGSMITH_API_KEY','LANGCHAIN_API_KEY'):os.environ[key]=''
for key in ('REDIS_ENABLED','LANGSMITH_TRACING','LANGCHAIN_TRACING_V2',
            'RERANK_LEGACY_EMBEDDING_CONFIG','RAG_QUERY_REWRITE_ENABLED','MEMORY_BACKGROUND_ENABLED'):os.environ[key]='false'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('offline','proxy'),default='offline')
    parser.add_argument('--split',choices=('development','calibration','test'),default='test')
    parser.add_argument('--output',type=Path,default=BACKEND/'evals/v2/strategy-reports')
    args=parser.parse_args()
    from evals.v2.dataset import load_cases,ROOT,digest,validate
    from evals.v2.safety import deny_network
    from scripts.eval_scenarios import metadata
    from app.services.rag.pipeline import HybridRAGPipeline,_embedding,_cosine
    from app.services.rag.retrieval import embedding_client
    from app.services.rag.model_reranker import model_reranker
    from app.services.rag.bm25 import BM25Index
    cases=load_cases();corpus=json.loads((ROOT/'corpus.json').read_text(encoding='utf-8'))
    if validate(cases,corpus):return 2
    cases=[c for c in cases if c['kind']=='rag' and c['split']==args.split]
    class Adapter(HybridRAGPipeline):
        def vector(self,qvec,k):
            ranked=sorted(((i,_cosine(qvec,_embedding(h.title+' '+h.content,1024))) for i,h in enumerate(self.documents)),key=lambda x:x[1],reverse=True)
            return [replace(self.documents[i],score=score,chunk_id=f'seed:{i}') for i,score in ranked[:k]]
    async def fake_embed(texts):return [_embedding(t,1024) for t in texts]
    async def fake_rerank(query,chunks,top_n=3):
        index=BM25Index([h.title+' '+h.content for h in chunks])
        return [replace(chunks[i],score=score,metadata={**chunks[i].metadata,'rerank_status':'proxy'})
                for i,score in index.rank(query,top_n) if score>0]
    async def evaluate():
        strategies={}
        for strategy in ('bm25','vector','hybrid','hybrid_rerank'):
            if args.mode=='offline' and strategy!='bm25':
                strategies[strategy]={'status':'unavailable','reason':'providers_disabled','metrics':None};continue
            rows=[]
            for case in cases:
                started=time.perf_counter()
                hits=await Adapter().asearch(case['input']['message'],top_k=4,strategy=strategy)
                evidence=case['expected']['evidence']
                matches=[any(h.source==e['source'] and e['snippet'] in h.content for e in evidence) for h in hits]
                covered=[any(h.source==e['source'] and e['snippet'] in h.content for h in hits) for e in evidence]
                rank=next((i for i,m in enumerate(matches,1) if m),None)
                rows.append({'id':case['id'],'answerable':case['expected']['answerable'],
                    'mrr':1/rank if rank else 0,'evidence_recall':sum(covered)/len(covered) if covered else None,
                    'candidate_precision':sum(matches)/len(hits) if hits else 0,
                    'sources':[h.source for h in hits],'latency_ms':round((time.perf_counter()-started)*1000,2)})
            positives=[r for r in rows if r['answerable']]
            def mean(field):return sum(r[field] for r in positives)/len(positives) if positives else None
            strategies[strategy]={'status':'diagnostic_proxy' if args.mode=='proxy' else 'measured_lexical',
                'metrics':{k:mean(k) for k in ('mrr','evidence_recall','candidate_precision')},'rows':rows}
        return strategies
    with deny_network() as audit:
        if args.mode=='proxy':
            with patch.object(embedding_client,'aembed',fake_embed),patch.object(model_reranker,'arerank',fake_rerank):result=asyncio.run(evaluate())
        else:result=asyncio.run(evaluate())
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/('strategies-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')+'.json')
    payload={'mode':args.mode,'split':args.split,'samples':len(cases),'k':4,'dataset_sha256':digest(ROOT/'cases.jsonl'),
             'environment':metadata(),'network_audit':audit,'real_provider_tested':False,
             'generation_evaluated':False,'strategies':result}
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    path.with_suffix('.md').write_text('# Four retrieval strategies\n\n'+
        f'Mode: {args.mode}; samples: {len(cases)}; no real provider or generation tested.\n\n'+
        '| Strategy | Status | MRR | Evidence recall | Candidate precision |\n|---|---|---:|---:|---:|\n'+
        '\n'.join(f'| {k} | {v["status"]} | {(v["metrics"] or {}).get("mrr")} | {(v["metrics"] or {}).get("evidence_recall")} | {(v["metrics"] or {}).get("candidate_precision")} |' for k,v in result.items()),encoding='utf-8')
    print(json.dumps({'report':str(path),'mode':args.mode,'samples':len(cases),'network':audit}))
    return 2 if audit['blocked_connections'] else 0


if __name__=='__main__':raise SystemExit(main())
