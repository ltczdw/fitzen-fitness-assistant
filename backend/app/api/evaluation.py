import ipaddress,json
from pathlib import Path
from fastapi import APIRouter,HTTPException,Request
from pydantic import BaseModel,ConfigDict
from typing import Literal
from app.services.evaluation_jobs import runner,SPECS,BACKEND
from app.services.rag.blocking import run_retrieval
from app.services.quality_groups import list_groups

router=APIRouter(tags=['evaluation'])


def local_only(request):
    try:valid=bool(request.client and ipaddress.ip_address(request.client.host).is_loopback)
    except ValueError:valid=False
    if not valid:raise HTTPException(403,'评测任务仅允许本机访问')


class StartJob(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:Literal['validate','core','all','rag-offline','rag-proxy']


@router.get('/evaluation',include_in_schema=False)
def page():
    from fastapi.responses import FileResponse
    return FileResponse(BACKEND/'app/static/evaluation.html')


@router.get('/evaluation/jobs')
def jobs(request:Request):
    local_only(request)
    return {'available':[{'name':k,'title':v[0]} for k,v in SPECS.items()],
            'jobs':[j.output() for j in reversed(list(runner.jobs.values()))],
            'providers_disabled':True,'real_model_tested':False}


@router.post('/evaluation/jobs')
async def start(payload:StartJob,request:Request):
    local_only(request)
    try:return await runner.start(payload.name)
    except RuntimeError:raise HTTPException(409,'已有评测正在运行') from None


@router.post('/evaluation/jobs/{jid}/cancel')
async def cancel(jid:str,request:Request):
    local_only(request)
    try:return await runner.cancel(jid)
    except KeyError:raise HTTPException(404,'任务不存在') from None


@router.get('/evaluation/baseline')
async def baseline(request:Request):
    local_only(request)
    def read():
        path=BACKEND/'evals/v2/reports/baseline.json'
        if not path.exists():return {'available':False}
        data=json.loads(path.read_text(encoding='utf-8'))
        return {'available':True,'version':data['dataset_version'],'environment':data['environment'],
                'summary':data['summary'],'simulation':{k:data.get('gpt6_luna_simulation',{}).get(k) for k in ('count','passed','mode')}}
    return await run_retrieval(read)


@router.get('/review/groups')
async def groups():
    try:return await run_retrieval(list_groups)
    except Exception as exc:
        from app.db.storage import StorageUnavailableError
        if isinstance(exc,StorageUnavailableError):raise HTTPException(503,'质量记录暂不可用') from None
        raise


@router.get('/evaluation/reports')
async def reports(request:Request):
    local_only(request)
    def read():
        paths=[]
        for folder,pattern in ((BACKEND/'evals/v2/reports','run-*.json'),(BACKEND/'evals/v2/strategy-reports','strategies-*.json')):
            paths.extend(p for p in folder.glob(pattern) if p.is_file() and not p.is_symlink() and p.stat().st_size<4_000_000)
        items=[]
        for path in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True)[:10]:
            try:
                data=json.loads(path.read_text(encoding='utf-8'))
                item={'name':path.name,'created':path.stat().st_mtime,'environment':data.get('environment'),
                      'summary':data.get('summary'),'mode':data.get('mode')}
                if 'strategies' in data:item['strategies']={k:{'status':v['status'],'metrics':v['metrics']} for k,v in data['strategies'].items()}
                items.append(item)
            except (OSError,ValueError,KeyError):continue
        return {'items':items,'real_provider_tested':False}
    return await run_retrieval(read)


@router.get('/evaluation/calibration')
async def calibration(request:Request):
    local_only(request)
    from app.core.config import settings
    def read():
        from evals.v2.scoring import calibrate
        folder=BACKEND/'evals/v2/reports'
        paths=sorted(folder.glob('run-*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in paths:
            if path.is_symlink() or path.stat().st_size>4_000_000:continue
            try:data=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError):continue
            if data.get('summary',{}).get('scenarios')!=400:continue
            result=calibrate(data['results'])
            return {'available':True,'report':path.name,'report_source':data.get('environment',{}).get('application_source_sha256'),
                    'threshold_in_use':settings.rag_answer_confidence_threshold,'recommendations':result,
                    'applied':False,'test_used_for_calibration':False}
        return {'available':False,'threshold_in_use':settings.rag_answer_confidence_threshold,'applied':False}
    return await run_retrieval(read)
