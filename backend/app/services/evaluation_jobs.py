"""Local, fixed-argv evaluation jobs inspired by MEWHELP/core/jobs.py.

Only safe offline jobs; bounded history/logs/runtime. No shell, supplied paths,
provider jobs, database maintenance or autonomous retries.
"""
import asyncio,json,os,re,sys,time,subprocess
from dataclasses import dataclass,field
from pathlib import Path
from uuid import uuid4

BACKEND=Path(__file__).resolve().parents[2]
SPECS={
    'validate':('数据完整性检查',('scripts/eval_scenarios.py','--validate-only')),
    'core':('80条核心场景',('scripts/eval_scenarios.py','--core','--with-simulations')),
    'all':('400条全部场景',('scripts/eval_scenarios.py','--with-simulations')),
    'rag-offline':('纯词法与不可用策略报告',('scripts/eval_rag_strategies.py','--mode','offline')),
    'rag-proxy':('四策略代理实验（不代表模型质量）',('scripts/eval_rag_strategies.py','--mode','proxy')),
}


def safe_environment():
    env=dict(os.environ)
    for key in ('DATABASE_URL','LLM_API_KEY','LLM_BASE_URL','RETRIEVAL_EMBEDDING_API_KEY',
                'RETRIEVAL_EMBEDDING_BASE_URL','RERANK_API_KEY','RERANK_BASE_URL',
                'LANGSMITH_API_KEY','LANGCHAIN_API_KEY','MCP_URL'):env[key]=''
    for key in ('REDIS_ENABLED','LANGSMITH_TRACING','LANGCHAIN_TRACING_V2','MEMORY_BACKGROUND_ENABLED','MEMORY_SUMMARY_ENABLED'):env[key]='false'
    for key in ('FITZEN_DB_TESTS','FITZEN_QUALITY_DB','PYTHONPATH','PYTHONSTARTUP'):env.pop(key,None)
    env['PYTHONIOENCODING']='utf-8'
    return env


@dataclass
class Job:
    id:str
    name:str
    status:str='queued'
    exit_code:int|None=None
    created:float=field(default_factory=time.time)
    lines:list=field(default_factory=list)
    task:object=None
    process:object=None

    def output(self):return {'id':self.id,'name':self.name,'status':self.status,'exit_code':self.exit_code,
                            'created':self.created,'lines':list(self.lines),'title':SPECS[self.name][0]}


class JobRunner:
    def __init__(self):self.jobs={};self.lock=asyncio.Lock()
    async def start(self,name):
        if name not in SPECS:raise ValueError('unknown_job')
        async with self.lock:
            if any(j.status in ('queued','running','stopping') for j in self.jobs.values()):raise RuntimeError('job_already_running')
            if len(self.jobs)>=20:
                oldest=min(self.jobs,key=lambda k:self.jobs[k].created);self.jobs.pop(oldest)
            job=Job(str(uuid4()),name);self.jobs[job.id]=job
            job.task=asyncio.create_task(self._run(job));return job.output()
    async def _run(self,job):
        try:
            job.status='running'
            job.process=await asyncio.create_subprocess_exec(sys.executable,*SPECS[job.name][1],
                cwd=BACKEND,env=safe_environment(),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT,
                limit=65536,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            async def read():
                while True:
                    line=await job.process.stdout.readline()
                    if not line:break
                    value=line.decode('utf-8',errors='replace').rstrip()[:2000]
                    value=re.sub(r'(?i)(?:sk-[a-z0-9_-]{12,}|bearer\s+\S+)','<redacted>',value)
                    job.lines.append(value);job.lines=job.lines[-200:]
                return await job.process.wait()
            job.exit_code=await asyncio.wait_for(read(),180)
            job.status='passed' if job.exit_code==0 else 'failed'
        except asyncio.TimeoutError:
            job.status='timeout';await self._terminate(job)
        except asyncio.CancelledError:
            job.status='cancelled';await self._terminate(job)
        except Exception as exc:
            job.status='failed';job.lines.append('runner_error:'+type(exc).__name__)
            await self._terminate(job)
    async def _terminate(self,job):
        if job.process is not None and job.process.returncode is None:
            job.process.terminate()
            try:await asyncio.wait_for(job.process.wait(),5)
            except asyncio.TimeoutError:job.process.kill();await job.process.wait()
    async def cancel(self,jid):
        job=self.jobs.get(jid)
        if job is None:raise KeyError(jid)
        if job.status in ('queued','running'):
            job.status='stopping';job.task.cancel();await asyncio.gather(job.task,return_exceptions=True)
            if job.status=='stopping':job.status='cancelled'
        return job.output()
    async def close(self):
        for job in list(self.jobs.values()):
            if job.status in ('queued','running','stopping'):await self.cancel(job.id)

runner=JobRunner()
