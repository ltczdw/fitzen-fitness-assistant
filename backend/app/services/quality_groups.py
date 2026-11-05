"""Read-only issue grouping inspired by MEWHELP's flywheel normalization.

Conservative canonical aliases only: no model/background job, no persisted
business merges. Every original case/run/audit remains authoritative.
"""
import hashlib,re,unicodedata
from collections import defaultdict
from sqlalchemy import select
from app.db import session as db
from app.db.storage import require_ready
from app.db.quality_models import ChatRun,FailureCase
from app.services.security.input_guard import inspect_input


def normalize_question(text):
    value=unicodedata.normalize('NFKC',inspect_input(text).sanitized_text).lower().strip()
    value=re.sub(r'^(?:请问|请告诉我|我想了解|想了解|麻烦问下|请教)[:：\s]*','',value)
    for before,after in (('蛋白含量','蛋白质含量'),('抗阻训练','力量训练'),('肌肉强化','力量训练')):
        value=value.replace(before,after)
    return re.sub(r'[\s，,。.!！？?；;：:]+','',value)


def group_cases(rows):
    groups={}
    for case,run in rows:
        normalized=normalize_question(run.question)
        # Numbers, task category, rejection state and failure reason survive.
        identity=(case.category,case.status,tuple(sorted(case.reasons or [])),normalized)
        key=hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
        group=groups.setdefault(key,{'group_id':key,'question':inspect_input(run.question).sanitized_text,
                                    'normalized_question':normalized,'category':case.category,
                                    'status':case.status,'occurrence_count':0,'case_ids':[],'run_ids':[]})
        group['occurrence_count']+=1;group['case_ids'].append(case.case_id);group['run_ids'].append(run.run_id)
    return sorted(groups.values(),key=lambda g:(-g['occurrence_count'],g['group_id']))


def list_groups(status='pending',limit=200):
    require_ready()
    with db.SessionLocal() as session:
        query=select(FailureCase,ChatRun).join(ChatRun).where(ChatRun.status.notin_(('deleted','cancelled')))
        if status is not None:query=query.where(FailureCase.status==status)
        rows=list(session.execute(query.order_by(FailureCase.created_at.desc()).limit(limit+1)))
        return {'groups':group_cases(rows[:limit]),'case_count':min(len(rows),limit),
                'truncated':len(rows)>limit,'grouping_mode':'conservative_read_only',
                'original_records_merged':False}
