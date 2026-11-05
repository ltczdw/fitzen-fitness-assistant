"""Check (default) or explicitly create the three additive budget tables.

No model calls and no existing queue/data changes. Back up and review before
using --apply on a real database; implementation tests use temporary schemas.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import inspect
from app.db import session as db
from app.db.model_budget_models import ModelQuota,ModelCallLedger,ModelResultCache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if db.engine is None:raise SystemExit('Database not configured')
    models=(ModelQuota,ModelCallLedger,ModelResultCache)
    if args.apply:
        with db.engine.begin() as conn:
            for model in models:model.__table__.create(conn,checkfirst=True)
    tables=set(inspect(db.engine).get_table_names())
    missing=[m.__tablename__ for m in models if m.__tablename__ not in tables]
    print({'ready':not missing,'missing_tables':missing,'applied':args.apply})
    return 1 if missing else 0


if __name__=='__main__':raise SystemExit(main())
