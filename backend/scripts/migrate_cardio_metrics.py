"""Check (default) or explicitly apply only the two nullable cardio columns.

No startup initialization, business DML, backfill, provider calls or credentials
in output. Run --apply only after approval for the actual database upgrade.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import inspect
from app.db import session as db
from app.db.cardio_metrics_schema import ensure_cardio_metrics_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        if db.engine is None:
            raise RuntimeError('database not configured')
        with db.engine.begin() as connection:
            inspector = inspect(connection)
            if not inspector.has_table('workout_session_sets'):
                raise RuntimeError('workout table not initialized')
            if args.apply:
                ensure_cardio_metrics_schema(connection)
            columns={item['name'] for item in inspect(connection).get_columns('workout_session_sets')}
            missing=sorted({'distance_km','duration_minutes'} - columns)
        print(json.dumps({'ready':not missing,'missing_columns':missing,'applied':args.apply}))
        return 0
    except Exception as exc:
        print(json.dumps({'ready':False,'error_type':type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
