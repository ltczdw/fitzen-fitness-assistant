"""Default read-only check; --apply explicitly adds one nullable profile column."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import inspect
from app.db import session as db
from app.db.profile_schema import ensure_profile_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        if db.engine is None:
            raise RuntimeError('database not configured')
        with db.engine.begin() as connection:
            if not inspect(connection).has_table('user_profiles'):
                raise RuntimeError('profile table not initialized')
            if args.apply:
                ensure_profile_schema(connection)
            columns={item['name'] for item in inspect(connection).get_columns('user_profiles')}
            missing=sorted({'body_fat_pct'}-columns)
        print(json.dumps({'ready':not missing,'missing_columns':missing,'applied':args.apply}))
        return 0
    except Exception as exc:
        print(json.dumps({'ready':False,'error_type':type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
