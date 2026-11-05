"""Local operator tool. Preview by default; apply only with explicit user approval.

Extends today's call allowance once, expires at Shanghai midnight. No model calls,
schema changes, credential/config edits, refunds, or personal-data access.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.services.llm.budget import BudgetStore

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day',required=True,help='Current date in Asia/Shanghai, YYYY-MM-DD')
    parser.add_argument('--additional-calls',type=int,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(BudgetStore().grant_daily_calls(args.additional_calls,day=args.day,apply=args.apply)))

if __name__=='__main__':main()
