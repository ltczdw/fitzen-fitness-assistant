"""Preview (default) failure cases created only by correct safety refusals.

The pre-fix runtime turned ``task_rejected_unauthorized`` into a degraded
request and an automatic failure case. This tool lists those historical
misreports read-only; ``--apply`` resolves each one through the audited review
flow, which appends a ReviewAudit row. Cases, runs and feedback are never
deleted.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db import session as db
from app.db.quality_models import FailureCase
from app.db.quality_repository import review
from app.api.quality import ReviewInput


REFUSAL_REASON = "task_rejected_unauthorized"
AUDIT_NOTE = "自动核对：正确的安全拒绝曾被误计为执行故障；本案例仅标记解决，原始记录保留。"


def pending_refusal_cases(limit: int):
    with db.SessionLocal() as session:
        rows = session.execute(
            select(FailureCase.case_id, FailureCase.run_id, FailureCase.version, FailureCase.reasons)
            .where(FailureCase.status == "pending")
            .order_by(FailureCase.created_at, FailureCase.case_id)
        ).all()
    cases = []
    for case_id, run_id, version, reasons in rows:
        reason_set = {str(reason) for reason in (reasons or [])}
        if reason_set == {REFUSAL_REASON}:
            cases.append({"case_id": case_id, "run_id": run_id, "version": version,
                          "reasons": sorted(reason_set)})
    return cases[:limit], len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Resolve the listed cases through the audited review flow.")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if db.SessionLocal is None:
        raise SystemExit("Database is not configured.")
    cases, total = pending_refusal_cases(args.limit)
    print(json.dumps({"candidates": total, "preview": cases, "apply": args.apply},
                     ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        return 0
    resolved, failures = 0, []
    for case in cases:
        try:
            review(case["case_id"], ReviewInput(decision="resolve",
                                                expected_version=case["version"],
                                                note=AUDIT_NOTE))
            resolved += 1
        except Exception as exc:  # report and continue; never delete anything
            failures.append({"case_id": case["case_id"], "error_type": type(exc).__name__})
    print(json.dumps({"resolved": resolved, "unchanged": total - resolved,
                      "failures": failures}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
