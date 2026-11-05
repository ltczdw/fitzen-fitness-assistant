"""Fail-closed database sessions and explicit commit uncertainty."""
from contextlib import contextmanager
import logging
from app.db import session as db

logger = logging.getLogger(__name__)

class StorageUnavailableError(RuntimeError):
    code = "storage_unavailable"

class CommitUnknownError(RuntimeError):
    code = "commit_unknown"

class IdempotencyConflict(RuntimeError):
    """Same idempotency key with different normalized parameters."""
    code = "idempotency_conflict"


class BusinessValidationError(ValueError):
    """Explicit input/ownership validation failure.

    ``storage_session`` propagates it before commit so the existing route
    contracts can answer 400 instead of reporting a false storage outage.
    """
    code = "business_validation"


class WorkoutAssociationError(BusinessValidationError):
    """A record/session/action identity cannot be established safely."""
    code = "workout_association_conflict"


class WorkoutDetailRequired(BusinessValidationError):
    """A heterogeneous record needs explicit per-set details for this edit."""
    code = "workout_detail_required"


def require_ready():
    if not db.DB_READY or db.SessionLocal is None:
        raise StorageUnavailableError("Database is not initialized or unavailable")


class _Session:
    def __init__(self, session):
        self.raw = session
        self.commit_attempted = False
    def __getattr__(self, name):
        return getattr(self.raw, name)
    def commit(self):
        self.commit_attempted = True
        return self.raw.commit()


@contextmanager
def storage_session():
    require_ready()
    tracked = None
    try:
        with db.SessionLocal() as raw:
            tracked = _Session(raw)
            yield tracked
    except (StorageUnavailableError, CommitUnknownError, IdempotencyConflict):
        raise
    except BusinessValidationError as exc:
        # Input/ownership validation is only a safe rejection before a commit
        # was attempted; afterwards the outcome is unknown, not an input error.
        if tracked and tracked.commit_attempted:
            raise CommitUnknownError("Commit outcome is unknown; verify before retrying") from exc
        raise
    except Exception as exc:
        logger.exception("database_operation_failed", extra={"commit_attempted": bool(tracked and tracked.commit_attempted)})
        if tracked and tracked.commit_attempted:
            raise CommitUnknownError("Commit outcome is unknown; verify before retrying") from exc
        raise StorageUnavailableError("Database operation failed before commit") from exc


def recover_committed(lookup, original):
    """Read in a fresh session. Never retry a write after ambiguous commit."""
    try:
        result = lookup()
    except Exception as exc:
        logger.exception("commit_verification_failed")
        raise CommitUnknownError("Commit verification unavailable") from exc
    if result is None:
        raise CommitUnknownError("No committed record confirmed by idempotency key") from original
    logger.warning("commit_verified_by_idempotency_key")
    return result
