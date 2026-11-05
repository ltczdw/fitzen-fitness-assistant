-- Additive budget schema only. Does not update/delete existing user data.

BEGIN;


CREATE TABLE IF NOT EXISTS model_quotas (
	quota_id VARCHAR(80) NOT NULL, 
	kind VARCHAR(24) NOT NULL, 
	calls INTEGER NOT NULL, 
	tokens BIGINT NOT NULL, 
	max_calls INTEGER NOT NULL, 
	max_tokens BIGINT NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (quota_id)
)

;


CREATE TABLE IF NOT EXISTS model_call_ledger (
	call_id VARCHAR(36) NOT NULL, 
	scope_id VARCHAR(80) NOT NULL, 
	kind VARCHAR(24) NOT NULL, 
	capability VARCHAR(20) NOT NULL, 
	model VARCHAR(100) NOT NULL, 
	stage_id VARCHAR(80) NOT NULL, 
	quota_ids JSON NOT NULL, 
	reserved_tokens BIGINT NOT NULL, 
	status VARCHAR(24) NOT NULL, 
	usage JSON, 
	error_code VARCHAR(80), 
	started_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	finished_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (call_id)
)

;

CREATE INDEX IF NOT EXISTS ix_model_call_ledger_scope_id ON model_call_ledger (scope_id);


CREATE TABLE IF NOT EXISTS model_result_cache (
	cache_id VARCHAR(64) NOT NULL, 
	scope_id VARCHAR(80) NOT NULL, 
	capability VARCHAR(20) NOT NULL, 
	owner_id VARCHAR(64), 
	result JSON NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (cache_id)
)

;

CREATE INDEX IF NOT EXISTS ix_model_result_cache_owner_id ON model_result_cache (owner_id);

CREATE INDEX IF NOT EXISTS ix_model_result_cache_scope_id ON model_result_cache (scope_id);

COMMIT;
