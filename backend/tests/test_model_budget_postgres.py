"""Optional real PostgreSQL locks, isolated random schema, zero provider calls."""
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from sqlalchemy import text
from quality_isolation import checked_engine as create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db.model_budget_models import ModelQuota,ModelCallLedger,ModelResultCache
from app.services.llm.budget import BudgetStore,model_scope,paid_attempt,ModelBudgetError


@pytest.mark.skipif(os.getenv('FITZEN_DB_TESTS')!='1',reason='PostgreSQL opt-in only')
def test_postgres_parallel_budget_reservation_is_atomic_and_durable():
    assert settings.database_url.startswith('postgresql')
    schema='test_model_budget_'+uuid4().hex
    assert schema.startswith('test_model_budget_') and schema.replace('_','').isalnum()
    admin=create_engine(settings.database_url,connect_args={'connect_timeout':3})
    with admin.begin() as connection:connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine=create_engine(settings.database_url,connect_args={'connect_timeout':3,'options':f'-csearch_path={schema},public'})
    try:
        with engine.begin() as connection:
            for model in (ModelQuota,ModelCallLedger,ModelResultCache):model.__table__.create(connection)
        store=BudgetStore(sessionmaker(bind=engine,expire_on_commit=False))
        def request(index):
            try:
                with model_scope('one-shared-job',store=store,max_calls=3):
                    with paid_attempt('chat','synthetic',{'index':index},0):pass
                return True
            except ModelBudgetError:return False
        with ThreadPoolExecutor(max_workers=8) as pool:result=list(pool.map(request,range(8)))
        assert sum(result)==3
        with model_scope('one-shared-job',store=BudgetStore(store.factory),max_calls=999):
            with pytest.raises(ModelBudgetError):
                with paid_attempt('chat','synthetic',{},0):pass
    finally:
        engine.dispose()
        with admin.begin() as connection:connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
