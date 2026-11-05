"""Deterministic current views, transient TTL and independently carried constraints."""
from __future__ import annotations

from datetime import timedelta, timezone
import hashlib
from uuid import uuid4

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.orm import aliased

from app.core.config import settings
from app.db.memory_models import MemoryState as State, MemoryTurn as Turn, UserFact as Fact
from app.db.models import UserProfileModel


def database_time(value):
    # SQLite stores datetimes as text; normalize precision before comparison.
    # PostgreSQL already compares native timezone-aware timestamps.
    from app.db import session as db
    return func.julianday(value) if db.engine.dialect.name == 'sqlite' else value


def visible(source, epoch):
    return (source.status == 'completed', source.hidden_at.is_(None), source.deleted_at.is_(None), source.memory_epoch == epoch)


def canonical_context_key(domain, key):
    """Normalize only the finite identifier framing around a state key.

    Models sometimes echo the server's ``context:<domain>:`` projection or an
    already-composed ``<domain>:`` prefix. Strip those framing prefixes from
    the attribute identity; this is identifier canonicalization, not a
    synonym/keyword table and never changes the attribute's meaning.
    """
    value = str(key or "").strip()
    prefixes = ("context:", str(domain or "") + ":")
    changed = True
    while changed and value:
        changed = False
        for prefix in prefixes:
            while prefix != ":" and value.startswith(prefix):
                value = value[len(prefix):]
                changed = True
    return value


def context_state_key(domain, key):
    return "context:" + str(domain) + ":" + canonical_context_key(domain, key)


def fact_top(epoch):
    source = aliased(Turn)
    ranked = select(Fact.fact_id, Fact.field, Fact.value, Fact.source_quote, Fact.valid_to,
        func.dense_rank().over(partition_by=Fact.field, order_by=database_time(Fact.valid_from).desc()).label('rank'))
    ranked = ranked.join(source, source.turn_id == Fact.source_turn_id).where(
        Fact.status == 'active', Fact.memory_epoch == epoch, database_time(Fact.valid_from) <= database_time(func.now()), *visible(source, epoch)).correlate(None).subquery()
    # Rank before expiry: an expired newest record must not revive an older one.
    return select(ranked).where(ranked.c.rank == 1,
        or_(ranked.c.valid_to.is_(None), database_time(ranked.c.valid_to) > database_time(func.now()))).subquery()


def fact_winners(epoch):
    top = fact_top(epoch)
    grouped = select(top.c.field, func.min(top.c.fact_id).label('fact_id'),
        func.count(func.distinct(top.c.source_quote)).label('quotes'),
        func.count(func.distinct(top.c.value)).label('variants')).group_by(top.c.field).subquery()
    return select(grouped.c.fact_id).where(grouped.c.variants == 1, grouped.c.quotes == 1)


def state_latest(kind, cid, epoch):
    source = aliased(Turn)
    return select(State.state_id, func.row_number().over(
        partition_by=(State.owner_id, State.conversation_id, State.scope, State.logical_key),
        order_by=State.version.desc()).label('rank')).join(source, source.turn_id == State.source_turn_id).where(
            State.owner_id == 'demo-user', State.conversation_id == cid,
            State.kind == kind, State.memory_epoch == epoch, *visible(source, epoch)).correlate(None).subquery()


def live_state_ids(cid, epoch):
    latest = state_latest('transient', cid, epoch)
    return select(State.state_id).join(latest, latest.c.state_id == State.state_id).where(
        latest.c.rank == 1, State.status == 'active', State.scope == 'conversation', State.expires_at > func.now()).correlate(None)


def current_page(turn, cid, epoch):
    bad_fact = select(Fact.fact_id).where(Fact.source_turn_id == turn.turn_id,
        Fact.memory_epoch == epoch, Fact.fact_id.not_in(fact_winners(epoch))).correlate(turn).exists()
    bad_state = select(State.state_id).where(State.source_turn_id == turn.turn_id,
        State.kind == 'transient', State.memory_epoch == epoch,
        State.state_id.not_in(live_state_ids(cid, epoch))).correlate(turn).exists()
    return and_(~bad_fact, ~bad_state)


def stamp(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def allocate_source_version(s, turn, *, owner, epoch):
    version, instant = s.execute(select(func.max(State.version), func.current_timestamp()).where(
        State.owner_id == owner, State.memory_epoch == epoch)).one()
    version = (version or 0) + 1
    s.add(State(state_id=str(uuid4()), owner_id=owner, conversation_id='', scope='sequence',
        logical_key='_source_order', kind='order', domain='general', version=version,
        value='', status='active', valid_from=stamp(instant), expires_at=None, memory_epoch=epoch,
        source_turn_id=turn.turn_id, source_message_id=turn.turn_id + ':user'))
    return version


def transfer_constraint(s, old, new, *, owner, epoch):
    """Explicit Wiki corrections retain the original field's durable policy."""
    policy = s.scalar(select(State).where(State.owner_id == owner, State.memory_epoch == epoch,
        State.kind == 'constraint', State.status == 'active', State.source_fact_id == old.fact_id))
    if policy is None:
        return
    version, instant = s.execute(select(func.max(State.version), func.current_timestamp()).where(
        State.owner_id == owner, State.memory_epoch == epoch)).one()
    write_versions(s, [dict(kind='constraint', conversation_id='', scope='global', logical_key=new.field,
        domain=policy.domain, value='', status='active', source_fact_id=new.fact_id,
        source_turn_id=new.source_turn_id, source_message_id=new.source_message_id)],
        owner=owner, epoch=epoch, version=(version or 0)+1, instant=instant)


def _canonical_cte(anchor_filter: str) -> str:
    """Shared finite prefix grammar for snapshot and write paths.

    The recursive prefix tests are substr equality, which is case-sensitive on
    both SQLite and PostgreSQL. SQLite LIKE is ASCII-case-insensitive and must
    not decide whether ``Training:`` is the same attribute as ``training:``.
    """
    return """
        WITH RECURSIVE stripped AS (
            SELECT ms.state_id, ms.conversation_id, ms.scope, ms.domain,
                   CAST(ms.logical_key AS VARCHAR(200)) AS key,
                   ms.version, ms.status, ms.expires_at
            FROM memory_states ms
            JOIN memory_turns t ON t.turn_id = ms.source_turn_id
            WHERE ms.owner_id = :owner
              AND ms.memory_epoch = :epoch
              AND ms.kind = 'transient'
              AND t.status = 'completed'
              AND t.hidden_at IS NULL
              AND t.deleted_at IS NULL
              AND t.memory_epoch = :epoch
              """ + anchor_filter + """
            UNION ALL
            SELECT state_id, conversation_id, scope, domain,
                   CAST(CASE
                     WHEN substr(key, 1, 8) = 'context:' THEN substr(key, 9)
                     WHEN substr(key, 1, length(domain) + 1) = domain || ':' THEN substr(key, length(domain) + 2)
                     ELSE NULL
                   END AS VARCHAR(200)),
                   version, status, expires_at
            FROM stripped
            WHERE key IS NOT NULL
              AND (substr(key, 1, 8) = 'context:'
                   OR substr(key, 1, length(domain) + 1) = domain || ':')
        ),
        terminal AS (
            SELECT state_id, conversation_id, scope, domain, key,
                   version, status, expires_at
            FROM stripped
            WHERE key IS NOT NULL
              AND key <> ''
              AND substr(key, 1, 8) <> 'context:'
              AND substr(key, 1, length(domain) + 1) <> domain || ':'
        )
    """


def _target_predicates(targets, alias='s'):
    where = []
    params = {}
    for index, (cid, scope, domain, key) in enumerate(targets):
        suffix = str(index)
        where.append(
            "(" + alias + ".conversation_id = :c" + suffix +
            " AND " + alias + ".scope = :s" + suffix +
            " AND " + alias + ".domain = :d" + suffix +
            " AND " + alias + ".key = :k" + suffix + ")")
        params["c" + suffix] = cid
        params["s" + suffix] = scope
        params["d" + suffix] = domain
        params["k" + suffix] = key
    return " OR ".join(where), params


def _anchor_predicates(targets):
    where = []
    params = {}
    for index, (cid, scope, _domain, _key) in enumerate(targets):
        suffix = str(index)
        where.append("(ms.conversation_id = :ac" + suffix +
                      " AND ms.scope = :as" + suffix + ")")
        params["ac" + suffix] = cid
        params["as" + suffix] = scope
    return "AND (" + " OR ".join(where) + ")", params


def _canonical_version_summary(s, owner, epoch, targets):
    """SQL GROUP BY canonical identity; only target max versions return."""
    targets = sorted(targets)
    if not targets:
        return {}
    anchor, anchor_params = _anchor_predicates(targets)
    target_where, target_params = _target_predicates(targets)
    statement = text(_canonical_cte(anchor) + """
        SELECT s.conversation_id, s.scope, s.domain, s.key, MAX(s.version)
        FROM terminal s
        WHERE """ + target_where + """
        GROUP BY s.conversation_id, s.scope, s.domain, s.key
    """)
    params = {"owner": owner, "epoch": epoch, **anchor_params, **target_params}
    return {(row[0], row[1], row[2], row[3]): int(row[4] or 0)
            for row in s.execute(statement, params)}


def _supersede_canonical_aliases(s, owner, epoch, targets, version):
    """UPDATE only active state_ids matching the requested canonical targets."""
    targets = sorted(targets)
    if not targets:
        return
    anchor, anchor_params = _anchor_predicates(targets)
    target_where, target_params = _target_predicates(targets)
    statement = text(_canonical_cte(anchor) + """
        UPDATE memory_states
        SET status = 'superseded'
        WHERE state_id IN (
            SELECT s.state_id
            FROM terminal s
            WHERE s.status = 'active'
              AND s.version < :version
              AND (""" + target_where + """)
        )
    """)
    params = {"owner": owner, "epoch": epoch, "version": version,
              **anchor_params, **target_params}
    s.execute(statement, params)


def snapshot(cid=None, *, epoch=None):
    from app.services.memory import compact
    with compact.session() as s:
        epoch = (s.scalar(select(UserProfileModel.memory_epoch).where(UserProfileModel.user_id == 'demo-user')) or 0) if epoch is None else epoch
        top = fact_top(epoch)
        rows = list(s.scalars(select(Fact).where(Fact.fact_id.in_(select(top.c.fact_id))).order_by(Fact.field, Fact.fact_id)))
        groups = {}
        for fact in rows:
            groups.setdefault(fact.field, []).append(fact)
        facts, conflicts = {}, {}
        for field, candidates in groups.items():
            if len({(f.value, f.source_quote) for f in candidates}) > 1:
                conflicts[field] = [{'source_message_id': f.source_message_id, 'source_quote': f.source_quote} for f in candidates]
                continue
            f = candidates[0]
            facts[field] = {'text': f.value, 'source_quote': f.source_quote, 'source_turn_id': f.source_turn_id,
                'source_message_id': f.source_message_id, 'kind': 'fact', 'domain': 'general', 'fact_id': f.fact_id}
        constraints, constraint_conflicts = {}, {}
        latest = state_latest('constraint', '', epoch)
        policies = s.execute(select(State, Fact).join(latest, latest.c.state_id == State.state_id)
            .join(Fact, Fact.fact_id == State.source_fact_id).where(latest.c.rank == 1, State.status == 'active',
                Fact.status == 'active', Fact.memory_epoch == epoch,
                or_(Fact.valid_to.is_(None), database_time(Fact.valid_to) > database_time(func.now()))))
        for policy, fact in policies:
            if fact.field in conflicts:
                constraint_conflicts[fact.field] = {'domain': policy.domain, 'candidates': conflicts[fact.field]}
            current = facts.get(fact.field)
            if current and current['source_quote'] == fact.source_quote:
                value = {**current, 'kind': 'constraint', 'domain': policy.domain, 'policy_version': policy.version}
                constraints[fact.field] = value
                facts[fact.field] = value
        states = {}
        request_version = None
        live_turn = compact.current_turn.get()
        if cid is not None and live_turn:
            request_version = s.scalar(select(Turn.messages[0]['app_meta']['_memory_version'].as_integer()).where(Turn.turn_id == live_turn))
        if cid is not None:
            dialect = s.bind.dialect.name
            expiry = ("ms.expires_at IS NOT NULL AND julianday(ms.expires_at) > julianday('now')"
                      if dialect == 'sqlite' else
                      "ms.expires_at IS NOT NULL AND ms.expires_at > CURRENT_TIMESTAMP")
            # Canonicalization and latest-version selection happen in SQL. The
            # recursive prefix stripper handles any mix of context:/domain:
            # framing; the final query returns only the active, unexpired
            # current set, so retired alias rows are never ORM-hydrated.
            statement = text(_canonical_cte(
                "AND ms.conversation_id = :cid AND ms.scope = 'conversation'") + """
                , canonical AS (
                    SELECT state_id, version,
                           ROW_NUMBER() OVER (
                               PARTITION BY domain, key ORDER BY version DESC
                           ) AS rn
                    FROM terminal
                )
                SELECT ms.*
                FROM memory_states ms
                JOIN canonical c ON c.state_id = ms.state_id
                WHERE c.rn = 1
                  AND ms.status = 'active'
                  AND """ + expiry + """
            """)
            loaded = s.execute(select(State).from_statement(statement),
                               {"owner": 'demo-user', "cid": cid, "epoch": epoch}).scalars().all()
            for state in loaded:
                key = context_state_key(state.domain, state.logical_key)
                states[key] = {'text': state.value, 'source_quote': state.value, 'source_turn_id': state.source_turn_id,
                    'source_message_id': state.source_message_id, 'kind': 'transient', 'domain': state.domain,
                    'state_id': state.state_id, 'state_version': state.version, 'expires_at': stamp(state.expires_at).isoformat()}
        return {'long_memory': {**facts, **states}, 'persistent_constraints': constraints, 'current_states': states,
                'all_persistent_constraints': constraints, 'all_current_states': states,
                'lifecycle_base_states': states, 'request_version': request_version,
                'memory_conflicts': conflicts, 'constraint_conflicts': constraint_conflicts,
                'versioned_view': True, 'memory_epoch': epoch}


def write_versions(s, items, *, owner, epoch, version, instant):
    """One aggregate, one set UPDATE and one bounded alias lookup.

    Canonical identity is evaluated over the active aliases returned by SQL,
    so a set/clear supersedes every framing variant without enumerating
    ``domain:context:...`` combinations or loading retired history.
    """
    if not items:
        return []
    items = [dict(item) for item in items]
    for item in items:
        item.pop('related_keys', None)
    global_keys = [item['logical_key'] for item in items if item['kind'] == 'constraint']
    if global_keys:
        current = {p.logical_key: p for p in s.scalars(select(State).where(State.kind == 'constraint',
            State.owner_id == owner, State.memory_epoch == epoch, State.status == 'active', State.logical_key.in_(global_keys)))}
        items = [item for item in items if item['kind'] != 'constraint' or item['logical_key'] not in current or
                 (current[item['logical_key']].source_fact_id, current[item['logical_key']].domain) !=
                 (item['source_fact_id'], item['domain'])]
        if not items:
            return []
    database_now = stamp(instant)
    constraint_items = [item for item in items if item['kind'] == 'constraint']
    versions = {}
    if constraint_items:
        constraint_identities = [(item['conversation_id'], item['scope'], item['logical_key'])
                                 for item in constraint_items]
        predicates = [and_(State.conversation_id == cid, State.scope == scope, State.logical_key == key)
                      for cid, scope, key in constraint_identities]
        old = list(s.execute(select(State.conversation_id, State.scope, State.logical_key, func.max(State.version))
            .where(State.owner_id == owner, State.memory_epoch == epoch, or_(*predicates))
            .group_by(State.conversation_id, State.scope, State.logical_key)))
        versions = {(cid, scope, key): version for cid, scope, key, version in old}
        s.execute(update(State).where(State.owner_id == owner, State.memory_epoch == epoch,
            or_(*predicates), State.status == 'active', State.version < version).values(status='superseded'))
    targets = {(item['conversation_id'], item['scope'], item['domain'],
                canonical_context_key(item['domain'], item['logical_key']))
               for item in items if item['kind'] == 'transient'}
    canonical_versions = _canonical_version_summary(s, owner, epoch, targets)
    _supersede_canonical_aliases(s, owner, epoch, targets, version)
    results: list[dict] = []
    for item in items:
        identity = (item['conversation_id'], item['scope'], item['logical_key'])
        expiry = None if item['kind'] == 'constraint' else database_now + timedelta(
            seconds=0 if item['scope'] == 'request' else settings.memory_transient_ttl_seconds)
        canonical_identity = (item['conversation_id'], item['scope'], item['domain'],
                              canonical_context_key(item['domain'], item['logical_key']))
        prior = versions.get(identity, 0) if item['kind'] == 'constraint' else canonical_versions.get(canonical_identity, 0)
        effective = prior <= version
        if not effective:
            item = {**item, 'status': 'superseded'}
        s.add(State(state_id=str(uuid4()), owner_id=owner, memory_epoch=epoch,
            version=version, valid_from=database_now, expires_at=expiry, **item))
        results.append({**item, 'version': version, 'effective': effective,
                        'conversation_id': item['conversation_id']})
    return results


def commit_context(s, row, semantic, *, owner, epoch, receipts):
    items = {}
    state_identities = set()
    if not semantic.get('current_is_critical'):
        for state in semantic.get('context_states', []):
            if state.get('action') in {'set', 'clear'}:
                domain = state['domain']
                state_identities.add((row.conversation_id, state['scope'],
                                      domain + ':' + canonical_context_key(domain, state['key'])))
        temporary = [*semantic.get('context_states', []),
            *[{'key': d['key'], 'domain': d.get('domain', 'general'), 'quote': d['quote'],
               'scope': 'conversation', 'action': 'set'} for d, receipt in receipts
              if d.get('kind') == 'transient' and receipt.get('status') == 'active']]
        for state in temporary:
            if state.get('action') == 'reference':
                # A reference selects the already-provided stable condition; it
                # must not create a new state or renew its TTL.
                continue
            domain = state['domain']
            logical_key = domain + ':' + canonical_context_key(domain, state['key'])
            item = {'kind': 'transient', 'conversation_id': row.conversation_id, 'scope': state['scope'],
                'domain': domain, 'logical_key': logical_key,
                'value': state['quote'], 'status': 'cleared' if state['action'] == 'clear' else 'active',
                'source_turn_id': row.turn_id, 'source_message_id': row.messages[0]['message_id']}
            items[(item['conversation_id'], item['scope'], item['logical_key'])] = item
    target_fields = {field for d, _ in receipts for field in d.get('target_fields', [])}
    inherited = {p.logical_key: p.domain for p in s.scalars(select(State).where(State.kind == 'constraint',
        State.owner_id == owner, State.memory_epoch == epoch, State.status == 'active', State.logical_key.in_(target_fields)))} if target_fields else {}
    for decision, receipt in receipts:
        domains = {inherited[field] for field in decision.get('target_fields', []) if field in inherited}
        domain = decision.get('domain', 'general') if decision.get('kind') == 'constraint' else (
            next(iter(domains)) if len(domains) == 1 else 'general' if domains else None)
        if domain is not None and receipt.get('status') == 'active' and receipt.get('binding'):
            fact = receipt['binding']
            item = {'kind': 'constraint', 'scope': 'global', 'conversation_id': '', 'logical_key': fact['field'],
                'domain': domain, 'value': '', 'status': 'active', 'source_fact_id': fact['fact_id'],
                'source_turn_id': fact['source_turn_id'], 'source_message_id': fact['source_message_id']}
            items[('', 'global', fact['field'])] = item
    fields = [entry['field'] for entry in semantic.get('constraint_fields', [])]
    if fields:
        sources = semantic.get('constraint_sources', {})
        current = {fact.field: fact for fact in s.scalars(select(Fact).where(Fact.field.in_(fields),
            Fact.fact_id.in_(fact_winners(epoch))))}
        for entry in semantic['constraint_fields']:
            fact = current.get(entry['field'])
            prior = sources.get(entry['field'], {})
            if fact and (fact.source_message_id, fact.source_quote) == (prior.get('source_message_id'), prior.get('source_quote')):
                items[('', 'global', fact.field)] = {'kind': 'constraint', 'scope': 'global', 'conversation_id': '',
                    'logical_key': fact.field, 'domain': entry['domain'], 'value': '', 'status': 'active',
                    'source_fact_id': fact.fact_id, 'source_turn_id': fact.source_turn_id, 'source_message_id': fact.source_message_id}
    version = row.messages[0].get('app_meta', {}).get('_memory_version')
    if items and version is not None:
        instant = s.scalar(select(State.valid_from).where(State.source_turn_id == row.turn_id, State.kind == 'order'))
        written = write_versions(s, list(items.values()), owner=owner, epoch=epoch,
                                 version=version, instant=instant)
        return [item for item in written
                if item['kind'] == 'transient'
                and (item['conversation_id'], item['scope'], item['logical_key']) in state_identities]
    return []


DOMAINS = {'training_plan': 'training', 'training_qa': 'training', 'workout_record': 'training',
           'nutrition_record': 'nutrition', 'nutrition_advice': 'nutrition', 'safety_triage': 'general'}


def channels(memory, task_type, scope):
    domain = DOMAINS.get(str(task_type))
    if not domain or scope.get('quoted') or scope.get('is_third_party') or scope.get('reply_mode') == 'conversation':
        return {'persistent_constraints': {}, 'current_states': {}}
    allowed = {'general', 'training', 'nutrition'} if domain == 'general' else {'general', domain}
    if any(value['domain'] in allowed for value in memory.get('constraint_conflicts', {}).values()):
        from app.services.memory.wiki_context import WikiReadError
        raise WikiReadError('persistent_constraint_conflict')
    return {name: {k: value for k, value in memory.get('all_' + name, memory.get(name, {})).items()
                   if value['domain'] in allowed} for name in ('persistent_constraints', 'current_states')}


def apply_labels(memory, understanding):
    if not memory.get('versioned_view'):
        return
    if (understanding.get('safety') or {}).get('level') == 'urgent':
        memory.update(persistent_constraints={}, current_states={})
        return
    constraints = dict(memory.get('all_persistent_constraints', {}))
    for entry in understanding.get('constraint_fields', []):
        field = entry['field']
        value = memory['long_memory'][field]
        if value.get('kind') != 'constraint':
            constraints[field] = {**value, 'kind': 'constraint', 'domain': entry['domain'], 'policy_pending': True}
    memory['all_persistent_constraints'] = constraints
    local_states, local_constraints, replaced, cleared = {}, {}, set(), set()
    if not understanding.get('current_is_critical'):
        temporary = [*understanding.get('context_states', []), *[
            {'key': d['key'], 'quote': d['quote'], 'scope': 'conversation', 'domain': d.get('domain', 'general')}
            for d in understanding.get('memory_requests', []) if d.get('kind') == 'transient']]
        chosen = {}
        for state in sorted(temporary, key=lambda item: item['scope'] == 'request'):
            canonical = canonical_context_key(state['domain'], state['key'])
            chosen[(state['domain'], canonical)] = {**state, 'key': canonical}
        for state in chosen.values():
            if state.get('action') == 'reference':
                # Reference keeps the already-loaded stable condition; it does
                # not create local state or renew the stored TTL.
                continue
            key = context_state_key(state['domain'], state['key'])
            if state.get('action') == 'clear':
                cleared.add(key)
                continue
            local_states[key] = {'text': state['quote'], 'source_quote': state['quote'], 'kind': 'transient',
                'domain': state['domain'], 'request_local': True, 'state_version': memory.get('request_version'),
                'source_message_id': 'current:' + str(memory.get('request_version'))}
    for decision in understanding.get('memory_requests', []):
        targets = decision.get('target_fields', [])
        inherited = any(field in constraints for field in targets)
        if decision.get('kind') != 'constraint' and not inherited:
            continue
        from app.services.memory.facts import semantic_field
        from app.services.memory.compact import digest
        field = semantic_field('demo-user', decision['key'], digest)
        if decision['action'] != 'replace' and field in memory['long_memory']:
            continue
        if decision['action'] == 'replace':
            replaced.update(targets)
        domains = {constraints[target]['domain'] for target in targets if target in constraints}
        domain = decision.get('domain', 'general') if decision.get('kind') == 'constraint' else (
            next(iter(domains)) if len(domains) == 1 else 'general')
        local_constraints['request:' + field] = {'text': decision['quote'], 'source_quote': decision['quote'],
            'kind': 'constraint', 'domain': domain, 'request_local': True,
            'source_message_id': 'current:' + str(memory.get('request_version'))}
    memory.update(lifecycle_local_states=local_states, lifecycle_local_constraints=local_constraints,
                  lifecycle_replaced_fields=sorted(replaced), lifecycle_cleared_states=sorted(cleared),
                  lifecycle_target_sources=understanding.get('memory_target_sources', {}))
    refresh_local(memory, memory)
    combined = {'persistent_constraints': {}, 'current_states': {}}
    for task in understanding.get('tasks', []):
        scoped = channels(memory, task['task_type'], task)
        for name in combined:
            combined[name].update(scoped[name])
    memory.update(combined)


def refresh_local(fresh, memory):
    """Current, source-validated input may override its prior conversation state."""
    states = dict(fresh.get('lifecycle_base_states', fresh.get('all_current_states', {})))
    for key, value in memory.get('lifecycle_local_states', {}).items():
        current = states.get(key)
        if current and value.get('state_version') is not None and current['state_version'] > value['state_version']:
            from app.services.memory.wiki_context import WikiReadError
            raise WikiReadError('memory_state_changed')
        states[key] = value
    for key in memory.get('lifecycle_cleared_states', []):
        if key in states and memory.get('request_version') is not None and states[key]['state_version'] > memory['request_version']:
            from app.services.memory.wiki_context import WikiReadError
            raise WikiReadError('memory_state_changed')
        states.pop(key, None)
    constraints = dict(fresh.get('all_persistent_constraints', {}))
    for key, value in memory.get('all_persistent_constraints', {}).items():
        if value.get('policy_pending'):
            constraints[key] = value
    for key in memory.get('lifecycle_replaced_fields', []):
        constraints.pop(key, None)
    constraints.update(memory.get('lifecycle_local_constraints', {}))
    fresh.update(all_current_states=states, all_persistent_constraints=constraints)
    stale_turns = {value['source_turn_id'] for key, value in memory.get('long_memory', {}).items()
                   if key in memory.get('lifecycle_replaced_fields', []) or key in memory.get('lifecycle_local_states', {})
                   or key in memory.get('lifecycle_cleared_states', [])}
    for page in memory.get('wiki_loaded_pages', {}).values():
        if page['turn_id'] in stale_turns:
            page['wiki_view'] = 'history'
    for message in memory.get('short_memory', []):
        if message.get('turn_id') in stale_turns:
            message.setdefault('app_meta', {})['wiki_view'] = 'history'
    for page in memory.get('retrieved_history', []):
        if page.get('turn_id') in stale_turns:
            page['wiki_view'] = 'history'


def binding(memory, understanding):
    if not memory.get('versioned_view'):
        return None
    fields = {field for task in understanding.get('tasks', []) for field in
              [*task.get('fact_fields', []), *task.get('critical_fact_fields', [])]}
    fields.update(understanding.get('critical_fact_fields', []))
    fields.update(understanding.get('memory_target_sources', {}))
    fields.update(memory.get('persistent_constraints', {})); fields.update(memory.get('current_states', {}))
    def identity(value):
        if value is None:
            return None
        return (value.get('source_message_id'), value.get('state_version'), value.get('domain', 'general'),
                hashlib.sha256(str(value.get('text', '')).encode()).hexdigest())
    values = {**memory['long_memory'], **memory.get('persistent_constraints', {}), **memory.get('current_states', {})}
    return {'conversation_id': memory.get('wiki_conversation_id'), 'memory_epoch': memory['memory_epoch'],
            'sources': {key: identity(values.get(key)) for key in sorted(fields)},
            'constraints': sorted(memory.get('persistent_constraints', {})), 'states': sorted(memory.get('current_states', {}))}


def validate_binding(memory, understanding):
    previous = understanding.get('__memory_view__')
    if previous is None:
        return
    if any(entry['field'] not in memory.get('long_memory', {}) for entry in understanding.get('constraint_fields', [])):
        from app.services.memory.wiki_context import WikiReadError
        raise WikiReadError('memory_state_changed')
    apply_labels(memory, understanding)
    current = binding(memory, understanding)
    # JSON snapshots turn tuples into arrays; compare canonical JSON values.
    import json
    if json.dumps(previous, sort_keys=True) != json.dumps(current, sort_keys=True):
        from app.services.memory.wiki_context import WikiReadError
        raise WikiReadError('memory_state_changed')
