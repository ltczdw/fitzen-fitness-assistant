"""Visible, current-conversation answer identities; no natural-language routing."""
from __future__ import annotations

RESULT_TYPES = frozenset({'nutrition_advice','training_plan','training_qa','safety_triage'})


def _target_reference_ids(target):
    return list(dict.fromkeys(mid for mid in [*target.get('source_user_ids', []),
        *target.get('context_message_ids', []), target.get('user_message_id'), target['message_id']] if mid))


def result_reference_ids(memory, message_id):
    target = next((source for source in result_sources(memory) if source['message_id'] == message_id), None)
    return _target_reference_ids(target) if target else []


def select_result_sources(memory: dict, ids) -> dict:
    """Promote only selected locators before the shared live visibility check."""
    current=list(memory.get('short_memory') or [])
    known={m.get('message_id') for m in current}
    wanted=set(ids)
    current += [m for m in memory.get('recent_result_dialogue') or []
                if m.get('message_id') in wanted and m.get('message_id') not in known]
    return {**memory,'short_memory':current,
            'result_sources_pending':any(m.get('content_status')=='not_loaded' for m in current
                                         if m.get('message_id') in wanted)}


def result_sources(memory: dict) -> list[dict]:
    sources = []
    user_id = None
    # An explicitly empty result inventory is authoritative (for example, all
    # visible turns were failed attempts). Do not promote those history replies
    # by falling back to the general conversation window.
    messages = (memory.get('recent_result_dialogue') if 'recent_result_dialogue' in memory
                else memory.get('short_memory')) or []
    for index, message in enumerate(messages):
        meta = message.get('app_meta') or {}
        if meta.get('result_context_only'):
            continue
        if meta.get('wiki_view') == 'history' or message.get('temporal_view') == 'history':
            continue
        mid = str(message.get('message_id') or f'history-{index}')
        if message.get('role') == 'user':
            user_id = mid
        elif message.get('role') == 'assistant' and user_id:
            preview = meta.get('plan_preview') or {}
            targets = meta.get('result_targets') or []
            types = list(dict.fromkeys(t.get('task_type') for t in targets
                                       if t.get('task_type') in RESULT_TYPES))
            if preview.get('preview_id') and not types:
                types = ['training_plan']
            sources.append({'message_id': mid, 'user_message_id': user_id,
                            'task_types': types, 'preview_id': preview.get('preview_id'),
                            'source_user_ids': list(dict.fromkeys(
                                source_id for t in targets for source_id in t.get('source_user_ids', []))),
                            'context_message_ids': list(dict.fromkeys(
                                mid for t in targets for mid in t.get('context_message_ids', [])))})
    return [{**item, 'is_latest': i == 0}
            for i, item in enumerate(reversed(sources[-6:]))]


def completed_result_targets(semantic: dict | None, response: dict | None,
                             user_message_id: str = '', memory: dict | None = None) -> list[dict]:
    """Only application-validated, completed read results become result markers."""
    if not semantic or semantic.get('__semantic_unavailable__') or not semantic.get('validated_tasks'):
        return []
    if ((response or {}).get('metrics') or {}).get('react_status') != 'complete':
        return []
    sources={s['message_id']:s for s in result_sources(memory or {})}
    targets=[]
    for t in semantic.get('tasks') or []:
        if (t.get('task_type') not in RESULT_TYPES or t.get('goal_role','result') != 'result'
                or t.get('action') not in {'question','answer'}):
            continue
        ref=t.get('result_reference') or {}
        previous=sources.get(ref.get('message_id'))
        contextual = ref.get('purpose', 'revise') == 'context'
        origins=([previous['user_message_id'], *previous.get('source_user_ids', [])]
                 if previous and not contextual else [user_message_id])
        target={'task_type':t['task_type'],'delivery_type':t.get('delivery_type'),
                'source_user_ids':list(dict.fromkeys(mid for mid in origins if mid))}
        context_ids = (t.get('context_ids', []) if contextual else
                       previous.get('context_message_ids', []) if previous else [])
        if context_ids:
            target['context_message_ids'] = list(dict.fromkeys(context_ids))
        targets.append(target)
    return targets


def validate_result_reference(task, relation: str, memory: dict) -> str | None:
    sources = result_sources(memory)
    latest = sources[0] if sources else None
    wrapper = memory.get('current_plan_preview') or {}
    previews = wrapper.get('available') or [wrapper.get('state') or wrapper]
    previews = [p for p in previews if isinstance(p,dict) and p.get('preview_id')]
    selected = next((p for p in previews if p.get('preview_id') == task.preview_ref), None)
    ref = task.result_reference
    if ref is None:
        if (selected and latest and latest.get('preview_id') != task.preview_ref
                and latest['message_id'] != selected.get('message_id')):
            return '旧训练预览并非最近一次回答。重新生成/修改必须提供result_reference；无明确更早对象时选择最新回答，不能自动选择旧训练预览。'
        if (latest and latest['task_types'] and relation == 'follow_up'
                and task.delivery_type in {'plan','preview'}
                and task.task_type not in latest['task_types']):
            return '当前follow_up的方案类型与最近已完成结果不同；请选择result_reference确认真正来源，新的独立目标应声明new。'
        return None
    if task.task_type not in RESULT_TYPES or task.action not in {'question','answer'}:
        return 'result_reference只能用于只读回答的修改或背景参考，不能授予记录/保存权限。'
    if ref.quote not in task.text:
        return 'result_reference.quote必须逐字引用当前task.text中的指向依据。'
    target = next((s for s in sources if s['message_id'] == ref.message_id), None)
    if target is None and ref.selection == 'explicit':
        p = next((p for p in previews if p.get('message_id') == ref.message_id), None)
        if p:
            target = {'message_id':ref.message_id,'task_types':['training_plan'],
                      'preview_id':p['preview_id']}
    if target is None:
        return 'result_reference只能选择本会话实际可见回答或应用提供的已有预览来源，不能自造或跨会话引用。'
    if ref.selection == 'latest' and (latest is None or target['message_id'] != latest['message_id']):
        return 'selection=latest必须指向result_sources中is_latest=true的回答；旧训练预览不能替代最新饮食方案。'
    contextual = ref.purpose == 'context'
    if not contextual and target['task_types'] and task.task_type not in target['task_types']:
        return 'purpose=revise时task_type须与所选回答一致；若仅参考旧回答生成另一个领域的新内容，使用purpose=context并保留当前任务类型。'
    if contextual and task.preview_ref:
        return '背景参考不能绑定preview_ref修改旧训练预览；使用context_ids引用背景，当前任务保持新目标。'
    if task.preview_ref and target.get('preview_id') != task.preview_ref:
        return 'preview_ref与result_reference必须指向同一份真实训练预览；普通饮食回答不能绑定训练预览。'
    if not task.preview_ref:
        ids = list(dict.fromkeys([*task.context_ids, *_target_reference_ids(target)]))
        if len(ids)>32:
            return '回答引用的必要来源超过单次来源数量上限；请重新完整说明当前目标，不能静默丢弃早期条件。'
        task.context_ids = ids
    return None
