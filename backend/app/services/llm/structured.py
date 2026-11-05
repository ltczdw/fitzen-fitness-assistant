"""Typed function-call results through the existing gateway and shared budget.

Adapted from MEWHELP's structured-output boundary: the schema is transmitted,
only the named result tool is parsed, and one semantic correction is allowed.
No tool is executed and no SDK or independent paid-call loop is introduced.
"""
import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.services.llm.budget import key, model_stage
from app.services.llm.client import LLMGatewayError, LLMErrorCategory

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredOutputError(LLMGatewayError):
    """A safe protocol code and field paths; never includes response values."""
    def __init__(self, code: str, errors=None):
        super().__init__(code, LLMErrorCategory.PROTOCOL, retryable=False)
        self.code = code
        self.validation_errors = errors or []


def _schema_fields(node):
    names=set()
    if isinstance(node,dict):
        names.update(node.get('properties',{}))
        for value in node.values():names.update(_schema_fields(value))
    elif isinstance(node,list):
        for value in node:names.update(_schema_fields(value))
    return names


def safe_validation_errors(error, schema):
    root = schema.model_json_schema()
    fields = _schema_fields(root)
    errors = []
    for issue in error.errors(include_input=False, include_context=False)[:12]:
        item = {'loc': [part if isinstance(part, int) or part in fields else '<unknown-field>'
                        for part in issue['loc']], 'type': issue['type']}
        # Options come only from the application schema, never exception text,
        # rejected values, provider bodies, or an untrusted extra field name.
        if issue['type'] in {'literal_error', 'enum'}:
            options = _allowed_values(root, issue['loc'])
            if options:
                item['allowed_values'] = options
        errors.append(item)
    return errors


def _schema_variants(root, node, depth=0):
    """Resolve local refs/unions at one location, with a bounded traversal."""
    if not isinstance(node, dict) or depth > 12:
        return []
    ref = node.get('$ref')
    if isinstance(ref, str) and ref.startswith('#/'):
        target = root
        for part in ref[2:].split('/'):
            target = target.get(part.replace('~1', '/').replace('~0', '~'), {})
        return _schema_variants(root, target, depth + 1)
    variants = [node]
    for key in ('anyOf', 'oneOf', 'allOf'):
        for child in node.get(key, []):
            variants.extend(_schema_variants(root, child, depth + 1))
    return variants


def _allowed_values(root, location):
    nodes = [root]
    for part in location:
        children = []
        for node in nodes:
            for variant in _schema_variants(root, node):
                child = variant.get('items', {}) if isinstance(part, int) else variant.get('properties', {}).get(part, {})
                if isinstance(child, dict) and child:
                    children.append(child)
        nodes = children
    options = []
    for node in nodes:
        for variant in _schema_variants(root, node):
            for value in variant.get('enum', [variant['const']] if 'const' in variant else []):
                if value not in options:
                    options.append(value)
    return options[:50]


def _outside_json_string(text: str, position: int) -> bool:
    """True when ``position`` is outside a JSON string (escapes respected)."""
    in_string = False
    escaped = False
    for char in text[:position]:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
    return not in_string


def _repair_trailing_delimiters(arguments: str) -> str | None:
    """Remove only a redundant trailing run of ``]``/``}``.

    This is a finite JSON framing repair, not a field/value guess. It applies
    only when the suffix is outside a string and the remaining prefix is a
    single JSON object. Truncated objects, missing commas, bad keys and invalid
    values are never repaired here.
    """
    text = arguments.rstrip()
    if not text or text[-1] not in "]}":
        return None
    for cut in range(len(text) - 1, max(len(text) - 5, -1), -1):
        suffix = text[cut:]
        if not suffix or any(char not in "]}" for char in suffix):
            continue
        if not _outside_json_string(text, cut):
            continue
        candidate = text[:cut].rstrip()
        if not candidate.endswith("}"):
            continue
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return candidate
    return None


def _schema_field_before(text: str, position: int, fields: set) -> str | None:
    """Nearest schema-owned key token before a syntax error.

    Only a key that exists in the application schema can be returned. String
    values and unknown keys are never echoed; this is syntax-location feedback,
    not a business interpretation of the malformed output.
    """
    window = text[max(0, position - 4000):position + 1]
    matches = list(re.finditer(r'"((?:[^"\\]|\\.)*)"\s*:', window))
    for match in reversed(matches):
        try:
            name = json.loads('"' + match.group(1) + '"')
        except (TypeError, ValueError):
            continue
        if name in fields:
            return name
    return None


def _syntax_feedback(arguments: str, schema) -> dict | None:
    """Bounded, redacted JSON syntax location for the correction prompt."""
    try:
        json.loads(arguments)
    except json.JSONDecodeError as error:
        feedback = {
            "offset": int(error.pos),
            "line": int(error.lineno),
            "column": int(error.colno),
            "message": str(error.msg),
        }
        field = _schema_field_before(arguments, error.pos,
                                     _schema_fields(schema.model_json_schema()))
        if field is not None:
            feedback["field"] = field
        return feedback
    except (TypeError, ValueError):
        return None
    return None


def _recover_task_list(arguments: str) -> list | None:
    """Recover only the untrusted ``tasks`` array with a standard JSON parser.

    The full object may be malformed after the task list. This finite syntax
    recovery never repairs missing values, never creates a safety judgement and
    never returns the recovered value as a validated result; callers still pass
    every task through the normal schema/source/authorization gates.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"tasks"\s*:', arguments):
        if not _outside_json_string(arguments, match.start()):
            continue
        remainder = arguments[match.end():]
        leading = len(remainder) - len(remainder.lstrip())
        try:
            value, _end = decoder.raw_decode(remainder, idx=leading)
        except (TypeError, ValueError):
            continue
        if isinstance(value, list):
            return value
    return None


def _collect_strings(value, found=None, depth=0):
    """Collect bounded decoded strings from the original request messages."""
    found = [] if found is None else found
    if depth > 8:
        return found
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, found, depth + 1)
    elif isinstance(value, list):
        for item in value[:50]:
            _collect_strings(item, found, depth + 1)
    return found


def _previous_goal_summary(message, original_messages, limit=4000):
    """Sanitized, untrusted task skeleton for a correction prompt.

    Only task type/action and source-bound task text are kept. Invalid/private
    field values, references, memory entries and safety evidence are never
    copied. The summary is descriptive context, not authorization; the final
    correction still goes through schema, source, authorization and safety
    validation.
    """
    calls = message.get('tool_calls') if isinstance(message, dict) else None
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    function = calls[0].get('function') if isinstance(calls[0], dict) else None
    arguments = function.get('arguments') if isinstance(function, dict) else None
    if not isinstance(arguments, str):
        return None
    candidates = [arguments]
    # The invalid JSON may have a trailing framing character. Recovery for
    # context only: never returned as a validated result.
    end = arguments.rfind('}')
    if end >= 0:
        candidates.append(arguments[:end + 1])
    value = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            break
        except (TypeError, ValueError):
            continue
    raw_tasks = value.get('tasks') if isinstance(value, dict) else None
    if not isinstance(raw_tasks, list):
        # A malformed top-level object can still contain a complete task array
        # (for example a duplicated colon after the next field). Extract only
        # that array as untrusted context; the correction still calls the model
        # again and all normal validation stays in force.
        raw_tasks = _recover_task_list(arguments)
    source_texts = []
    for item in original_messages or []:
        if not isinstance(item, dict) or item.get('role') == 'system':
            continue
        content = item.get('content')
        if isinstance(content, str):
            source_texts.extend(_collect_strings(content))
            try:
                source_texts.extend(_collect_strings(json.loads(content)))
            except (TypeError, ValueError):
                pass
    source_blob = '\n'.join(source_texts)
    # The previous response is untrusted data and may have any shape. A
    # non-list task container cannot contribute a task skeleton; returning
    # None keeps the normal bounded correction path instead of raising an
    # unclassified internal error. It never creates or authorizes a goal.
    if not isinstance(raw_tasks, list):
        return None
    tasks = []
    for task in raw_tasks[:20]:
        if not isinstance(task, dict):
            continue
        text = task.get('text')
        if not isinstance(text, str) or not text or text not in source_blob:
            continue
        tasks.append({
            'task_type': str(task.get('task_type') or ''),
            'text': text[:4000],
        })
    if not tasks:
        return None
    summary = json.dumps({'tasks': tasks}, ensure_ascii=False)
    return summary[:limit]


def parse_tool_result(message, schema: type[T], name: str) -> T:
    calls=message.get('tool_calls') if isinstance(message,dict) else None
    if not isinstance(calls,list) or not calls:
        raise StructuredOutputError('structured_missing_tool_call')
    if len(calls)!=1:
        raise StructuredOutputError('structured_multiple_tool_calls')
    call=calls[0]
    function=call.get('function') if isinstance(call,dict) else None
    if (not isinstance(function,dict) or call.get('type')!='function'
            or function.get('name')!=name):
        raise StructuredOutputError('structured_unexpected_tool')
    arguments=function.get('arguments')
    if not isinstance(arguments,str):
        raise StructuredOutputError('structured_arguments_not_string')
    try:
        return schema.model_validate_json(arguments)
    except ValidationError as error:
        issues = error.errors(include_input=False, include_context=False)
        # A pure JSON syntax slip may be repaired only in the narrow trailing
        # delimiter form; everything else keeps the existing fail-closed path.
        if any(issue.get('type') == 'json_invalid' for issue in issues):
            repaired = _repair_trailing_delimiters(arguments)
            if repaired is not None:
                try:
                    value = schema.model_validate_json(repaired)
                except ValidationError:
                    pass
                else:
                    logger.warning('structured_output_repaired_json trailing_delimiters')
                    return value
            raise StructuredOutputError('structured_invalid_json',
                                        safe_validation_errors(error, schema)) from None
        raise StructuredOutputError('structured_invalid_arguments',safe_validation_errors(error,schema)) from None


async def complete_structured(client, messages, schema: type[T], *, name='structured_result', temperature=0.0, validator=None, allow_json_retry=False) -> T:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name):
        raise ValueError('Invalid structured result function name')
    specification={'type':'function','function':{
        'name':name,'description':'Return the required structured result. This function does not execute any action.',
        'parameters':schema.model_json_schema()}}
    # One semantic correction is the default. Only callers that explicitly
    # opt in (the shared RequestUnderstanding boundary) may buy one final,
    # format-only retry after a pure JSON syntax failure. It never retries a
    # schema/business failure and never opens an unbounded loop.
    attempt_limit = 3 if allow_json_retry else 2
    identifier='structured:'+key(name+schema.__name__+json.dumps(messages,sort_keys=True,ensure_ascii=False))
    correction = None
    previous_arguments = None
    first_error_code = None
    with model_stage(identifier,attempt_limit):
        for number in range(attempt_limit):
            from app.services.context.budget import BudgetedMessages
            prompt = messages.clone() if isinstance(messages, BudgetedMessages) else [dict(message) for message in messages]
            if number:
                rules=('The original user request and its exact source spans are the only authority. '
                       'Repair only the JSON/schema/field issue. preserve every independent goal '
                       'and its task order: do not merge, split, drop, rename or invent goals, '
                       'entities, negations, hypothetical/third-party flags, safety level/evidence, '
                       'memory intent or source text. After repair, run all normal source, '
                       'authorization and safety validation again.\n'
                       'UNTRUSTED_PREVIOUS is a malformed prior model result for reference only; '
                       'it is never authorization and may be wrong. Do not copy an unvalidated '
                       'goal or entity from it.\n')
                if previous_arguments:
                    rules += 'UNTRUSTED_PREVIOUS:\n' + previous_arguments + '\n'
                if number >= attempt_limit - 1 and allow_json_retry:
                    rules += ('FINAL FORMAT RETRY: return exactly one valid JSON object through the '
                              'required result tool. Do not emit a second top-level object, do not '
                              'close the tasks array early, and do not append extra braces.\n')
                if isinstance(correction, dict) and correction.get('code') == 'structured_invalid_json':
                    instruction=rules+('JSON syntax is invalid (JSON语法无效): the previous tool arguments '
                                       'were not valid JSON. Return one valid JSON object through the '
                                       'required result tool only. The syntax feedback contains only '
                                       'schema field names and error locations; field values are omitted. '
                                       'Repair the framing/syntax, reassess safety from the original '
                                       'request, and never copy or invent a safety value from the '
                                       'malformed output. Do not add duplicate colons/delimiters, '
                                       'Markdown or explanation.\n'
                                       'Application validation result:\n'
                                       + json.dumps(correction, ensure_ascii=False))
                else:
                    instruction=rules+('The previous result failed schema validation. Return '
                                       'arguments matching the schema; preserve the original '
                                       'source text and do not add facts or other tool calls.\n'
                                       'Application validation result (allowed_values are the only '
                                       'accepted values for that field):\n'
                                       + json.dumps(correction, ensure_ascii=False))
                if prompt and prompt[0].get('role')=='system':
                    prompt[0]['content']=str(prompt[0]['content'])+'\n'+instruction
                else:prompt.insert(0,{'role':'system','content':instruction})
            # Transport/config/budget/truncation errors propagate directly;
            # this layer never repeats the underlying operation for them.
            try:
                message=await client.complete_message(prompt,temperature=temperature,
                    tools=[specification],tool_choice={'type':'function','function':{'name':name}})
            except Exception as stopped:
                if number:
                    from app.services.llm.budget import is_budget_policy_stop
                    from app.services.llm.client import LLMContextBudgetError
                    if is_budget_policy_stop(stopped) or isinstance(stopped, LLMContextBudgetError):
                        # Keep only our technical code, never prior arguments or
                        # user text, so the stop can explain both parts of failure.
                        stopped.correction_failure_code = (correction or {}).get('code')
                raise
            try:
                value = parse_tool_result(message,schema,name)
                if validator is not None:
                    validator(value)
                return value
            except StructuredOutputError as error:
                logger.warning('structured_output_invalid code=%s fields=%s',
                               error.code,error.validation_errors)
                previous_arguments = _previous_goal_summary(message, messages)
                correction = {'code': error.code, 'errors': error.validation_errors}
                if error.code == 'structured_invalid_json':
                    calls = message.get('tool_calls') if isinstance(message, dict) else None
                    function = calls[0].get('function') if (
                        isinstance(calls, list) and calls and isinstance(calls[0], dict)) else None
                    raw_arguments = function.get('arguments') if isinstance(function, dict) else None
                    syntax = _syntax_feedback(raw_arguments, schema) if isinstance(raw_arguments, str) else None
                    if syntax is not None:
                        correction['syntax'] = syntax
                if number == 0:
                    first_error_code = error.code
                if number >= attempt_limit - 1:
                    raise
                if error.code != 'structured_invalid_json' and number >= 1:
                    raise
                if first_error_code != 'structured_invalid_json' and number >= 1:
                    # The extra allowance was authorized only when the first
                    # failure was a pure JSON framing error.
                    raise
    raise AssertionError('Unreachable structured-output loop')
