"""Valid structured plan fixtures shared by ordinary mocked worker tests."""
import json


def untimed_plan(*names):
    return {'sessions': [{'label': f'训练日{i+1}', 'warmup': ['关节活动'],
        'main': {'exercises': [{'name': name, 'sets': 3, 'reps': 10}]},
        'cooldown': ['轻柔放松'], 'without_timing': True} for i, name in enumerate(names)]}


def plan_message(*names):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'mock-plan', 'type': 'function', 'function': {
            'name': 'return_training_plan', 'arguments': json.dumps(untimed_plan(*names), ensure_ascii=False)}}]}
