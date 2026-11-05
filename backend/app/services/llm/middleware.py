"""Pure ASGI scope propagation, including SSE and independent model APIs."""
import json
from contextlib import nullcontext
from uuid import uuid4
from app.services.llm.budget import model_scope,model_task,ModelBudgetError,key


class ModelBudgetMiddleware:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        request_key=headers.get(b'idempotency-key',b'').decode(errors='ignore')[:100] or uuid4().hex
        started=False
        # New HTTP roots aggregate real usage and use per-task/internal
        # preparation allowances. Chat requests are split into business tasks
        # by the agent; wrapping the whole chat request here would make sibling
        # goals share one allowance. Other model APIs are one business task.
        path = scope.get('path','')
        with model_scope('api:'+scope.get('method','')+path+':'+request_key,
                         policy='per_task') as budget:
            budget.owner_id = key('api-owner:' + request_key)
            chat_request = path.startswith('/api/chat')
            task_context = (nullcontext() if chat_request else
                            model_task('api-task:'+request_key,
                                       task_type='independent_api'))
            body=[]
            async def bounded_receive():
                message=await receive()
                if message['type']=='http.disconnect':budget.cancelled=True
                if message['type']=='http.request' and len(body)<64:
                    body.append(message.get('body',b''))
                    if not message.get('more_body') and path.startswith('/api/chat'):
                        try:
                            value=json.loads(b''.join(body)).get('request_id')
                            if isinstance(value,str) and 0<len(value)<=100:budget.name='request:'+value
                        except (ValueError,AttributeError):pass
                return message
            async def bounded_send(message):
                nonlocal started
                if message['type']=='http.response.start':started=True
                await send(message)
            try:
                with task_context:
                    await self.app(scope,bounded_receive,bounded_send)
            except ModelBudgetError as exc:
                if started:raise
                payload=json.dumps({'detail':'本次AI处理已达到调用限制或计量暂不可用。','code':exc.code}).encode()
                await send({'type':'http.response.start','status':429,'headers':[(b'content-type',b'application/json')]})
                await send({'type':'http.response.body','body':payload})
            finally:
                # Tasks inherited from an HTTP request may not keep spending
                # after the response. Background jobs must create their own scope.
                budget.cancelled=True
