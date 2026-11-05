import asyncio
from threading import get_ident
from app.services.agent.events import event_sink
from app.services.agent.runtime import execute_request


async def stream_response(agent, request):
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    owner = get_ident()

    def publish(event):
        if get_ident() == owner:
            queue.put_nowait(("event", event))
        else:
            loop.call_soon_threadsafe(queue.put_nowait, ("event", event))

    async def run():
        token = event_sink.set(publish)
        try:
            response = await execute_request(agent, request)
            await queue.put(("response", response))
        except Exception as exc:
            from app.services.agent.failure_details import transport_failure
            await queue.put(("error", {**transport_failure(exc),
                "model_used": getattr(exc, "_request_model_used", None)}))
        finally:
            event_sink.reset(token)

    task = asyncio.create_task(run())
    try:
        yield {"event": "status", "data": {"message": "正在处理请求"}}
        while True:
            kind, value = await queue.get()
            if kind == "event":
                yield value
            elif kind == "error":
                yield {"event": "error", "data": {**value, "message": value["detail"]}}
                yield {"event": "done", "data": {"status": "failed", "code": value["code"],
                    "model_used": value["model_used"]}}
                return
            else:
                result = value.model_dump()
                yield {"event": "meta", "data": {k: v for k, v in result.items() if k != "answer"}}
                # Output guard already processed the entire answer once. These
                # are transport chunks, not claims of provider token streaming.
                for i in range(0, len(value.answer), 32):
                    yield {"event": "token", "data": {"content": value.answer[i:i + 32]}}
                yield {"event": "done", "data": {k: v for k, v in result.items() if k != "answer"}}
                return
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
