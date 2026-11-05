"""Progress notifications carry task status only, never authorization or inputs."""
from contextvars import ContextVar

event_sink = ContextVar("execution_event_sink", default=None)


def emit(event, data):
    sink = event_sink.get()
    if sink:
        sink({"event": event, "data": data})
