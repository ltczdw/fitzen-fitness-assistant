"""Event-loop-local connection pools. No SDK retries, no inherited headers.

Adapted from MEWHELP's async embedding approach; use our existing protocols.
"""
import asyncio
from weakref import WeakKeyDictionary
import httpx

_clients=WeakKeyDictionary()


def client():
    loop=asyncio.get_running_loop()
    value=_clients.get(loop)
    if value is None or value.is_closed:
        value=httpx.AsyncClient(follow_redirects=False,
            limits=httpx.Limits(max_connections=8,max_keepalive_connections=4))
        _clients[loop]=value
    return value


async def close():
    value=_clients.pop(asyncio.get_running_loop(),None)
    if value is not None:await value.aclose()
