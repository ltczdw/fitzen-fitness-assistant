"""Deny app socket connections while permitting CPython's local wakeup pair."""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import socket
from unittest.mock import patch

building_pair=ContextVar('evaluation_building_asyncio_wakeup_pair',default=False)


@contextmanager
def deny_network():
    original_connect=socket.socket.connect
    original_connect_ex=socket.socket.connect_ex
    original_pair=socket.socketpair
    original_getaddrinfo=socket.getaddrinfo
    audit={'blocked_connections':0,'blocked_dns':0}
    def reject(*args):
        audit['blocked_connections']+=1
        raise RuntimeError('evaluation_network_prohibited')
    def reject_dns(*args,**kwargs):
        # R006：应用侧 DNS 解析同样被审计拦截。域名解析本身即是外联探测信号，
        # 不必等到 connect 才失败。asyncio 的本地唤醒对（socketpair）不经过
        # getaddrinfo，事件循环内部机制不受影响。
        audit['blocked_dns']+=1
        raise RuntimeError('evaluation_dns_prohibited')
    def connect(sock,address):
        # Windows socket.socketpair() uses an ephemeral loopback connection.
        # Only its internal, synchronous construction is allowed. Application
        # calls (including direct localhost calls) never receive this context.
        if building_pair.get() and isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
            return original_connect(sock,address)
        return reject()
    def connect_ex(sock,address):
        if building_pair.get() and isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
            return original_connect_ex(sock,address)
        return reject()
    def pair(*args,**kwargs):
        token=building_pair.set(True)
        try:return original_pair(*args,**kwargs)
        finally:building_pair.reset(token)
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket,'connect',connect))
        stack.enter_context(patch.object(socket.socket,'connect_ex',connect_ex))
        stack.enter_context(patch.object(socket,'create_connection',reject))
        stack.enter_context(patch.object(socket,'getaddrinfo',reject_dns))
        stack.enter_context(patch.object(socket,'socketpair',pair))
        yield audit
