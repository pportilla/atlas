"""Helpers for exercising the public HTTP API on a private loopback server."""
from contextlib import contextmanager
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine.algebra_basis_store import AlgebraBasisStore
from engine.app import AtlasServer, Handler


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


RING = {"field": "QQ", "variables": ["x", "y"], "order": "lex"}


@contextmanager
def endpoint(**limits):
    with AtlasServer(("127.0.0.1", 0), QuietHandler) as server:
        server.algebra_bases = AlgebraBasisStore(**limits)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/api/algebra"
        finally:
            server.shutdown()
            thread.join()


def post(url, operation, operands, ring=RING, **extra):
    request = {"operation": operation, "operands": operands, "ring": ring, "compute": "cpu", **extra}
    req = Request(url, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
    try:
        response = urlopen(req, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)
