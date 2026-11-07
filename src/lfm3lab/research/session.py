"""Small Python entry point shared by local and HTTP research backends."""
from __future__ import annotations

import threading


class ResearchSession:
    def __init__(self, backend):
        if not all(callable(getattr(backend, name, None)) for name in ("run", "status", "close")):
            raise TypeError("backend must provide run, status, and close")
        self.backend = backend
        self._closed = False
        self._mutex = threading.RLock()

    @classmethod
    def local(cls, **kwargs):
        from .backends import LFM3Backend
        return cls(LFM3Backend(**kwargs))

    @classmethod
    def remote(cls, url, **kwargs):
        from .worker import RemoteBackend
        return cls(RemoteBackend(url, **kwargs))

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Session is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, operation, payload):
        with self._mutex:
            if self._closed:
                raise RuntimeError("Session is closed")
            return self.backend.run(operation, payload)

    def status(self):
        with self._mutex:
            return self.backend.status()

    def close(self):
        with self._mutex:
            if not self._closed:
                self._closed = True
                self.backend.close()

    def score(self, **payload):
        # Convenience accepts the schema object; the wire protocol stays JSON.
        from .schema import Objective
        if isinstance(payload.get("objective"), Objective):
            payload["objective"] = payload["objective"].to_dict()
        return self.run("score", payload)

    def probe(self, **payload):
        return self.run("probe", payload)

    def attribute(self, **payload):
        return self.run("attribute", payload)

    def generate(self, **payload):
        return self.run("generate", payload)

    def sweep(self, **payload):
        return self.run("sweep", payload)
