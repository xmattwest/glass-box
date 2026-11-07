"""JSON-only, serialized research transport. The server does not own its backend.

Call shutdown() from another thread, then server_close(), before backend.close().
Socket and queue timeouts bound waiting, not an already-running model operation.
For nonloopback deployment use TLS at a trusted reverse proxy or an SSH tunnel.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from .schema import SCHEMA_VERSION

OPERATIONS = frozenset({"score", "probe", "attribute", "generate", "sweep"})
MAX_BODY_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
SOCKET_TIMEOUT = 15
QUEUE_TIMEOUT = 600
MAX_CONNECTIONS = 16
STATUS_PATH = "/api/research/status"
RUN_PATH = "/api/research/run"


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(raw, parse_constant=invalid, object_pairs_hook=pairs)
    _json(value)  # Also rejects overflowing numeric literals (e.g. 1e999).
    return value


def _request(operation, payload):
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError("Unsupported research operation")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    body = _json({"schema_version": SCHEMA_VERSION, "operation": operation, "payload": payload})
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("Request exceeds body limit")
    return body


class RemoteError(RuntimeError):
    """A worker rejected a request; preserve HTTP status and backend error type."""

    def __init__(self, status, error_type, message):
        self.status_code = status
        self.error_type = error_type
        super().__init__(f"Worker HTTP {status} {error_type}: {message}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteBackend:
    def __init__(self, url, token=None, timeout=600):
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Worker URL must be an http(s) origin without credentials or path")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("timeout must be positive and at most 3600 seconds")
        _validate_token(token)
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._closed = False
        # Do not forward credentials through redirects or ambient HTTP proxies.
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _call(self, path, body=None):
        if self._closed:
            raise RuntimeError("RemoteBackend is closed")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.url + path, data=body, headers=headers)
        try:
            response = self._opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            response = exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Worker connection failed: {exc}") from exc
        with response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("Worker response exceeds limit")
            if response.headers.get_content_type() != "application/json":
                raise RuntimeError("Worker returned a non-JSON response")
            try:
                envelope = _decode(raw)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise RuntimeError("Invalid worker JSON response") from exc
            if not isinstance(envelope, dict) or type(envelope.get("schema_version")) is not int or envelope.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("Unsupported worker response schema")
            if response.status != 200:
                error = envelope.get("error", {})
                if not isinstance(error, dict):
                    raise RuntimeError("Malformed worker error response")
                raise RemoteError(response.status, error.get("type", "Error"), error.get("message", "Request failed"))
            if set(envelope) != {"schema_version", "result"} or not isinstance(envelope["result"], dict):
                raise RuntimeError("Malformed worker result")
            return envelope["result"]

    def run(self, operation, payload):
        return self._call(RUN_PATH, _request(operation, payload))

    def status(self):
        return self._call(STATUS_PATH)

    def close(self):
        # A proxy close must never unload the shared worker.
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("RemoteBackend is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def _validate_token(token):
    if token is not None and (not isinstance(token, str) or not token or
                              any(ord(c) < 33 or ord(c) > 126 for c in token)):
        raise ValueError("Bearer token must be nonempty printable ASCII without whitespace")


def _loopback(host):
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ResearchHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            try:
                request.settimeout(SOCKET_TIMEOUT)
                body = _json({"schema_version": SCHEMA_VERSION, "error": {"type": "Busy", "message": "Connection limit reached"}})
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()


class _Handler(BaseHTTPRequestHandler):
    # One request per connection avoids persistent idle clients holding slots.
    protocol_version = "HTTP/1.0"

    def setup(self):
        self.request.settimeout(SOCKET_TIMEOUT)
        super().setup()

    def log_message(self, format, *args):
        # Access logs contain no payload/token; errors remain explicit JSON.
        super().log_message(format, *args)

    def _reply(self, status, result=None, error=None):
        body = _json({"schema_version": SCHEMA_VERSION, **({"result": result} if error is None else {"error": error})})
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Backend response exceeds transport limit")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def send_error(self, code, message=None, explain=None):
        self._reply(code, error={"type": "HTTPError", "message": message or self.responses.get(code, ("Error",))[0]})

    def _guard(self):
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.server.allowed_hosts:
            self.send_error(403, "Host is not allowed")
            return False
        # API is not a browser application. Reject every browser Origin,
        # including null; do not implement permissive CORS/preflight.
        if self.headers.get_all("Origin") or self.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            self.send_error(403, "Browser origins are not allowed")
            return False
        if self.server.token is not None:
            values = self.headers.get_all("Authorization", [])
            if len(values) != 1 or not hmac.compare_digest(values[0].encode("utf-8"), ("Bearer " + self.server.token).encode("ascii")):
                self.send_error(401, "Bearer authentication required")
                return False
        return True

    def _backend(self, operation=None, payload=None):
        if not self.server.backend_lock.acquire(timeout=QUEUE_TIMEOUT):
            self.send_error(503, "Backend queue timeout")
            return
        try:
            try:
                result = self.server.backend.status() if operation is None else self.server.backend.run(operation, payload)
                if not isinstance(result, dict):
                    raise TypeError("Backend must return a JSON object")
                self._reply(200, result=result)
            except (BrokenPipeError, ConnectionResetError):
                # Computation is finished; disconnect must not unload backend.
                self.log_error("Client disconnected before result delivery")
            except Exception as exc:
                self.log_error("Backend %s: %s", type(exc).__name__, exc)
                status = 400 if isinstance(exc, ValueError) else 500
                self._reply(status, error={"type": type(exc).__name__, "message": str(exc)})
        finally:
            self.server.backend_lock.release()

    def do_GET(self):
        if not self._guard():
            return
        if self.path != STATUS_PATH:
            self.send_error(404, "Unknown endpoint")
            return
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0") != "0":
            self.send_error(400, "GET body is not allowed")
            return
        self._backend()

    def do_POST(self):
        if not self._guard():
            return
        if self.path != RUN_PATH:
            self.send_error(404, "Unknown endpoint")
            return
        if self.headers.get_all("Transfer-Encoding"):
            self.send_error(400, "Transfer-Encoding is not supported")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            self.send_error(411, "A single Content-Length is required")
            return
        if len(lengths[0]) > 10:
            self.send_error(413, "Content-Length exceeds body limit")
            return
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            self.send_error(413, "Request exceeds body limit")
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(415, "Content-Type must be application/json")
            return
        try:
            deadline = time.monotonic() + SOCKET_TIMEOUT
            parts = []
            remaining = length
            while remaining:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    raise TimeoutError("Request body deadline exceeded")
                self.connection.settimeout(wait)
                part = self.rfile.read1(min(65536, remaining))
                if not part:
                    raise ValueError("Incomplete request body")
                parts.append(part)
                remaining -= len(part)
            self.connection.settimeout(SOCKET_TIMEOUT)
            raw = b"".join(parts)
            value = _decode(raw)
            if not isinstance(value, dict) or set(value) - {"schema_version", "operation", "payload"}:
                raise ValueError("Unknown request schema fields")
            if type(value.get("schema_version", SCHEMA_VERSION)) is not int or value.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
                raise ValueError("Unsupported request schema version")
            operation, payload = value.get("operation"), value.get("payload")
            _request(operation, payload)
        except (ValueError, UnicodeError, RecursionError) as exc:
            self.send_error(400, str(exc))
            return
        except (TimeoutError, socket.timeout):
            self.send_error(408, "Request body timeout")
            return
        self._backend(operation, payload)


def create_server(backend, host="127.0.0.1", port=0, token=None):
    """Bind a worker without starting threads or loading/closing a backend.

    Only the bound authority is accepted in Host (loopback aliases are allowed).
    Nonloopback binds require an explicit token. server_close drains handlers.
    """
    _validate_token(token)
    if not isinstance(host, str) or not host or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid worker bind address")
    if not _loopback(host) and token is None:
        raise ValueError("Nonloopback workers require an explicit bearer token")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    server_type = type("BoundResearchHTTPServer", (ResearchHTTPServer,), {"address_family": family})
    server = server_type((host, port), _Handler)
    server.backend = backend
    server.token = token
    server.backend_lock = threading.Lock()
    server.connection_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    bound_host, bound_port = server.server_address[:2]
    names = {host.lower(), bound_host.lower()}
    if _loopback(host):
        names |= {"localhost", "127.0.0.1", "::1"}
    elif host in ("0.0.0.0", "::"):
        # Wildcard workers remain accessible through a local SSH tunnel.
        # Public/NAT authorities must be explicitly added to allowed_hosts.
        names |= {"localhost", "127.0.0.1", "::1"}
    server.allowed_hosts = {f"[{name}]:{bound_port}" if ":" in name else f"{name}:{bound_port}" for name in names}
    if bound_port == 80:
        server.allowed_hosts |= {f"[{name}]" if ":" in name else name for name in names}
    return server
