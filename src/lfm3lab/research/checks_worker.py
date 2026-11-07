"""CPU-only loopback checks of transport, safety boundaries and CLI routing."""
from __future__ import annotations

import contextlib
from http.client import HTTPConnection
import io
import json
import socket
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from . import cli, worker


class FakeBackend:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.calls = []
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, operation, payload):
        if self.closed:
            raise RuntimeError("fake backend unloaded early")
        if set(payload) - {"echo", "fail", "block"}:
            raise ValueError("Unknown fake payload fields")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if payload.get("fail"):
                raise RuntimeError("visible fake failure")
            if payload.get("block"):
                self.entered.set()
                if not self.release.wait(5):
                    raise TimeoutError("fixture release not signaled")
            self.calls.append(operation)
            return {"operation": operation, "echo": payload.get("echo"), "fixture": True}
        finally:
            self.active -= 1

    def status(self):
        if self.closed:
            raise RuntimeError("fake backend is closed")
        return {"fixture": True, "revision": "fake-v1", "calls": len(self.calls)}

    def close(self):
        if self.active:
            raise RuntimeError("unload with active work")
        self.closed = True


class ScoreFixture(FakeBackend):
    def status(self):
        return {"backend": "fake", "model_id": "transport-fixture", "revision": "fake-v1", "device": "cpu", "dtype": "float32", "adapter": "synthetic", "implementation": "checks-worker-v1", "fixture": True}

    def run(self, operation, payload):
        if operation != "score":
            raise ValueError("Only score supported by this fixture")
        self.calls.append(operation)
        return {"value": 1.0, "objective": payload["objective"], "revision": "fake-v1", "device": "cpu", "dtype": "float32", "input_ids": [1], "prompt_length": 1, "elapsed_s": 0.0, "fixture": True}


@contextlib.contextmanager
def serving(token=None, backend=None):
    backend = backend if backend is not None else FakeBackend()
    server = worker.create_server(backend, token=token)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield backend, server, url
    finally:
        backend.release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        backend.close()
        if thread.is_alive():
            raise RuntimeError("Fixture server did not stop")


def http(url, method="POST", body=None, headers=None, path=worker.RUN_PATH):
    port = int(url.rsplit(":", 1)[1])
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": "application/json", **(headers or {})}
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


class WorkerChecks(unittest.TestCase):
    def test_all_operations_and_context(self):
        with serving() as (backend, server, url):
            with worker.RemoteBackend(url) as remote:
                self.assertTrue(remote.status()["fixture"])
                for op in sorted(worker.OPERATIONS):
                    self.assertEqual(remote.run(op, {"echo": ["α", 1, None]}), {"operation": op, "echo": ["α", 1, None], "fixture": True})
            self.assertEqual(len(backend.calls), 5)
            self.assertFalse(backend.closed)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                remote.status()

    def test_remote_failures_visible(self):
        with serving() as (_, _, url):
            remote = worker.RemoteBackend(url)
            with self.assertRaises(worker.RemoteError) as caught:
                remote.run("probe", {"fail": True})
            self.assertEqual(caught.exception.status_code, 500)
            self.assertEqual(caught.exception.error_type, "RuntimeError")
            self.assertIn("visible fake failure", str(caught.exception))
            with self.assertRaises(worker.RemoteError) as caught:
                remote.run("score", {"unknown": 1})
            self.assertEqual(caught.exception.status_code, 400)

    def test_auth_and_nonloopback_policy(self):
        with self.assertRaisesRegex(ValueError, "bearer"):
            worker.create_server(FakeBackend(), "0.0.0.0")
        for token in ("", "has space", "newline\n", "é"):
            with self.assertRaises(ValueError):
                worker.RemoteBackend("http://localhost:1234", token=token)
        with serving(token="fixture-secret") as (_, _, url):
            for token in (None, "wrong"):
                with self.assertRaises(worker.RemoteError) as caught:
                    worker.RemoteBackend(url, token=token).status()
                self.assertEqual(caught.exception.status_code, 401)
            self.assertTrue(worker.RemoteBackend(url, token="fixture-secret").status()["fixture"])
            self.assertEqual(http(url, "GET", headers={"Authorization": "Bearer é"}, path=worker.STATUS_PATH)[0], 401)

    def test_host_origin_and_restricted_endpoints(self):
        with serving() as (backend, _, url):
            for headers in ({"Host": "attacker.invalid"}, {"Origin": "null"}, {"Origin": url}, {"Sec-Fetch-Site": "cross-site"}):
                self.assertEqual(http(url, "GET", headers=headers, path=worker.STATUS_PATH)[0], 403)
            for method, path in (("GET", "/"), ("POST", worker.STATUS_PATH), ("GET", worker.STATUS_PATH + "?x=1")):
                self.assertEqual(http(url, method, {}, path=path)[0], 404)
            self.assertEqual(http(url, "DELETE", path=worker.RUN_PATH)[0], 501)
            self.assertEqual(backend.calls, [])

    def test_operation_schema_and_json_rejection(self):
        with serving() as (backend, _, url):
            bodies = [
                {"operation": "exec", "payload": {"code": "print(1)"}},
                {"operation": "score", "payload": []},
                {"operation": "score", "payload": {}, "schema_version": 2},
                {"operation": "score", "payload": {}, "schema_version": True},
                {"operation": "score", "payload": {}, "code": "exec"},
                [], {}, b'{"operation":"score","operation":"probe","payload":{}}',
                b'{"operation":"score","payload":{"echo":NaN}}',
                b'{"operation":"score","payload":{"echo":1e999}}',
                b'{"operation":"score","payload":{"echo":Infinity}}',
                b'not json', b'\xff', b'[' * 2000,
            ]
            for body in bodies:
                with self.subTest(body=str(body)[:80]):
                    self.assertEqual(http(url, body=body)[0], 400)
            self.assertEqual(backend.calls, [])
            self.assertEqual(http(url, body={"operation": "score", "payload": {"echo": 3}})[0], 200)

    def test_body_content_type_and_connection_limits(self):
        with serving() as (_, server, url):
            self.assertEqual(http(url, body=b"", headers={"Content-Length": str(worker.MAX_BODY_BYTES + 1)})[0], 413)
            self.assertEqual(http(url, body=b"{}", headers={"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(http(url, body=b"{}", headers={"Transfer-Encoding": "chunked"})[0], 400)
            self.assertEqual(http(url, body=b"{}", headers={"Content-Length": "-1"})[0], 411)
            self.assertEqual(http(url, body=b"", headers={"Content-Length": "9" * 5000})[0], 413)
            with self.assertRaisesRegex(ValueError, "limit"):
                worker.RemoteBackend(url).run("score", {"echo": "x" * worker.MAX_BODY_BYTES})
            for _ in range(worker.MAX_CONNECTIONS):
                self.assertTrue(server.connection_slots.acquire(False))
            try:
                self.assertEqual(http(url, "GET", path=worker.STATUS_PATH)[0], 503)
            finally:
                for _ in range(worker.MAX_CONNECTIONS):
                    server.connection_slots.release()

    def test_serialized_concurrent_requests_and_status(self):
        with serving() as (backend, _, url), ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(worker.RemoteBackend(url).run, "probe", {"block": True})
            self.assertTrue(backend.entered.wait(2))
            other = pool.submit(worker.RemoteBackend(url).run, "score", {"echo": 2})
            status = pool.submit(worker.RemoteBackend(url).status)
            self.assertFalse(other.done())
            self.assertFalse(status.done())
            backend.release.set()
            self.assertTrue(first.result(5)["fixture"])
            self.assertEqual(other.result(5)["echo"], 2)
            self.assertTrue(status.result(5)["fixture"])
            self.assertEqual(backend.peak, 1)

    def test_server_close_drains_before_backend_close(self):
        with serving() as (backend, server, url), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker.RemoteBackend(url).run, "attribute", {"block": True})
            self.assertTrue(backend.entered.wait(2))
            server.shutdown()
            closed = threading.Event()

            def close():
                server.server_close()
                backend.close()
                closed.set()

            closer = threading.Thread(target=close)
            closer.start()
            self.assertFalse(closed.wait(0.05))
            self.assertFalse(backend.closed)
            backend.release.set()
            self.assertTrue(future.result(5)["fixture"])
            closer.join(5)
            self.assertTrue(closed.is_set())

    def test_queue_and_socket_timeouts(self):
        with serving() as (_, server, url):
            server.backend_lock.acquire()
            try:
                with patch.object(worker, "QUEUE_TIMEOUT", 0.02):
                    self.assertEqual(http(url, "GET", path=worker.STATUS_PATH)[0], 503)
            finally:
                server.backend_lock.release()
            with patch.object(worker, "SOCKET_TIMEOUT", 0.05):
                sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2)
                try:
                    request = f"POST {worker.RUN_PATH} HTTP/1.0\r\nHost: 127.0.0.1:{server.server_address[1]}\r\nContent-Type: application/json\r\nContent-Length: 10\r\n\r\n{{"
                    sock.sendall(request.encode())
                    self.assertIn(b"408", sock.recv(4096))
                finally:
                    sock.close()

    def test_client_bad_urls_and_connection_failure(self):
        for url in ("file:///tmp/code", "http://user:pass@localhost", "http://localhost/run", "http://localhost?token=x", "ftp://localhost"):
            with self.assertRaises(ValueError):
                worker.RemoteBackend(url)
        for timeout in (0, -1, float("nan"), 4000):
            with self.assertRaises(ValueError):
                worker.RemoteBackend("http://localhost", timeout=timeout)
        with serving() as (_, _, url):
            with self.assertRaises(ValueError):
                worker.RemoteBackend(url).run("exec", {})
        with self.assertRaisesRegex(RuntimeError, "connection failed"):
            worker.RemoteBackend(url, timeout=1).status()

    def test_backend_invalid_results_and_response_limits(self):
        with serving() as (backend, _, url):
            remote = worker.RemoteBackend(url)
            for value, code in (({"bad": float("nan")}, 400), ([1, 2], 500), ({"bad": object()}, 500)):
                with patch.object(backend, "status", return_value=value):
                    with self.assertRaises(worker.RemoteError) as caught:
                        remote.status()
                    self.assertEqual(caught.exception.status_code, code)
            with patch.object(worker, "MAX_RESPONSE_BYTES", 256), patch.object(backend, "status", return_value={"big": "x" * 1000}):
                with self.assertRaises(worker.RemoteError) as caught:
                    remote.status()
                self.assertEqual(caught.exception.status_code, 400)

    def test_cli_proxy_never_creates_local_backend(self):
        class Session:
            def __init__(self, backend):
                self.backend = backend

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.backend.close()

        session_module = types.ModuleType("lfm3lab.research.session")
        session_module.ResearchSession = Session
        runner_module = types.ModuleType("lfm3lab.research.runner")

        def run_study(study, session, output, *, split, resume):
            self.assertEqual(split, "discovery")
            result = session.backend.run("score", {"echo": study.id})
            result["splits"] = {split: {"status_counts": {"success": 1, "failed": 0, "interrupted": 0, "missing": 0}}}
            return result

        runner_module.run_study = run_study
        with serving() as (backend, _, url), tempfile.TemporaryDirectory() as temp:
            study = Path(temp) / "study.json"
            study.write_text(json.dumps({"id": "fake", "question": "Does transport work?", "prediction": "Echo agrees.", "cases": [{"id": "a", "prompt": "hello", "target": 1, "foil": 2}], "conditions": [{"id": "clean"}]}))
            with patch.dict(sys.modules, {session_module.__name__: session_module, runner_module.__name__: runner_module}), patch.object(cli, "_local", side_effect=AssertionError("local model creation")), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["run", str(study), "--output", temp, "--remote-url", url]), 0)
            self.assertEqual(json.loads(output.getvalue())["echo"], "fake")
            self.assertFalse(backend.closed)

    def test_cli_model_free_validate_and_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["validate", "/does/not/exist/study.json"]), 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["type"], "FileNotFoundError")
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["unknown-command"]), 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["type"], "ValueError")
        args = cli.parser().parse_args(["run", "a.json", "--output", "out", "--device", "cpu"])
        with self.assertRaisesRegex(ValueError, "allow-local-model"):
            cli._local(args)

    def test_cli_exports_bundled_examples_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as stdout:
            dest = Path(temp) / "examples"
            self.assertEqual(cli.main(["examples", "--output", str(dest)]), 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["synthetic_walkthrough"])
            self.assertEqual({p.name for p in dest.iterdir()},
                             {"code_retrieval.json", "tool_continuation.json", "fake_walkthrough.py"})
            before = {p.name: p.read_bytes() for p in dest.iterdir()}
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main(["examples", "--output", str(dest)]), 1)
            self.assertEqual(json.loads(error.getvalue())["error"]["type"], "FileExistsError")
            self.assertEqual(before, {p.name: p.read_bytes() for p in dest.iterdir()})

    def test_cli_real_runner_remote_resume_and_report(self):
        with serving(backend=ScoreFixture()) as (backend, _, url), tempfile.TemporaryDirectory() as temp:
            study = Path(temp) / "study.json"
            study.write_text(json.dumps({"id": "remote-fixture", "question": "Do saved remote scores agree?", "prediction": "Identity preserves the fixture score.", "cases": [{"id": "d", "prompt": "hello", "target": 1, "foil": 2}, {"id": "h", "prompt": "holdout", "target": 1, "foil": 2, "split": "holdout"}], "conditions": [{"id": "clean"}, {"id": "identity", "patch": {"rules": []}}], "metadata": {"synthetic": True}}))
            output_path = str(Path(temp) / "evidence")
            with patch.object(cli, "_local", side_effect=AssertionError("local model creation")):
                for arguments in (["validate", str(study)], ["run", str(study), "--output", output_path, "--remote-url", url], ["run", str(study), "--output", output_path, "--remote-url", url], ["report", output_path]):
                    with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as errors:
                        self.assertEqual(cli.main(arguments), 0, errors.getvalue())
                    result = json.loads(output.getvalue())
                self.assertEqual(result["splits"]["discovery"]["status_counts"]["success"], 2)
                self.assertEqual(result["splits"]["holdout"]["started_trials"], 0)
                self.assertEqual(len(backend.calls), 2)
                self.assertTrue((Path(output_path) / "report.md").exists())

    def test_cli_requested_split_exit_codes_keep_json_reports(self):
        from .runner import score_payload
        from .schema import Study
        from .store import EvidenceStore, backend_identity, output_lock, trial_key

        def invoke(arguments):
            with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = cli.main(arguments)
            report = json.loads(stdout.getvalue())
            self.assertNotIn("error", report)
            self.assertNotIn('"error": {', stderr.getvalue())
            return code, report

        for scenario, expected in (("failed", 1), ("budget", 3), ("interrupted", 1), ("zero-case", 0), ("other-split-failed", 0)):
            with self.subTest(scenario=scenario), serving(backend=ScoreFixture()) as (backend, _, url), tempfile.TemporaryDirectory() as temp:
                study = Study.from_dict({"id": "exit-fixture", "question": "Do CLI exits reflect journaled outcomes?", "prediction": "Only the requested split determines the exit.", "cases": [{"id": "d", "prompt": "hello", "target": 1, "foil": 2}, {"id": "h", "prompt": "holdout", "target": 1, "foil": 2, "split": "holdout"}], "conditions": [{"id": "clean"}, {"id": "identity"}], "budget": {"max_trials": 1 if scenario == "budget" else 8}})
                study_path = Path(temp) / "study.json"
                study_path.write_text(json.dumps(study.to_dict()))
                output_path = str(Path(temp) / "evidence")
                if scenario in ("interrupted", "other-split-failed"):
                    with output_lock(output_path) as path:
                        store = EvidenceStore(path)
                        store.initialize(study, backend_identity(backend.status()), resume=True)
                        case = study.cases[0 if scenario == "interrupted" else 1]
                        key = trial_key(case.split, case.id, "clean")
                        store.append("intent", key, split=case.split, case_id=case.id, condition_id="clean", payload=score_payload(study, case, study.conditions[0]), reserved_seconds=study.budget.max_seconds)
                        if scenario == "other-split-failed":
                            store.append("failure", key, elapsed_s=0.0, error_type="RuntimeError", error="stored fixture failure", interrupted=False)
                requested = "validation" if scenario == "zero-case" else "discovery"
                run_args = ["run", str(study_path), "--output", output_path, "--remote-url", url, "--split", requested]
                with patch.object(cli, "_local", side_effect=AssertionError("local model creation")):
                    if scenario == "failed":
                        with patch.object(backend, "run", side_effect=RuntimeError("journaled fixture failure")):
                            code, result = invoke(run_args)
                    else:
                        code, result = invoke(run_args)
                    self.assertEqual(code, expected)
                    counts = result["splits"][requested]["status_counts"]
                    if scenario == "failed":
                        self.assertEqual(counts["failed"], 2)
                        # A resumed stored failure remains a report, not an exception.
                        self.assertEqual(invoke(run_args)[0], 1)
                    elif scenario == "budget":
                        self.assertEqual(counts["success"], 1)
                        self.assertEqual(counts["missing"], 1)
                        self.assertEqual(result["run"]["stop_reason"], "max_trials")
                    elif scenario == "interrupted":
                        self.assertEqual(counts["interrupted"], 1)
                        self.assertEqual(counts["missing"], 1)  # Failure precedence over incomplete.
                        self.assertEqual(len(backend.calls), 0)
                    elif scenario == "zero-case":
                        self.assertEqual(result["splits"][requested]["expected_trials"], 0)
                    else:
                        self.assertEqual(result["splits"]["holdout"]["status_counts"]["failed"], 1)
                        self.assertEqual(counts["success"], 2)
                    report_code, saved_report = invoke(["report", output_path])
                    self.assertEqual(report_code, 0)
                    self.assertEqual(saved_report["splits"], result["splits"])


def run_checks() -> dict:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WorkerChecks)
    names = [test._testMethodName.removeprefix("test_") for test in suite]
    with contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {"passed": result.wasSuccessful(), "ok": result.wasSuccessful(), "checks": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "model_free": True, "local_model_loads": 0, "tests": names}
    if not result.wasSuccessful():
        summary["details"] = stream.getvalue()
    return summary


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
