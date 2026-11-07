"""Model-free defaults and explicit local/remote execution for fixed studies.

Run exits: 0 = requested split complete (including zero cases), 1 = failed or
interrupted trials / command error, 3 = incomplete with missing trials. Failed
or interrupted trials take precedence over missing trials. Reports remain JSON
on stdout; the model-free report command exits 0 even for partial/failed studies.
"""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
import signal
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .schema import SPLITS, Study
from .worker import RemoteBackend, _decode, create_server


def _emit(value, stream=None):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), file=stream or sys.stdout, flush=True)


def _study(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Study exceeds 8 MiB limit")
    return Study.from_dict(_decode(raw))


def _local(args):
    if args.device != "cuda" and not args.allow_local_model:
        raise ValueError("CPU/MPS real weights require --allow-local-model")
    from .backends import LFM3Backend
    kwargs = {"device": args.device, "allow_local_model": args.allow_local_model, "dtype": args.dtype}
    if args.revision is not None:
        kwargs["revision"] = args.revision
    return LFM3Backend(**kwargs)


def _model_options(parser):
    parser.add_argument("--allow-local-model", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f"Invalid command line: {message}")


def parser():
    root = _Parser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Run model-free contract checks")
    examples = commands.add_parser("examples", help="Copy bundled protocol examples and synthetic walkthrough; no model weights")
    examples.add_argument("--output", required=True, help="Destination directory; refuses to overwrite existing files")
    validate = commands.add_parser("validate")
    validate.add_argument("study")
    run = commands.add_parser("run", epilog="Exit codes: 0 requested split complete (including zero cases); 1 failed/interrupted trials; 3 incomplete with missing trials. JSON report is retained on stdout.")
    run.add_argument("study")
    run.add_argument("--output", required=True)
    run.add_argument("--split", choices=SPLITS, default="discovery")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--remote-url")
    target.add_argument("--device", choices=("cuda", "cpu", "mps"))
    run.add_argument("--token", help="Bearer token (prefer LFM3_RESEARCH_TOKEN environment variable)")
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--no-resume", action="store_true")
    _model_options(run)
    report = commands.add_parser("report")
    report.add_argument("output")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--token", help="Bearer token (prefer LFM3_RESEARCH_TOKEN environment variable)")
    serve.add_argument("--allowed-host", action="append", default=[], help="Additional exact Host authority, including external port, for a trusted NAT/proxy")
    serve.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    _model_options(serve)
    return root


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command == "check":
            from .checks import run_checks
            result = run_checks()
            _emit(result)
            return 0 if result.get("passed", result.get("ok", True)) else 1
        if args.command == "examples":
            source_examples = Path(__file__).resolve().parents[3] / "examples" / "mechanistic"
            bundled = files("lfm3lab.research").joinpath("examples")
            source = bundled if bundled.is_dir() else source_examples
            names = ("code_retrieval.json", "tool_continuation.json", "fake_walkthrough.py")
            dest = Path(args.output)
            if any((dest / name).exists() for name in names):
                raise FileExistsError("Bundled examples would overwrite an existing file; choose a new output directory")
            contents = [source.joinpath(name).read_bytes() for name in names]
            dest.mkdir(parents=True, exist_ok=True)
            for name, content in zip(names, contents):
                with (dest / name).open("xb") as handle:
                    handle.write(content)
            _emit({"ok": True, "synthetic_walkthrough": True, "files": [str(dest / name) for name in names]})
        elif args.command == "validate":
            study = _study(args.study)
            _emit({"ok": True, "study_id": study.id, "fingerprint": study.fingerprint})
        elif args.command == "report":
            from .report import build_report
            _emit(build_report(args.output))
        elif args.command == "run":
            study = _study(args.study)
            from .runner import run_study
            from .session import ResearchSession
            if args.remote_url:
                backend = RemoteBackend(args.remote_url, token=args.token or os.environ.get("LFM3_RESEARCH_TOKEN"), timeout=args.timeout)
            else:
                backend = _local(args)
            with ResearchSession(backend) as session:
                result = run_study(study, session, args.output, split=args.split, resume=not args.no_resume)
            _emit(result)
            counts = result["splits"][args.split]["status_counts"]
            if counts["failed"] or counts["interrupted"]:
                return 1
            return 3 if counts["missing"] else 0
        elif args.command == "serve":
            backend = _local(args)
            server = None
            previous_sigterm = None
            previous_sigint = None
            try:
                server = create_server(backend, args.host, args.port, args.token or os.environ.get("LFM3_RESEARCH_TOKEN"))
                for authority in args.allowed_host:
                    parsed = urlsplit("http://" + authority)
                    if (not parsed.hostname or parsed.username or parsed.password or parsed.path
                            or parsed.query or parsed.fragment or parsed.port is None):
                        raise ValueError("--allowed-host must be an exact hostname:port authority")
                    server.allowed_hosts.add(authority.lower())
                previous_sigterm = signal.getsignal(signal.SIGTERM)
                previous_sigint = signal.getsignal(signal.SIGINT)

                def terminate(signum, frame):
                    raise KeyboardInterrupt

                signal.signal(signal.SIGTERM, terminate)
                _emit({"schema_version": 1, "serving": True, "host": server.server_address[0], "port": server.server_address[1], "authenticated": server.token is not None})
                try:
                    server.serve_forever(poll_interval=0.1)
                except KeyboardInterrupt:
                    pass
            finally:
                # ThreadingHTTPServer has non-daemon handlers and joins them.
                # Never release the model's lifetime lock while hooks are active.
                if previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    if server is not None:
                        server.server_close()
                finally:
                    try:
                        backend.close()
                    finally:
                        if previous_sigterm is not None:
                            signal.signal(signal.SIGTERM, previous_sigterm)
                            signal.signal(signal.SIGINT, previous_sigint)
        return 0
    except Exception as exc:
        _emit({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}, sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
