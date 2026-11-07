# Glass-Box

Glass-Box is a Python library for inspecting language-model activations, making
targeted edits, and recording fixed, auditable studies. The instruments help you
investigate; the study framework supplies protocols, sessions, execution,
receipts and reports.

## Try it without a model

Python 3.12 on macOS or Linux. From this checkout:

```sh
python3.12 -m pip install .
glass-box examples --output ./my-studies
glass-box validate ./my-studies/code_retrieval.json
python3.12 ./my-studies/fake_walkthrough.py --output ./synthetic-demo --summary
glass-box report ./synthetic-demo/code-retrieval-mechanics-v2
python3.12 ./my-studies/audit_controls.py ./synthetic-demo/code-retrieval-mechanics-v2 --tolerance 0.000001
```

This runs invented fixture arithmetic, not LFM3. It needs no weights, API key,
PyTorch or worker. After installing the local wheel, the same commands work
outside the checkout. This project is not yet published to a package index.

The output directory retains `protocol.json`, `journal.jsonl`, `raw/*.json` and
derived `report.json`/`report.md` for each study. Reports verify saved hashes;
they do not rescore. The audit checks declared identity pairs against the explicit
fixture tolerance, not generated correctness. Holdout stays unrun. Without
`--output`, the walkthrough discards temporary receipts; omit `--summary` only
if you want all nested reports on stdout. Use a new output for a fresh walkthrough.

`validate` means schema-valid, not tokenizer- or model-ready. The demo also makes
one direct smoke score per study outside its Study budget; the summary counts
those separately. Direct session operations, loading and billing are not metered
by a Study. See the guide's [preflight checklist](docs/library-guide.md#before-real-model-work).

## What is included

- **Instruments:** activation probes, local gradient attribution, finite
  ablations, temporary patches and greedy generation with telemetry replay.
- **Fixed studies:** JSON/Python protocols, explicit splits and budgets,
  local/remote sessions, resumption, hash-verified receipts and paired reports.
- **Examples and CPU checks:** two unrun model protocols and a synthetic
  walkthrough. No research results or model weights are bundled.

The first adapter targets only
`LiquidAI/LFM3-1.7B@40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
Python imports retain the original `lfm3lab` namespace for compatibility.
`lfm3-research` remains a CLI alias. No browser studio, cloud provisioning,
automatic search or general model adapter is included.

**A likelihood change is not generated correctness, a native circuit or a
practical benefit.** Historical GPU receipts identify earlier source versions;
they do not validate this extracted package. Current verification is CPU-only.

## Installation options

| Install | Use |
| --- | --- |
| `pip install .` | Protocols, reports, synthetic examples, remote clients |
| `pip install '.[instruments]'` | PyTorch instruments and the full CPU fixture suite |
| `pip install '.[local]'` | Model loading on an explicitly authorized host |

Installation and status do not load weights. A local session operation or CLI
`run` can; loading executes the pinned checkpoint's custom code. CPU/MPS model
loading requires an extra explicit opt-in. The worker transport connects to
existing hardware and never rents it. Model terms are separate from the library
license.

## Development and guide

```sh
uv sync --extra instruments
uv run python scripts/build_docs.py --check
uv run glass-box check
uv run python -m unittest discover -s tests -v
uv build
uv run python scripts/wheel_smoke.py dist/glass_box-0.1.0-py3-none-any.whl
```

Start at the [documentation index](docs/index.html) or
[visual walkthrough](docs/how-we-investigate-lfm3.html).
Use the [API reference](docs/api/index.html) to look up signatures, fields,
methods and constants. Agents should read the [agent guide](docs/agents.md),
then the [Markdown API](docs/api/reference.md); [llms.txt](llms.txt) indexes the
docs and [llms-full.txt](llms-full.txt) combines them into one agent-readable file.
These Markdown/JSON references are also bundled in the wheel, readable through
`importlib.resources.files("lfm3lab").joinpath("docs")` without loading a model.

Read the [library guide](docs/library-guide.md),
[extraction and verification report](docs/extraction.md), and
[first-release checklist](docs/release-checklist.md).

Code license: [Apache-2.0](LICENSE), selected by the project owner.
See [NOTICE](NOTICE) and [provenance](docs/provenance.md) for the decoder
adaptation and separate model terms. No public remote has been created.

To rebuild the API pages and agent reference after an interface change, run
`uv run python scripts/build_docs.py`. The standard-library generator parses
source without importing it; `--check` catches stale generated documentation.
