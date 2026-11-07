# Library guide

Use Glass-Box to inspect a model and test a written hypothesis. Use a `Study` to
freeze the comparison and retain what happened, including failures.

For exact module interfaces, use the [API reference](api/index.html) or
[Markdown API](api/reference.md). Agents should start with the
[agent guide](agents.md), which explains entry-point selection, operation payloads
and failure handling. [The documentation index](index.html) links all formats.

## Start with the model-free path

```sh
glass-box examples --output ./my-studies
glass-box validate ./my-studies/code_retrieval.json
python3.12 ./my-studies/fake_walkthrough.py
```

The default walkthrough runs discovery and validation in a temporary directory,
then discards the invented receipts. To keep them:

```sh
python3.12 ./my-studies/fake_walkthrough.py --output ./synthetic-demo
glass-box report ./synthetic-demo/code-retrieval-mechanics-v1
```

The visible synthetic holdout runs only with `--include-holdout`. These scores
and token IDs are programmed arithmetic. They test orchestration, not attention,
tokenization, convolution, residual dynamics or model behavior. Example export
refuses to overwrite existing files. `glass-box check` requires the
`instruments` extra; validation, export, fake walkthrough and reports do not.

## Python pieces

| Piece | Role | Import |
| --- | --- | --- |
| `Unit`, `PatchSpec`, `GlassboxHooks` | Address, specify and temporarily edit activations | `lfm3lab.glassbox` |
| `Case`, `Condition`, `Objective`, `Budget`, `Study` | Describe a fixed test | `lfm3lab.research` |
| `ResearchSession` | Serialize a local or remote connection | `lfm3lab.research.session` |
| `run_study` | Execute fixed case × condition trials | `lfm3lab.research.runner` |
| `EvidenceStore`, `output_lock` | Store and verify immutable source evidence | `lfm3lab.research.store` |
| `build_report` | Regenerate paired summaries without a model | `lfm3lab.research.report` |

The root `lfm3lab` imports no model stack and exports no unrelated archive
helpers. Numerical modules import PyTorch when explicitly requested; the loader
imports Transformers and Hugging Face Hub only when `load` is called.

This constructs a condition without inference:

```python
from lfm3lab.research import Condition

condition = Condition(
    id="donor_L22_boundary",
    input_variant="corrupt",
    patch={"norm_mode": "native", "rules": [
        {"unit": "L22.res.mid", "op": "replace",
         "source": "clean", "scope": "boundary"}
    ]},
).validate()
```

`Condition.validate` checks protocol structure, not architecture or tokenization.
`PatchSpec.from_value` checks native coordinates; the adapter checks token counts,
vocabulary and exact target/foil tokenization. A character-length match is not
token alignment. The two example protocols are unrun on real weights.

## Inspect, then freeze the hypothesis

Each session has `probe`, `attribute`, `sweep`, `score`, `generate`,
`run(operation, payload)`, `status` and `close`.

| Operation | Question | Boundary |
| --- | --- | --- |
| `probe` | What is active here? | Projections and routing are clues, not necessity tests |
| `attribute` | What is locally sensitive? | Activation × gradient of a fixed token margin, not continuation attribution |
| `sweep` | What changes under finite zero ablation? | Heads, operators, MLPs or blocks; not circuit discovery |
| `score` | How does a supplied target's likelihood change? | Fixed token margin, token logprob or continuation sum/mean logprob |
| `generate` | Which tokens does greedy decoding choose? | Cached decoding plus replay; no clean-source donors |

When a clue suggests a hypothesis, specify the question, prediction, cases,
conditions, objective, controls and budget in a new `Study`. Then execute fixed
conditions by split. The runner does not pick the next hypothesis or adapt a
target to patched outputs.

```python
import json
from lfm3lab.research import Study
from lfm3lab.research.session import ResearchSession
from lfm3lab.research.runner import run_study
from lfm3lab.research.report import build_report

with open("my-studies/code_retrieval.json") as handle:
    study = Study.from_dict(json.load(handle))

# Requires an existing, authorized worker. This URL is an example, not a service.
with ResearchSession.remote("http://127.0.0.1:18765") as session:
    run_study(study, session, "my-evidence", split="discovery")
report = build_report("my-evidence")
```

That last block requests real scores if the worker is real. For CPU orchestration
only, use `FakeBackend` in the exported walkthrough instead. Keep validation and
holdout as separately requested runs. Holdout is visible JSON: the boundary is
procedural, not access control.

## Coordinates and temporary hooks

The pinned adapter has 24 hybrid layers. `L#.h#` names an attention head,
`L#.c#` a convolution channel, `L#.n#` an MLP neuron, and
`L#.res.in/mid/out` a residual checkpoint. Native bounds are 16 heads per
attention layer, 2,048 channels per conv layer and 7,680 neurons per MLP. Head and
channel coordinates must match the layer family. Conv kernels have seven taps;
a lag is not an independently editable channel. There are 49 physical residual
checkpoints; adjacent `out`/`in` names alias the same site.

| Family | Supported edits |
| --- | --- |
| Head | `gate_clamp`, `logit_offset`, `scale`, `zero` |
| Conv channel | `clamp`, `scale`, `zero` |
| MLP neuron | `clamp`, `scale`, `zero`, `mean_scale` with explicit calibrated `mu` |
| Residual | `replace` with explicit hidden-size vector or clean donor, `scale`, `zero` |

Scopes select **predictor input positions**, not the predicted tokens.
`boundary` is the final prompt position. `prompt` excludes it; `answer` includes
it; `first4` includes it and the next three predictor positions. `all` selects
everything. Explicit absolute `positions` narrow the scope. Rules apply in
order; later overlapping rules win. Internal batch `row` coordinates are reserved
for native sweeps, not study conditions.

`GlassboxHooks` is a context manager. It restores hooks and any midpoint forward
override on exit, including exceptions. A midpoint replacement changes both the
next MLP input and the residual skip path, not just an FFN normalization input.
Callers using primitives on their own `Loaded` object must hold `inference_lock`
through loading, use, cleanup and unloading, and serialize hooks. Prefer the
session adapter for ownership; do not nest concurrent hook contexts on one model.

`norm_mode="native"` is the default. `"frozen"` fixes only post-operator and
post-FFN RMS denominators from an unpatched pass on the same receiver input.
Other norms, including final normalization, stay native. A clean donor is not
that denominator reference. Frozen mode is an engineered condition and needs
its own identity control. Clean donors are supported by `score` and `probe`,
with equal token counts and absolute-position matching, not semantic alignment.

## Receipts, budgets and resumption

`run_study` detaches a validated protocol snapshot, journals intent before every
call, and saves JSON responses before recording completion. An evidence output
contains `protocol.json`, `journal.jsonl`, `raw/*.json` and derived
`report.json`/`report.md`. Keep raw evidence unchanged. `EvidenceStore` callers
must hold `output_lock`; runners and reports do this for you.

Budgets count attempts, including failures, across all splits. The token budget
includes teacher-forced predictor tokens: prompt plus all but the last supplied
continuation token. The time budget stops new trials, not an in-flight call.
An interrupted trial charges its remaining time reservation and is not retried.
This is bounded resumption, not exactly-once distributed execution.

Resume verifies protocol and backend identity, journal chaining and raw receipt
hashes. Started pairs, successful or not, are never silently rescored. Changed
protocol, revision, dtype, device, numerical source or Torch/Transformers versions
requires a new output. A hash proves integrity relative to a recorded chain, not
honesty or an externally authenticated history. Retain a trusted journal head
outside the bundle if tamper evidence matters.

Reports pair conditions with the explicit baseline and retain failed, interrupted
and missing counts. Use `metadata.secondary_comparisons` for predeclared
`[[left_id, right_id]]` contrasts. The older example's nested
`metadata.controls.secondary_comparisons` is descriptive metadata only; the
runner does not promote it into report contrasts. Freeze an amended protocol
before moving it. Controls are declarations, not automatically passed tests.

CLI `run` exits 0 when the requested split is complete, 1 for failed/interrupted
trials, and 3 for an incomplete split. `report` can exit 0 while describing
failures accurately. Standard errors describe paired-case variability, not token
uncertainty or population evidence from two examples.

## Local loading and remote transport

Install the `local` extra only on a host where model use is authorized. The
session adapter defaults to the pinned instruct model, BF16 and CUDA. CPU/MPS
real weights require `allow_local_model=True` or CLI `--allow-local-model`.
Other explicit revisions are accepted but status marks them unverified. The
low-level loader retains its old unpinned default and base alias for internal
compatibility; neither implies a supported mechanistic adapter. Use the session
adapter, not an unpinned direct load, for studies.

On an **already authorized** worker host, `glass-box serve --device cuda`
binds loopback port 8766 by default. An SSH tunnel can forward researcher port
18765 to that worker. `glass-box run FILE --output OUTPUT --remote-url URL`
uses only the remote backend. `LFM3_RESEARCH_TOKEN` supplies a bearer token.
Do not commit tokens or put them in example URLs; environment variables avoid
command-line exposure.

The transport accepts only GET `/api/research/status` and POST
`/api/research/run` with a versioned JSON operation/payload. It rejects arbitrary
code, nonfinite/duplicate JSON, browser origins, unexpected hosts and oversized
bodies. Nonloopback binds require a token; use TLS or SSH on untrusted networks.
This small standard-library worker is not a multi-tenant production service.
Remote `close` closes the proxy, not the shared worker. Owners must stop the
server, drain handlers, then close the model backend. There is no provisioning,
distributed scheduler or paid-resource integration.

## Verification limits

The extracted package is tested using synthetic float32/BF16 CPU modules,
exception/ownership fixtures, evidence-corruption tests and loopback transport.
Cached real-model generation, GPU kernels and numerical agreement with historical
GPU receipts are not revalidated here. No real inference is authorized for the
first extraction. Inspect the [extraction report](extraction.md) for commands and
the [release checklist](release-checklist.md) before distribution.
