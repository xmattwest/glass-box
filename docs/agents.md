# Glass-Box for agents

Read this first when using the library. It is consumer documentation, not the
repository's development instructions. Then look up exact interfaces in the
[generated API reference](api/reference.md) or [structured inventory](api/reference.json).
Humans can start at the [documentation index](index.html) and
[library guide](library-guide.md).

## Identity and scope

- Distribution: `glass-box` 0.1.0. Python namespace: `lfm3lab`.
- Purpose: inspect → human/agent-written hypothesis → targeted intervention →
  fixed, auditable Study. There is no automatic hypothesis search or circuit finder.
- Initial adapter: `LiquidAI/LFM3-1.7B@40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
  Other explicit revisions are accepted but marked unverified. Base/arbitrary
  loader aliases do not imply supported model adapters.
- Python 3.12, POSIX (macOS/Linux). Core has no numerical dependencies.
  `instruments` adds PyTorch; `local` adds model-loading dependencies.
- A documentation example or status capability is not authorization to run a
  model, use a paid worker, inspect holdout or publish. Obtain authorization from
  the caller. Do not invent a worker URL, credential or active service.
- No real inference was performed for extraction or documentation verification.
  Historical GPU receipts identify earlier source versions, not this package.

## Choose the smallest entry point

| Task | API | Effects |
| --- | --- | --- |
| Describe/validate a test | `from lfm3lab.research import Case, Condition, Objective, Budget, Study` | Model-free; no coordinate/tokenizer validation |
| Inspect native coordinates | `from lfm3lab.glassbox import Unit, PatchSpec` | Imports PyTorch; no weights |
| Connect to a backend | `from lfm3lab.research import ResearchSession` | local construction is lazy; remote construction makes no HTTP call |
| Inspect/score/generate/sweep | `session.probe/attribute/score/generate/sweep` | Can load local weights or request real remote inference |
| Execute fixed trials | `from lfm3lab.research import run_study` | Calls scores and writes immutable evidence plus derived reports |
| Verify/report saved evidence | `from lfm3lab.research import build_report` | Model-free; verifies evidence and rewrites derived reports |
| Advanced store/worker use | `lfm3lab.research.store` / `.worker` | Respect locks, authentication and server ownership |

Prefer sessions. Low-level Glass-Box functions require a caller-owned `Loaded`
model and `inference_lock` covering **loading, every call, cleanup and unloading**.
Never overlap hook contexts on the same model. Use context managers; `close()`
waits for active serialized work. Remote close closes the client, not the worker.

## Safe first task: model-free protocol

```python
from lfm3lab.research import Budget, Case, Condition, Objective, Study

study = Study(
    id="code-margin-draft-v1",
    question="Does a declared midpoint edit change the fixed code-token margin?",
    prediction="Identity is unchanged; the finite edit may change the margin.",
    cases=[Case("d0", "Record: amber code = 7. Code:", target="7", foil="2")],
    conditions=[
        Condition("clean"),
        Condition("identity", patch={"rules": [
            {"unit": "L0.res.mid", "op": "scale", "value": 1, "scope": "boundary"}
        ]}),
        Condition("mid_zero", patch={"rules": [
            {"unit": "L0.res.mid", "op": "zero", "scope": "boundary"}
        ]}),
    ],
    objective=Objective("token_margin"),
    budget=Budget(max_trials=3, max_seconds=30, max_input_tokens=128),
    metadata={"status": "unrun draft; no model findings",
              "controls": {"no_op": ["identity"], "correctness": []}},
).validate()
protocol = study.to_dict()
protocol_id = study.fingerprint
```

This constructs a draft only. It does not tokenize, discover a unit, load weights
or execute a trial. The fingerprint changes when fields change; reading it does
not make the Python object immutable. `run_study` detaches a fixed snapshot.
Use multiple separately declared discovery/validation/holdout cases for an actual
study. Fix targets, controls, tolerances and comparisons before reading results.

For an executable CPU orchestration example after installation:

```sh
glass-box examples --output ./my-studies
glass-box validate ./my-studies/code_retrieval.json
python3.12 ./my-studies/fake_walkthrough.py
```

The scores and token IDs are invented fixture arithmetic. Never cite them as model
findings. Export refuses overwrite. The default walkthrough leaves holdout unused.

## Backend operation payload contract

This describes the **initial LFM3Backend**, not every custom backend. A session
does not translate all fields from a low-level primitive signature. It supplies
no `lm` argument. All payloads sent via `run` or HTTP must be JSON-compatible;
only `session.score` accepts an `Objective` instance as a convenience.

Common fields: `prompt`, `chat`, `thinking`, `target`, `foil`, `patch`,
`max_input_tokens`. `prompt` must be nonempty unless probe/attribute uses explicit
`input_ids`. Session defaults: `chat=False`, `thinking=False`,
`max_input_tokens=256`. Low-level primitives instead default to `chat=True`.
Unsupported fields are rejected. `sweep` rejects `patch`.

| Operation | Additional fields | Required/fixed measurement |
| --- | --- | --- |
| `score` | `objective`, `continuation`, `donor_prompt` | Default objective is token_margin: fixed distinct target and foil. token_logprob needs target. continuation_logprob needs nonempty supplied continuation; sum or mean reduction |
| `probe` | `position`, `top_k`, `pins`, `routing`, `input_ids`, `prompt_length`, `donor_prompt`, `trace_start`, `chosen_ids`, `vocabulary_units` | Selected-position readout; fixed target/foil recommended. Omitted pair can be selected from logits for observation only |
| `attribute` | `position`, `top_k`, `input_ids`, `prompt_length`, `full_scores` | Local sensitivity of a token margin, not continuation attribution |
| `generate` | `max_new_tokens`, `top_k`, `pins` | Greedy tokens with telemetry replay, not a benchmark |
| `sweep` | `scope`, `norm_mode`, `batch_size`, `units`, `mode` | Zero ablations; modes heads/layers/mlp_layers/blocks; units only in heads mode |

Limits: score predictor tokens <= `max_input_tokens` (1..4096). For continuation,
predictors are prompt plus continuation IDs except the last; continuation is encoded
separately, not boundary-retokenized. Probe prompt <=512, explicit IDs <=768;
attribute/sweep <=256; generate prompt <=512. Session `max_input_tokens` can
lower these limits but cannot lift primitive caps. `top_k` is 1..64, pins <=32,
vocabulary_units <=4 c/n units. Routing needs <=128 tokens and 1..8 pinned heads.
Generation requests 1..128 new tokens. Sweep batches are 1..16; MPS additionally
limits batch × prompt tokens <=1024. Explicit IDs must be one nonempty flat list
of nonnegative vocabulary IDs; tensors are not accepted over the wire.

Score returns a dict with `value`, `objective`, `input_ids`, `prompt_length`,
`predictor_positions`, `patch`, `elapsed_s`, repository/revision/device/dtype and
implementation/runtime identity. Continuation scores additionally return
`continuation_ids`, `token_logprobs`; donors return `donor_input_ids`.
Telemetry operations return richer JSON dicts documented in the API reference.
Do not assume every backend returns every optional diagnostic field.

## Patch contract

`patch = {"rules": [...], "norm_mode": "native"}`. Rules run in order; later
overlapping rules win. Each rule has `unit`, `op`, optional `value`, `mu`, `scope`,
`positions`, `source`; internal `row` is not allowed in session/study patches.

| Address | Operations |
| --- | --- |
| `L2.h0` head | gate_clamp/logit_offset/scale/zero |
| `L0.c0` conv channel | clamp/scale/zero |
| `L0.n0` MLP neuron | clamp/scale/zero/mean_scale (explicit finite calibrated mu) |
| `L12.res.mid` residual | replace/scale/zero |

Only attention layers have heads and only conv layers have channels. Bounds and
layout are pinned, not guessed. Seven conv taps are lags, not separately editable
channels. Residual adjacent out/in names alias a physical checkpoint. Direct
`Unit`/`PatchSpec` dataclass constructors do not validate; use their validated
factory methods. Study schema validation alone does not check native bounds.

Scopes select **predictor inputs**: boundary is final prompt; prompt excludes it;
answer includes it; first4 includes it and the next three predictors; all is the
whole sequence. Explicit positions narrow scope. Midpoint editing affects both
the MLP input and residual skip. A clean residual replacement uses
`source="clean"`; direct session calls require explicit `donor_prompt`, while
run_study supplies the clean case prompt. Donor/receiver token counts must match,
with absolute-position alignment, not semantic alignment. Clean-source donors
work only for score/probe. Explicit residual vectors must match hidden width.

Native normalization is default. Frozen mode fixes only unpatched **receiver**
post-operator/post-FFN denominators. It does not freeze final normalization or
use donor denominators. Treat frozen mode as an engineered condition with its
own identity control.

## Execution, evidence and failures

- `run_study(study, session, output, split="discovery", resume=True)` scores only
  the explicitly requested split; holdout is a caller decision, not hidden data.
- Budgets count attempts, including failed/interrupted calls, across all splits.
  An in-flight call is not cancelled by the study time budget. Interrupted intent
  consumes its reserved remaining time and is not automatically retried.
- Never repair/overwrite immutable protocol, journal or raw receipts to resume.
  Protocol/backend mismatch or corrupted hashes means preserve output and
  diagnose, not retry. Changed objective/conditions/budget/identity needs a new
  output/protocol; runtime/source changes also affect adapter identity.
- Reports retain raw signed paired deltas and failed/interrupted/missing counts.
  Use `metadata.secondary_comparisons` for fixed [left, right] pairs; the old
  example's nested controls.secondary_comparisons is metadata only.
- Control labels are declarations, not automatic correctness or no-op validation.
  Retain identity errors, dtype and tolerances. Report uncertainty by paired case,
  not token; repeated inputs and tiny samples do not support population claims.
- CLI run: 0 complete requested split, 1 failed/interrupted/error, 3 incomplete.
  CLI report may exit 0 while faithfully describing failures.
- Common failure types: ValueError for invalid requests, PermissionError for
  CPU/MPS loading without opt-in, EvidenceError/OutputLocked for evidence,
  RemoteError with status_code/error_type for worker errors, RuntimeError for
  closed backends, transport failures or malformed envelopes. Do not blindly retry.
- A projection/gradient is a clue. A likelihood delta is an objective-specific
  intervention effect. Neither is generated correctness, a native circuit or
  practical benefit. Hashes establish relative integrity, not experiment honesty.

## Discover docs after installation

The wheel bundles this guide, the library guide, generated Markdown/JSON API and
the combined agent reference. Read them without importing numerical modules:

```python
from importlib.resources import files
docs = files("lfm3lab").joinpath("docs")
orientation = docs.joinpath("agents.md").read_text()
api_reference = docs.joinpath("api", "reference.md").read_text()
api_inventory = docs.joinpath("api", "reference.json").read_text()
combined = docs.joinpath("llms-full.txt").read_text()
```

In a checkout, `llms.txt` indexes documents and `llms-full.txt` is the single-file
orientation + full API reference. Source links in the generated reference refer
to checkout paths. Private underscore helpers, third-party inherited methods and
fixture implementation modules are not part of this documented surface.
