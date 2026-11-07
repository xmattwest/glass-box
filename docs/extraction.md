# Extraction and local verification

Glass-Box is a small, library-only extraction of reference commit
`af21465351079e72f5c1dc4ba7bd154f9d8b66df`. It starts in fresh local Git history.
The research archive is only a read-only information source. No archive results
or historical GPU receipts are copied or presented as validation of this package.

## Actual dependency closure

| Root or edge | Selected source | Reason |
| --- | --- | --- |
| Instruments | `glassbox.py` | Addresses, patch validation, temporary hooks, probe/gradient/ablation/generation; includes the original 24-layer CPU fixture |
| Numerical dependency | `jacobian_lens.py:sync, unembed` | Device synchronization and final normalization/projection only; no Jacobian fitter |
| Model ownership dependency | `model.py:Loaded, pick_device, MODELS, load` | Fixture carrier and adapter loader; hub/Transformers imports deferred into `load` |
| Protocols | `research/schema.py`, `research/__init__.py` | JSON schema and protocol fingerprinting |
| Session/adapter | `research/session.py`, `backends.py` | Serialized connection, lazy pinned adapter and numerical implementation identity |
| Execution/evidence | `research/runner.py`, `store.py`, `report.py` | Bounded trials, locked/hash-verified storage and paired offline reports |
| Transport/CLI | `research/worker.py`, `cli.py` | Existing JSON worker/client contract, installed CLI and example export |
| Model-free verification | `research/checks*.py` | Schema, tiny-adapter math, evidence/resume, worker and example contracts |
| Minimal examples | Two JSON protocols and `fake_walkthrough.py` | Unrun likelihood protocols and explicitly invented CPU orchestration |

Runtime closure: core schema/session/store/runner/report/transport use the standard
library. Instruments and numerical fixtures need PyTorch. Real loading additionally
needs Transformers and Hugging Face Hub. The package requires Python 3.12 and
POSIX file locks. Windows, other Python versions and generic architecture support
are not claimed. Broad lower-version ranges are dependency declarations, not a
fully tested compatibility matrix.

Excluded: all other lab modules, Jacobian calibration/fitting, plotting libraries,
datasets, notebooks, router/refusal tools, unrelated experiments, studio, paid
worker/cloud utilities, social assets, model weights, research studies/results,
credentials, machine configs and archive Git history. The allowed documentation
was read for meaning, not copied as a historical-results bundle.

## Preservation and deliberate extraction changes

[source-manifest.json](source-manifest.json) lists 20 selected source files and
their original SHA-256 values. It also records **95 unchanged top-level function
or class bodies**, hashed as Python ASTs without line numbers or comments.
These include the numerical primitives, hook implementation, adapter math,
runner/store/report and transport. The regression test compares these bodies
without reopening the archive. It establishes source preservation, not GPU
numerical validation.

Deliberate changes:

- Replaced unrelated eager root exports with a lightweight compatibility
  namespace. Numerical imports are explicit; `Loaded` no longer imports a model
  framework. Kept loader call behavior, including its old unpinned low-level
  default; the study adapter still passes an explicit revision.
- Selected only `sync`/`unembed` from the Jacobian module. Kept their import paths
  so callers and implementation identity do not rely on an unrelated fitter.
- Split dependencies into dependency-free core, `instruments` and `local` extras.
  Removed all research-heavy direct dependencies outside the closure.
- Added `glass-box`, preserved `lfm3-research`, explicitly bundled three example
  resources and prefer installed resources over coincidental source-like paths.
- Added the original 24-layer instrument fixture to CLI `check`, an actionable
  missing-PyTorch error, extraction regressions, wheel smoke and release audits.
- Removed a test-only dependency on a pre-existing inference-lock directory;
  fresh installed environments now use an ordinary temporary fixture output.
- Added explicit Apache decoder attribution after owner license selection and
  rewrote docs for the library. Historical scores and studio links are absent.

The example JSON bytes and fingerprints are unchanged. The retrieval example's
nested control comparisons remain metadata, not report contrasts. Promoting them
to `metadata.secondary_comparisons` would amend a frozen protocol, so it is
documented rather than silently done.

## Local verification

Environment: Python **3.12.12**, PyTorch **2.14.1**, macOS ARM64 CPU. The fixture
environment installs no Transformers, Hub or NumPy. Torch emits an optional
NumPy-initialization warning; these checks use tensors only and do not require
NumPy. No real checkpoint is loaded. The wheel smoke guards the real loader and
sets model-download offline flags.

| Check | Evidence and scope |
| --- | --- |
| Original 24-layer CPU fixture | Float32 hybrid; four/seven-tap reconstruction, 72→49 aliases, scoped cached cursors, endpoint gates, frozen identity, residual donor restoration, projection identities, gradient finite differences and cleanup |
| Tiny backend fixture | Three-layer float32/BF16; fixed score math, tokenization/predictor boundaries, receiver-vs-donor norms, ownership/serialization and unloaded resume for all three objectives |
| Evidence/runner suite | 16 tests; starts/failures count, interrupted reservations, no retries, split boundaries, fingerprint/backend mismatch, locks and raw/protocol/journal corruption rejection |
| Worker/CLI suite | 16 tests; five operations, auth, host/origin restrictions, bounded JSON, explicit errors, queue/socket limits, serialization/draining, CLI exits and remote resume/report |
| Example suite | Two original protocols; 60 invented trials when all synthetic splits explicitly included; no model claims |
| Extraction regressions | Five tests: lightweight imports, deferred loader dependencies, midpoint/partial-registration cleanup and 95-symbol source preservation |
| Installed-wheel smoke | Fresh dependency-free core and PyTorch-only environments outside checkout; export/validate, 40 invented discovery/validation trials, offline reports, full installed CPU suite and loader guard |

Reproduce with the commands in [release-checklist.md](release-checklist.md).
Machine-local smoke JSON is written to ignored `docs/verification-local.json`
when `--output` is used. Study evidence remains temporary or explicitly synthetic,
never part of the wheel. Public documentation summarizes software verification;
the detailed local receipt is not a research finding.

The initial extraction's staged audit covered **38 files**, a **27-file wheel** and the portable
source distribution. No hard-coded machine paths or recognizable credential
formats were found. Wheel modules/resources, license notices and source
distribution contents are checked byte-for-byte against the local sources.
This pattern-based scan is not a guarantee that every possible secret is absent.

The adapted HTML explainer was reviewed at desktop and phone widths. The scope
table scrolls locally, not the page; disclosure controls use native keyboard
behavior. The page has no external resource requests. System font fallbacks are
intentional, so no Warm Precision font assets enter the package.

## Release boundary

Local artifacts are `dist/glass_box-0.1.0-py3-none-any.whl` and
`dist/glass_box-0.1.0.tar.gz`. LICENSE and NOTICE travel with both. The wheel
contains library modules, model-free checks, three example resources and portable
agent/API documentation;
the source distribution also contains portable docs, tests and audit scripts.

The root library was relicensed only after the owner explicitly chose Apache-2.0.
Confirm original-code rights and the adapted decoder's file-header licensing
before public publication. No remote, public upload, live inference or paid
worker is authorized by this extraction. The roadmap recommendation is the
bounded inspect-to-study draft helper in the release checklist, not an implemented
feature or an automatic circuit-search promise.
