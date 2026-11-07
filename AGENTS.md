# Glass-Box

Public, library-only project. Read README.md and the relevant section of
docs/library-guide.md. For consumer-agent usage read docs/agents.md, then the
generated docs/api/reference.md or reference.json. This file is contributor
guidance, not a replacement for consumer API documentation.
Source provenance and extraction boundaries are in docs/extraction.md; research
ownership is in docs/research-workflow.md; release gates are in
docs/release-checklist.md.

- Preserve the existing numerical and wire contracts before adding features.
  `lfm3lab` is the compatibility namespace, not the archive's whole lab.
- API signatures/fields/constants are generated statically by scripts/build_docs.py.
  Edit docs/api-contracts.json for semantic notes, not the generated API files.
  Run the builder and --check after interface/docs changes. Never import a model
  to build docs; agent references must remain readable from an installed wheel.
- Archive access requires explicit authorization and is read-only when granted.
  Do not copy its Git history, lab log, results, social assets, cloud helpers,
  secrets or weights. Existing study workspaces and immutable evidence stay intact.
- Default to model-free engineering. Engineering, model execution and publication
  are separate authorizations. No real inference, model downloads, paid resources,
  holdout access, public remote creation or push without explicit owner approval
  for that action. Contributor guidance does not grant any of these permissions.
- Keep implementation, portable contracts and synthetic regressions/examples here.
  Put protocols, cases, analysis and exact environment locks in a separate private
  studies repository. Study runtimes, raw evidence, models and secrets never enter
  library commits or releases. Library `uv.lock` remains ignored.
- Studies consume identified installed wheels outside this checkout. Reduce a
  research observation to a model-free regression before a general library fix;
  evaluate a new build independently. Never repair immutable evidence in place.
- Use Python 3.12. Run `uv sync --extra instruments` then `uv run glass-box check`
  and `uv run python -m unittest discover -s tests -v`. Verify an installed wheel
  with `uv run python scripts/wheel_smoke.py dist/glass_box-0.1.0-py3-none-any.whl`.
- Hold the inference lock throughout model ownership. Serialize hooks; close
  sessions, drain worker requests and restore hooks on errors.
- Record protocol/backend identity, dtype, budgets, missingness and receipt
  hashes. Historical GPU receipts do not validate this extracted package.
- Likelihood effects are not generated correctness, native circuits or practical
  benefit. Fixtures measure software behavior, never model findings.
- Keep public prose specific, restrained and free of em dashes. Follow Warm
  Precision principles without importing its workspace or runtime dependencies.
- Preserve Git author/committer identity and history. Sanitize personal details
  and machine paths in publishable content, not attribution or legal notices.
- Local engineering changes do not authorize publication. Review the exact staged
  contents and artifacts under docs/release-checklist.md before an authorized
  publication. Do not commit machine-local verification or approval outputs.
