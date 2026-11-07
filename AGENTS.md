# Glass-Box

Public, library-only project. Read README.md and the relevant section of
docs/library-guide.md. Source provenance and extraction boundaries are in
docs/extraction.md; release gates are in docs/release-checklist.md.

- Preserve the existing numerical and wire contracts before adding features.
  `lfm3lab` is the compatibility namespace, not the archive's whole lab.
- The research archive is a read-only information source. Do not copy its Git
  history, lab log, results, social assets, cloud helpers, secrets or weights.
- Default to model-free work. No real inference, paid worker, publication,
  public remote creation or push without Matt's explicit authorization.
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
- Local commits are permitted for this extraction. Ask before publishing or
  creating/pushing a remote. Do not commit machine-local verification outputs.
