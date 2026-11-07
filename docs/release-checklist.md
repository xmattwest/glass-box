# First public release

The public repository is [xmattwest/glass-box](https://github.com/xmattwest/glass-box).
The owner confirmed original-code redistribution rights and retained Apache-2.0
decoder attribution and the separate model-license notice. The documentation site
is [GitHub Pages](https://xmattwest.github.io/glass-box/).
Package-index upload, model execution and private evidence publication remain
separate decisions.

## Historical local readiness

These checks describe the initial local extraction, not verification of subsequent
publication-preparation edits. Re-run the exact-build gates below before release.

- [x] Fresh Git repository with no archive ancestry; initially no public remote.
- [x] Explicit dependency closure, original source hashes and unchanged numerical
  symbol checks. No results, credentials, weights, studio or cloud helpers.
- [x] Root imports are model-free. Numerical and loading dependencies are extras.
- [x] Original `lfm3lab` namespace and `lfm3-research` CLI retained; `glass-box`
  added. Examples export from the installed package without overwriting files.
- [x] Pinned initial adapter and unverified-revision/capability limits explicit.
- [x] CPU fixtures, hook cleanup, budget/resume, receipt corruption and loopback
  transport verified. No real inference or paid worker used.
- [x] Owner selected Apache-2.0. LICENSE/NOTICE included in package metadata.
- [x] Concise README, Python guide, visual explainer and provenance report.
- [x] Human API module pages and consumer-agent guide; source-derived Markdown/JSON
  API and combined agent reference bundled in the wheel. Docs freshness and safe
  protocol examples are tested without model imports.
- [x] Initial staged-file and distribution audit passed for that extraction build.
- [x] Initial installed-wheel smoke passed outside the checkout in fresh base and
  instrument environments. These historical results do not validate a new build.

## Commands before publication

```sh
uv sync --extra instruments --group docs
uv run python scripts/build_docs.py --check
uv run glass-box check
uv run python -m unittest discover -s tests -v
uv run --group docs python scripts/build_site.py --output .site
uv build
uv run python scripts/wheel_smoke.py dist/glass_box-0.1.0-py3-none-any.whl
# Stage only the intended library, docs, tests, guidance and build metadata.
git add .gitignore .python-version AGENTS.md README.md LICENSE NOTICE pyproject.toml llms.txt llms-full.txt src examples docs tests scripts .github/workflows/check.yml .github/workflows/pages.yml
uv run python scripts/release_audit.py dist/glass_box-0.1.0-py3-none-any.whl dist/glass_box-0.1.0.tar.gz
# After authorized remote creation, append:
# --expected-remote-url https://github.com/xmattwest/glass-box.git
git diff --cached --check
git diff --cached --stat
```

The audit checks every indexed file, including unchanged files, staged/working
parity, complete package membership, license metadata, source/resource byte parity,
forbidden artifacts, machine paths and recognizable credential formats. It is not
a full legal or secret-scanning
guarantee. Review the actual staged diff and package lists. Never force-add
ignored evidence or machine-local receipts.

## Publication gates and status

- [x] Owner confirmed original-code redistribution rights and selected Apache-2.0.
  Retain decoder attribution and the separate model notice; see provenance.md.
- [x] Owner selected `xmattwest/glass-box` for first public GitHub publication.
  `glass-box` is the distribution name, not a package-index availability claim.
- [x] Reviewed the exact indexed source, final notices, generated docs and
  distribution contents. Personal content/machine paths are sanitized; Git
  author/committer metadata, history and required attribution are retained.
- [x] Model-free CPU contracts, 53 regression tests, installed-wheel smoke and the
  full-index/distribution audit passed locally for the initial public snapshot
  `be2145cca284469575f8eb195fe516bd156da90f`. Wheel SHA-256:
  `e721bba8c408854747df2c5e1afd4dca8d93361e8d138d6148e24466382c2735`.
- [x] Created the authorized public remote and pushed the reviewed source.
- [x] Deployed and browser-verified the public Pages site, including mobile layout,
  walkthrough, API navigation and rendered guides.
- [ ] Record tagging/release upload only if separately included in owner approval.
- [ ] Package-index publication requires a separate destination and approval.

Do not stage or publish private research, evidence, model assets, secrets, local
approval records or machine configuration. The independent studies workspace is
local/private with no remote by default. See
[research-workflow.md](research-workflow.md) for installed-wheel consumption,
exact research environment locks, immutable evidence and separate engineering,
model/research and publication permissions. Historical GPU receipts do not validate
this build; any new real-model test requires explicit bounded authorization.

## Recommended next feature, not yet implemented

**An explicit inspect-to-study draft helper.** Given a saved probe receipt, a
human-written question/prediction and a selected patch, produce a new Study draft
with source-receipt hash, fixed objective, baseline/identity conditions and visible
split/budget fields. Refuse incomplete targets, unsupported donor operations and
missing controls; require review/freeze before any execution.

Keep it bounded to the current adapter and fixed likelihood objectives. Test
offline that draft creation never invokes a backend, that provenance and selected
coordinates survive roundtrips, and that edits produce a new fingerprint.
Do not infer a native circuit, select validation results, score holdout, perform
automatic search or broaden model support. This reduces the manual handoff
between a useful observation and a fixed test without inventing a research policy.
