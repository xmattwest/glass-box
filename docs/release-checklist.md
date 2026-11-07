# First public release

The deliverable is a usable **local** Glass-Box 0.1.0 wheel and source
distribution. Creating a public remote, pushing, tagging a public release or
uploading to a package index requires a separate owner decision.

## Local readiness

- [x] Fresh Git repository, no archive ancestry or public remote.
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
- [x] Final staged-file and distribution audit passes for the exact release build.
- [x] Installed-wheel smoke passes outside the checkout in fresh base and
  instrument environments. Re-run after changes affecting package contents.

## Commands before publication

```sh
uv sync --extra instruments
uv run glass-box check
uv run python -m unittest discover -s tests -v
uv build
uv run python scripts/wheel_smoke.py dist/glass_box-0.1.0-py3-none-any.whl
# Stage only the intended library, docs, tests, guidance and build metadata.
git add .gitignore .python-version AGENTS.md README.md LICENSE NOTICE pyproject.toml src examples docs tests scripts
uv run python scripts/release_audit.py dist/glass_box-0.1.0-py3-none-any.whl dist/glass_box-0.1.0.tar.gz
git diff --cached --check
git diff --cached --stat
```

The audit checks staged/working parity, explicit package membership, license
metadata, source/resource byte parity, forbidden artifacts, machine paths and
recognizable credential formats. It is not a full legal or secret-scanning
guarantee. Review the actual staged diff and package lists. Never force-add
ignored evidence or machine-local receipts.

## Owner decisions still required

- [ ] Confirm rights to redistribute the archive-derived original code and the
  adapted decoder under the selected license. The upstream modeling file says
  Apache-2.0; the model repository has separate LFM terms. See provenance.md.
- [ ] Choose/confirm the public repository and package-index names; `glass-box`
  is the local distribution name, not a claim that an index name is available.
- [ ] Approve the public remote before creation or push.
- [ ] Approve publication separately. Verify final license notices, contents and
  checksum; add public URLs only after the destination is chosen.
- [ ] If promising GPU support beyond the inherited contract, authorize and record
  a bounded test of this exact source build. Historical receipts are insufficient.

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
