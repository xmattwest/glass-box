# Documentation site

The GitHub Pages site is a static view of the library documentation. It contains
no model runtime, experiment dashboard, private evidence or cloud controls.

## Edit the source, not generated pages

- `scripts/build_docs.py` generates the landing page and API HTML/Markdown/JSON
  from source ASTs and `docs/api-contracts.json`. It never imports the library.
- `docs/how-we-investigate-lfm3.html` is the authored visual walkthrough. Its
  Markdown companion records editorial scope; the site builder preserves the
  authored HTML instead of replacing it with a Markdown rendering.
- Other Markdown guides remain the editable source. The Pages build renders them
  as HTML while retaining raw Markdown for agents.
- `docs/assets/workflow.svg` is the README/desktop workflow illustration. The
  landing page recomposes the same sequence as readable steps on narrow screens.
  System font fallbacks keep the site independent of a font service or design kit.

Warm Precision informs hierarchy, semantic color, restrained emphasis, readable
measure and evidence-specific language. No Warm Precision workspace, executable
code, fonts or runtime dependencies are imported.

## Build and preview locally

Python 3.12. Markdown is a pinned documentation dependency, not a library runtime
requirement:

```sh
uv sync --group docs
uv run python scripts/build_docs.py
uv run python scripts/build_docs.py --check
uv run --group docs python scripts/build_site.py --output .site
uv run python -m http.server 8080 --bind 127.0.0.1 --directory .site
```

Open `http://127.0.0.1:8080/`. Choose an unused port if needed. The site builder
requires a new or empty output directory so stale files cannot enter a deployment.
`.site/` is generated and ignored; remove only that generated output before a
fresh rebuild. Never use the source tree or a research workspace as output.

Inspect desktop and phone layouts, keyboard focus, the walkthrough disclosures,
API search, and navigation between rendered guides. Necessary code/table overflow
should scroll within its region, not widen the page. Follow local links and
fragments; a passing build alone is not visual verification.

## GitHub Pages and continuous checks

`.github/workflows/pages.yml` builds with documentation dependencies only and
uploads a fresh static directory to GitHub Pages. `scripts/build_site.py` selects
public documentation, root reference/license files and Python source text; it
excludes the machine-local verification receipt. It maps links correctly from
both the root landing page and nested documentation, and points HTML source links
to GitHub. No model operations execute in a Pages build.

`.github/workflows/check.yml` runs Python 3.12 CPU instruments, regression tests,
static documentation, installed-wheel verification and the complete-index/package
publication audit. The Linux instruments extra routes only PyTorch through an
explicit CPU index; build dependencies stay on PyPI. Hosted wheel verification
uses the helper's explicit `--online` mode for Python dependencies and
`UV_TORCH_BACKEND=cpu`. Model downloads and real inference remain blocked by
offline model settings and the loader guard. No checkpoint, GPU worker, research
credentials or study evidence is needed. Workflow
actions are pinned to reviewed release commits.

Repository publication and Pages enablement require explicit owner approval.
Library publication does not authorize private research publication or model
execution. See [release checks](release-checklist.md) and
[research boundaries](research-workflow.md).
