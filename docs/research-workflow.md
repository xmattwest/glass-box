# Library and research workflow

Glass-Box is the public library. Research belongs in a separate local/private
studies repository, for example `~/glassbox-studies`. That sample path is not a
required location. Neither repository grants permission to inspect an archive,
load a model, access holdout, spend money or publish anything.

## Ownership and storage

| Location | Owned content | Excluded content |
| --- | --- | --- |
| Library repository | General implementation, numerical/wire contracts, portable documentation, synthetic model-free regressions and examples | Research cases, analysis, exact research environment locks, raw evidence, model assets, secrets and machine configuration |
| Private studies repository | Frozen protocols, case definitions, analysis code, study documentation and committed exact environment locks | Credentials, model weights, machine-local logs, runtime state and raw evidence in Git |
| Separate private evidence storage | Raw immutable records, protocol copies, journals and trusted receipt heads | In-place corrections or silent replacement |
| Local runtime/storage | Environments, downloaded models and authorized orchestration state | Library commits, wheels, source distributions or public site assets |

The library implements protocol and evidence machinery. This does not make actual
research protocols or evidence library assets. Explicitly invented examples and
reduced fixtures may be committed here when they contain no real observations,
private cases or model data. Ignore rules are a guardrail, not a publication audit.
Never force-add ignored research outputs to a library release. Existing dated
study workspaces and evidence remain untouched; a new workspace is not a migration.

## Reproducible installed-wheel studies

1. Identify the library source commit and build a new wheel through the library's
   release process. Record its filename and SHA-256, not just a version string.
2. Create a study environment outside the library checkout. Install the identified
   wheel, not an editable checkout, source-path import or mutable branch. Record
   the installed distribution and backend implementation identity. Keep the wheel
   in private artifact storage, not the studies Git repository.
3. Commit exact research environment locks, including Python and resolved runtime
   dependencies. Unlike the library's ignored `uv.lock`, the study's `uv.lock` or
   equivalent exact lock is part of its reproducibility record. Record platform,
   device/runtime versions and installation method as well.
4. Freeze the protocol and its fingerprint, cases/splits, model identifier and
   exact revision, tokenizer/config identity, dtype, device, objectives, controls,
   budgets and missingness policy before authorized execution. Record whether any
   model revision or capability is unverified. A library license is not a model
   license or execution approval.
5. Record trusted receipt heads and their hashes in the private study record.
   Validate protocol, backend and raw/journal hashes against those independently
   retained heads before resume or analysis. A hash stored beside replaceable data
   alone is not an independent trust anchor.
6. Keep raw evidence immutable and separate from Git. Analysis may derive new
   outputs with explicit source hashes, but must not rewrite raw evidence to make
   it pass a current implementation. Do not publish evidence implicitly.

For a new protocol, cases, split, objective or budget policy, create a new protocol
identity and new output location. For a changed library/backend build or runtime
identity, use a new output location and identify that build independently, even
when the scientific protocol remains unchanged. Never resume a mismatched build
into old evidence. Preserve the old record and report any invalidity rather than
repairing receipts, journals or protocol files in place. Follow the library's
existing fingerprint and backend checks rather than inventing bypasses.

## From observation to library change

An authorized experiment may reveal a software defect. Reduce it to a portable,
model-free regression with invented inputs and a clear contract. Implement the
general library fix here, preserving unaffected numerical and wire contracts.
Build a new wheel, then independently evaluate it from the private study workspace
with new outputs and explicit provenance. Old experimental evidence does not
validate the new build, and a synthetic regression is not a model finding.
Changes to scientific methodology belong in the studies repository; changes to
general software behavior belong in the library. Do not copy private cases into a
public regression as a shortcut.

## Separate authorization gates

- **Engineering:** permission to edit a named repository does not permit editing
  the other repository. A studies agent must request a scoped library change from
  its owner rather than modifying the library checkout itself.
- **Model/research execution:** require explicit approval for the model, revisions,
  data/splits, resources and budgets. No implicit model downloads, inference,
  holdout access, cloud orchestration or paid resources. Archive inspection needs
  its own explicit approval and remains read-only when granted.
- **Publication:** require approval for the destination and exact content. Local
  commits or a library publication approval do not authorize publishing private
  studies, evidence, secrets, models or independent projects.

Keep local approval records and machine paths out of public content. Preserve Git
identity/history and required public attribution/legal notices. See
[extraction.md](extraction.md), [provenance.md](provenance.md) and
[release-checklist.md](release-checklist.md) for source and publication gates.
