# Glass-Box, explained

Glass-Box helps you inspect a language model, form a hypothesis, make a targeted
activation edit and turn the comparison into a fixed study. The instruments help
you ask the question. The study framework keeps the test and its receipts.

This is the editable outline for
[the visual walkthrough](how-we-investigate-lfm3.html). The filename retains the
initial adapter's name; the page describes this library, not the archive's pilot.

## A clue becomes a fixed comparison

1. **Inspect:** probe a prompt, read local gradient sensitivity or measure a
   finite ablation. A ranked unit is a clue, not a discovered circuit.
2. **Write a hypothesis:** state the question and prediction; choose the score,
   input cases, patch, identity controls and budget.
3. **Freeze a Study:** keep discovery, validation and holdout explicit. Changed
   conditions create a new protocol fingerprint, not a rewritten result.
4. **Execute and retain:** the session sends fixed scoring requests. The runner
   writes intent, raw response and completion/failure to the evidence store.
5. **Report offline:** verify protocol and receipt hashes, compare pairs, retain
   failed and missing denominators. A time budget stops new calls, not a running
   model operation. Interrupted attempts are charged and not silently retried.

The software dataflow is Study → runner → session/backend → receipts → report.
It is not a native model circuit. The current runner executes fixed likelihood
trials; it does not automatically turn a probe into a hypothesis or protocol.

## Where an intervention attaches

The first adapter targets the pinned LFM3-1.7B instruct checkpoint. Its 24 blocks
mix 13 convolution, 5 sliding-attention and 6 global-attention operators. Each
also has an MLP. Across the architecture there are 176 heads, 26,624 conv channels
and 184,320 MLP neurons. These counts come from pinned adapter constants and
bounds, not this extraction's inference results.

Within a block: input residual → normalized operator → post-operator norm →
addition with input residual → midpoint → normalized MLP → post-FFN norm →
addition with midpoint → output residual. Both skip paths matter. Replacing the
midpoint changes both the MLP's read and the state carried around it.

`Unit` addresses heads, channels, neurons or residual checkpoints. `PatchSpec`
specifies the edit. `GlassboxHooks` applies it temporarily and restores on exit.
The 72 residual in/mid/out names identify 49 physical checkpoints because adjacent
output/input names alias. Conv lags 0 through 6 are seven kernel taps, not seven
independently editable units.

## Scope says when, not just where

Positions index predictor inputs. The final prompt token predicts answer token
zero. `boundary` selects it, `prompt` excludes it, `first4` includes it and the
next three predictors, `answer` includes it and later predictors, and `all`
selects the whole sequence. The visual's timeline is illustrative, not measured
tokenization. A clean donor must match receiver token counts.

Native normalization is the default. Frozen mode fixes only receiver
post-operator/post-FFN RMS denominators, not final normalization and not donor
denominators. This is an engineered condition with its own identity control.

## Five calls, different evidence

`probe` describes activations and projections. `attribute` estimates local
activation-times-gradient sensitivity. `sweep` measures finite ablation deltas.
`score` measures a supplied target or continuation, not generated text.
`generate` returns actual greedy tokens, but is not itself a benchmark harness.
Clean-source residual donors are supported only for `score` and `probe`.

No likelihood result here establishes generated correctness, a native circuit
or practical benefit. Receipts establish integrity relative to the recorded
chain, not experiment honesty. Historical GPU receipts belong to earlier source
versions and are not live validation of the extracted library.

## Try the model-free example

After installing the package, export examples, validate a protocol and run the
synthetic walkthrough. This requires no weights or worker. See the README for
installation and the library guide for Python APIs and operational limits.

## Editorial changes from the archive

Retained the conceptual workflow, block anatomy, coordinate key, scopes and
operation distinctions. Removed pilot scores, recovery plots, GPU receipts,
studio/experiment links and archive build dependencies. They are not needed to
understand or validate this public extraction. The page carries no model
findings. Warm Precision informs hierarchy, color roles and voice; no workspace
or fonts are copied. Named font stacks use system fallbacks offline.
