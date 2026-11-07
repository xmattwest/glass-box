# Code provenance and licensing

The owner selected **Apache-2.0** for Glass-Box's original code during extraction.
`LICENSE` contains the full text; `NOTICE` carries the retained attribution.
This is a provenance audit, not a legal opinion or proof of every contributor's
ownership. Confirm redistribution rights before public publication.

## Extracted source

The reference is archive commit
`af21465351079e72f5c1dc4ba7bd154f9d8b66df`. The new Git history starts locally
and contains no archive ancestors. [source-manifest.json](source-manifest.json)
lists source paths, original SHA-256 hashes and the selection for every extracted
file. The archive had no root LICENSE or NOTICE at that reference. An absence of
a license does not establish ownership; the project owner's license selection
is the authorization used here.

The mechanistic modules contain no vendored third-party packages. One small
adaptation deserves explicit attribution: `GlassboxHooks.mid_forward` mirrors
the pinned `Lfm3DecoderLayer.forward` operation sequence, adding midpoint edits,
scope/capture and exception-safe restoration. The upstream source header says
`Copyright 2026 The HuggingFace Team. All rights reserved.` and Apache-2.0:

https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/modeling_lfm3.py

The adaptation omits the upstream generalized sandwich-norm branches because the
adapter targets the pinned sandwich-norm configuration. The source file now
carries a prominent origin/change notice. No complete decoder, convolution
kernel, tokenizer or checkpoint implementation is bundled.

The archive's Jacobian fitter referenced Anthropic's Jacobian-lens implementation.
That fitter, its reduction algorithm, calibration helpers and gate edits are
excluded. Only `sync` and `unembed`, ordinary device synchronization and final
normalization/projection helpers, are selected. CPU fixtures resemble the native
RMSNorm/MLP/block structure for contract testing; the decoder attribution is
retained conservatively for that operation sequence as well.

## External dependencies and models

PyTorch (BSD-style), Transformers and Hugging Face Hub (Apache-2.0) are installed
dependencies, not copied source. Their transitive packages keep their own terms;
our license does not replace them. Package metadata lists explicit extras and
the wheel does not vendor those distributions.

The model repository's root license is **LFM Open License v1.0**, distinct from
the modeling-code header. It includes a commercial-use revenue threshold of
US$10 million and other obligations. Review the exact terms before using or
redistributing model artifacts:

https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/LICENSE

Both pinned URLs were inspected during this extraction. No weights were fetched.
Library licensing does not license weights, tokenizer artifacts, model outputs
or trademarks. Because the repository-level and file-header licenses differ,
confirm that relying on the explicit Apache modeling-code header for this small
adaptation is acceptable before public publication; do not call the whole model
Apache-licensed or OSI open source.

Warm Precision was consulted for presentation principles and voice only. No
workspace, fonts, assets, build tools or executable code were copied.
