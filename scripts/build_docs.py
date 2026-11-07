"""Build human and agent API references by parsing source, never importing it.

Standard library only. Editorial contracts live in docs/api-contracts.json;
signatures, fields, constants, docstrings and source hashes come from Python ASTs.
Generated pages and agent text are committed. --check rejects stale output.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_METHODS = {"__enter__", "__exit__"}


def slug(name):
    return name.replace(".", "-")


def signature(node, name=None, bound=False):
    args = copy.deepcopy(node.args)
    if bound:
        positional = args.posonlyargs or args.args
        if positional and positional[0].arg in ("self", "cls"):
            positional.pop(0)
    text = f"{name or node.name}({ast.unparse(args)})"
    if node.returns:
        text += " -> " + ast.unparse(node.returns)
    return text


def parameters(node, bound=False):
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    result = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if bound and index == 0 and arg.arg in ("self", "cls"):
            continue
        result.append({"name": arg.arg, "kind": "positional-only" if index < len(args.posonlyargs) else "positional-or-keyword",
                       "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
                       "default": ast.unparse(default) if default is not None else None})
    if args.vararg:
        result.append({"name": args.vararg.arg, "kind": "variadic-positional", "annotation": ast.unparse(args.vararg.annotation) if args.vararg.annotation else None, "default": None})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        result.append({"name": arg.arg, "kind": "keyword-only", "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
                       "default": ast.unparse(default) if default is not None else None})
    if args.kwarg:
        result.append({"name": args.kwarg.arg, "kind": "variadic-keyword", "annotation": ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else None, "default": None})
    return result


def build_inventory(root=ROOT):
    spec = json.loads((root / "docs/api-contracts.json").read_text())
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    modules, names, payload_fields = [], set(), {}
    for module, overview in spec["modules"].items():
        path = root / "src" / Path(*module.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = path.read_text()
        tree = ast.parse(source)
        local_classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        if module == "lfm3lab.research.backends":
            validate = next(n for n in local_classes["LFM3Backend"].body if isinstance(n, ast.FunctionDef) and n.name == "_validate")
            literals = {n.targets[0].id: ast.literal_eval(n.value) for n in validate.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"common", "extra"}}
            payload_fields = {op: sorted((literals["common"] | extra) - ({"patch"} if op == "sweep" else set())) for op, extra in literals["extra"].items()}
        relative = path.relative_to(root).as_posix()
        record = {"name": module, "overview": overview, "source": relative,
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "docstring": ast.get_docstring(tree), "exports": [], "symbols": []}

        def add(node, name, kind, **details):
            qualified = f"{module}.{name}"
            contract = spec["symbols"].get(qualified, {})
            doc = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
            summary = contract.get("summary") or (doc.split("\n\n", 1)[0].replace("\n", " ") if doc else "")
            entry = {"name": qualified, "short_name": name, "kind": kind, "source": relative,
                     "line": node.lineno, "summary": summary, "docstring": doc, "contract": contract, **details}
            record["symbols"].append(entry)
            names.add(qualified)
            return entry

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                if isinstance(node, ast.FunctionDef):
                    add(node, node.name, "function", signature=signature(node), parameters=parameters(node))
                    continue
                decorators = [ast.unparse(d) for d in node.decorator_list]
                dataclass = any(d == "dataclass" or d.startswith("dataclass(") for d in decorators)
                fields = []
                for field in node.body:
                    if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and not field.target.id.startswith("_"):
                        default = ast.unparse(field.value) if field.value is not None else None
                        factory = None
                        if isinstance(field.value, ast.Call) and isinstance(field.value.func, ast.Name) and field.value.func.id == "field":
                            factory = next((ast.unparse(k.value) for k in field.value.keywords if k.arg == "default_factory"), None)
                        fields.append({"name": field.target.id, "annotation": ast.unparse(field.annotation), "default": default, "factory": factory})
                init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
                constructor = signature(init, node.name, True) if init else None
                if dataclass:
                    constructor = node.name + "(" + ", ".join(f["name"] + ": " + f["annotation"] + (" = <factory>" if f["factory"] else " = " + f["default"] if f["default"] is not None else "") for f in fields) + ")"
                instance = set()
                if init:
                    for child in ast.walk(init):
                        if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store) and isinstance(child.value, ast.Name) and child.value.id == "self" and not child.attr.startswith("_"):
                            instance.add(child.attr)
                add(node, node.name, "class", signature=constructor, parameters=parameters(init, True) if init else [],
                    fields=fields, instance_attributes=sorted(instance), bases=[ast.unparse(b) for b in node.bases], decorators=decorators)
                if dataclass:
                    record["symbols"][-1]["parameters"] = [{"name": f["name"], "kind": "positional-or-keyword", "annotation": f["annotation"],
                                                             "default": "<factory>" if f["factory"] else f["default"], "factory": f["factory"]} for f in fields]
                own_methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
                inherited = {}
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in local_classes:
                        inherited.update({n.name: (n, base.id) for n in local_classes[base.id].body if isinstance(n, ast.FunctionDef) and n.name not in own_methods})
                for method, origin in [*((m, None) for m in own_methods.values()), *inherited.values()]:
                    if method.name.startswith("_") and method.name not in CONTEXT_METHODS:
                        continue
                    dec = [ast.unparse(d) for d in method.decorator_list]
                    kind = "property" if "property" in dec else "classmethod" if "classmethod" in dec else "method"
                    entry = add(method, f"{node.name}.{method.name}", kind, signature=f"{node.name}.{method.name}" if kind == "property" else signature(method, f"{node.name}.{method.name}", True),
                                parameters=[] if kind == "property" else parameters(method, True), inherited_from=f"{module}.{origin}" if origin else None)
                    if not entry["summary"] and origin == "JsonSpec":
                        entry["summary"] = f"Inherited schema {method.name}; see JsonSpec serialization/validation contract."
                    if not entry["summary"] and method.name == "validate":
                        entry["summary"] = "Validate this object; returns the same instance or raises for invalid fields."
                for field in fields:
                    field_node = next(n for n in node.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == field["name"])
                    add(field_node, f"{node.name}.{field['name']}", "field", signature=field["name"] + ": " + field["annotation"], annotation=field["annotation"], default=field["default"], factory=field["factory"])
                for attribute in node.body:
                    if isinstance(attribute, ast.Assign):
                        for target in attribute.targets:
                            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                                add(attribute, f"{node.name}.{target.id}", "class attribute", expression=ast.unparse(attribute.value))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if name == "__all__":
                        for exported in ast.literal_eval(node.value):
                            origin = spec["exports"].get(f"{module}.{exported}")
                            for imp in tree.body:
                                if isinstance(imp, ast.ImportFrom):
                                    match = next((alias for alias in imp.names if (alias.asname or alias.name) == exported), None)
                                    if match:
                                        parent = module.split(".")[:len(module.split(".")) - imp.level + 1] if imp.level else []
                                        origin = ".".join(parent + ([imp.module] if imp.module else []) + [match.name])
                            record["exports"].append({"name": exported, "origin": origin})
                    elif not name.startswith("_") or name == "__version__":
                        alias = isinstance(node.value, ast.Name) and any(isinstance(n, ast.FunctionDef) and n.name == node.value.id for n in tree.body)
                        add(node, name, "alias" if alias else "constant", expression=ast.unparse(node.value), alias_of=f"{module}.{node.value.id}" if alias else None)
        modules.append(record)
    unknown = set(spec["symbols"]) - names
    assert not unknown, f"Editorial contracts refer to missing symbols: {sorted(unknown)}"
    return {"format_version": 1, "project": "glass-box", "version": version, "generator": "static Python 3.12 AST; no library imports or inference",
            "limits": "Declared public surface only. Return contracts are editorial, not inferred type annotations. Dataclass <factory> defaults create fresh objects. Private helpers and external inherited APIs are excluded.", "lfm3_operation_payload_fields": payload_fields, "modules": modules}


def markdown(inventory):
    lines = ["# Glass-Box API reference", "", f"Library version: `{inventory['version']}`. Generated from Python source, without importing the library.", "", inventory["limits"], "",
              "Read `docs/agents.md` for operation payloads, safety, workflow and evidence contracts. Most interfaces are unannotated; this reference does not invent static return types. Method signatures omit self/cls: call instance methods on an instance, classmethods on the class, and properties without parentheses.", "",
             "## Modules", ""]
    lines += [f"- `{m['name']}`: {m['overview']}" for m in inventory["modules"]]
    lines += ["", "## Initial LFM3 operation payload fields", "", "Field names come from the adapter source; required values/defaults/limits are in the agent guide. This is not a JSON Schema or a claim that every listed combination is valid. sweep rejects patch.", ""]
    lines += [f"- `{op}`: " + ", ".join(f"`{field}`" for field in fields) for op, fields in inventory["lfm3_operation_payload_fields"].items()]
    for module in inventory["modules"]:
        lines += ["", f"## {module['name']}", "", module["overview"], "", f"Source: [{module['source']}](../../{module['source']}); SHA-256 `{module['source_sha256']}`.", ""]
        for export in module["exports"]:
            lines += [f"- Re-export `{export['name']}` from `{export['origin']}`."]
        for symbol in module["symbols"]:
            lines += ["", f"### {symbol['name']}", "", f"Kind: {symbol['kind']}. Source: `{symbol['source']}:{symbol['line']}`.", ""]
            if symbol.get("signature"):
                lines += ["```python", symbol["signature"], "```", ""]
            if symbol.get("expression"):
                lines += ["```python", f"{symbol['short_name']} = {symbol['expression']}", "```", ""]
            if symbol["summary"]:
                lines += [symbol["summary"], ""]
            if symbol.get("fields"):
                lines += ["Fields (defaults are source expressions; factories create fresh values):", "", "| Field | Annotation | Default |", "| --- | --- | --- |"]
                lines += [f"| `{f['name']}` | `{f['annotation'].replace('|', r'\|')}` | `{f['default'] if f['default'] is not None else 'required'}` |" for f in symbol["fields"]]
                lines.append("")
            if symbol.get("instance_attributes"):
                lines += ["Public instance attributes assigned by the constructor: " + ", ".join(f"`{n}`" for n in symbol["instance_attributes"]) + ". These are state, not extra constructor keywords.", ""]
            if symbol.get("inherited_from"):
                lines += [f"Inherited from `{symbol['inherited_from']}`.", ""]
            contract = symbol["contract"]
            if contract.get("returns"):
                lines += ["Documented return contract: " + contract["returns"] + ".", ""]
            if contract.get("notes"):
                lines += [contract["notes"], ""]
            if symbol["docstring"]:
                lines += ["Source docstring:", "", "```text", symbol["docstring"], "```", ""]
    return "\n".join(lines).rstrip() + "\n"


STYLE = """<style>
:root{--canvas:#FFFFF0;--ink:#0F0F0F;--secondary:#1E3447;--structure:#2C3E50;--well:#F5EEDC;--rule:rgb(44 62 80 / .22)}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:400 1.125rem/1.65 Inter,Arial,sans-serif}
main{max-width:1040px;margin:auto;padding:48px 24px 80px}a{color:var(--structure);text-underline-offset:3px}:focus-visible{outline:3px solid var(--structure);outline-offset:4px}
.skip{position:absolute;top:-100px;left:24px;padding:8px;background:var(--structure);color:var(--canvas)}.skip:focus{top:8px}
nav{display:flex;flex-wrap:wrap;gap:8px 24px;margin-bottom:32px}nav a{min-height:44px;display:inline-flex;align-items:center;font-size:1rem}
h1{font-size:2.5rem;line-height:1.2;letter-spacing:-.02em;overflow-wrap:anywhere}h2{font-size:1.5rem;line-height:1.3;overflow-wrap:anywhere}h3{font-size:1.125rem;overflow-wrap:anywhere}
p{max-width:66ch}section{padding-top:32px;margin-top:24px;border-top:1px solid var(--rule)}.meta{font-size:.875rem;color:var(--secondary);overflow-wrap:anywhere}
code{font:.9em/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}pre{max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;background:var(--well);padding:16px 20px;border-left:2px solid var(--structure);font-size:.875rem}pre code{font-size:1em}
.scroll{max-width:100%;overflow-x:auto}table{border-collapse:collapse;min-width:520px;width:100%;font-size:1rem}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule)}th{border-bottom:2px solid var(--structure)}td code{white-space:nowrap;overflow-wrap:normal}
ul{padding-left:24px}.symbols li{margin:8px 0;overflow-wrap:anywhere}label{display:block;font-weight:600}input{font:inherit;width:100%;max-width:640px;border:2px solid var(--structure);border-radius:8px;padding:10px 12px;background:var(--canvas);color:var(--ink)}
summary{min-height:44px;cursor:pointer;color:var(--structure);font-weight:600}details{margin:20px 0}[hidden]{display:none!important}.notice{border-left:3px solid var(--structure);padding:12px 20px;background:var(--well)}
@media(max-width:600px){main{padding-top:24px}h1{font-size:2rem}}@media print{nav,.skip,.search{display:none}}
</style>"""


def inline(text):
    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def paragraphs(text):
    return "".join(f"<p>{inline(p.replace(chr(10), ' '))}</p>" for p in text.split("\n\n") if p)


def page(title, body, api=True):
    nav = '<a href="../index.html">Documentation</a><a href="index.html">API modules</a><a href="../agents.md">Agent guide</a><a href="reference.md">Full Markdown API</a>' if api else '<a href="how-we-investigate-lfm3.html">Visual walkthrough</a><a href="library-guide.md">Library guide</a><a href="api/index.html">API reference</a><a href="agents.md">Agent guide</a>'
    return '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>' + html.escape(title) + ' · Glass-Box</title>' + STYLE + '</head><body><a class="skip" href="#content">Skip to content</a><main id="content"><nav aria-label="Documentation">' + nav + '</nav><h1>' + html.escape(title) + '</h1>' + body + '</main></body></html>\n'


HOME_STYLE = """<style>
.site-header{display:flex;align-items:center;justify-content:space-between;gap:24px;padding-bottom:24px;border-bottom:1px solid var(--rule)}
.brand{font:600 1.75rem/1 Newsreader,Georgia,serif;letter-spacing:-.05em;text-decoration:none}.site-header nav{margin:0;gap:24px}
.kicker{font-size:.75rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--secondary)}
.hero{padding:64px 0 24px}.hero h1{font:600 clamp(2.75rem,6.2vw,4.5rem)/1.08 Newsreader,Georgia,serif;letter-spacing:-.04em;margin:18px 0 24px;max-width:920px}
.lede{font-size:1.25rem;line-height:1.6;max-width:640px}.actions{display:flex;flex-wrap:wrap;align-items:center;gap:12px 24px;margin:28px 0}
.button{display:inline-flex;align-items:center;min-height:48px;padding:10px 20px;border:2px solid var(--structure);border-radius:8px;background:var(--structure);color:var(--canvas);font-weight:600;text-decoration:none}
.facts{font-size:.875rem;color:var(--secondary)}.workflow{margin:24px 0 48px}.workflow img{display:block;width:100%;height:auto}.workflow figcaption{font-size:.875rem;color:var(--secondary);margin-top:12px;max-width:72ch}
.workflow-steps{display:none;list-style:none;margin:0;padding:0}.workflow-steps li{display:grid;grid-template-columns:88px minmax(0,1fr);gap:16px;padding:14px 16px;border-top:2px solid var(--structure);font-size:1rem}.workflow-steps li:nth-child(3){background:#F39C12;border:2px solid var(--structure);border-radius:8px}
.capabilities{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:32px}.capabilities dt{font-weight:600;margin-bottom:12px}.capabilities dd{margin:0;font-size:1rem}
.start{display:grid;grid-template-columns:minmax(220px,.8fr) minmax(0,1.5fr);gap:40px}.start>div{min-width:0}.start pre{margin-top:0;white-space:pre;overflow-x:auto;overflow-wrap:normal;font-size:.8125rem}
.start p{font-size:1rem}.resources{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px 40px;padding:0;list-style:none}.resources a{font-weight:600}.resources span{display:block;font-size:1rem;color:var(--secondary);margin-top:6px}
.limits{border-left:3px solid var(--structure);padding:20px 24px;background:var(--well);margin:32px 0}.limits p:last-child{margin-bottom:0}
.site-footer{margin-top:48px;padding-top:24px;border-top:1px solid var(--rule);font-size:.875rem;color:var(--secondary)}.site-footer p{max-width:84ch}
@media(max-width:700px){main{padding:28px 24px 48px}.site-header{align-items:flex-start;flex-direction:column;gap:16px}.site-header nav{gap:8px 24px}.hero{padding-top:36px}.hero h1{font-size:2.75rem}.capabilities,.start,.resources{grid-template-columns:minmax(0,1fr);gap:24px}.workflow{margin-bottom:28px}.workflow img{display:none}.workflow-steps{display:grid;gap:12px}.start pre{font-size:.8125rem}.lede{font-size:1.125rem}}
@media print{.site-header nav,.actions{display:none}.hero{padding-top:24px}.hero h1{font-size:3rem}.workflow{break-inside:avoid}}
</style>"""


def home_page(version):
    body = """<header class="site-header"><a class="brand" href="index.html">glass-box.</a><nav aria-label="Main navigation"><a href="how-we-investigate-lfm3.html">Walkthrough</a><a href="api/index.html">API</a><a href="https://github.com/xmattwest/glass-box">GitHub</a></nav></header>
<div class="hero"><p class="kicker">Python library · @VERSION@ · initial adapter: LFM3-1.7B</p><h1>Inspect activations.<br>Test a fixed hypothesis.</h1><p class="lede">Read internal signals, make a scoped edit, and compare fixed likelihood scores. You choose the question and controls. Glass-Box retains the protocol, attempts and receipts.</p><div class="actions"><a class="button" href="how-we-investigate-lfm3.html">Explore the walkthrough</a><a href="#start">Start without a model</a></div><p class="facts">Python 3.12 · macOS and Linux · Apache-2.0 · no weights bundled</p></div>
<figure class="workflow"><img src="assets/workflow.svg" width="1000" height="240" alt="Inspect, specify, freeze, score and audit: the user chooses the hypothesis and the runner executes only declared trials."><ol class="workflow-steps" aria-label="Study workflow"><li><strong>Inspect</strong><span>Read activations</span></li><li><strong>Specify</strong><span>Choose the question and controls</span></li><li><strong>Freeze</strong><span>Fix cases, edits and budget</span></li><li><strong>Score</strong><span>Execute declared trials only</span></li><li><strong>Audit</strong><span>Verify and compare saved evidence</span></li></ol><figcaption>A software workflow, not a discovered model circuit. The runner does not choose hypotheses or adapt comparisons to outcomes.</figcaption></figure>
<section aria-labelledby="pieces"><h2 id="pieces">Instruments for the question. A framework for the test.</h2><dl class="capabilities"><div><dt>Inspect and intervene</dt><dd>Activation probes, local gradient attribution, finite ablations and temporary patches. Select a native site and predictor-position scope.</dd></div><div><dt>Freeze the comparison</dt><dd>Explicit cases, conditions, objectives, discovery/validation splits and budgets. Serialized local or remote sessions execute fixed scoring trials.</dd></div><div><dt>Keep the evidence</dt><dd>Protocol fingerprints, attempt journals and raw receipts. Resume without silently retrying started trials; verify and report offline.</dd></div></dl></section>
<section id="start" aria-labelledby="start-title"><h2 id="start-title">Start without a model</h2><div class="start"><div><p>Install the core library, export the examples, and run a complete synthetic study.</p><p>No weights, PyTorch, API key or worker are needed. The invented scores test software behavior, not a model finding.</p><p><a href="library-guide.md">Installation options and Python usage</a></p></div><div><pre><code>git clone https://github.com/xmattwest/glass-box.git
cd glass-box
python3.12 -m pip install .

glass-box examples --output ./my-studies
glass-box validate ./my-studies/code_retrieval.json
python3.12 ./my-studies/fake_walkthrough.py \\
  --output ./synthetic-demo --summary
glass-box report ./synthetic-demo/code-retrieval-mechanics-v2
python3.12 ./my-studies/audit_controls.py \\
  ./synthetic-demo/code-retrieval-mechanics-v2 --tolerance 0.000001</code></pre></div></div><p class="meta">The output retains protocol.json, journal.jsonl and raw receipts. Reports verify saved evidence without rescoring. Schema validation is not tokenizer or model readiness; holdout stays unrun.</p></section>
<div class="limits"><p><strong>A likelihood change is not generated correctness, a native circuit or a practical benefit.</strong></p><p>The initial adapter targets one pinned LiquidAI/LFM3-1.7B revision. Greedy generation is an instrument, not a correctness benchmark. Model loading, downloads and paid resources require separate authorization.</p><p><a href="library-guide.md#before-real-model-work">Read the real-model preflight checklist</a></p></div>
<section aria-labelledby="guides"><h2 id="guides">Choose the reference for your task</h2><ul class="resources"><li><a href="how-we-investigate-lfm3.html">Visual walkthrough</a><span>Workflow, native sites, block anatomy and predictor scopes.</span></li><li><a href="library-guide.md">Library guide</a><span>Python usage, ownership, budgets and evidence.</span></li><li><a href="api/index.html">Searchable API reference</a><span>Signatures, fields, methods and constants, generated statically.</span></li><li><a href="agents.md">Agent guide</a><span>Smallest entry points, operation payloads, failures and safe usage.</span></li></ul><p class="meta"><a href="../llms.txt">Documentation map</a> · <a href="api/reference.md">API Markdown</a> · <a href="api/reference.json">Structured API</a> · <a href="../llms-full.txt">Combined agent reference</a></p></section>
<section aria-labelledby="contribute"><h2 id="contribute">Develop the library. Keep experiments separate.</h2><p>Public source includes portable instruments, contracts, synthetic fixtures and documentation. Study plans, analysis, exact environments and immutable live evidence belong in a separate research workspace.</p><p><a href="research-workflow.md">Research and library boundaries</a> · <a href="release-checklist.md">Release checks</a> · <a href="site-guide.md">Maintain the documentation site</a></p></section>
<footer class="site-footer"><p><a href="https://github.com/xmattwest/glass-box">Source on GitHub</a> · <a href="../LICENSE">Apache-2.0</a> · <a href="provenance.md">Licensing and provenance</a> · <a href="extraction.md">Extraction and verification</a></p><p>Model terms are separate. No model weights, research results or cloud provisioning are included. Warm Precision informs hierarchy, color and voice; this site uses system font fallbacks and no design-system runtime.</p></footer>"""
    return '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Python instruments for activation inspection, scoped interventions and fixed, auditable likelihood studies. Start without a model."><link rel="icon" href="data:,"><title>Glass-Box · Activation instruments and fixed studies</title>' + STYLE + HOME_STYLE + '</head><body><a class="skip" href="#content">Skip to content</a><main id="content">' + body.replace('@VERSION@', html.escape(version)) + '</main></body></html>\n'


def module_page(module):
    body = paragraphs(module["overview"]) + '<p class="meta">Source: <a href="../../' + module["source"] + '">' + module["source"] + '</a>. Generated statically; no model execution.</p>'
    if module["exports"]:
        body += '<h2>Convenience exports</h2><ul>' + ''.join('<li><code>' + e["name"] + '</code> from <code>' + html.escape(e["origin"] or "source") + '</code></li>' for e in module["exports"]) + '</ul>'
    body += '<h2>On this page</h2><ul class="symbols">' + ''.join(f'<li><a href="#{slug(s["name"])}"><code>{html.escape(s["short_name"])}</code></a> <span class="meta">{s["kind"]}</span></li>' for s in module["symbols"]) + '</ul>'
    for s in module["symbols"]:
        body += f'<section id="{slug(s["name"])}"><h2><code>{html.escape(s["short_name"])}</code></h2><p class="meta">{s["kind"]} · source line {s["line"]}</p>'
        code = s.get("signature") or (s["short_name"] + " = " + s["expression"] if s.get("expression") else None)
        if code:
            body += '<pre><code>' + html.escape(code) + '</code></pre>'
        if s["summary"]:
            body += paragraphs(s["summary"])
        if s.get("fields"):
            body += '<div class="scroll" role="region" aria-label="Dataclass fields" tabindex="0"><table><thead><tr><th>Field</th><th>Annotation</th><th>Default expression</th></tr></thead><tbody>' + ''.join('<tr><td><code>' + html.escape(f["name"]) + '</code></td><td><code>' + html.escape(f["annotation"]) + '</code></td><td><code>' + html.escape(f["default"] if f["default"] is not None else "required") + '</code></td></tr>' for f in s["fields"]) + '</tbody></table></div><p class="meta">&lt;factory&gt; constructor defaults create fresh values; fields show the source factory expressions.</p>'
        if s.get("instance_attributes"):
            body += '<p>Public state assigned by the constructor: ' + ', '.join('<code>' + n + '</code>' for n in s["instance_attributes"]) + '. These are not extra constructor keywords.</p>'
        if s.get("inherited_from"):
            body += '<p class="meta">Inherited from <code>' + s["inherited_from"] + '</code>.</p>'
        for key, prefix in (("returns", "Documented return contract: "), ("notes", "")):
            if s["contract"].get(key):
                body += paragraphs(prefix + s["contract"][key])
        if s["docstring"]:
            body += '<details><summary>Source docstring</summary><pre>' + html.escape(s["docstring"]) + '</pre></details>'
        body += '</section>'
    return page(module["name"], body)


def outputs(root=ROOT):
    inventory = build_inventory(root)
    reference = markdown(inventory)
    body = '<p>Look up modules, classes, methods, fields, constants and declared signatures. Start with the library guide for the investigation workflow.</p><p class="notice">Built from source without imports or inference. Return contracts are documented prose, not invented type annotations. Method signatures omit self/cls; use an instance unless marked classmethod. Private helpers and external inherited APIs are excluded.</p><div class="search"><label for="query">Find an API symbol</label><input id="query" type="search" placeholder="Study, score, PatchSpec" aria-describedby="results"><p id="results" class="meta" aria-live="polite"></p></div>'
    for module in inventory["modules"]:
        body += '<section data-module><h2><a href="' + slug(module["name"]) + '.html">' + module["name"] + '</a></h2>' + paragraphs(module["overview"]) + '<ul class="symbols">'
        for s in module["symbols"]:
            body += '<li data-search="' + html.escape((s["name"] + ' ' + s["summary"]).lower(), quote=True) + '"><a href="' + slug(module["name"]) + '.html#' + slug(s["name"]) + '"><code>' + html.escape(s["short_name"]) + '</code></a> <span class="meta">' + s["kind"] + '</span></li>'
        body += '</ul></section>'
    body += """<script>
const input=document.querySelector('#query'), rows=[...document.querySelectorAll('[data-search]')];
function filter(){const q=input.value.toLowerCase().trim();let n=0;for(const row of rows){row.hidden=!row.dataset.search.includes(q);if(!row.hidden)n++}for(const group of document.querySelectorAll('[data-module]')){group.hidden=q!==''&&![...group.querySelectorAll('[data-search]')].some(row=>!row.hidden)&&!group.querySelector('h2').textContent.toLowerCase().includes(q)}document.querySelector('#results').textContent=n+(n===1?' symbol shown':' symbols shown');}input.addEventListener('input',filter);filter();
</script>"""
    combined = '# Glass-Box agent documentation and API\n\n' + (root / 'docs/agents.md').read_text().rstrip() + '\n\n---\n\n' + reference
    result = {"docs/api/index.html": page("API reference", body), "docs/api/reference.md": reference,
              "docs/api/reference.json": json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", "llms-full.txt": combined}
    result.update({f"docs/api/{slug(m['name'])}.html": module_page(m) for m in inventory["modules"]})
    result["docs/index.html"] = home_page(inventory["version"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = outputs()
    stale = [name for name, data in expected.items() if not (ROOT / name).exists() or (ROOT / name).read_text() != data]
    extras = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs/api").glob("*.html") if str(p.relative_to(ROOT)) not in expected)
    if args.check:
        if stale or extras:
            raise SystemExit("Stale generated docs: " + ", ".join(stale + extras) + "; run python scripts/build_docs.py")
        print(f"Documentation current: {len(expected)} generated files; no library imports")
        return
    assert not extras, f"Remove/reconcile obsolete generated pages explicitly: {extras}"
    for name, data in expected.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    print(f"Built {len(expected)} documentation files from source ASTs")


if __name__ == "__main__":
    main()
