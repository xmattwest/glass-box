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
    result["docs/index.html"] = page("Glass-Box documentation", '<p>Inspect model activations, write a hypothesis, then run a fixed study with auditable receipts. Choose the documentation for the task, whether you are a person or an agent.</p><section><h2>Understand the library</h2><p><a href="how-we-investigate-lfm3.html">Visual walkthrough</a>: workflow, intervention sites and scope.</p><p><a href="library-guide.md">Library guide</a>: Python usage, ownership, budgets and evidence.</p></section><section><h2>Look up an interface</h2><p><a href="api/index.html">Searchable API reference</a>: modules, classes, functions, methods, fields and constants. Generated from the source without loading a model.</p></section><section><h2>Use it as an agent</h2><p><a href="agents.md">Agent guide</a>: smallest entry points, payload contracts, safe examples, failure handling and claim limits.</p><p><a href="../llms-full.txt">Single-file agent guide and complete API</a> · <a href="api/reference.md">API Markdown</a> · <a href="api/reference.json">Structured API inventory</a> · <a href="../llms.txt">Documentation map</a></p></section><section><h2>Check release provenance</h2><p><a href="extraction.md">Extraction report</a> · <a href="provenance.md">License/provenance</a> · <a href="release-checklist.md">Release checklist</a></p><p>No model weights, research results or paid worker are included. Documentation and fixtures do not authorize real inference.</p></section>', api=False)
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
