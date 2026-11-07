"""Documentation coherence and safe examples, without real models."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glass_box_docs_builder", ROOT / "scripts/build_docs.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])


class DocumentationChecks(unittest.TestCase):
    def test_generated_documents_are_current(self):
        expected = builder.outputs()
        self.assertEqual(expected, builder.outputs())
        for name, content in expected.items():
            self.assertEqual((ROOT / name).read_text(), content, f"Stale: {name}")

    def test_builder_never_imports_library_or_model_stack(self):
        code = '''
import importlib.abc, runpy, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'lfm3lab','torch','transformers','huggingface_hub'}:
            raise AssertionError('docs imported runtime library: ' + fullname)
sys.meta_path.insert(0, Deny())
namespace = runpy.run_path(sys.argv[1], run_name='docs_test')
assert namespace['outputs']()
'''
        subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "scripts/build_docs.py")], check=True, capture_output=True)

    def test_inventory_covers_fields_inheritance_properties_and_payloads(self):
        inventory = builder.build_inventory()
        records = {s["name"]: s for m in inventory["modules"] for s in m["symbols"]}
        case = records["lfm3lab.research.schema.Case"]
        self.assertIn("metadata: dict = <factory>", case["signature"])
        self.assertEqual(case["parameters"][0]["name"], "id")
        self.assertEqual(case["parameters"][-1]["factory"], "dict")
        self.assertEqual(records["lfm3lab.research.schema.Case.from_dict"]["inherited_from"], "lfm3lab.research.schema.JsonSpec")
        self.assertEqual(records["lfm3lab.glassbox.Unit.address"]["kind"], "property")
        self.assertEqual(records["lfm3lab.glassbox.probe_pass"]["alias_of"], "lfm3lab.glassbox.run_glassbox_probe")
        self.assertIn("max_input_tokens", inventory["lfm3_operation_payload_fields"]["score"])
        self.assertNotIn("patch", inventory["lfm3_operation_payload_fields"]["sweep"])
        self.assertNotIn("input_ids", inventory["lfm3_operation_payload_fields"]["score"])
        self.assertNotIn("lfm3lab.model.chat_inputs", records)
        self.assertNotIn("lfm3lab.glassbox._reference", records)
        self.assertEqual(len(inventory["modules"]), 14)

    def test_agent_protocol_example_is_valid_and_model_free(self):
        text = (ROOT / "docs/agents.md").read_text()
        example = re.findall(r"```python\n(.*?)\n```", text, re.S)[0]
        namespace = {}
        exec(compile(example, "agent_protocol_example", "exec"), namespace)
        self.assertEqual(namespace["study"].budget.max_trials, 3)
        self.assertEqual(namespace["protocol_id"], namespace["study"].fingerprint)
        guard = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','transformers','huggingface_hub'}:
            raise AssertionError('protocol example imported model stack: ' + fullname)
sys.meta_path.insert(0, Deny())
exec(compile(sys.argv[1], 'agent_protocol_example', 'exec'))
'''
        subprocess.run([sys.executable, "-c", guard, example], check=True, capture_output=True)

    def test_local_html_links_and_fragments_resolve(self):
        paths = [ROOT / name for name in builder.outputs() if name.endswith(".html")]
        parsed = {}
        for path in paths:
            parser = Links()
            parser.feed(path.read_text())
            parsed[path.resolve()] = parser
        for path, parser in parsed.items():
            for link in parser.links:
                url = urlsplit(link)
                if url.scheme or url.netloc:
                    continue
                destination = (path.parent / unquote(url.path)).resolve() if url.path else path
                self.assertTrue(destination.is_file(), f"Missing link in {path.name}: {link}")
                if url.fragment and destination in parsed:
                    self.assertIn(unquote(url.fragment), parsed[destination].ids, f"Broken anchor: {link}")

    def test_keyword_only_and_positional_only_signature(self):
        node = __import__("ast").parse("def changed(a, /, *, b=2, **kwargs): pass").body[0]
        self.assertEqual(builder.signature(node), "changed(a, /, *, b=2, **kwargs)")
        self.assertEqual(builder.parameters(node)[1]["kind"], "keyword-only")

    def test_source_signature_change_changes_generated_reference(self):
        with tempfile.TemporaryDirectory(prefix="glass-box-docs-test-") as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            shutil.copy(ROOT / "pyproject.toml", root / "pyproject.toml")
            shutil.copy(ROOT / "docs/api-contracts.json", root / "docs/api-contracts.json")
            shutil.copy(ROOT / "docs/agents.md", root / "docs/agents.md")
            for module in builder.build_inventory()["modules"]:
                dest = root / module["source"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / module["source"], dest)
            before = builder.outputs(root)["docs/api/reference.md"]
            schema = root / "src/lfm3lab/research/schema.py"
            source = schema.read_text()
            changed = source.replace("def fingerprint(value):", "def fingerprint(value, salt=None):")
            self.assertNotEqual(source, changed)
            schema.write_text(changed)
            after = builder.outputs(root)["docs/api/reference.md"]
            self.assertNotEqual(before, after)
            self.assertIn("fingerprint(value, salt=None)", after)


if __name__ == "__main__":
    unittest.main()
