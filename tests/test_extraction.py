"""Focused extraction regressions; tiny CPU modules, never real weights."""
import importlib
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


class ExtractionChecks(unittest.TestCase):
    def test_selected_source_symbol_bodies_preserved(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "docs" / "source-manifest.json").read_text())
        count = 0
        for row in manifest["files"]:
            expected = row.get("unchanged_symbol_ast_sha256", {})
            if not expected:
                continue
            nodes = {n.name: n for n in ast.parse((root / row["destination"]).read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
            for name, digest in expected.items():
                actual = hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, digest, f"Changed extracted numerical/contract body: {row['destination']}:{name}")
                count += 1
        self.assertGreater(count, 90)

    def test_core_imports_need_no_numerical_packages(self):
        script = '''
import importlib, importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise AssertionError('eager numerical import: ' + fullname)
sys.meta_path.insert(0, Deny())
for name in ['lfm3lab', 'lfm3lab.research', 'lfm3lab.research.schema',
             'lfm3lab.research.session', 'lfm3lab.research.runner',
             'lfm3lab.research.store', 'lfm3lab.research.report',
             'lfm3lab.research.worker', 'lfm3lab.research.cli']:
    importlib.import_module(name)
from lfm3lab.research.backends import LFM3Backend
with_model = LFM3Backend(device='cpu')
assert not with_model.status()['loaded']
with_model.close()
'''
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)

    def test_loaded_fixture_import_needs_no_model_dependencies(self):
        script = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'transformers', 'huggingface_hub'}:
            raise AssertionError('model dependency import: ' + fullname)
sys.meta_path.insert(0, Deny())
from lfm3lab.model import Loaded
assert Loaded(None, None, 'synthetic', 'synthetic', 'cpu').device == 'cpu'
'''
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)

    def test_midpoint_override_cleanup_on_forward_exception(self):
        from lfm3lab.glassbox import GlassboxHooks
        from lfm3lab.research.checks_backend import _tiny
        import torch
        lm = _tiny()
        block = lm.model.model.layers[0]
        original = block.forward.__func__
        cap = GlassboxHooks(lm.model, 2, {"rules": [
            {"unit": "L0.res.mid", "op": "zero"}]})
        with patch.object(block.feed_forward, "forward", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with cap:
                    lm.model.model(input_ids=torch.tensor([[1, 2]]))
        self.assertIs(block.forward.__func__, original)
        self.assertNotIn("forward", block.__dict__)
        self.assertFalse(cap.handles)
        self.assertFalse(cap.forwards)
        self.assertFalse(cap.points)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in lm.model.modules()))

    def test_partial_hook_registration_cleanup(self):
        from lfm3lab.glassbox import GlassboxHooks
        from lfm3lab.research.checks_backend import _tiny
        lm = _tiny()
        block = lm.model.model.layers[0]
        cap = GlassboxHooks(lm.model, 2, {"rules": [
            {"unit": "L0.res.mid", "op": "zero"}]})
        with patch.object(block.feed_forward.w2, "register_forward_pre_hook", side_effect=RuntimeError("registration failure")):
            with self.assertRaisesRegex(RuntimeError, "registration failure"):
                with cap:
                    self.fail("partial registration entered")
        self.assertNotIn("forward", block.__dict__)
        self.assertFalse(cap.handles)
        self.assertFalse(cap.forwards)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in lm.model.modules()))


if __name__ == "__main__":
    unittest.main()
