"""Consumer-visible URL boundaries for project-subpath documentation deployment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glass_box_site_builder", ROOT / "scripts/build_site.py")
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class SiteLinks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="glass-box-site-links-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        for name in ("docs/index.html", "docs/api/module.html", "docs/library-guide.md", "docs/user guide.md", "README.md", "src/lfm3lab/glassbox.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")

    def test_root_landing_preserves_document_and_fragment_destination(self):
        link = site.rewrite_link("library-guide.md#before-real-model-work", self.root / "docs/index.html", Path("index.html"), self.root)
        self.assertEqual(link, "docs/library-guide.html#before-real-model-work")

    def test_nested_document_keeps_repository_root_reference(self):
        link = site.rewrite_link("../../README.md", self.root / "docs/api/module.html", Path("docs/api/module.html"), self.root)
        self.assertEqual(link, "../../README.html")

    def test_escaped_filename_query_and_fragment_survive_deployment(self):
        link = site.rewrite_link("user%20guide.md?mode=read&theme=light#scope", self.root / "docs/index.html", Path("index.html"), self.root)
        self.assertEqual(link, "docs/user%20guide.html?mode=read&theme=light#scope")

    def test_source_links_open_the_public_repository_at_requested_line(self):
        link = site.rewrite_link("../../src/lfm3lab/glassbox.py#L42", self.root / "docs/api/module.html", Path("docs/api/module.html"), self.root)
        self.assertEqual(link, "https://github.com/xmattwest/glass-box/blob/main/src/lfm3lab/glassbox.py#L42")

    def test_missing_and_outside_targets_fail_instead_of_publishing_broken_links(self):
        source = self.root / "docs/index.html"
        for value in ("missing.md", "../../private-study.md"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                site.rewrite_link(value, source, Path("index.html"), self.root)

    def test_html_link_rewrite_preserves_encoded_query_and_code_text(self):
        text = '<a href="user%20guide.md?x=1&amp;y=2#scope">Guide</a><script>if (a < b && b > 0) run();</script>'
        result = site.rewrite_html(text, self.root / "docs/index.html", Path("index.html"), self.root)
        self.assertEqual(result, '<a href="docs/user%20guide.html?x=1&amp;y=2#scope">Guide</a><script>if (a < b && b > 0) run();</script>')


if __name__ == "__main__":
    unittest.main()
