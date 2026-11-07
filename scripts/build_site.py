"""Build a portable static documentation site, without importing the library.

Markdown is a build-only dependency. Published files come from explicit source
roots; local verification receipts and generated working directories stay out.
"""
from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import posixpath
import shutil
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
ROOT_DOCUMENTS = ("README.md", "LICENSE", "NOTICE", "llms.txt", "llms-full.txt")


def rewrite_link(value, source, destination, root=ROOT):
    url = urlsplit(value)
    if url.scheme or url.netloc or not url.path:
        return value
    target = (source.parent / unquote(url.path)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Link leaves the source tree: {source.name}: {value}")
    if not target.is_file():
        raise ValueError(f"Missing link target: {source.name}: {value}")
    relative = target.relative_to(root.resolve())
    if relative.parts[0] in {"src", "scripts"} or relative.name == "AGENTS.md":
        return urlunsplit(("https", "github.com", "/xmattwest/glass-box/blob/main/" + quote(relative.as_posix()), url.query, url.fragment))
    if relative.suffix == ".md":
        relative = relative.with_suffix(".html")
    path = posixpath.relpath(relative.as_posix(), destination.parent.as_posix())
    return urlunsplit(("", "", quote(path), url.query, url.fragment))


class SiteHTML(HTMLParser):
    def __init__(self, source, destination, root=ROOT):
        super().__init__(convert_charrefs=False)
        self.source, self.destination, self.root = source, destination, root
        self.parts = []

    def tag(self, tag, attrs, close):
        values = []
        for name, value in attrs:
            if value is None:
                values.append(name)
            else:
                if name in {"href", "src"}:
                    value = rewrite_link(value, self.source, self.destination, self.root)
                values.append(name + '="' + html.escape(value, quote=True) + '"')
        self.parts.append("<" + tag + (" " + " ".join(values) if values else "") + close)

    def handle_starttag(self, tag, attrs):
        self.tag(tag, attrs, ">")

    def handle_startendtag(self, tag, attrs):
        self.tag(tag, attrs, " />")

    def handle_endtag(self, tag):
        self.parts.append("</" + tag + ">")

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append("&" + name + ";")

    def handle_charref(self, name):
        self.parts.append("&#" + name + ";")

    def handle_comment(self, data):
        self.parts.append("<!--" + data + "-->")

    def handle_decl(self, decl):
        self.parts.append("<!" + decl + ">")


class GuideHTML(SiteHTML):
    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.parts.append('<div class="scroll" role="region" aria-label="Scrollable table" tabindex="0">')
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        super().handle_endtag(tag)
        if tag == "table":
            self.parts.append("</div>")


def rewrite_html(text, source, destination, root=ROOT):
    parser_class = GuideHTML if source.suffix == ".md" else SiteHTML
    parser = parser_class(source, destination, root)
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)


def build_site(output, root=ROOT):
    import markdown

    root = root.resolve()
    output = output.resolve()
    if output == root or root.is_relative_to(output) or output.is_relative_to(root / "docs") or output.is_relative_to(root / "src"):
        raise ValueError("Site output must not contain or overwrite source files")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new or empty site output directory")
    spec = importlib.util.spec_from_file_location("glass_box_docs_builder", root / "scripts/build_docs.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    expected = builder.outputs(root)
    stale = [name for name, content in expected.items() if (root / name).read_text() != content]
    if stale:
        raise ValueError("Regenerate documentation before building the site: " + ", ".join(stale))

    sources = [root / name for name in ROOT_DOCUMENTS]
    sources += [path for path in (root / "docs").rglob("*") if path.is_file() and path.name != "verification-local.json"]
    sources += [path for path in (root / "src/lfm3lab").rglob("*.py") if "__pycache__" not in path.parts]
    renderer = markdown.Markdown(extensions=["fenced_code", "tables", "toc"])
    for source in sorted(sources):
        relative = source.relative_to(root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".html":
            destination.write_text(rewrite_html(source.read_text(), source, relative, root))
        else:
            shutil.copyfile(source, destination)
        if source.suffix == ".md" and not source.with_suffix(".html").is_file():
            text = source.read_text()
            title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), source.stem)
            content = renderer.reset().convert(text)
            links = [("docs/index.html", "Documentation"), ("docs/how-we-investigate-lfm3.html", "Walkthrough"), ("docs/api/index.html", "API"), ("docs/agents.md", "Agent guide")]
            nav = "".join('<a href="' + posixpath.relpath(path, relative.parent.as_posix()) + '">' + label + "</a>" for path, label in links)
            page = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>' + html.escape(title) + " · Glass-Box</title>" + builder.STYLE + '</head><body><a class="skip" href="#content">Skip to content</a><main id="content"><nav aria-label="Documentation">' + nav + "</nav>" + content + "</main></body></html>\n"
            rendered = relative.with_suffix(".html")
            (output / rendered).write_text(rewrite_html(page, source, rendered, root))
    home = root / "docs/index.html"
    (output / "index.html").write_text(rewrite_html(home.read_text(), home, Path("index.html"), root))
    (output / ".nojekyll").touch()
    return len([path for path in output.rglob("*") if path.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".site")
    args = parser.parse_args()
    count = build_site(args.output)
    print(f"Built {count} static files; no library imports or model execution")


if __name__ == "__main__":
    main()
