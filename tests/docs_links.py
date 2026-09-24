"""Check local links and fragment targets in a built project-site directory."""

from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urljoin, urlparse


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "a" and values.get("name"):
            self.ids.add(values["name"])
        for key in ("href", "src", "srcset"):
            if values.get(key):
                self.targets.append(values[key])


def main(directory):
    root = Path(directory).resolve()
    pages = {}
    for path in root.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path] = parser

    failures = []
    for path, parser in pages.items():
        source_url = "/Nocturne/" + path.relative_to(root).as_posix()
        if path.name == "index.html":
            source_url = source_url.removesuffix("index.html")
        for target in parser.targets:
            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc or target.startswith("//"):
                continue
            destination = urlparse(urljoin(source_url, target))
            if not destination.path.startswith("/Nocturne/"):
                failures.append(f"{path.relative_to(root)}: outside baseurl: {target}")
                continue
            relative = unquote(destination.path.removeprefix("/Nocturne/"))
            resolved = (root / relative).resolve()
            if root not in resolved.parents and resolved != root:
                failures.append(f"{path.relative_to(root)}: escaped site: {target}")
                continue
            if resolved.is_dir():
                resolved /= "index.html"
            if not resolved.is_file():
                failures.append(f"{path.relative_to(root)}: missing: {target}")
            elif destination.fragment and resolved.suffix == ".html":
                destination_page = pages.get(resolved)
                if destination_page and unquote(destination.fragment) not in destination_page.ids:
                    failures.append(f"{path.relative_to(root)}: missing fragment: {target}")
    if failures:
        raise SystemExit("\n".join(failures[:100]))
    print(f"Validated local links and fragments across {len(pages)} HTML pages.")


if __name__ == "__main__":
    main(sys.argv[1])
