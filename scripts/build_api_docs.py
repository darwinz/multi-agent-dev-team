"""Build the pdoc API reference into docs/api/.

    make docs        # or: PYTHONHASHSEED=0 uv run --with pdoc==16.0.0 python -m scripts.build_api_docs

Two things keep the committed HTML clean and reproducible:

- Module constants that are paths render with their value, which would bake this machine's absolute
  checkout path into the HTML. Every occurrence of the repo root is rewritten to ".".
- Set and frozenset values render in hash order, so the build must run with a fixed PYTHONHASHSEED
  (the Makefile sets it). Without it, rebuilding produces spurious diffs.
- Function reprs in annotations (`<function merge_dicts at 0x...>`) carry a memory address. It is
  dropped in pdoc's annotation formatter, before the search index is built from those strings.
"""

import os
import re
import shutil
import sys
from pathlib import Path

import pdoc
import pdoc.doc
import pdoc.render

_ADDRESS = re.compile(r" at 0x[0-9a-f]+")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "api"
MODULES = [
    "agents",
    "app",
    "memory",
    "observability",
    "orchestration",
    "resilience",
    "tools",
    "!tools._runner",  # child-process script: it reads sys.argv and writes a file on import
]


def _stable_annotations() -> None:
    """Wrap pdoc's annotation formatter so rendered annotations carry no memory addresses."""
    original = pdoc.doc.formatannotation

    def formatannotation(annotation) -> str:
        return _ADDRESS.sub("", original(annotation))

    pdoc.doc.formatannotation = formatannotation


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("build_api_docs: run with PYTHONHASHSEED=0 (use `make docs`)", file=sys.stderr)
        return 2
    shutil.rmtree(OUT, ignore_errors=True)
    _stable_annotations()
    pdoc.render.configure(docformat="google")
    pdoc.pdoc(*MODULES, output_directory=OUT)
    root = str(ROOT)
    for path in OUT.rglob("*"):
        if path.suffix in {".html", ".js"}:
            text = path.read_text(encoding="utf-8")
            clean = _ADDRESS.sub("", text.replace(root, "."))
            if clean != text:
                path.write_text(clean, encoding="utf-8")
    print(f"build_api_docs: wrote {sum(1 for _ in OUT.rglob('*.html'))} pages to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
