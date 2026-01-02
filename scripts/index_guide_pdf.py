import argparse
import hashlib
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader
from pypdf.errors import PdfReadWarning


@dataclass(frozen=True)
class OutlineItem:
    title: str
    page_1_indexed: int
    level: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _outline_title(item: Any) -> str | None:
    try:
        if hasattr(item, "title"):
            t = getattr(item, "title")
            if t is not None:
                t = str(t).strip()
                return t if t else None
        if isinstance(item, dict):
            t = item.get("/Title") or item.get("Title")
            if t is not None:
                t = str(t).strip()
                return t if t else None
    except Exception:
        return None
    return None


def _outline_page_1(reader: PdfReader, item: Any) -> int:
    try:
        page_idx = reader.get_destination_page_number(item)
        return page_idx + 1
    except Exception:
        return -1


def _flatten_outline_items(reader: PdfReader, outline: Any) -> list[OutlineItem]:
    # pypdf outline is typically a list like:
    # [Destination, [child1, child2], Destination, [child...], ...]
    if outline is None:
        return []

    if not isinstance(outline, list):
        outline = [outline]

    out: list[OutlineItem] = []

    def walk(items: list[Any], level: int) -> None:
        last_destination: Any | None = None
        for elem in items:
            if isinstance(elem, list):
                if last_destination is not None:
                    walk(elem, level + 1)
                else:
                    walk(elem, level)
                continue

            title = _outline_title(elem)
            if title is not None:
                out.append(
                    OutlineItem(
                        title=title,
                        page_1_indexed=_outline_page_1(reader, elem),
                        level=level,
                    )
                )
                last_destination = elem
            else:
                last_destination = None

            for child_attr in ("children", "_children"):
                if hasattr(elem, child_attr):
                    try:
                        children = getattr(elem, child_attr)
                        if isinstance(children, list) and children:
                            walk(children, level + 1)
                    except Exception:
                        pass

    walk(outline, 0)
    return out


def extract_page_text(reader: PdfReader, page_index: int) -> str:
    page = reader.pages[page_index]
    try:
        text = page.extract_text() or ""
    except Exception:
        text = ""

    # Normalize line endings a bit.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index a guide PDF into docs/vendor/library/<doc-id> with md index + extracted text."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file to index",
    )
    parser.add_argument(
        "--doc-root",
        required=True,
        help="Document root folder (e.g. docs/vendor/library/<doc-id>)",
    )
    parser.add_argument(
        "--write-pages",
        action="store_true",
        help="Write per-page text files under txt/pages/",
    )

    args = parser.parse_args()

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=PdfReadWarning)

    pdf_path = Path(args.pdf)
    doc_root = Path(args.doc_root)

    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)

    pdf_hash = sha256_file(pdf_path)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # Extract outline if present.
    outline_raw = None
    try:
        outline_raw = reader.outline
    except Exception:
        outline_raw = None

    outline_items = _flatten_outline_items(reader, outline_raw)

    # Extract text.
    all_pages_text: list[str] = []
    pages_dir = doc_root / "txt" / "pages"
    for i in range(n_pages):
        page_text = extract_page_text(reader, i)
        all_pages_text.append(page_text)

        if args.write_pages:
            page_file = pages_dir / f"page_{i+1:03d}.txt"
            header = f"[page {i+1}/{n_pages}]\n"
            write_text(page_file, header + page_text)

    fulltext = "\n\n".join(
        f"[page {i+1}/{n_pages}]\n{t}" for i, t in enumerate(all_pages_text)
    )
    write_text(doc_root / "txt" / "fulltext.txt", fulltext)

    # Generate markdown index.
    md_lines: list[str] = []
    md_lines.append("# UVR/MDX/Demucs/GSEP Guide (Indexed)")
    md_lines.append("")
    md_lines.append("## Source")
    md_lines.append("")
    md_lines.append(f"- **PDF**: `{pdf_path.as_posix()}`")
    md_lines.append(f"- **SHA256**: `{pdf_hash}`")
    md_lines.append(f"- **Pages**: `{n_pages}`")
    md_lines.append(f"- **Indexed at**: `{now}`")
    md_lines.append("")
    md_lines.append("## Files")
    md_lines.append("")
    md_lines.append(f"- `txt/fulltext.txt` (all pages)")
    if args.write_pages:
        md_lines.append(f"- `txt/pages/page_###.txt` (one file per page)")
    md_lines.append("")

    md_lines.append("## Outline / Table of Contents")
    md_lines.append("")

    if outline_items:
        for item in outline_items:
            indent = "  " * min(item.level, 6)
            page = f"p.{item.page_1_indexed}" if item.page_1_indexed > 0 else "p.?"
            md_lines.append(f"- {indent}**{item.title}** ({page})")
    else:
        md_lines.append("No embedded PDF outline was found.")

    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- This index is auto-generated from the PDF using `pypdf`.")
    md_lines.append("- If the outline is missing in the PDF, only extracted text is available.")

    write_text(doc_root / "md" / "index.md", "\n".join(md_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
