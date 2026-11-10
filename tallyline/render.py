"""A small Markdown-to-HTML renderer for case documents and results, with passage highlighting.

Input is escaped first, so document text can never inject markup into a page.
"""
from __future__ import annotations

import html
import re
from typing import Iterable


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[#*|`-]", " ", text).lower().split())


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)


def _is_highlighted(line: str, chunks: list[str]) -> bool:
    normal = _normal(line)
    if len(normal) < 3:
        return False
    return any(normal in chunk or chunk in normal for chunk in chunks)


def _table(rows: list[str], chunks: list[str]) -> str:
    out = ['<div class="table-wrap"><table>']
    for number, row in enumerate(rows):
        if re.fullmatch(r"\|?[\s:|-]+\|?", row):
            continue
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        tag = "th" if number == 0 else "td"
        css = ' class="hl"' if number > 0 and _is_highlighted(row, chunks) else ""
        out.append(f"<tr{css}>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def render_markdown(text: str, highlights: Iterable[str] = ()) -> str:
    """Render headings, paragraphs, lists and tables. Lines inside a highlight get class hl."""
    chunks = [_normal(chunk) for chunk in highlights if chunk.strip()]
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(_table(rows, chunks))
        elif re.match(r"^\s*[-*] ", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*] ", lines[i]):
                item = re.sub(r"^\s*[-*] ", "", lines[i])
                css = ' class="hl"' if _is_highlighted(item, chunks) else ""
                items.append(f"<li{css}>{_inline(item)}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
        else:
            heading = re.match(r"^(#{1,4}) (.*)$", line)
            css = ' class="hl"' if _is_highlighted(line, chunks) else ""
            if heading:
                level = min(len(heading.group(1)) + 2, 6)
                out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            else:
                out.append(f"<p{css}>{_inline(line)}</p>")
            i += 1
    return "\n".join(out)
