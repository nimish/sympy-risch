"""Executable checks for public documentation examples and local links.

Python fences in one document share a namespace and execute in document order,
so tutorials may build on earlier examples.  Every Python fence is executable
by default.  Context-dependent or schematic fences must be listed explicitly
in ``NON_EXECUTABLE_PYTHON_BLOCKS`` with a reason.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
MARKDOWN_DIRECTORIES = ("docs", "plans", "references")

# Keys use a document-relative path and a zero-based Python-fence ordinal.
NON_EXECUTABLE_PYTHON_BLOCKS = {
    ("docs/design.md", 0): "schematic function signature, not a complete program",
    (
        "docs/understanding-decisions.md",
        0,
    ): "usage pattern intentionally leaves integrand and derivation to the reader",
}


@dataclass(frozen=True, slots=True)
class _Fence:
    """One fenced code block in a Markdown document."""

    info: str
    source: str
    source_line: int


def _markdown_paths() -> tuple[Path, ...]:
    """Return the maintained Markdown documents in deterministic order."""
    paths = [REPOSITORY_ROOT / "README.md"]
    for directory_name in MARKDOWN_DIRECTORIES:
        directory = REPOSITORY_ROOT / directory_name
        paths.extend(directory.rglob("*.md"))
    return tuple(sorted(paths))


def _opening_fence(line: str) -> tuple[str, int, str] | None:
    """Parse a CommonMark-style opening fence, if present."""
    content = line.rstrip("\r\n")
    stripped = content.lstrip(" ")
    if len(content) - len(stripped) > 3 or not stripped:
        return None

    marker = stripped[0]
    if marker not in {"`", "~"}:
        return None

    length = len(stripped) - len(stripped.lstrip(marker))
    if length < 3:
        return None
    return marker, length, stripped[length:].strip()


def _is_closing_fence(line: str, marker: str, minimum_length: int) -> bool:
    """Return whether a line closes the active fenced code block."""
    content = line.rstrip("\r\n")
    stripped = content.lstrip(" ")
    if len(content) - len(stripped) > 3:
        return False

    length = len(stripped) - len(stripped.lstrip(marker))
    return length >= minimum_length and not stripped[length:].strip()


def _fences(markdown: str) -> tuple[_Fence, ...]:
    """Parse fenced code blocks without relying on a multiline regex."""
    blocks: list[_Fence] = []
    marker: str | None = None
    minimum_length = 0
    info = ""
    source_line = 0
    source: list[str] = []

    for line_number, line in enumerate(markdown.splitlines(keepends=True), start=1):
        if marker is None:
            opening = _opening_fence(line)
            if opening is None:
                continue
            marker, minimum_length, info = opening
            source_line = line_number + 1
            source = []
            continue

        if _is_closing_fence(line, marker, minimum_length):
            blocks.append(_Fence(info, "".join(source), source_line))
            marker = None
            continue

        source.append(line)

    if marker is not None:
        raise AssertionError(f"unclosed {marker * minimum_length} code fence")
    return tuple(blocks)


def _python_fences(path: Path) -> tuple[_Fence, ...]:
    """Return Python fences from one document."""
    markdown = path.read_text(encoding="utf-8")
    return tuple(block for block in _fences(markdown) if block.info == "python")


def _documents_with_python_fences() -> tuple[Path, ...]:
    """Return every maintained document containing a Python fence."""
    return tuple(path for path in _markdown_paths() if _python_fences(path))


@pytest.mark.parametrize(
    "document",
    _documents_with_python_fences(),
    ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)),
)
def test_python_documentation_executes_in_order(document: Path) -> None:
    """Execute runnable Python fences sequentially within each document."""
    relative_path = str(document.relative_to(REPOSITORY_ROOT))
    namespace = {"__name__": "__main__"}

    for ordinal, block in enumerate(_python_fences(document)):
        if (relative_path, ordinal) in NON_EXECUTABLE_PYTHON_BLOCKS:
            continue

        # Keep tracebacks aligned with the Markdown document.
        source = "\n" * (block.source_line - 1) + block.source
        exec(compile(source, str(document), "exec"), namespace)


def test_non_executable_python_fence_policy_is_current() -> None:
    """Require every documented exception to identify an existing Python fence."""
    actual_blocks = {
        (str(path.relative_to(REPOSITORY_ROOT)), ordinal)
        for path in _markdown_paths()
        for ordinal, _ in enumerate(_python_fences(path))
    }
    stale_exceptions = NON_EXECUTABLE_PYTHON_BLOCKS.keys() - actual_blocks
    assert not stale_exceptions, f"stale Python-fence exceptions: {stale_exceptions}"


def _prose(markdown: str) -> str:
    """Replace fenced code with blank lines before scanning Markdown links."""
    output: list[str] = []
    marker: str | None = None
    minimum_length = 0

    for line in markdown.splitlines(keepends=True):
        if marker is None:
            opening = _opening_fence(line)
            if opening is None:
                output.append(line)
                continue
            marker, minimum_length, _ = opening
        elif _is_closing_fence(line, marker, minimum_length):
            marker = None
        output.append("\n" if line.endswith("\n") else "")

    return "".join(output)


def _find_balanced(text: str, start: int, opening: str, closing: str) -> int | None:
    """Find a matching Markdown delimiter, respecting nesting and escapes."""
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _link_destination(contents: str) -> str:
    """Extract the destination from Markdown inline-link contents."""
    contents = contents.lstrip()
    if contents.startswith("<"):
        end = contents.find(">")
        return "" if end < 0 else contents[1:end]

    escaped = False
    depth = 0
    for index, character in enumerate(contents):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character.isspace() and depth == 0:
            return contents[:index]
    return contents


def _markdown_links(markdown: str) -> Iterator[tuple[int, str]]:
    """Yield line numbers and destinations for inline Markdown links."""
    prose = _prose(markdown)
    index = 0
    while index < len(prose):
        if prose[index] != "[" or (index > 0 and prose[index - 1] == "\\"):
            index += 1
            continue

        label_end = _find_balanced(prose, index, "[", "]")
        if label_end is None:
            break

        target_start = label_end + 1
        while target_start < len(prose) and prose[target_start].isspace():
            target_start += 1
        if target_start >= len(prose) or prose[target_start] != "(":
            index = label_end + 1
            continue

        target_end = _find_balanced(prose, target_start, "(", ")")
        if target_end is None:
            break

        destination = _link_destination(prose[target_start + 1 : target_end])
        line_number = prose.count("\n", 0, index) + 1
        yield line_number, destination
        index = target_end + 1


def _local_link_path(document: Path, destination: str) -> Path | None:
    """Resolve a local link destination, ignoring URLs and anchors."""
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return document.parent / unquote(parsed.path)


def test_local_markdown_links_resolve() -> None:
    """Keep repository-local inline Markdown links from going stale."""
    missing: list[str] = []
    for document in _markdown_paths():
        markdown = document.read_text(encoding="utf-8")
        for line_number, destination in _markdown_links(markdown):
            target = _local_link_path(document, destination)
            if target is not None and not target.exists():
                relative_document = document.relative_to(REPOSITORY_ROOT)
                missing.append(f"{relative_document}:{line_number}: {destination}")

    assert not missing, "broken local Markdown links:\n" + "\n".join(missing)
