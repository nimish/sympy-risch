"""Executable checks for the public documentation examples."""

import re
from pathlib import Path


def _first_python_block_after(markdown: str, heading: str) -> str:
    """Return the first Python fence following one exact Markdown heading."""
    _, separator, section = markdown.partition(heading)
    if not separator:
        raise AssertionError(f"missing documentation heading: {heading}")
    match = re.search(r"```python\n(.*?)\n```", section, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"missing Python example after: {heading}")
    return match.group(1)


def test_readme_minimal_example_executes() -> None:
    """Keep the public-package quick start synchronized with the API."""
    readme = Path(__file__).parents[1] / "README.md"
    source = _first_python_block_after(
        readme.read_text(encoding="utf-8"),
        "## Minimal example",
    )

    namespace = {"__name__": "__main__"}
    exec(compile(source, str(readme), "exec"), namespace)
