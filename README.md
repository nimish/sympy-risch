# sympy-risch

An upstream-oriented implementation of the Risch algorithm built directly on
SymPy. Development begins with the polynomial and rational cases, then proceeds
through transcendental and algebraic differential extensions.

The current mathematical architecture and staged implementation plan are in
[docs/design.md](docs/design.md).

## Development

```console
uv sync --no-editable
uv run --no-sync pytest
```
