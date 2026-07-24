# sympy-risch

An upstream-oriented implementation of the Risch algorithm built directly on
SymPy. Development begins with the polynomial and rational cases, then proceeds
through transcendental and algebraic differential extensions.

`python-flint` is the required exact-arithmetic backend.  The project does not
maintain compatibility with SymPy's slower pure-Python ground types.

The current mathematical architecture and staged implementation plan are in
[docs/design.md](docs/design.md).

## Development

```console
uv sync --no-editable
uv run --no-sync pytest
```

Skip the computationally expensive algebraic stress tests with
`uv run --no-sync pytest -m "not slow"`, or run only those tests with
`uv run --no-sync pytest -m slow`.
