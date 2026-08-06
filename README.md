# sympy-risch

[![Quality](https://github.com/nimish/sympy-risch/actions/workflows/ruff.yml/badge.svg)](https://github.com/nimish/sympy-risch/actions/workflows/ruff.yml)

An experimental implementation of the Risch algorithm built on SymPy's exact
domains.

The project is research software, not a drop-in replacement for
`sympy.integrate`. Its goal is a complete, upstreamable implementation that
distinguishes three outcomes:

- `ElementaryPrimitive`: a reconstructible elementary antiderivative;
- `NonElementary`: a theorem-backed Liouville obstruction; and
- `NotSupported`: an expression or algorithm outside the implemented cell.

An unsupported computation is never reported as a proof of
non-elementarity.

## Quick start

The project currently requires Python 3.13 or newer, SymPy 1.14, and
`python-flint`. It is not yet released on PyPI, so install it from a checkout:

```console
git clone https://github.com/nimish/sympy-risch.git
cd sympy-risch
uv sync --locked --all-groups
```

`integrate_risch` is the public expression-level entry point:

```python
from sympy import diff, exp, log, sin, symbols

from sympy_risch import (
    ElementaryPrimitive,
    NonElementary,
    NotSupported,
    integrate_risch,
)

x = symbols("x")

result = integrate_risch(1 / (x * log(x)), x)
assert isinstance(result, ElementaryPrimitive)
assert result.primitive == log(log(x))
assert diff(result.primitive, x) == 1 / (x * log(x))

non_elementary = integrate_risch(exp(x**2), x)
assert isinstance(non_elementary, NonElementary)

unsupported = integrate_risch(sin(x), x)
assert isinstance(unsupported, NotSupported)
```

The returned primitive is a local differential identity. The frontend does
not silently apply branch-sensitive identities such as
`log(a*b) = log(a) + log(b)`.

## Current scope

The automatic frontend accepts a one-variable grammar of rational arithmetic,
rational powers, and nested `log`, `exp`, and `tan` calls. Other free symbols
are treated as generic algebraically independent constants where the selected
backend supports them.

The implemented mathematical core includes:

- rational integration, including Hermite reduction and independent LRT
  implementations;
- complete explicit primitive, hyperexponential, and hypertangent towers over
  supported exact finite constant fields;
- exact derivative, logarithmic derivative, scalar and parametric RDE,
  limited integration, and the coupled systems required for real tangent
  integration on that P/E/T cell;
- algebraic Hermite reduction, maximal orders, places, valuations, divisors,
  Riemann--Roch spaces, residue reconstruction, and exactness decisions for
  admitted direct algebraic curves;
- scalar and parametric RDEs on admitted algebraic curves; and
- a first mixed-tower cell for terminal diagonal simple radicals over P/E/T
  towers.

Important remaining boundaries include general `RootOf` presentations,
automatic algebraic composita, algebraic constant-field rebasing in mixed
towers, monomial-over-algebraic execution, arbitrary algebraic/transcendental
interleavings, and global branch-aware real presentation.

The low-level APIs also allow callers to construct immutable differential
towers and algebraic function-field models explicitly. See the documentation
for their admission and certificate requirements.

## Development

Run the same main checks used by CI:

```console
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q -n auto --dist worksteal
uv build
```

The codebase targets exact arithmetic. `python-flint` is required; the slower
pure-Python SymPy ground types are not supported.

## Documentation

- [Getting started](docs/getting-started.md) introduces the public API and
  explicit tower construction.
- [Understanding decisions](docs/understanding-decisions.md) explains
  constructive, non-elementary, and unsupported results.
- [Design](docs/design.md) connects the implementation to the mathematical
  algorithms.
- [Algebraic infrastructure](docs/algebraic-infrastructure.md) documents the
  function-field engine.
- [Testing contract](docs/testing.md) states the correctness and performance
  requirements.
- [Full roadmap](plans/001-full-risch-roadmap.md) tracks the path to arbitrary
  algebraic/transcendental towers and eventual SymPy upstreaming.
- [Documentation index](docs/README.md) links the remaining guides and
  research notes.
