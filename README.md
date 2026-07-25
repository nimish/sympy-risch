# sympy-risch

`sympy-risch` is an experimental, upstream-oriented implementation of the
Risch algorithm built directly on SymPy's exact domains. It is not yet a
drop-in replacement for `sympy.integrate`: callers currently construct the
differential tower explicitly.

The implemented end-to-end dispatcher handles rational functions, recursively
stacked primitive extensions, direct hyperexponential and hypertangent
extensions over `QQ(x)`, and hyperexponential extensions over recursively
primitive lower towers with rational logarithmic derivatives in the top
primitive generator. Lower-level modules also provide recursive radical and
parametric logarithmic-derivative decisions over primitive towers, primitive
RDE weak normalization and cancellation, algebraic Hermite reduction, Trager
finite-residue polynomials for supported algebraic models, generalized
monomial residue reduction, family-aware exact-derivative decisions, and Risch
differential-equation solvers.

`python-flint` is the required exact-arithmetic backend. The project does not
support SymPy's slower pure-Python ground types.

## Install and test

From a checkout:

```console
uv sync --no-editable
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync pytest -q -n auto --dist worksteal -m "not slow"
```

The non-editable install avoids an editable-install issue on Python versions
that ignore hidden `.pth` files. Pytest still reads the current source tree
through the configured `src` path. Ruff checks the complete checkout, while ty
checks the production package under `src`. Run the complete suite, including
the generated rational stress test, with:

```console
uv run --no-sync pytest -q -n auto --dist worksteal
```

The automatic worker count was fastest on the development machine. Use plain
`pytest -q` when debugging a failure serially.

## Minimal example

```python
from sympy import Poly, log, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    DifferentialTower,
    certify_primitive_extension,
    integrate_elementary,
)

x = symbols("x")

# Certify QQ < QQ(x), with Dx = 1, before adjoining x.
constants = DifferentialTower(QQ)
base_analysis = certify_primitive_extension(
    Poly(1, x, domain=QQ),
    constants.derivation(),
)
tower = constants.extend(base_analysis.extension)
field = tower.field()
integrand = field.from_sympy(-1/(x - 1)**2 + 3/x)

result = integrate_elementary(integrand, tower.derivation())

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == 1/(x - 1) + 3*log(x)
```

See the [getting-started guide](docs/getting-started.md) for nested logarithms,
non-elementarity obstructions, and the supported tower shapes.

## Documentation

- [Getting started](docs/getting-started.md): construct towers, integrate, and
  interpret results.
- [Architecture and roadmap](docs/design.md): mathematical design, implemented
  algorithms, and remaining work.
- [Testing contract](docs/testing.md): correctness criteria and test commands.
- [Tutoring notes](docs/tutoring-notes.md): a guided introduction to the
  mathematics.
- [Algebraic-over-hypertangent research note](docs/algebraic-hypertangent.md):
  a proposed future real-algebraic branch.
- [Literature archive](references/README.md): locally available primary papers,
  their provenance, and cited works that still need an authorized copy.
