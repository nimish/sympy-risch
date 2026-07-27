# sympy-risch

`sympy-risch` is an experimental, upstream-oriented implementation of the
Risch algorithm built directly on SymPy's exact domains. It is not yet a
drop-in replacement for `sympy.integrate`. The narrow one-variable grammar of
rational arithmetic, integer powers, and nested `log`/`exp`/`tan` calls can
use `integrate_expression`; other callers construct the differential tower
explicitly.

The implemented end-to-end dispatcher handles rational functions and explicit
finite towers of primitive, hyperexponential, and hypertangent extensions
over `QQ(x)`, in any order when their constant fields have been certified as
preserved. It also normalizes the exact hyperbolic law
`Dt = eta*(1 - t**2)` to an isomorphic hyperexponential coordinate. When a
caller supplies two checked linear special-factor certificates, it can also
normalize a preserved-constants quadratic/Riccati top level explicitly and
integrate through its hyperexponential coordinate when the transformed target
is admitted by the P/E/T contract, via
`build_riccati_mobius_normal_form` and
`integrate_riccati_mobius_normal_form`. It does not yet search for those
factors automatically. A separate local API,
`reduce_certified_linear_special_poles`, consumes one checked linear special
factor to cancel a currently exposed principal part when its lower-field RDE
has a solution; an unsolved local RDE remains an explicit boundary, not a
non-elementarity claim.
Lower-level modules also provide
recursive radical and parametric logarithmic-derivative decisions over
primitive towers, primitive RDE weak normalization and cancellation,
algebraic Hermite reduction, certified algebraic maximal orders and
fractional ideals, ramified and infinite places, exact valuations and
divisors, good-prime finite Jacobian torsion decisions over `QQ`, Trager
finite-residue polynomials for supported algebraic models,
an independent Czichowski Gröbner-basis checker for rational LRT logarithmic
parts, generalized monomial residue reduction, family-aware
exact-derivative decisions, and Risch differential-equation solvers.
Hyperexponential RDEs include general weak normalization, normal and special
denominator reduction, SPDE, and the Laurent recurrence core. Real coupled
RDEs include Gaussian scalar certificates over primitive towers and direct
Chapter 8 cancellation recurrences for hyperexponential Laurent modes and
hypertangent polynomial systems. Together with the immutable Gaussian Cayley
normal form, those solvers close the declared explicit P/E/T tower contract.
The expression frontend reuses its checked extension factories and returns
only local, branch-preserving presentations. Its separate candidate-DAG pass
now recognizes rational-power algebraic levels and both mixed
algebraic/transcendental orders; the native compiler executes terminal
algebraic-over-P/E/T towers and reports still-missing opposite-order backends
separately. Complete general-polynomial special-factor reduction, algebraic
composita, and global branch-aware expression normalization remain future
work.

`python-flint` is the required exact-arithmetic backend. The project does not
support SymPy's slower pure-Python ground types.

## Install and test

From a checkout:

```console
uv sync --no-editable
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest -q -n auto --dist worksteal
```

The non-editable install avoids an editable-install issue on Python versions
that ignore hidden `.pth` files. If the environment already contains an
editable installation, add `--reinstall-package sympy-risch` to the sync
command when switching it to a non-editable install. Pytest still reads the
current source tree through the configured `src` path. Ruff checks the complete
checkout, while ty checks the production package under `src`.

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
integrand = field.from_sympy(-1 / (x - 1) ** 2 + 3 / x)

result = integrate_elementary(integrand, tower.derivation())

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == 1 / (x - 1) + 3 * log(x)
```

See the [getting-started guide](docs/getting-started.md) for nested logarithms,
non-elementarity obstructions, the narrow automatic frontend, and the
supported tower shapes.

## Documentation

- [Getting started](docs/getting-started.md): construct towers, integrate, and
  interpret results.
- [Understanding solver decisions](docs/understanding-decisions.md): tell a
  constructive answer from a proved obstruction or an unsupported input.
- [Architecture and roadmap](docs/design.md): mathematical design, implemented
  algorithms, and remaining work.
- [Mixed transcendental towers](docs/mixed-transcendental-towers.md): the
  explicit P/E/T tower contract.
- [Testing contract](docs/testing.md): correctness criteria and test commands.
- [Tutoring notes](docs/tutoring-notes.md): a guided introduction to the
  mathematics.
- [Algebraic-over-hypertangent research note](docs/algebraic-hypertangent.md):
  a proposed future real-algebraic branch.
- [Algebraic function-field infrastructure](docs/algebraic-infrastructure.md):
  exact orders, ideals, places, local expansions, divisors, and differentials.
- [Literature archive](references/README.md): locally available primary papers,
  their provenance, and cited works that still need an authorized copy.
