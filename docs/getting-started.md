# Getting started

`sympy-risch` integrates elements of an explicitly supplied differential
tower. The library works with native SymPy domain elements: the tower records
what each formal generator means to the derivation, while ordinary field
arithmetic remains SymPy arithmetic.

This is a research API. It does not yet inspect an arbitrary SymPy expression,
discover all of its generators, or choose a presentation automatically.

## Construct the rational base

The standard base is `QQ(x)` with `Dx = 1`:

```python
from sympy import Poly, exp, log, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    IntegralBasis,
    IntegrationObstructionKind,
    MonomialExtension,
    ParametricLogarithmicDerivativeOutcome,
    integrate_elementary,
    primitive_logarithmic_derivative_signature,
    solve_parametric_logarithmic_derivative,
    solve_parametric_logarithmic_derivative_heuristic,
    trager_algebraic_residue_polynomial,
)

x, t, u, y, z = symbols("x t u y z")

base = DifferentialTower(QQ).extend(
    MonomialExtension(
        Poly(1, x, domain=QQ),
        ConstantFieldStatus.PRESERVED,
    )
)
```

`DifferentialTower` and its extensions are immutable. Calling `extend` returns
a new tower and leaves `base` unchanged.

Inputs must be elements of the current exact field:

```python
field = base.field()
integrand = field.from_sympy(-1/(x - 1)**2 + 3/x)
result = integrate_elementary(integrand, base.derivation())

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == 1/(x - 1) + 3*log(x)
```

The rational reducer has separated the exact rational derivative from the
simple-pole logarithmic part.

## Add primitive levels

A primitive level `K(t)` has `Dt = eta` for some `eta` in the lower field.
For `t = log(x)`, use `Dt = 1/x`:

```python
log_x = base.extend(
    MonomialExtension(
        Poly(1/x, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
```

The generator is formal. Its mathematical meaning comes from its derivative
law and the construction assumption. Add `u = log(t)` in the same way:

```python
log_log_x = log_x.extend(
    MonomialExtension(
        Poly(1/(x*t), u, domain=log_x.field()),
        ConstantFieldStatus.PRESERVED,
    )
)

field = log_log_x.field()
result = integrate_elementary(
    field.from_sympy(1/(x*t*u)),
    log_log_x.derivation(),
)

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == log(u)
```

After the presentation substitutions `t = log(x)` and
`u = log(log(x))`, this is the familiar identity

```text
integral dx/(x log(x) log(log(x))) = log(log(log(x))).
```

The recursive primitive reducer is tested with deeper stacks as well; it is
not limited to this two-log example.

## Read the result

`integrate_elementary` returns an immutable `ElementaryIntegralResult`.
The most useful properties are:

| Property | Meaning |
| --- | --- |
| `is_elementary` | No terminal Liouville obstruction was found. |
| `reconstructs` | The exact tower identity reconstructs the input. |
| `antiderivative` | Encoded elementary answer, or `None` on a proved obstruction. |
| `total_exact_part` | Exact-derivative contributions from every level. |
| `all_logarithmic_terms` | Constructive logarithmic terms from every level. |
| `unintegrated_remainder` | Zero on success; otherwise the terminal remainder. |
| `obstruction` | Structured reason and level for a negative result. |
| `certificate` | The family-specific reduction object for the current level. |

`reconstructs` uses the tower derivation and is the primary exact check.
Ordinary SymPy differentiation treats `t` and `u` as independent symbols, so
differentiate `antiderivative` with `diff` only after substituting the
functions represented by those generators.

## Inspect a non-elementarity obstruction

In the same tower, `1/u` represents `1/log(log(x))`. The implemented residue
criterion proves that this example has no elementary antiderivative under the
recorded no-new-constants assumptions:

```python
result = integrate_elementary(
    field.from_sympy(1/u),
    log_log_x.derivation(),
)

assert not result.is_elementary
assert result.antiderivative is None
assert result.reconstructs
assert result.unintegrated_remainder == field.from_sympy(1/u)
assert result.obstruction is not None
assert result.obstruction.level == 2
assert result.obstruction.kinds == (
    IntegrationObstructionKind.NONCONSTANT_RESIDUE,
)
```

This is different from an unsupported input. A structured obstruction records
a completed negative decision for the supported case. Unsupported tower
shapes raise `NotImplementedError` instead.

## Hyperexponential and hypertangent levels

The same `MonomialExtension` class classifies a level from the shape of `Dt`.
For example, `Dt = t` is hyperexponential and represents `t = exp(x)` up to a
nonzero constant:

```python
exp_x = base.extend(
    MonomialExtension(
        Poly(t, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
result = integrate_elementary(
    exp_x.field().from_sympy((x + 1)*t),
    exp_x.derivation(),
)

assert result.antiderivative == x*t
assert result.reconstructs
```

Likewise, `Dt = 1 + t**2` is hypertangent and represents `t = tan(x)` up to a
constant phase:

```python
tan_x = base.extend(
    MonomialExtension(
        Poly(1 + t**2, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
result = integrate_elementary(
    tan_x.field().from_sympy(2*t),
    tan_x.derivation(),
)

assert result.antiderivative == log(t**2 + 1)
assert result.reconstructs
```

The direct hyperexponential and hypertangent dispatchers currently require
the standard rational base in the general case. One mixed hyperexponential
branch is also implemented: the lower field may be primitive when the new
logarithmic derivative is polynomial of positive degree in the top primitive
generator.

For example, start from the earlier `log_x` tower and adjoin
`u = exp(t**2)`. Since `Dt = 1/x`, its derivative law is
`Du = (2*t/x)*u`:

```python
exp_log_squared = log_x.extend(
    MonomialExtension(
        Poly(2*t*u/x, u, domain=log_x.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
field = exp_log_squared.field()
result = integrate_elementary(
    field.from_sympy((1 + 2*t)*u),
    exp_log_squared.derivation(),
)

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == x*u
assert result.antiderivative.subs(
    {t: log(x), u: exp(log(x)**2)},
    simultaneous=True,
) == x*exp(log(x)**2)
```

The lower-field RDE solved here is

```text
Dy + (2t/x)y = 1 + 2t,
```

with solution `y = x`. The implementation also handles rational
right-hand sides in the primitive generator through normal-denominator and
SPDE reduction. Coefficients with poles in the primitive generator, and the
cancellation case where the coefficient lies entirely in the next lower
field, remain explicit `NotImplementedError` boundaries.

## Test a parametric logarithmic derivative

Bronstein's parametric logarithmic-derivative problem asks whether, for a
proposed hyperexponential generator `theta`,

```text
n*f = D(v)/v + m*D(theta)/theta
```

has a solution with integers `n != 0`, `m`, and a nonzero rational function
`v`. The complete rational-base entry point uses Bronstein's Section 7.3
algorithm as its fast path:

```python
theta = symbols("theta")
eta = x**2 + 1/x
theta_extension = MonomialExtension(
    Poly(eta*theta, theta, domain=base.field())
)

decision = solve_parametric_logarithmic_derivative(
    (1/(x - 1) + 3*eta)/2,
    theta_extension,
    base.derivation(),
)

assert decision.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION
assert (decision.n, decision.m) == (2, 3)
assert decision.argument == base.field().from_sympy(x - 1)
assert decision.reconstructs
```

If the fast path determines `m/n`, its certificate records whether that ratio
came from the high-degree polynomial comparison or the repeated-denominator
congruence. Every successful witness is rechecked by exact differentiation.

When the fast path is inconclusive, the public solver compares the remaining
constant part and every non-logarithmic residue coordinate. For Bronstein's
Example 7.3.2, `D(theta)/theta = 1` and `f = 11`:

```python
constant_theta = MonomialExtension(
    Poly(theta, theta, domain=base.field())
)
fast_result = solve_parametric_logarithmic_derivative_heuristic(
    11,
    constant_theta,
    base.derivation(),
)
decision = solve_parametric_logarithmic_derivative(
    11,
    constant_theta,
    base.derivation(),
)

assert fast_result.outcome is ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE
assert decision.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION
assert (decision.n, decision.m, decision.argument) == (
    1,
    11,
    base.field().one,
)
```

Thus the complete `QQ(x)` entry point has no inconclusive result.

## Inspect a primitive logarithmic signature

The primitive structure-theorem building block works recursively over stacks
of primitive levels. At each level `K(t)`, it decomposes an element into its
polynomial part and every principal part at each irreducible polynomial in
`t`. It then descends into `K` to test whether some positive integer multiple
is a logarithmic derivative:

```python
signature = primitive_logarithmic_derivative_signature(
    log_x.field().from_sympy(1/(2*x) + 1/(3*x*t)),
    log_x.derivation(),
)

assert signature.has_solution
assert signature.minimal_power == 6
assert signature.relation_argument == log_x.field().from_sympy(x**3*t**2)
assert signature.reconstructs_decomposition
assert signature.reconstructs_relation
```

Here multiplication by six clears the lower residue `1/2` at `x` and the
top-level residue `1/3` at `t`. On failure, the immutable signature
distinguishes a positive-degree polynomial part, a repeated pole, a simple
residue not proportional to `Dp`, and an obstruction inherited from
`QQ(x)`.

`solve_radical_logarithmic_derivative(f, derivation)` is the uniform entry
point. It returns the minimal positive integer `n` and an argument `v`
satisfying `n*f = D(v)/v`, whether `derivation` is on `QQ(x)` or at the top of
a recursively primitive tower. Its immutable certificate retains the rational
result or complete primitive signature from the current level.

The public parametric solver consumes these coordinates recursively when its
lower field is a primitive tower over `QQ(x)`. For example, the repeated pole
in `eta` is not itself logarithmic, but it cancels at the uniquely determined
ratio:

```python
primitive_eta = 1/t + 1/t**2
primitive_theta = MonomialExtension(
    Poly(primitive_eta*theta, theta, domain=log_x.field())
)
decision = solve_parametric_logarithmic_derivative(
    2*primitive_eta + 1/(x*t),
    primitive_theta,
    log_x.derivation(),
)

assert decision.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION
assert (decision.n, decision.m) == (1, 2)
assert decision.argument == log_x.field().from_sympy(t)
assert decision.reconstructs
```

This decision is complete for one primitive level carrying a
`PRESERVED` constant-field certificate. Deeper primitive lower towers remain
unsupported.

## Compute finite residues of an algebraic differential

Algebraic Hermite reduction leaves a differential with a square-free common
denominator in an integral basis. For a monic polynomial model whose finite
poles are unbranched, Trager's nested resultant computes all finite residues
without first splitting the algebraic curve into individual places:

```python
sqrt_x_extension = AlgebraicExtension(
    Poly(y**2 - x, y, domain=base.field())
)
sqrt_x = base.extend(sqrt_x_extension)
basis = IntegralBasis.power_basis(sqrt_x_extension)

residues = trager_algebraic_residue_polynomial(
    y/(x - 1),
    sqrt_x.derivation(),
    basis,
    z,
)

assert residues.residue_polynomial == Poly(z**2 - 1, z, domain=QQ)
assert residues.reconstructs_input
```

At `x = 1`, the two places have `y = 1` and `y = -1`, so the residues are
`1` and `-1`. The result retains the defining polynomial, rationalized
numerator and denominator, discriminant, branch-locus gcd, inner resultant,
raw outer resultant, and final monic residue polynomial as an immutable
certificate.

This is currently a finite-residue component, not a complete algebraic
integrator. It requires `Dx = 1`, a polynomial model monic in `y`, a
square-free integral-basis denominator, and denominator coprime to the model
discriminant. It explicitly rejects branch poles and does not yet construct
the algebraic constant extension, residue divisors, logarithmic terms, or the
contribution at infinity.

## Mathematical assumptions

Risch structure theorems require proof that a transcendental adjunction does
not enlarge the constant field. Until the automatic tower builder can prove
this, manual examples record the hypothesis with
`ConstantFieldStatus.PRESERVED`.

This flag is not an inferred fact. Use it only when the construction justifies
the no-new-constants assumption. Algorithms that need the hypothesis reject
`UNKNOWN` and `ENLARGED` rather than issuing an unsound certificate.

General nonlinear monomial reduction has a similar explicit
`SpecialPolynomialStatus.ABSENT` precondition. It means the caller has
established that the level has no nonconstant irreducible polynomial `p` with
`p` dividing `Dp`.

## Current coverage

| Tower shape or operation | Status |
| --- | --- |
| Polynomial and rational integration over `QQ(x)` | End-to-end |
| Recursively stacked primitive levels | End-to-end for the implemented reduction |
| One hyperexponential level over `QQ(x)` | End-to-end |
| Hyperexponential over a primitive level, with positive-degree polynomial logarithmic derivative | End-to-end |
| One hypertangent level over `QQ(x)` | End-to-end |
| Nonlinear monomial with certified absence of special polynomials | End-to-end over supported lower towers |
| Parametric logarithmic derivative | Complete decision over `QQ(x)` and one primitive level |
| Primitive radical logarithmic signature | Low-level API over one primitive level above `QQ(x)` |
| Algebraic Hermite reduction over `QQ(x)` | Low-level API |
| Algebraic finite-residue polynomial | Low-level API for monic polynomial models with square-free unbranched finite poles |
| Algebraic levels in `integrate_elementary` | Not implemented |
| Other nested hyperexponential shapes or nested hypertangent levels | Not implemented |
| Automatic tower discovery and back-substitution | Not implemented |
| Complete global real branch management | Not implemented |

## Low-level entry points

The package exports the component algorithms used by the dispatcher. The
principal groups are:

| Task | Entry points |
| --- | --- |
| Polynomial integration | `integrate_polynomial` |
| Rational reduction and integration | `hermite_reduce_rational`, `integrate_rational`, `integrate_rational_real` |
| Rational logarithmic analysis | `lazard_rioboo_trager`, `rothstein_trager_residue_polynomial`, `solve_rational_logarithmic_derivative` |
| Tower construction and differentiation | `DifferentialTower`, `MonomialExtension`, `AlgebraicExtension`, `Derivation` |
| Monomial reduction | `hermite_reduce_monomial`, `residue_reduce_monomial`, and the family-specific `reduce_*_monomial` functions |
| Exact derivatives and RDEs | `solve_exact_derivative`, `solve_risch_differential_equation`, and the field-specific RDE solvers |
| Primitive logarithmic analysis | `primitive_logarithmic_derivative_signature` |
| Parametric logarithmic derivative | `solve_parametric_logarithmic_derivative`; `solve_parametric_logarithmic_derivative_heuristic` exposes the Section 7.3 fast path |
| Algebraic Hermite reduction | `hermite_reduce_algebraic`, `lazy_hermite_reduce_algebraic`, `simple_radical_integral_basis` |
| Algebraic finite residues | `trager_algebraic_residue_polynomial` |

These functions return structured immutable results rather than only a printed
antiderivative. Their dataclass fields are the executable reduction
certificates used by the test suite. Consult the function docstrings and the
corresponding focused test file for preconditions and exact postconditions.

See the [architecture and roadmap](design.md) for the algorithms behind these
boundaries and the planned path toward the full algebraic cases.
