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
    AlgebraicFunctionField,
    ConstantFieldStatus,
    DifferentialTower,
    IntegralBasis,
    IntegrationObstructionKind,
    MonomialExtension,
    ParametricLogarithmicDerivativeOutcome,
    TowerAdmission,
    TowerAdmissionOutcome,
    TowerAdmissionReason,
    assess_elementary_admission,
    certify_primitive_extension,
    integrate_elementary,
    primitive_logarithmic_derivative_signature,
    solve_parametric_logarithmic_derivative,
    solve_parametric_logarithmic_derivative_heuristic,
    solve_exact_derivative,
    solve_hyperexponential_coupled_laurent_rde,
    solve_hypertangent_coupled_cancellation_rde,
    solve_radical_logarithmic_derivative,
    solve_risch_differential_equation,
    trager_algebraic_residue_polynomial,
)

x, t, u, y, z = symbols("x t u y z")

constants = DifferentialTower(QQ)
base_analysis = certify_primitive_extension(
    Poly(1, x, domain=QQ),
    constants.derivation(),
)
base = constants.extend(base_analysis.extension)
```

`DifferentialTower` and its extensions are immutable. Calling `extend` returns
a new tower and leaves `constants` unchanged. Here the factory proves that
`1` is not a derivative in `QQ`, so adjoining `x` preserves the constants.

Inputs must be elements of the current exact field:

```python
field = base.field()
integrand = field.from_sympy(-1 / (x - 1) ** 2 + 3 / x)
result = integrate_elementary(integrand, base.derivation())

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == 1 / (x - 1) + 3 * log(x)
```

The rational reducer has separated the exact rational derivative from the
simple-pole logarithmic part.

## Check the admission boundary

Before relying on the elementary dispatcher, ask whether the explicit tower
belongs to its certified P/E/T contract:

```python
admission = assess_elementary_admission(base.derivation())

assert isinstance(admission, TowerAdmission)
assert admission.outcome is TowerAdmissionOutcome.ACCEPTED
```

`TowerAdmissionOutcome.ACCEPTED` says that the tower is supported, not that a
particular integrand has an elementary antiderivative. If the outcome is
`TowerAdmissionOutcome.UNSUPPORTED`, the failed assumption is represented by
`TowerAdmissionReason`; that result also says nothing about integrability.
The admitted family is primitive, hyperexponential, or hypertangent levels
with preserved constants, together with an exact hyperbolic Möbius
normalization at the active top level described below. General nonlinear
monomials, including a dispatcher experiment with
`SpecialPolynomialStatus.ABSENT`, are deliberately outside this total public
contract.

See the [mixed-tower admission contract](mixed-transcendental-towers.md) for
the precise boundary and its relation to integration results.

## Add primitive levels

A primitive level `K(t)` has `Dt = eta` for some `eta` in the lower field.
For `t = log(x)`, use `Dt = 1/x`:

```python
log_x_analysis = certify_primitive_extension(
    Poly(1 / x, t, domain=base.field()),
    base.derivation(),
)
log_x = base.extend(log_x_analysis.extension)
```

The generator is formal. Its mathematical meaning comes from its derivative
law. The factory proves that `1/x` has no rational antiderivative, so this is
a valid primitive extension. Add `u = log(t)` in the same way:

```python
log_log_x_analysis = certify_primitive_extension(
    Poly(1 / (x * t), u, domain=log_x.field()),
    log_x.derivation(),
)
log_log_x = log_x.extend(log_log_x_analysis.extension)

field = log_log_x.field()
result = integrate_elementary(
    field.from_sympy(1 / (x * t * u)),
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
    field.from_sympy(1 / u),
    log_log_x.derivation(),
)

assert not result.is_elementary
assert result.antiderivative is None
assert result.reconstructs
assert result.unintegrated_remainder == field.from_sympy(1 / u)
assert result.obstruction is not None
assert result.obstruction.level == 2
assert result.obstruction.kinds == (IntegrationObstructionKind.NONCONSTANT_RESIDUE,)
```

This is different from an unsupported input. A structured obstruction records
a completed negative decision for the supported case. Query
`assess_elementary_admission` first when the caller needs that distinction:
an unsupported `TowerAdmission` is a support-boundary result, not a negative
mathematical conclusion. Direct calls outside the advertised boundary can
still raise `NotImplementedError` from a lower-level implementation gap.

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
    exp_x.field().from_sympy((x + 1) * t),
    exp_x.derivation(),
)

assert result.antiderivative == x * t
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
    tan_x.field().from_sympy(2 * t),
    tan_x.derivation(),
)

assert result.antiderivative == log(t**2 + 1)
assert result.reconstructs
```

The hyperbolic law `Dt = 1 - t**2` represents `t = tanh(x)` (or `coth(x)` on
another presentation). It is neither primitive nor hypertangent, but its
Möbius coordinate

```text
u = (1 + t)/(1 - t)
```

has `Du = 2*u`. The dispatcher performs that exact, immutable normalization
internally and maps its certificate back to the original `t` coordinate:

```python
tanh_x = base.extend(
    MonomialExtension(
        Poly(1 - t**2, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
result = integrate_elementary(tanh_x.field().one, tanh_x.derivation())

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == x
```

This covers precisely that Möbius-normalizable law. It is not yet a general
decision procedure for arbitrary polynomial or Riccati monomials.

Exactness is stricter than elementary integrability. The same dispatcher
recovers derivatives in both families while retaining logarithmic residues or
unsolved special modes as exactness obstructions:

```python
planted = exp_x.field().from_sympy(x * t)
exact = solve_exact_derivative(
    exp_x.derivation()(planted),
    exp_x.derivation(),
)
logarithmic = solve_exact_derivative(
    tan_x.field().from_sympy(2 * t),
    tan_x.derivation(),
)

assert exact.has_solution
assert exp_x.derivation()(exact.exact_part) == exp_x.derivation()(planted)
assert not logarithmic.has_solution
assert logarithmic.obstruction == tan_x.field().from_sympy(2 * t)
```

Hypertangent and hyperexponential levels may also sit over recursively
primitive lower towers. Recursive dependency analysis rejects rational and
algebraic presentations before reduction.

For example, use the earlier `t = log(x)` level and adjoin the formal
hypertangent `u` with

```text
D(u) = (1 + u**2)/(x*t).
```

This represents `u = tan(log(log(x)))` up to a constant phase:

```python
tan_log_log = log_x.extend(
    MonomialExtension(
        Poly(
            (1 + u**2) / (x * t),
            u,
            domain=log_x.field(),
        ),
        ConstantFieldStatus.PRESERVED,
    )
)
field = tan_log_log.field()
D = tan_log_log.derivation()
planted = field.from_sympy(x * u)
result = integrate_elementary(D(planted), D)

assert result.is_elementary
assert result.reconstructs
assert field.from_sympy(result.antiderivative) == planted
```

Internally the tangent special place is converted into conjugate Cayley
modes. Each mode produces a real coupled RDE over `QQ(x,t)`. The solver
temporarily rebases the immutable primitive tower over `QQ_I`, solves one
ordinary scalar RDE, and converts the exact real and imaginary coordinates
back. Its direct block-matrix test oracle does not use that conversion.

For example, start from the earlier `log_x` tower and adjoin
`u = exp(t**2)`. Since `Dt = 1/x`, its derivative law is
`Du = (2*t/x)*u`:

```python
exp_log_squared = log_x.extend(
    MonomialExtension(
        Poly(2 * t * u / x, u, domain=log_x.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
field = exp_log_squared.field()
result = integrate_elementary(
    field.from_sympy((1 + 2 * t) * u),
    exp_log_squared.derivation(),
)

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == x * u
assert result.antiderivative.subs(
    {t: log(x), u: exp(log(x) ** 2)},
    simultaneous=True,
) == x * exp(log(x) ** 2)
```

The lower-field RDE solved here is

```text
Dy + (2t/x)y = 1 + 2t,
```

with solution `y = x`. The implementation also handles rational
right-hand sides in the primitive generator through normal-denominator and
SPDE reduction. When the coefficient lies entirely in the next lower field,
the solver now applies Bronstein's primitive cancellation recurrence. It
either:

1. recognizes `a = D(z)/z` and solves the equivalent exact-derivative problem
   `D(z*y) = z*b`; or
2. removes each leading power of the primitive generator by recursively
   solving `D(s) + a*s = lc(b)` in the lower field.

This primitive recurrence is the specialized lower-field route. In the wider
supported P/E/T contract, the RDE dispatcher uses the corresponding
hyperexponential or hypertangent reductions and coupled-RDE fallback when a
lower level is not primitive. Coefficients with poles in the current primitive
generator are weakly normalized before normal-denominator reduction.
Positive-integer logarithmic residues are removed by an exact change of
unknown; other residues remain in the normalized equation and are handled by
the denominator, degree-bound, and SPDE stages.

At a hyperexponential level `K(t)` with `Dt = eta*t`, the RDE dispatcher
completely handles Laurent coefficients and right-hand sides. A lower-field
coefficient preserves each power, which is solved independently:

```text
D(y_j) + (a + j*eta)*y_j = b_j.
```

For example:

```python
exp_x = base.extend(
    MonomialExtension(
        Poly(t, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
field = exp_x.field()
D = exp_x.derivation()
result = solve_risch_differential_equation(
    field.from_sympy(t + 1),
    field.one,
    D,
)
assert result.has_solution
assert result.solution == field.from_sympy(1 / t)
assert result.reconstructs
assert result.certificate.minimum_power_bound == -1
assert result.certificate.exceptional_power == -1
```

Here the boundary relation `-1 = -eta` permits the exceptional power `-1`.
For a coefficient with a positive highest power, the solver applies forced
corrections downward; for one with a negative lowest power, it works upward.
The certificate retains the bounds, direction, every correction, and the
before/after residuals. In the diagonal lower-field case it instead contains
one recursive lower-field RDE result per Laurent power.

A failed recurrence leaves its solved corrections in `partial_solution` and
the exact obstruction in `residual`.

General rational coefficients and right-hand sides use the full
hyperexponential frontend. For example, the normal factor `t + 1` and the
special factor `t` may occur together:

```python
coefficient = field.from_sympy(t + 1 / t)
planted = field.from_sympy((x + 1) / (t**2 * (t + 1) ** 2))
right_hand_side = D(planted) + coefficient * planted

result = solve_risch_differential_equation(
    coefficient,
    right_hand_side,
    D,
)
assert result.solution == planted
assert result.certificate.normal_denominator_bound == Poly(
    (t + 1) ** 2,
    t,
    domain=base.field(),
)
assert result.certificate.special_denominator_power == 2
```

The certificate records the weak normalizer, normal denominator bound,
special power bound, polynomial degree bound, terminal Laurent result, and
any failed SPDE division. The P/E/T dispatcher composes this path with the
appropriate lower-level solver; see [mixed transcendental towers]
(mixed-transcendental-towers.md) for the exact supported scope.

For example, let `t = log(x)` and adjoin a formal hyperexponential `u` with

```text
D(u)/u = 1/(t - 1).
```

The coefficient has a pole in `t`, but the complete primitive RDE now proves
the following elementary integral:

```python
pole_exp_over_log = log_x.extend(
    MonomialExtension(
        Poly(u / (t - 1), u, domain=log_x.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
field = pole_exp_over_log.field()
result = integrate_elementary(
    field.from_sympy((1 + x / (t - 1)) * u),
    pole_exp_over_log.derivation(),
)
assert result.is_elementary
assert result.antiderivative == x * u
assert result.reconstructs
```

For example, adjoining `u = exp(x)` above `QQ(x)(t)`, where `t = log(x)`,
exercises the lower-field cancellation branch:

```python
exp_x_over_log = log_x.extend(
    MonomialExtension(
        Poly(u, u, domain=log_x.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
field = exp_x_over_log.field()
result = integrate_elementary(
    field.from_sympy((t + 1 / x) * u),
    exp_x_over_log.derivation(),
)

assert result.is_elementary
assert result.reconstructs
assert result.antiderivative == t * u
```

## Solve direct coupled cancellation systems

Chapter 8 coupled systems use the real matrix operator

```text
D(p) + a*p - lambda*q = r,
D(q) + lambda*p + a*q = s.
```

At a hyperexponential level `K(u)` with `Du = eta*u`, lower-field `a` and
`lambda` preserve every Laurent power. The direct solver handles each power
through a lower coupled system with shifted diagonal `a + m*eta`:

```python
exp_x = base.extend(
    MonomialExtension(
        Poly(u, u, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
field = exp_x.field()
D = exp_x.derivation()
p = field.from_sympy((x + 1) * u**2 + (x - 1) / u)
q = field.from_sympy(x * u**2 + 2 / u)
a = field.from_sympy(x)
rotation = field.from_sympy(x + 1)

coupled = solve_hyperexponential_coupled_laurent_rde(
    a,
    rotation,
    D(p) + a * p - rotation * q,
    D(q) + rotation * p + a * q,
    D,
)
assert coupled.real_solution == p
assert coupled.imaginary_solution == q
assert coupled.reconstructs
```

For a hypertangent generator `u` with `Du = 1 + u**2`, the exceptional
polynomial cancellation has diagonal `b0 - n*u`. Bronstein's worked
Example 8.4.1 is:

```python
tan_x = base.extend(
    MonomialExtension(
        Poly(1 + u**2, u, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
coupled = solve_hypertangent_coupled_cancellation_rde(
    0,
    4 * x,
    -(u**2) + 2 * u - 8 * x**2 + 1,
    2 * (1 - 2 * x),
    tan_x.derivation(),
    2,
)

assert coupled.real_solution == Poly(
    u - 1,
    u,
    domain=base.field(),
)
assert coupled.imaginary_solution == Poly(
    2 * x,
    u,
    domain=base.field(),
)
assert coupled.reconstructs
```

This is the terminal recurrence used by the complete rational coupled-RDE
frontend. That frontend performs paired weak normalization, denominator
reduction, degree bounds, and SPDE before reaching this stage. It also uses
the immutable all-level Cayley normalization needed for nested hypertangent
levels.

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
eta = x**2 + 1 / x
theta_extension = MonomialExtension(Poly(eta * theta, theta, domain=base.field()))

decision = solve_parametric_logarithmic_derivative(
    (1 / (x - 1) + 3 * eta) / 2,
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
constant_theta = MonomialExtension(Poly(theta, theta, domain=base.field()))
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
    log_x.field().from_sympy(1 / (2 * x) + 1 / (3 * x * t)),
    log_x.derivation(),
)

assert signature.has_solution
assert signature.minimal_power == 6
assert signature.relation_argument == log_x.field().from_sympy(x**3 * t**2)
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
satisfying `n*f = D(v)/v` throughout the documented P/E/T tower contract.
Its immutable certificate retains the rational result, a complete primitive
signature, or the mixed-tower relation data appropriate to the current level.

For an additional primitive level with `Du = 1/(x*t)`, the same entry point
combines residue denominators from every level:

```python
log_log_x = log_x.extend(
    MonomialExtension(
        Poly(1 / (x * t), u, domain=log_x.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
radical = solve_radical_logarithmic_derivative(
    1 / (2 * x) + 1 / (3 * x * t) + 1 / (5 * x * t * u),
    log_log_x.derivation(),
)

assert radical.minimal_power == 30
assert radical.argument == log_log_x.field().from_sympy(x**15 * t**10 * u**6)
assert radical.reconstructs
```

The public parametric solver consumes these coordinates recursively when its
lower field is a primitive tower over `QQ(x)`. For example, the repeated pole
in `eta` is not itself logarithmic, but it cancels at the uniquely determined
ratio:

```python
primitive_eta = 1 / t + 1 / t**2
primitive_theta = MonomialExtension(
    Poly(primitive_eta * theta, theta, domain=log_x.field())
)
decision = solve_parametric_logarithmic_derivative(
    2 * primitive_eta + 1 / (x * t),
    primitive_theta,
    log_x.derivation(),
)

assert decision.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION
assert (decision.n, decision.m) == (1, 2)
assert decision.argument == log_x.field().from_sympy(t)
assert decision.reconstructs
```

The primitive signature route is complete over recursively primitive towers
when every primitive level carries a `PRESERVED` constant-field certificate.
For mixed P/E/T lower towers the public parametric solver automatically uses
the general relation-system route; algebraic and other unsupported lower
families remain explicit `UNSUPPORTED` outcomes.

## Compute finite residues of an algebraic differential

Algebraic Hermite reduction leaves a differential with a square-free common
denominator in an integral basis. For a monic polynomial model whose finite
poles are unbranched, Trager's nested resultant computes all finite residues
without first splitting the algebraic curve into individual places:

```python
sqrt_x_extension = AlgebraicExtension(Poly(y**2 - x, y, domain=base.field()))
sqrt_x = base.extend(sqrt_x_extension)
basis = IntegralBasis.power_basis(sqrt_x_extension)

residues = trager_algebraic_residue_polynomial(
    y / (x - 1),
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

## Build the algebraic function-field model

The general curve infrastructure normalizes the field before constructing
places. It therefore supports ramified fibers and infinity independently of
the plane-model restrictions in the Trager shortcut:

```python
curve = AlgebraicFunctionField(sqrt_x_extension)
branch_place = curve.finite_places(Poly(x, x, domain=QQ))[0]
infinity = curve.infinite_places()[0]

assert branch_place.ramification_index == 2
assert branch_place.valuation(sqrt_x_extension.field.generator) == 1
assert infinity.valuation(sqrt_x_extension.field.from_sympy(x)) == -2
assert curve.principal_divisor(sqrt_x_extension.field.from_sympy(x)).degree == 0
assert curve.genus == 0
```

The model also exposes the certified maximal order, fractional ideals, local
reduction and Laurent expansion, differents, differential residues, and
finite ideal/divisor conversion. See the
[algebraic infrastructure contract](algebraic-infrastructure.md). These are
low-level prerequisites for future algebraic Risch algorithms; they do not
perform logarithmic reconstruction, principal-divisor solving, or an
algebraic RDE.

## Mathematical assumptions

Risch structure theorems require proof that a transcendental adjunction does
not enlarge the constant field. For primitive levels, use
`certify_primitive_extension`: it proves `PRESERVED` when `Dt` is not exact in
the lower field, returns `ENLARGED` with the witness `v` when `Dt = D(v)`, and
leaves unsupported or uncertified lower towers `UNKNOWN`.

Other monomial families still use explicitly supplied
`ConstantFieldStatus.PRESERVED` metadata after their family-specific
dependency analysis. Algorithms that need the hypothesis reject `UNKNOWN`
and `ENLARGED` rather than issuing an unsound certificate.

General nonlinear monomial reduction has a similar
`SpecialPolynomialStatus.ABSENT` precondition. It means the level has no
nonconstant irreducible polynomial `p` with `p` dividing `Dp`. For a primitive
level with a certified preserved constant field, the special-polynomial
analyzer proves that fact automatically; other monomial families still need
their family theorem or explicit evidence.

The current SymPy domain backend cannot yet represent a monomial extension
above a function-field `AlgebraicExtension`, or a nested
`AlgebraicExtension`. The library therefore rejects those tower compositions
immediately with `NotImplementedError`, rather than constructing a tower that
would fail later during coefficient coercion. This is a backend representation
boundary, not a mathematical claim about the integral.

## Current coverage

| Tower shape or operation | Status |
| --- | --- |
| Polynomial and rational integration over `QQ(x)` | End-to-end |
| Explicit certified primitive/hyperexponential/hypertangent towers over `QQ(x)`, in any finite order | End-to-end |
| Admission of the certified P/E/T contract and top-level hyperbolic Möbius normalization | `assess_elementary_admission` reports support separately from elementary integrability |
| Scalar RDE over hyperexponential levels, general rational coefficient and right-hand side | Complete decision in the explicit certified P/E/T contract |
| Coupled RDE in the explicit certified P/E/T contract | Complete decision |
| Coupled RDE over a hyperexponential level, lower-field matrix and Laurent right-hand sides | Complete terminal decision |
| Coupled polynomial RDE over a hypertangent level, lower-field or `b0 - n*eta*t` diagonal | Complete terminal decision |
| General rational coupled RDE over a hypertangent level | Complete decision in the explicit certified P/E/T contract |
| Exact derivatives in supported rational, primitive, hyperexponential, and hypertangent levels | Complete decision |
| Nonlinear monomial with certified absence of special polynomials | End-to-end over supported lower towers |
| Radical and parametric logarithmic derivatives in the supported P/E/T tower contract | Complete decision with explicit unsupported outcomes outside that contract |
| Algebraic Hermite reduction over `QQ(x)` | Low-level API |
| Algebraic finite-residue polynomial | Low-level API for monic polynomial models with square-free unbranched finite poles |
| Algebraic orders, ideals, all places, valuations, local expansions, and divisors | Complete low-level infrastructure over supported `C(x)` constants |
| Algebraic constant-field analysis and all-place residues | Low-level API: exact Hermite success, plus complete pole data for admitted preserved direct `C(x)` simple-radical presentations; other nonzero cases return an explicit boundary with no partial list |
| Algebraic levels in `integrate_elementary` | Not implemented |
| Mixed algebraic/monomial prefix representation | `analyze_tower_representation` proves native representability for a monomial prefix plus an optional terminal algebraic level and reports monomial-over-algebraic as a structured backend limitation; nested algebraic extension construction remains rejected by the backend |
| Hypertangent over hyperexponential or hypertangent lower levels | End-to-end in the explicit certified P/E/T contract |
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
| Elementary-dispatch admission | `assess_elementary_admission`, `TowerAdmission`, `TowerAdmissionOutcome`, `TowerAdmissionReason` |
| Monomial reduction | `hermite_reduce_monomial`, `residue_reduce_monomial`, and the family-specific `reduce_*_monomial` functions |
| Exact derivatives and RDEs | `solve_exact_derivative`, `solve_risch_differential_equation`, and the field-specific RDE solvers |
| Primitive logarithmic analysis | `solve_radical_logarithmic_derivative`, `primitive_logarithmic_derivative_signature` |
| Parametric logarithmic derivative | `solve_parametric_logarithmic_derivative`; `solve_parametric_logarithmic_derivative_heuristic` exposes the Section 7.3 fast path |
| Hyperbolic normal form | `build_hyperbolic_mobius_normal_form` for the exact law `Dt = eta*(1 - t**2)` |
| Special-polynomial evidence | `certify_special_polynomial`, `analyze_special_polynomials`, `certify_primitive_no_special_polynomials` (primitive preserved-constant levels have a complete absence theorem; other families are positive-evidence-only) |
| Algebraic Hermite reduction | `hermite_reduce_algebraic`, `lazy_hermite_reduce_algebraic`, `simple_radical_integral_basis` |
| Algebraic finite residues | `trager_algebraic_residue_polynomial` |
| Algebraic constants and all-place residues | `analyze_algebraic_constant_field`, `collect_algebraic_all_place_residues` |
| Algebraic function-field infrastructure | `AlgebraicFunctionField`, `integral_closure`, `AlgebraicFractionalIdeal`, `AlgebraicDivisor`, `AlgebraicDifferential` |
| Mixed-prefix backend feasibility | `TowerRepresentationProposal`, `analyze_tower_representation`, `TowerRepresentationOutcome` |

These functions return structured immutable results rather than only a printed
antiderivative. Their dataclass fields are the executable reduction
certificates used by the test suite. Consult the function docstrings and the
corresponding focused test file for preconditions and exact postconditions.

For a plain-language guide to interpreting success, a mathematical
non-elementarity certificate, and an unsupported input, see
[Understanding solver decisions](understanding-decisions.md).

See the [architecture and roadmap](design.md) for the algorithms behind these
boundaries and the planned path toward the full algebraic cases.
