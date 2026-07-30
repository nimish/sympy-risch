# Getting started

`sympy-risch` integrates elements of an explicitly supplied differential
tower. The library works with native SymPy domain elements: the tower records
what each formal generator means to the derivation, while ordinary field
arithmetic remains SymPy arithmetic.

For a checked one-variable fragment, it can also start from a SymPy expression
and construct the tower for you. Rational powers now create algebraic levels,
including algebraic-over-transcendental towers. General algebraic objects,
arbitrarily interleaved mixed backends, and global branch-aware presentation
remain future work.

## Start from a SymPy expression

`integrate_expression` is the smallest user-facing entry point. It accepts one
integration symbol, rational constants, addition, multiplication, rational
powers, and nested `log`, `exp`, and `tan` calls. A nonintegral rational power
proposes an algebraic relation; a transcendental call becomes a P/E/T level
only after the existing checked extension factory certifies it. The compiler
can also reclassify a finite-power exponential dependency as algebraic.

```python
from sympy import log, symbols

from sympy_risch import ExpressionIntegrationOutcome, integrate_expression

x = symbols("x")
result = integrate_expression(1 / (x * log(x)), x)

assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
assert result.antiderivative == log(log(x))
assert result.elementary_result is not None
assert result.elementary_result.reconstructs
```

The returned antiderivative is a *local differential presentation*: ordinary
SymPy differentiation verifies it for this example, but the frontend never
uses identities such as `log(a*b) = log(a) + log(b)` or `atan(tan(a)) = a`.
Those identities require branch or interval assumptions. If a new syntactic
call is already dependent on the lower tower, such as `log(exp(x))` after
`exp(x)` or a second related tangent, the compiler returns `UNSUPPORTED` with
`DEPENDENT_EXTENSION` rather than silently rewrite it.

`ExpressionIntegrationOutcome.NON_ELEMENTARY` is a certified Liouville
obstruction in the compiled P/E/T tower. `UNSUPPORTED` instead means that the
expression grammar, constant field, dependency decision, or dispatcher does
not yet cover the request; it is never a non-elementarity claim.

Tower discovery can be inspected independently of integration:

```python
from sympy import exp, log, sqrt, symbols

from sympy_risch import discover_expression_tower

x = symbols("x")
discovery = discover_expression_tower(
    sqrt(x + log(x)) + exp(sqrt(x)),
    x,
)

assert discovery.discovered
assert [candidate.kind.value for candidate in discovery.candidates] == [
    "primitive",
    "algebraic",
    "algebraic",
    "hyperexponential",
]
```

Sibling order follows SymPy's deterministic expression order; dependencies
always precede their users. The first mixed term can already compile as
`QQ(x)(log(x))[sqrt(x + log(x))]`. The second demonstrates the opposite
algebraic-then-exponential order: discovery succeeds, while its lower
algebraic logarithmic-derivative decision is still an explicit backend
boundary. This distinction lets callers inspect the right mathematical tower
without mistaking an unfinished integrator for failed structure discovery.

A simple algebraic expression already runs end to end:

```python
from sympy import Rational, sqrt

radical = integrate_expression(sqrt(x), x)
assert radical.outcome is ExpressionIntegrationOutcome.ELEMENTARY
assert radical.antiderivative == 2 * x ** Rational(3, 2) / 3
```

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
    build_riccati_mobius_normal_form,
    certify_primitive_extension,
    certify_special_polynomial,
    integrate_elementary,
    integrate_riccati_mobius_normal_form,
    primitive_logarithmic_derivative_signature,
    solve_limited_integration,
    solve_parametric_logarithmic_derivative,
    solve_parametric_logarithmic_derivative_heuristic,
    solve_parametric_risch_differential_equation,
    solve_exact_derivative,
    solve_hyperexponential_coupled_laurent_rde,
    solve_hypertangent_coupled_cancellation_rde,
    solve_radical_logarithmic_derivative,
    solve_rational_parametric_risch_differential_equation,
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

`TowerAdmissionOutcome.ACCEPTED` says that the tower has a supported
structural shape, not that a particular integrand has an elementary
antiderivative or that every current recursive solver applies. If the outcome
is `TowerAdmissionOutcome.UNSUPPORTED`, the failed assumption is represented
by `TowerAdmissionReason`; that result also says nothing about integrability.
The admitted family is primitive, hyperexponential, or hypertangent levels
with replay-proved preserved constants, together with an exact hyperbolic Möbius
normalization at the active top level described below. The explicit
certificate-driven Riccati route below is intentionally separate: it first
changes coordinates, then checks the resulting hyperexponential target.
For the automatic hyperbolic law, the derivative law itself constructs and
checks its two linear special factors. Radical and parametric logarithmic
decisions are total across the checked P/E/T family orderings; later scalar
and parametric RDE cells can still report an explicit unsupported boundary.
That outcome is not non-elementarity. General nonlinear monomials remain
outside this public contract until the library implements a complete
derivation-bound no-special proof.

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
replayed no-new-constants hypotheses:

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

This automatic dispatcher path covers precisely that hyperbolic law. It is
not a general decision procedure for arbitrary polynomial or Riccati
monomials.

### Explicit certified Riccati normalization

There is a broader *explicit* coordinate-change API when you already know two
linear special factors. For a quadratic law `Dt = Q(t)`, a factor `t-a` is
special when `D(t-a)` is divisible by `t-a`; equivalently, `D(a) = Q(a)`.
If certificates prove this for two distinct roots `a` and `b`, then

```text
u = (t-a)/(t-b)
Du/u = A*(a-b), where A is the t**2 coefficient of Q.
```

For example, neither factor below is discovered by the library. We supply
and verify them before asking for the coordinate change:

```python
riccati = base.extend(
    MonomialExtension(
        Poly(t**2 + (1 / x - x) * t, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
derivation = riccati.derivation()
numerator = certify_special_polynomial(
    Poly(t - x, t, domain=base.field()),
    derivation,
)
denominator = certify_special_polynomial(
    Poly(t, t, domain=base.field()),
    derivation,
)
normal_form = build_riccati_mobius_normal_form(
    derivation,
    u,
    numerator,
    denominator,
)

# u = (t-x)/t and Du = x*u.
planted = riccati.field().from_sympy((t**2 + x) / (t + 1))
result = integrate_riccati_mobius_normal_form(
    derivation(planted),
    normal_form,
)

assert result.is_elementary
assert result.reconstructs
```

The builder retains both input certificates and verifies the forward map
`t = (b*u-a)/(u-1)`, its inverse, and the tower derivation.
`RiccatiMobiusNormalForm` also exposes those verification methods for
inspection. Its integration function uses the normal P/E/T dispatcher only
after the target tower passes the existing admission check. The target `u`
must be a fresh SymPy `Symbol` over the whole source field. The checked
factor pair supplies the coordinate evidence, and preservation is replayed on
the hyperexponential target; source status metadata is only advisory. A
missing factor certificate, an unavailable replay, or a target outside that
contract is an unsupported algorithmic boundary—not a non-elementarity
result.

### Certified local special-pole reduction

One linear special factor can also be useful without a second factor or a
coordinate change. Given a checked `p=t-a` with `D(p)=h*p`, the library can
cancel an exposed principal part (the negative-power portion at `p`) when its
necessary lower-field Risch differential equation (RDE) has a solution. For
the Riccati factor above, `p=t-x` and
`h=t+1/x`, so the local coefficient is `h mod p = x+1/x`:

```python
from sympy_risch import reduce_certified_linear_special_poles

pole_exact = riccati.field().from_sympy(x / (t - x) ** 2)
local = reduce_certified_linear_special_poles(
    derivation(pole_exact),
    derivation,
    numerator,  # the checked certificate for t-x
)

assert local.is_fully_reduced
assert local.reconstructs
assert local.exact_part == pole_exact
assert local.remainder == riccati.field().zero
```

At pole order `m`, it solves `D(q) - m*hbar*q = r` in the lower field, where
`r` is the leading coefficient and `hbar = h mod p`. It then subtracts the
*full* derivative of `q/p**m`, because a `t`-dependent `h` can alter the next
lower pole order. This is intentionally a local reduction, not a full
elementary integration decision: `UNRESOLVED_RDE` and
`UNSUPPORTED_LOWER_RDE` preserve the exact work completed but do not prove
that the original integral is non-elementary. Higher-order cancellation can
make an integrand exact even when its apparent leading local RDE is unsolved.
`local.is_fully_reduced` means only that the selected factor has disappeared;
an unrelated remainder can still be present.

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
modes. Each mode produces a real coupled RDE. For a provably real finite
constant field `C`, the solver temporarily rebases the immutable tower over
the exact compositum `C(i)`, solves one ordinary scalar RDE, and converts the
exact real and imaginary coordinates back over `C`. For `C = QQ`, this is
SymPy's `QQ_I`. Its direct block-matrix test oracle does not use that
conversion.

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

This primitive recurrence is the specialized lower-field route. For the
documented mixed P/E/T shapes, the RDE dispatcher uses the corresponding
hyperexponential or hypertangent reductions and coupled-RDE fallback when a
lower level is not primitive; other recursive paths retain the boundary
described in [mixed transcendental towers](mixed-transcendental-towers.md).
Coefficients with poles in the current primitive generator are weakly
normalized before normal-denominator reduction.
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

This is the terminal recurrence used by the rational coupled-RDE frontend for
its documented dispatch shapes. That frontend performs paired weak
normalization, denominator reduction, degree bounds, and SPDE before reaching
this stage. It also uses the immutable Cayley normalization needed for nested
hypertangent levels.

## Solve a rational parametric RDE

A parametric RDE asks for the complete constant-vector solution space of

```text
D(y) + a*y = c_1*b_1 + ... + c_m*b_m.
```

The right-hand sides must be reduced together: individually unsolvable terms
can cancel. For example, neither `1/x + 1` nor `1/x` is a rational derivative,
but their difference is:

```python
parametric = solve_rational_parametric_risch_differential_equation(
    0,
    (1 / x + 1, 1 / x),
    x,
)

assert parametric.parameter_basis[0].parameters == (1, -1)
assert parametric.parameter_basis[0].solution == x
assert parametric.homogeneous_basis[0].solution == 1
```

The solver weakly normalizes `a`, derives one common denominator and a
complete polynomial degree bound, and computes the exact constant-field
nullspace. `parameter_basis` gives independent solvable right-hand-side
combinations; `homogeneous_basis` separately spans solutions of
`D(y) + a*y = 0`. Native finite algebraic constants are selected with the
`domain=` keyword.

The recursive entry point performs the same calculation over a checked
primitive tower. It keeps all right-hand sides together while denominator
reduction, SPDE, and the descending polynomial recurrence move through the
tower. For example, in `QQ(x, t)` with `Dt = 1/x`, subtracting the second
right-hand side from the first leaves `t`, which is `D(1) + t*1`:

```python
field = log_x.field()
primitive_parametric = solve_parametric_risch_differential_equation(
    field.from_sympy(t),
    (field.from_sympy(t + 1), field.one),
    log_x.derivation(),
)

direction = primitive_parametric.parameter_basis[0]
assert direction.parameters == (1, -1)
assert direction.solution == field.one
assert primitive_parametric.homogeneous_basis == ()
```

At a lower-field cancellation, the solver recursively computes another
parametric solution space rather than choosing one scalar solution. This is
what preserves combinations that only become solvable after several tower
levels.

Hyperexponential levels use the same public entry point. Normal-pole
remainders, special powers of the exponential generator, Laurent support
bounds, and SPDE divisibility conditions all restrict the full parameter
space. Here the shared `1/x` obstruction cancels before the forced leading
`t` term yields the solution `y = 1`:

```python
field = exp_x.field()
theta = exp_x.levels[-1].generator
hyperexponential_parametric = solve_parametric_risch_differential_equation(
    field.from_sympy(theta),
    (field.from_sympy(theta + 1 / x), field.from_sympy(1 / x)),
    exp_x.derivation(),
)

direction = hyperexponential_parametric.parameter_basis[0]
assert direction.parameters == (1, -1)
assert direction.solution == field.one
```

Hypertangent levels use the same entry point and return a basis over the
original real constant field. Internally, the solver uses the exact Cayley
normal form over `C(i)`, then splits both solutions and parameter coordinates
back into real directions:

```python
field = tan_x.field()
planted = field.from_sympy((u**2 + x) / (u + 1))
obstruction = field.from_sympy(1 / u)
tangent_parametric = solve_parametric_risch_differential_equation(
    field.zero,
    (tan_x.derivation()(planted) + obstruction, obstruction),
    tan_x.derivation(),
)

direction = tangent_parametric.parameter_basis[0]
assert direction.parameters == (1, -1)
assert direction.solution == planted
```

Parametric RDE propagation is therefore complete through recursively checked
P/E/T prefixes over the admitted finite constant fields.

Limited integration uses that complete vector space to solve the affine
problem

```text
D(v) = f + c_1*g_1 + ... + c_m*g_m.
```

For example, neither `1 + 1/x` nor `1/x` has a rational antiderivative by
itself, but their difference does:

```python
field = base.field()
limited = solve_limited_integration(
    field.from_sympy(1 + 1 / x),
    (field.from_sympy(1 / x),),
    base.derivation(),
)

assert limited.has_solution
assert limited.particular.coefficients == (-1,)
assert limited.particular.antiderivative == field.from_sympy(x)
```

`direction_basis` describes every allowable change to both the constants and
the antiderivative. If `proves_no_solution` is true, completeness of the
underlying PRDE basis proves only that this affine problem has no solution in
the current field; it is not by itself a non-elementarity claim.

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
point for its implemented level and recursive shapes. It returns the minimal
positive integer `n` and an argument `v` satisfying `n*f = D(v)/v`; outside
those shapes it reports the documented unsupported boundary. Its immutable
certificate retains the rational result, a complete primitive signature, or
the mixed-tower relation data appropriate to the current level.

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
when replay proves constant preservation at every primitive level. Where the
relation-system route is implemented for a mixed P/E/T lower tower, the public
parametric solver uses it automatically; other recursive shapes, algebraic
lower fields, and other unsupported families retain explicit boundaries.

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
[algebraic infrastructure contract](algebraic-infrastructure.md).

## Integrate the first algebraic logarithmic cases

`integrate_algebraic` combines algebraic Hermite reduction, all-place
residues, and complete reduced divisor-class arithmetic. Its current constructive
scope is a direct `C(x)` presentation with `C = QQ` or a finite algebraic
extension of `QQ`, provided every local residue already lies in the displayed
constant field. The algorithm expands those constants in an exact `QQ` basis
and constructs one verified logarithmic term for each principal coordinate
divisor. For `y**2 = x`, the ramified point over `x = 0` and infinity have
residues `2` and `-2`, so the divisor is `div(y)`:

```python
from sympy import log
from sympy_risch import integrate_algebraic

decision = integrate_algebraic(
    1 / x,
    sqrt_x.derivation(),
    basis,
)

assert decision.is_elementary
assert decision.antiderivative == 2 * log(y)
assert decision.reconstructs
```

For a nonprincipal coordinate divisor, `integrate_algebraic` automatically
reduces the curve and class at two supported good primes. Compatible finite
orders determine one characteristic-zero candidate, which is then verified
by general divisor-class iteration; incompatible orders prove nontorsion.
This does not special-case hyperelliptic or Pell--Abel curves.
`torsion_order_bound` remains an optional explicit bounded-search override.
The genus-2 degree-29 example in the algebraic infrastructure guide needs no
order hint.

Non-rational local residues return `NEEDS_CONSTANT_BASE_CHANGE`, and a nonzero residue-free
remainder is tested for an algebraic primitive. If its primitive-pole space
has a complete Riemann--Roch certificate, `integrate_algebraic` either adds
that primitive or retains `RESIDUE_FREE_REMAINDER` with an exact in-field
`NO_PRIMITIVE` decision. Those are honest algorithmic boundaries, not
non-elementarity certificates. Algebraic levels are not yet part of the
recursive `integrate_elementary` dispatcher. The direct-`C(x)`
integrating-factor RDE and the narrower terminal simple-radical RDE descent
below are separate low-level endpoints.

For example, on the same curve `y**2 = x`, the residue-free differential
`dx / y` is `D(2*y)`, so the integration result is now constructive without
any manually supplied ansatz:

```python
exact_decision = integrate_algebraic(
    sqrt_x_extension.field.one / sqrt_x_extension.field.generator,
    sqrt_x.derivation(),
    basis,
)

assert exact_decision.is_elementary
assert exact_decision.exact_part == 2 * sqrt_x_extension.field.generator
assert exact_decision.reconstructs
```

## Solve a first mixed algebraic--monomial RDE

For a terminal pure radical `theta**n = u` over an implemented lower P/E/T
tower, `solve_relative_simple_radical_rde` descends the equation
`D(v) + a*v = b` when `a` lies in the lower field. With
`D(theta) = eta*theta`, it solves one scalar lower RDE for each power-basis
coordinate, using the shifted coefficients `a + j*eta`. The returned result
reconstructs in the full algebraic field.

The generic exact-derivative dispatcher uses this path when `a = 0`, so a
planted derivative at the terminal algebraic level is available through the
usual entry point:

```python
from sympy import Poly, symbols
from sympy.polys.domains import QQ
from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_exact_derivative,
)

x, t, theta = symbols("x t theta")
mixed_base = DifferentialTower(QQ).extend(
    MonomialExtension(Poly(1, x, domain=QQ), ConstantFieldStatus.PRESERVED)
)
mixed_lower = mixed_base.extend(
    MonomialExtension(
        Poly(1 / x, t, domain=mixed_base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
mixed = mixed_lower.extend(
    AlgebraicExtension(Poly(theta**2 - t, theta, domain=mixed_lower.field()))
)
v = mixed.field().from_sympy(x + theta)
exact = solve_exact_derivative(mixed.derivation()(v), mixed.derivation())

assert exact.has_solution
assert exact.antiderivative == v
```

This terminal tower has `Dt = 1/x` and `theta**2 = t`. The lower field is used
only for the coordinate equations; the library does not treat it as `C(x)`.
Non-radical terminal extensions, non-lower coefficients, relative
Hermite/residue work, and recursive elementary integration remain explicit
boundaries. See
[the algebraic infrastructure contract](algebraic-infrastructure.md#relative-simple-radical-rde-descent)
for the complete current scope and a nonzero-coefficient example.

### Continue through profile-selected algebraic constant extensions

Omit `factor` and the integrator factors the square-free all-place residue
profile. It tries supported rationally defined factors in deterministic order.
If one target still needs more residue constants, the target is profiled
again and a new exact compositum is constructed. The result makes every field
change explicit: an elementary target answer is an answer in the final
`C'(x, y)`, not automatically one in the original `QQ(x, y)`.

```python
from sympy_risch import integrate_algebraic_with_constant_base_change

continued = integrate_algebraic_with_constant_base_change(
    y / (x * (x - 2)),
    sqrt_x.derivation(),
    basis,
)

assert continued.source_result.status.value == "needs_constant_base_change"
assert len(continued.steps) == 1
assert continued.base_change is not None and continued.base_change.certified
assert continued.target_result is not None
assert continued.is_elementary
assert continued.reconstructs
```

Pass an explicit `Poly` as `factor` when a particular constant field is
desired; that requests one step. Automatic selection can build successive
composita from `QQ` and existing SymPy algebraic number fields. It currently
adjoins only factors defined over `QQ` and retains deliberately narrow
finite-place splitting support. Factors with genuinely algebraic
coefficients and general splitting fields remain future algebraic work.
For curves over `QQ`, positive divisor-class torsion is selected
algorithmically by supported good-prime reductions and verified over
characteristic zero. Factors genuinely defined over algebraic constants,
prime ideals of algebraic number fields, and unsupported bad models remain
future work.

## Certify a first global algebraic space

The global-space kernel deliberately certifies only cases with a direct
completeness theorem. For the same rational curve `y**2 = x`, the generator
`y` has one pole of degree one, so its powers form a complete basis of each
corresponding Riemann--Roch space:

```python
from sympy_risch.algebraic_geometry.riemann_roch import compute_riemann_roch_space

parameter = sqrt_x_extension.field.generator
pole_divisor = curve.principal_divisor(parameter).negative_part
space = compute_riemann_roch_space(
    curve,
    pole_divisor.scale(3),
    parameter=parameter,
)

assert space.complete
assert space.dimension == 4
assert space.basis == tuple(parameter**power for power in range(4))
```

For arbitrary divisors, the function returns an empty `UNSUPPORTED` result
until the finite/infinite lattice-intersection algorithm is implemented. It
does not mistake a bounded candidate search for a complete basis.

## Mathematical assumptions

Risch structure theorems require proof that a transcendental adjunction does
not enlarge the constant field. For primitive levels, use
`certify_primitive_extension`: it proves `PRESERVED` when `Dt` is not exact in
the lower field, returns `ENLARGED` with the witness `v` when `Dt = D(v)`, and
leaves unsupported or uncertified lower towers `UNKNOWN`.

For every currently supported primitive, hyperexponential, hypertangent, or
hyperbolic level, algorithms replay the family-specific constant-field
criterion at its exact tower position. `ConstantFieldStatus` is cached
diagnostic metadata: factories populate it, but a caller-supplied enum never
admits or rejects a theorem application.

A checked positive special-polynomial certificate is usable evidence, and
the primitive structure theorem can produce a derivation-bound absence
certificate after constant preservation is replayed. General nonlinear
absence remains unsupported rather than becoming a negative conclusion from
caller metadata.

The current SymPy domain backend cannot yet represent a monomial extension
above a function-field `AlgebraicExtension`, or a nested
`AlgebraicExtension`. The library therefore rejects those tower compositions
immediately with `NotImplementedError`, rather than constructing a tower that
would fail later during coefficient coercion. This is a backend representation
boundary, not a mathematical claim about the integral.

## Current coverage

| Tower shape or operation | Status |
| --- | --- |
| Polynomial integration over `QQ[x]`; rational integration over standard `C(x)` for supported finite algebraic constants | End-to-end |
| Explicit certified primitive/hyperexponential/hypertangent towers over admitted finite constants | Structure, scalar-RDE, parametric-RDE, and required coupled-RDE decisions are complete across the declared P/E/T solver cell; elementary integration covers the documented recursive cells pending the executable operation-totality matrix |
| Explicit two-factor quadratic/Riccati normalization at the active top level | End-to-end once two checked linear special factors yield a replay-preserved hyperexponential target; ordinary general-polynomial dispatch remains unsupported |
| Caller-certified linear special-pole reduction | Local exact principal-part reduction over one active linear special factor; unresolved lower RDEs are structured boundaries, not integration verdicts |
| Conservative automatic P/E/T expression compilation | End-to-end for one variable, `QQ` constants, rational arithmetic, integer powers, and nested `log`/`exp`/`tan`; dependent or branch-sensitive forms return a structured boundary |
| Admission of the certified P/E/T contract and top-level hyperbolic Möbius normalization | `assess_elementary_admission` reports support separately from elementary integrability |
| Scalar-RDE dispatch admission | Checks the complete recursive prefix over every supported finite algebraic `C/QQ`: P/E levels generally, plus hypertangent levels when `C` is provably real, with replay-proved preserved constants above the base |
| Scalar RDE over recursive P/E/T levels, general rational coefficient and right-hand side | Complete for the certified finite-constant P/E/T cell |
| Rational, primitive, hyperexponential, and hypertangent parametric RDE | Complete full constant-vector solution space over supported finite algebraic constants and recursively checked P/E/T prefixes; tangent results descend exactly to the original real field |
| Limited integration over recursive P/E/T levels | Complete affine solution family, or a PRDE-backed proof that no solution exists in the current field |
| Coupled RDE in the explicit certified P/E/T contract | Complete for the documented dispatch shapes over provably real finite constants; the scalar fallback uses the exact compositum `C(i)` |
| Coupled RDE over a hyperexponential level, lower-field matrix and Laurent right-hand sides | Complete terminal decision |
| Coupled polynomial RDE over a hypertangent level, lower-field or `b0 - n*eta*t` diagonal | Complete terminal decision |
| General rational coupled RDE over a hypertangent level | Complete for the documented dispatch shapes |
| Exact derivatives in supported rational, primitive, hyperexponential, and hypertangent levels | Complete for the documented recursive shapes; a terminal pure-radical algebraic level also descends to implemented lower scalar RDEs |
| General nonlinear monomial with absence of special polynomials | Not implemented; positive special-factor certificates support only their documented local reductions |
| Radical and parametric logarithmic derivatives in the supported P/E/T tower contract | Complete for every recursively checked P/E/T ordering over the admitted finite constant fields; tangent relations are direct real relations with an independent Gaussian oracle |
| Algebraic Hermite reduction over `QQ(x)` | Low-level API |
| Algebraic finite-residue polynomial | Low-level API for monic polynomial models with square-free unbranched finite poles |
| Algebraic orders, ideals, all places, valuations, local expansions, and divisors | Complete low-level infrastructure over supported `C(x)` constants |
| Algebraic constant-field analysis and all-place residues | Low-level API: exact Hermite success, plus complete pole data for admitted preserved direct `C(x)` simple-radical presentations; other nonzero cases return an explicit boundary with no partial list |
| Certified algebraic Riemann--Roch spaces | Low-level API for degree-negative/zero divisors, degree-zero principal witnesses, nonspecial supplied bases, degree-one-pole rational parameters, and Hess ideal intersections; unavailable ideal arithmetic returns `UNSUPPORTED` |
| Finite algebraic valuation constraints | Low-level exact `V ∩ L(D)` kernel over a supplied finite constant subspace, including native coordinates at constructed higher-degree closed places; bounded nonspecial lattice searches are certified only when they reach the Riemann--Roch dimension |
| Algebraic divisor classes and torsion | General Riemann--Roch reduction; supported good-prime finite Jacobians select the unique characteristic-zero candidate or prove nontorsion, and every positive candidate is replayed exactly |
| Algebraic principality decision | Generic AFF decision through primitive coefficient content and reduced class addition, with complete direct `L(-D)` fallback |
| Algebraic displayed-constant residue integration | Constructive direct-`C(x)` algebraic Hermite/logarithmic integration for `C = QQ` or a finite algebraic extension; principal coordinate divisors are automatic and `QQ` torsion orders are selected from good finite reductions |
| Algebraic residue-free exactness | Uses the generic complete primitive-pole Riemann--Roch space; constructs a primitive or proves no primitive in the present algebraic field, without claiming non-elementarity |
| Algebraic integrating-factor RDE | A low-level direct-`C(x)` endpoint for `Du + a*u = b` when a verified `h` in the field satisfies `a = D(h)/h`; it reduces to algebraic exactness and leaves general curve RDEs unsupported |
| Relative simple-radical RDE | A low-level terminal algebraic-over-P/E/T endpoint for `theta**n = u` and lower-field `a`; it diagonalizes to scalar lower RDEs, while non-lower coefficients and general algebraic relations remain unsupported |
| Relative simple-radical normal Hermite reduction | Repeated normal poles are removed coordinatewise over the immediate lower monomial field; special poles and resonant leading equations remain explicit boundaries |
| One-step algebraic constant-base-change integration | Constructive `QQ(x, y) -> QQ(alpha)(x, y)` continuation for an explicit or profile-selected factor that splits in its root field; a returned elementary answer is explicitly target-field-valued |
| Terminal pure simple-radical algebraic level | Constructive positive route in `integrate_elementary`: exact nonconstant power-basis modes descend to lower scalar RDEs, then the lower coordinate is integrated recursively; general algebraic logarithmic/residue cases remain unsupported |
| Mixed algebraic/monomial prefix representation | `analyze_tower_representation` proves native representability for a monomial prefix plus an optional terminal algebraic level and reports monomial-over-algebraic as a structured backend limitation; nested algebraic extension construction remains rejected by the backend |
| Hypertangent over hyperexponential or hypertangent lower levels | End-to-end for the documented certified shapes |
| Automatic expression tower discovery and back-substitution | Candidate DAGs cover rational powers and nested P/E/T calls in either mixed order; the native compiler supports rational-power algebraic levels, terminal algebraic-over-P/E/T towers, and finite-power exponential reclassification. Algebraic `RootOf`, algebraic composita, monomial-over-algebraic execution, full minimization, and branch metadata remain open |
| Complete global real branch management | Not implemented |

## Low-level entry points

The package exports the component algorithms used by the dispatcher. The
principal groups are:

| Task | Entry points |
| --- | --- |
| Polynomial integration | `integrate_polynomial` |
| Rational reduction and integration | `hermite_reduce_rational`, `integrate_rational`, `integrate_rational_real` |
| Rational logarithmic analysis | `lazard_rioboo_trager`, `rothstein_trager_residue_polynomial`, `solve_rational_logarithmic_derivative` |
| Expression tower discovery | `discover_expression_tower`, `ExpressionTowerDiscovery`, `ExpressionTowerCandidate` |
| Tower construction and differentiation | `DifferentialTower`, `MonomialExtension`, `AlgebraicExtension`, `Derivation` |
| Algorithm admission | `assess_elementary_admission`, `assess_scalar_rde_admission`, `TowerAdmission`, `TowerAdmissionOutcome`, `TowerAdmissionReason` |
| Monomial reduction | `hermite_reduce_monomial`, `residue_reduce_monomial`, and the family-specific `reduce_*_monomial` functions |
| Exact derivatives and RDEs | `solve_exact_derivative`, `solve_risch_differential_equation`, and the field-specific RDE solvers |
| Primitive logarithmic analysis | `solve_radical_logarithmic_derivative`, `primitive_logarithmic_derivative_signature` |
| Parametric logarithmic derivative | `solve_parametric_logarithmic_derivative`; `solve_parametric_logarithmic_derivative_heuristic` exposes the Section 7.3 fast path |
| Hyperbolic normal form | `build_hyperbolic_mobius_normal_form` for the exact law `Dt = eta*(1 - t**2)` |
| Certified Riccati normal form | `build_riccati_mobius_normal_form`, `integrate_riccati_mobius_normal_form` with two `SpecialPolynomialCertificate` inputs |
| Special-polynomial evidence | `certify_special_polynomial`, `analyze_special_polynomials`, `certify_primitive_no_special_polynomials` (primitive preserved-constant levels have a complete absence theorem; other families are positive-evidence-only) |
| Certified linear special poles | `reduce_certified_linear_special_poles`, `LinearSpecialPoleReduction`, `LinearSpecialPoleReductionStatus`, `LinearSpecialPoleStep` |
| Algebraic Hermite reduction | `hermite_reduce_algebraic`, `lazy_hermite_reduce_algebraic`, `simple_radical_integral_basis` |
| Algebraic finite residues | `trager_algebraic_residue_polynomial` |
| Algebraic constants and all-place residues | `analyze_algebraic_constant_field`, `collect_algebraic_all_place_residues` |
| Algebraic all-place residue profiles | `build_algebraic_residue_profile` |
| Algebraic displayed-constant residue integration | `AlgebraicIntegrationResult`, `AlgebraicLogarithmicTerm`, `integrate_algebraic`, `reconstruct_constant_residue_logarithms` |
| Algebraic integrating-factor RDE | `find_algebraic_integrating_factor`, `solve_algebraic_integrating_factor_rde` |
| Relative simple-radical RDE | `recognize_relative_simple_radical`, `solve_relative_simple_radical_rde`, `solve_relative_simple_radical_exact_derivative` |
| Relative simple-radical normal Hermite reduction | `hermite_reduce_relative_simple_radical` |
| Relative simple-radical integration | `integrate_relative_simple_radical` and the corresponding terminal-algebraic `integrate_elementary` route |
| Certified algebraic global spaces | `RiemannRochSpace`, `compute_riemann_roch_space` |
| Algebraic valuation constraints | `solve_algebraic_valuation_constraints`, `finite_lattice_slice`, `search_nonspecial_lattice_slice` |
| Algebraic divisor classes | `ReducedAlgebraicDivisorClass`, `reduce_algebraic_divisor_class`, `AlgebraicTorsionSearch`, `search_algebraic_divisor_torsion` |
| Algebraic principality decision | `AlgebraicPrincipalityDecision`, `decide_algebraic_principality` |
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
