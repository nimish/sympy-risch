# Mixed transcendental towers

This checkpoint closes the recursive transcendental core for monomial towers
over the standard rational base `QQ(x)`, `Dx = 1`. The supported families are

| family | derivative law |
| --- | --- |
| primitive | `Dt = eta` |
| hyperexponential | `Dt = eta*t` |
| hypertangent | `Dt = eta*(1 + t**2)` |

Every non-base level must carry a certified
`ConstantFieldStatus.PRESERVED`. Within that scope, the exact-derivative,
radical-logarithmic-derivative, Risch differential-equation, and recursive
elementary-integration solvers may descend through any finite ordering of
these three families.

## Admission is not integrability

`assess_elementary_admission` exposes this support boundary before a caller
asks for an integral. It returns an immutable `TowerAdmission` whose
`TowerAdmissionOutcome` is one of the following:

| Outcome | Meaning |
| --- | --- |
| `TowerAdmissionOutcome.ACCEPTED` | The explicit tower satisfies the current base, constant-field, and P/E/T-family contract, including an exact hyperbolic Möbius normalization only at its active top level. |
| `TowerAdmissionOutcome.UNSUPPORTED` | The tower is outside that contract. Its failed requirement is recorded as a `TowerAdmissionReason` when one applies. |

Admission answers only whether the tower is a supported input shape. An
accepted tower can still have either an elementary antiderivative or a proved
Liouville/Risch obstruction for a particular integrand. Conversely, an
unsupported admission is not a non-elementarity result and must never be
reinterpreted as one. Call `integrate_elementary` only to obtain the
mathematical integration result; use the admission result to choose supported
control flow without treating `NotImplementedError` as a proof.

The exact law `Dt = eta*(1 - t**2)` is admitted only when it is the active
top level, because the dispatcher normalizes that level before its recursive
descent; it cannot yet normalize a hyperbolic level below another extension.
Its special-polynomial metadata may be `UNKNOWN` or `PRESENT`, but `ABSENT`
is rejected because the law itself exhibits two linear special factors.
An active quadratic/Riccati level with two caller-supplied checked linear
special factors has a separate explicit route:
`build_riccati_mobius_normal_form` constructs its hyperexponential coordinate,
and `integrate_riccati_mobius_normal_form` checks admission of that *target*.
This does not make the source tower accepted by `assess_elementary_admission`,
and it does not search for the factors.
By contrast, the broader dispatcher has an
experimental nonlinear branch when the caller supplies
`SpecialPolynomialStatus.ABSENT`; it is not part of this total public
admission contract. An admission result must reject arbitrary nonlinear
monomials rather than treat that metadata as a completeness proof.

## Why tangent levels compose

For a hypertangent level, the Cayley coordinate

```text
z = (1 + i*t)/(1 - i*t)
```

has `Dz/z = 2*i*eta`. When a lower tower already contains tangent levels, the
implementation builds an immutable Gaussian normal form for *every* active
tangent generator before invoking the hyperexponential solver. It never
mutates the real tower. A Gaussian solution is split back into exact real and
imaginary coordinates, so public elementary results remain in the original
real field.

This handles, for example, a tower of the form

```text
QQ(x)(e)(t)(u)
De = x*e
Dt = (e + 1)*(1 + t**2)
Du = (t + 1)*(1 + u**2).
```

The direct real tangent reducer remains the normal production path. The
Gaussian form supplies its all-level coupled-RDE fallback and a proof object
for the coordinate change.

## Hyperexponential relations

At a hyperexponential level, the special generator contributes the
logarithmic direction `Dt/t = eta`. With more than one such level, deciding

```text
n*f = D(v)/v + m*eta
```

cannot safely be reduced to one recursive ratio. The relation compiler in
`sympy_risch.relations` collects all normal-factor valuations and every
hyperexponential special direction in one rational linear system. A kernel
vector supplies an integer witness; lack of a vector with nonzero `f`
coordinate is a negative certificate in the supported tower scope.

Tangent lower levels enter this same calculation through the Gaussian normal
form. The integer valuation lattice remains over `QQ`, not `QQ(i)`.

## Certified fundamental signatures

`build_fundamental_signature` exposes the certified structural directions of
an active P/E/T tower prefix as one immutable object.  On a supported prefix,
directions are ordered by tower level and retain native parent-field elements:

| family | recorded source direction |
| --- | --- |
| rational base / primitive | `Dt` |
| hyperexponential | `Dt/t` |
| hypertangent | `eta` in `Dt = eta*(1 + t**2)` |

The signature is supported only for the standard rational base and when every
non-base active level has certified preserved constants.  An unsupported
signature records why the narrow representation is unavailable; it is not a
negative theorem about a relation or an integral.

The public entry point is `build_fundamental_signature(derivation)`.  Its
`directions` property is ordered by active tower level, while the three
family-specific tuples retain the native field element needed by each
structure theorem.  Checked primitive, hyperexponential, and hypertangent
factory results expose the lower signature as
`lower_fundamental_signature` only when that lower tower is already within
this certified P/E/T scope; `None` never implies preserved constants.

For a hypertangent level, the recorded direction remains real.  The Gaussian
normal form independently verifies that its Cayley coordinate has
logarithmic derivative `2*i*eta`; this conversion is an adapter check, not a
second source of signature data.  The direct relation compiler consults the
signature for the active family and hyperexponential generator direction while
retaining its existing normal-factor valuation logic.  In particular, a
signature is not a catalogue of all Darboux polynomials and does not decide
non-generator special factors.

## Checked construction

The dependency module provides immutable checked factories:

```python
from sympy_risch.dependency import (
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
)
```

They return a `TranscendentalExtensionAnalysis` containing a fresh extension,
the dependency certificate when available, and an explanatory detail when the
lower tower is outside the implemented scope. A logarithmic dependency sets
the returned extension status to `ENLARGED`; otherwise the relevant structure
theorem certifies `PRESERVED`. They deliberately return `UNKNOWN` rather than
misclassifying an unsupported lower family as independent.

## Boundaries

This checkpoint does not make algebraic levels part of the transcendental
recursion. Algebraic Hermite reduction, residues, and curve-level Risch
equations remain a separately coordinated track. The expression front end
that discovers, orders, and rewrites `log`, `exp`, and `tan` expressions is
also separate: callers currently construct an explicit immutable tower.

General nonlinear monomials with an arbitrary nonconstant special polynomial
are likewise outside this P/E/T contract. The tangent case is covered because
its special polynomial has the explicit Cayley hyperexponential
normalization above.
