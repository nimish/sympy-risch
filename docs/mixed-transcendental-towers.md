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
