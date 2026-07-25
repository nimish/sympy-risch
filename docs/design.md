# Architecture and implementation roadmap

This document records the mathematical architecture of `sympy-risch`, the
parts that are implemented, and the boundaries that remain. It complements
the [getting-started guide](getting-started.md), which shows the current public
workflow.

The goal is a full elementary-integration decision procedure that can
eventually be proposed upstream to SymPy. The current package is an
experimental implementation of substantial Risch components, not yet a
drop-in replacement for `sympy.integrate`.

## Current status

The top-level entry point is:

```python
integrate_elementary(integrand, derivation) -> ElementaryIntegralResult
```

It operates on a manually constructed differential tower and supports:

- the standard rational base `QQ(x)` with `Dx = 1`;
- recursively stacked primitive extensions;
- one hyperexponential or hypertangent level directly over `QQ(x)`;
- a hyperexponential level over a recursively primitive lower tower with a
  rational logarithmic derivative in the top primitive generator; and
- a nonlinear polynomial monomial over supported lower levels when absence of
  nonconstant special polynomials has been certified.

The dispatcher returns exact and logarithmic parts, recursively descends into
the lower field, and preserves a structured Liouville obstruction when a
supported case is proved non-elementary. It explicitly rejects unsupported
tower shapes instead of treating an incomplete reduction as a negative
decision.

The lower-level library additionally implements:

- polynomial and rational integration over `QQ(x)`;
- Horowitz--Ostrogradsky and recursive rational Hermite reduction;
- Rothstein--Trager and Lazard--Rioboo--Trager logarithmic reduction;
- rational exact-derivative and logarithmic-derivative decisions;
- immutable differential towers over native SymPy domains;
- monomial classification and normal/special denominator decomposition;
- generalized monomial Hermite and residue reduction;
- primitive, hyperexponential, hypertangent, and nonlinear-no-special
  reducers;
- recursive radical logarithmic-derivative decisions over primitive towers,
  including complete principal parts and exact minimal-power witnesses;
- rational and polynomial Risch differential equations;
- weak normalization, normal-denominator, SPDE, and polynomial RDE reduction
  over a primitive field for rational coefficients, including Bronstein's
  lower-field cancellation recurrence;
- complete hyperexponential RDE reduction when the coefficient is in the
  lower field and the right-hand side is Laurent in the top generator;
- a recurrence implementation for polynomial RDEs and an independent matrix
  implementation used as a test oracle;
- rational coupled RDEs for the real hypertangent case;
- recursive dependency tests for hyperexponential generators over primitive
  towers and rational dependency tests for hypertangent generators;
- an exact result and verification contract for Bronstein's parametric
  logarithmic-derivative equation, distinguishing a proved negative result
  from an inconclusive heuristic and an unsupported field;
- Bronstein's Section 7.3 degree and denominator heuristic for that equation
  over `QQ(x)`, with exact radical-logarithmic verification;
- a complete rational fallback that solves the heuristic's remaining constant
  and non-logarithmic residue constraints over `QQ`;
- a complete parametric logarithmic-derivative decision over recursively
  primitive towers, with local signature constraints and independent
  one-level and nested global linear-system oracles;
- validated finite algebraic extensions and implicit differentiation;
- algebraic Hermite reduction over `QQ(x)` with explicit or lazily refined
  integral bases;
- Trager's nested-resultant finite-residue polynomial for polynomial algebraic
  models with certified square-free unbranched poles; and
- simple-radical recognition and integral-basis construction.

The required exact ground-type backend is `python-flint`.

Recent implementation checkpoints, newest first, are:

```text
693c0bb  Solve hyperexponential Laurent RDEs
609b55a  Extend exact decisions through exp and tan
4bab9ef  Certify primitive extension constants
f1fe842  Type the Risch solver kernel
e98ab35  Parallelize generated oracle tests
3b2744b  Document primitive weak normalization
4d732c6  Complete primitive RDE weak normalization
bc763f1  Reconcile recursive primitive checkpoint
889e4f6  Complete primitive RDE cancellation
c3225a2  Generalize primitive parametric decisions recursively
a9e62ff  Generalize primitive logarithmic signatures recursively
71fe12e  Compute algebraic finite residues
6e8e693  Complete primitive parametric logarithmic derivatives
5556d95  Add primitive logarithmic signatures
f47439e  Add independent parametric logarithmic oracle
444aebf  Complete rational parametric logarithmic derivatives
0a48352  Implement parametric logarithmic derivative heuristic
92599b0  Define parametric logarithmic derivative results
7b67c9f  Add full Risch roadmap and literature archive
bf9a424  Reduce hyperexponentials over primitive fields
25e7c2c  Reduce nested primitive extensions
820f9bd  Add recursive elementary integration dispatcher
```

## Design principles

1. Use SymPy's exact `Domain`, `Poly`, fraction-field, and finite-extension
   elements directly. Do not maintain a parallel element hierarchy.
2. Keep differential towers immutable. Recursion selects an explicit level;
   it never mutates a shared “current field.”
3. Classify extensions from their differential algebra, not from the name of
   the function that produced them.
4. Keep expression discovery and branch-sensitive presentation outside the
   exact differential-field core.
5. Require executable evidence for structure-theorem hypotheses such as
   preservation of constants or absence of special polynomials.
6. Distinguish constructive success, a proved obstruction, an inconclusive
   partial algorithm, and an unsupported input class.
7. Verify every constructive result by exact differentiation in the same
   tower.
8. Keep mathematical checkpoints small enough to review and revert
   independently.

## Architecture

The intended full system has three layers:

```text
SymPy expression
      |
      v
Tower construction and presentation metadata        planned
      |
      v
Native SymPy fields + immutable DifferentialTower   implemented
      |
      v
Reduction and decision algorithms                   partly implemented
      |
      v
Back-substitution and real branch presentation      planned
```

The middle layer is deliberately useful without the two presentation layers.
Manually constructed towers allow the mathematical algorithms to be tested
without conflating them with generator discovery.

### Implemented recursive kernel

The current implementation is organized by mathematical responsibility:

| Module | Responsibility |
| --- | --- |
| `_typing.py` | Structural types for native SymPy domain elements |
| `tower.py` | Immutable extension metadata and level-bound derivations |
| `exact.py` | Level-independent exact-derivative dispatch |
| `monomial.py` | Primitive signatures, radical logarithmic decisions, and family reducers |
| `dependency.py` | Parametric logarithmic decisions and generator-dependency certificates |
| `rde.py` | Polynomial, rational, and primitive Risch differential equations |
| `elementary.py` | Recursive elementary integration and obstruction propagation |

These modules form a descending recursion, not a mutable solver context. Every
recursive entry point receives a `Derivation` identifying the current tower
level, constructs an immutable result, and calls the same decision at
`derivation.lower()` when it reaches a lower-field coordinate. The currently
complete recursive path is

```text
QQ(x) <- primitive <- primitive <- ...
```

with explicit `Const(K(t)) = Const(K)` evidence at every primitive level.
Unsupported intervening families do not silently become negative decisions.

SymPy's exact domains return several backend-specific element classes:
`python-flint` ground values, fraction-field elements, and algebraic-extension
elements. The internal `FieldElement` protocol describes the arithmetic shared
by those values without erasing them to `Any`. Public coercion boundaries
accept `object`, then validate or convert it before field operations. Result
objects, recursive callbacks, and certificates carry concrete structural or
algorithm-specific types. Ruff enforces annotations in production code and ty
checks the complete `src` tree.

The important consumer relationships are:

```text
radical logarithmic decision ----> hyperexponential dependency
             |                                  |
             v                                  v
primitive RDE cancellation <-------- hyperexponential reduction
             |
             v
     lower-field RDE or exact-derivative decision
```

In particular, dependency analysis rejects rational or algebraic proposed
hyperexponential generators before integration. A genuine generator then
reaches the family reducer, whose coefficient equations use the same
recursive RDE and exact-derivative contracts.

### Result and certificate contracts

Public decisions keep construction, proof data, and unsupported scope
separate:

| Decision | Constructive witness | Negative or unsupported evidence |
| --- | --- | --- |
| Radical logarithmic derivative | least `n > 0` and `v` with `n*f = D(v)/v` | canonical rational result or nested primitive signature |
| Parametric logarithmic derivative | normalized `(n, m, v)` with `n > 0` | explicit `PROVED_NO_SOLUTION`, `INCONCLUSIVE`, or `UNSUPPORTED` outcome |
| Risch differential equation | `y` with `D(y) + a*y = b` | denominator, SPDE, polynomial, lower-field, or exact-derivative obstruction |
| Elementary integration | exact part and logarithmic terms | terminal canonical remainder and obstruction kind |

`RadicalLogarithmicDerivativeResult` contains either the rational base
certificate or a `PrimitiveLogarithmicDerivativeSignature`. A primitive
signature stores its lower-field result, so the certificate tree has the same
shape as the tower. On success, `minimal_power` is minimal, not merely a
clearing multiple, and `reconstructs` verifies the identity in the native
field.

`ParametricLogarithmicDerivativeResult` uses
`PrimitiveParametricLogarithmicDerivativeCertificate` at recursive primitive
levels. That certificate records the top-level linear constraints, a nested
lower parametric result when the ratio is free, and the final radical
signature for the chosen residual. Only `PROVED_NO_SOLUTION` is safe for a
caller to consume as a mathematical negative answer.

Primitive RDE results similarly retain the weak normalizer, normalized
coefficient and right-hand side, denominator bound, SPDE or polynomial result,
every recursively solved leading-coefficient equation, and the
logarithmic/exact decision used by the cancellation transform. Result objects
validate constructive witnesses on creation or expose a `reconstructs`
property; tests also differentiate them independently.

## Differential towers

`DifferentialTower` stores a constant domain and an immutable tuple of
extensions. The first extension may represent the independent variable:

```text
QQ < QQ(x) < QQ(x)(t) < QQ(x)(t)(u) < ...
```

`tower.field(level)` returns the exact field at a level.
`tower.derivation(level)` returns a `Derivation` bound to that level.
`tower.extend(extension)` returns a new tower.

Consecutive transcendental levels are physically represented by a flat SymPy
fraction field such as `QQ.frac_field(x, t, u)`. The tower metadata still
records the conceptual coefficient-field boundaries. This avoids coercion
problems in deeply nested SymPy fraction fields while preserving the recursive
mathematics.

### Implemented extension interface

`DifferentialExtension` is a structural protocol. Two concrete immutable
extension types currently implement it:

- `MonomialExtension`, for a transcendental generator `t` with `Dt` in
  `K[t]`;
- `AlgebraicExtension`, for a finite separable algebraic generator defined by
  a minimal polynomial over `K`.

Named `LogExtension`, `ExpExtension`, and `TanExtension` classes do not
currently exist. A future builder may provide named constructors or
presentation metadata, but the reducers should continue to dispatch on the
mathematical family.

### Derivation

For

```text
p(t) = sum(a_i t^i),    a_i in K,
```

the full derivative is

```text
D(p) = sum(D(a_i) t^i) + (partial p/partial t) Dt.
```

Coefficient differentiation retains only the first term. Both operations are
needed: Risch reduction repeatedly separates differentiation in the lower
field from differentiation caused by the top generator.

Fractions use the quotient rule:

```text
D(p/q) = (D(p)q - pD(q))/q^2.
```

All results remain native elements of the relevant SymPy domain.

## Monomial families

A `MonomialExtension` stores `Dt` as a univariate `Poly` over its parent
field. Its exact shape determines the family:

| Derivative law | `MonomialFamily` |
| --- | --- |
| `Dt = eta`, with `eta` in `K` | `PRIMITIVE` |
| `Dt = eta*t` | `HYPEREXPONENTIAL` |
| `Dt = eta*(1 + t**2)` | `HYPERTANGENT` |
| any other polynomial in `t` | `POLYNOMIAL` |

This classification is intentionally weaker than named-function recognition.
For example, `Dt = eta*t` identifies a hyperexponential generator, but
presenting it as `exp(u)` requires solving `Du = eta` in the lower field.

Derivative laws determine a generator only up to constants:

```text
Dt = Du/u                 t = log(u) + c
Dt = Du*t                 t = c exp(u)
Dt = Du(1 + t^2)          t = tan(u + c)
```

An expression front end must retain the original presentation when branch or
phase information matters.

### Constant-field evidence

Liouville and Risch structure theorems generally require

```text
Const(K(t)) = Const(K).
```

`ConstantFieldStatus` records whether that condition is `PRESERVED`,
`ENLARGED`, or `UNKNOWN`. Manual towers currently supply the evidence.
Algorithms that rely on the condition reject unproved cases.

For a primitive proposal `K(t)` with `Dt = eta` in `K`, this condition is
decidable using machinery already present in the lower tower. If
`eta = D(v)` for some `v` in `K`, then `t - v` is a new constant, so the
proposed generator is dependent and the level must be rejected or recorded as
`ENLARGED`. If the lower exact-derivative decision proves that no such `v`
exists, constants are preserved. Applying that test from the bottom upward
also detects dependencies between primitive generators; for example, after
`Dt1 = 1/x`, a proposal `Dt2 = 2/x` is rejected because
`2/x = D(2*t1)`.

### Normal and special polynomials

For an irreducible `p` in `K[t]`:

- `p` is normal when `gcd(p, Dp) = 1`;
- `p` is special when `p` divides `Dp`.

This is the monomial analogue of separating repeated rational poles from the
part that survives ordinary Hermite reduction. The common monomial reducer
lowers repeated normal factors, applies the generalized residue criterion to
the remaining square-free normal part, and leaves special factors and the
polynomial part for the family-specific algorithm.

For general nonlinear monomials, `SpecialPolynomialStatus.ABSENT` is explicit
evidence for the structure-theorem case with no nonconstant irreducible
special polynomials.

## Parametric logarithmic derivatives

For a hyperexponential generator `theta` over `K`, write
`eta = D(theta)/theta`. Bronstein's parametric logarithmic-derivative problem
is to decide whether

```text
n*f = D(v)/v + m*eta
```

for integers `n != 0`, `m`, and `v` in `K*`.

`solve_parametric_logarithmic_derivative_heuristic` implements the fast
Section 7.3 algorithm for `K = QQ(x)`, `Dx = 1`. For
`f = p + a/d` and `eta = q + b/e`, it first tries to determine `m/n` from
the positive-degree coefficients of `p` and `q`. If those do not suffice, it
uses the necessary congruence modulo

```text
gcd(lcm(d, e), derivative(lcm(d, e), x)).
```

Once a unique rational ratio is found, the rational
logarithmic-derivative solver decides whether an integer multiple of the
residual is `D(v)/v`. The result retains the coefficient comparison and that
final logarithmic certificate, and a success is independently verified in the
native fraction field.

The heuristic alone is not complete. Constant polynomial parts and
square-free denominators may leave `m/n` undetermined; Bronstein's
Example 7.3.2 is the smallest regression. The public
`solve_parametric_logarithmic_derivative` function invokes a complete
rational fallback in that case.

At the heuristic's `failed` exit, the common denominator is square-free and
both polynomial parts are constant. The fallback therefore constructs the
finite rational system

```text
constant_part(f) = c*constant_part(eta)
nonlog_residue_i(f) = c*nonlog_residue_i(eta)
```

for `c = m/n`. At each irreducible factor, the ordinary logarithmic direction
spanned by the factor derivative is projected away. An inconsistent system
proves that no ratio exists. Otherwise the remaining rational residues are
cleared by their denominator least common multiple, a rational logarithmic
argument is constructed, and the final witness is verified exactly.

Consequently the public solver is total over `QQ(x)`: it returns either a
verified solution or a proved negative result.

At each primitive level `K(t)`, the recursive structure fallback computes
signatures for `f` and `eta` and compares:

- every positive-degree coefficient in `t`;
- every repeated principal numerator at each irreducible `p`;
- every simple numerator modulo the rational-constant direction `QQ*Dp`; and
- the remaining constant-in-`t` components through the same complete solver
  over `K`.

The top constraints either determine `m/n`, prove inconsistency, or impose no
condition. In the last case, the lower-field problem chooses the ratio
recursively.
The selected integer residual is passed back through the primitive signature
solver, which constructs the logarithmic argument and verifies the complete
identity. Thus the public solver is total over recursively primitive towers
whose levels all carry certified preserved constants. General mixed
transcendental towers still require the Chapter 9 structure-theorem machinery.

The word *total* here is deliberately scoped. It means that every valid input
in the stated recursively primitive domain receives either a verified witness
or a proved obstruction. It does not mean that an intervening
hyperexponential, hypertangent, or algebraic level is coerced into this
algorithm; the parametric result reports `UNSUPPORTED` for such a tower.

## Recursive elementary integration

Each level reducer establishes an exact identity

```text
f = D(v) + ell + r,
```

where `ell` is the derivative of the encoded logarithmic terms. On a
successful level reduction, `r` belongs to the lower field and recursion
continues. On a negative decision, `r` carries the terminal obstruction.

`ElementaryIntegralResult` preserves both the local reduction and its lower
result. Its flattened invariants are:

```text
integrand
  = D(total_exact_part)
  + total_logarithmic_derivative
  + unintegrated_remainder.
```

The `reconstructs` property checks this identity exactly. `antiderivative` is
available only when the unintegrated remainder is zero.

The obstruction kinds currently distinguish:

- a nonconstant finite residue;
- an infinity residue;
- a nonzero reduced polynomial;
- an unsolvable Risch differential equation; and
- a nonzero element left in the constant-field base case.

### Exact-derivative dispatch

`solve_exact_derivative` is complete for the supported rational, recursively
primitive, hyperexponential, and hypertangent tower shapes. At a
hyperexponential level it reduces every nonzero Laurent mode through
`Dy + n*eta*y = b`; at a hypertangent level it uses the real coupled RDE for
each conjugate Cayley pair. The power-zero mode is passed recursively to the
lower exact-derivative decision.

Constant normal residues and the hypertangent residue at infinity are
logarithmic derivatives, not in-field derivatives, so they remain exactness
obstructions even though elementary integration can remove them. A
`TranscendentalExactDerivativeCertificate` retains the family reduction and
the lower exact result. If a nonconstant normal residue already proves
non-exactness, the dispatcher separately reduces the disjoint special and
lower-field part; this keeps the exact part maximal and the final obstruction
idempotent without discarding the residue certificate.

### Recursive primitive levels

For a primitive extension `K(t)` with `Dt = eta`, polynomial reduction needs
to decide lower-field coefficients modulo exact derivatives and the constant
direction `C*eta`. The implementation recursively solves exact derivatives in
the lower tower and uses a fixed, iterated Laurent coefficient at infinity as
the projection coordinate for the `eta` class.

This makes the decomposition deterministic and supports towers such as:

```text
Dx = 1
Dt = 1/x
Du = 1/(x*t)
Dv = 1/(x*t*u).
```

The recurrence is verified by planted exact derivatives, idempotence tests,
generated mixed cohomology classes, and full end-to-end examples.

### Hyperexponential levels

For `Dt = eta*t`, the reducer performs normal residue reduction followed by
Laurent reduction in `t`. Its coefficient equations are Risch differential
equations of the form

```text
Dy + a*y = b.
```

The end-to-end implementation solves these equations over `QQ(x)` and over
recursively primitive lower towers when `eta` is rational in the current
primitive generator. Recursive radical-logarithmic and exact-derivative
decisions first classify the proposed hyperexponential generator.

For a rational RDE coefficient `a`, Bronstein weak normalization constructs
`q` such that

```text
a_normal = a - D(q)/q
```

has no positive-integer residue at a normal finite factor. With `z = q*y`,
the equation becomes

```text
D(z) + a_normal*z = q*b.
```

Normal-denominator reduction and Rothstein SPDE then produce a polynomial
RDE. The full primitive degree bound includes both limited-integration
cancellation tests from Section 6.3. A positive-degree polynomial coefficient
uses the no-cancellation recurrence. For a coefficient in the next lower
field, the Section 6.6 cancellation algorithm applies:

1. if `a = D(z)/z`, transform `D(y) + a*y = b` into
   `D(z*y) = z*b` and invoke the exact-derivative solver;
2. otherwise solve `D(s) + a*s = lc(b)` recursively for each leading
   coefficient and subtract `s*t**degree(b)`;
3. stop with a certified lower-field obstruction when a coefficient equation
   has no solution.

The transformation in step 1 is taken only for an ordinary logarithmic
derivative (`minimal_power == 1`). A merely radical relation
`n*a = D(z)/z` does not justify replacing `D(y) + a*y` by an exact derivative
inside the current field. In step 2, each returned lower-field RDE is retained
in `lower_field_results`; this is both an audit trail and the recursive
certificate for the constructed polynomial solution.

The production recurrence is cross-checked with an independent bounded global
matrix ansatz, including generated success and nonsolution families. The
oracle accepts a caller-supplied top denominator, so weak-normalization cases
are checked without sharing the production normalizer or denominator theorem.

### Hyperexponential Laurent RDEs

At a hyperexponential level `K(t)` with `Dt = eta*t`, the implemented scalar
RDE branch accepts `a in K` and a Laurent right-hand side

```text
b = sum(b_j*t**j).
```

The operator `D + a` preserves each Laurent mode:

```text
(D + a)(y_j*t**j)
    = (D(y_j) + (a + j*eta)*y_j)*t**j.
```

The solver therefore invokes the general RDE dispatcher on the lower field
for each shifted coefficient `a + j*eta`. Successful modes are accumulated
in a partial solution; unsuccessful modes remain in an exact residual, and
every lower-field result is retained in the certificate.

This is a complete rational decision under the stated restrictions, not only
a Laurent ansatz. A rational solution cannot have a normal finite pole in the
top generator: differentiation would increase that pole order while
multiplication by `a in K` would not, and the Laurent right-hand side has no
such pole. The hyperexponential special factor `t` accounts precisely for the
allowed negative Laurent powers.

An independent bounded matrix oracle handles one hyperexponential level over
`QQ(x)`. It uses a single global bivariate ansatz and does not split modes or
call the production lower-field recurrence. Generated planted and
nonsolution examples compare the two implementations.

The next scalar checkpoint extends this terminal Laurent problem from
`a in K` to `a in K[t, t**-1]`. Its valuation bounds and leading/trailing
recurrences must exist before the rational frontend: weak normalization,
normal-denominator reduction, and SPDE ultimately reduce a general rational
coefficient to precisely that Laurent core. Once the core is complete, those
normal-factor stages complete the rational hyperexponential RDE.

### Hypertangent levels

For `Dt = eta*(1 + t**2)`, the irreducible real special polynomial is
`t**2 + 1`. The implementation retains this real quadratic instead of making
a complex-exponential rewrite the production representation.

The special reduction uses the Cayley correspondence as mathematics and as a
test oracle, while solving the resulting two-component coupled RDE over the
rational lower field. General lower fields and global real branch correction
remain unimplemented.

### Nonlinear monomials without special polynomials

When `deg_t(Dt) >= 2` and absence of special polynomials is certified, the
reducer combines:

1. normal Hermite reduction;
2. generalized Rothstein--Trager residue reduction;
3. residue-at-infinity reduction; and
4. Bronstein's polynomial degree reduction.

A nonzero canonical remainder is a Liouville obstruction under the recorded
hypotheses.

## Rational and RDE foundations

The rational base has two independent Hermite reducers:

- Horowitz--Ostrogradsky is the production implementation;
- recursive Hermite reduction is retained as an independent oracle.

The logarithmic part is likewise exercised through Rothstein--Trager and an
independent Lazard--Rioboo--Trager implementation. Complex algebraic residues
are encoded exactly with `RootSum`; a separate limited real presenter handles
rational denominators whose irreducible factors have degree at most two.

Polynomial Risch differential equations use Bronstein's recurrence in
production. A bounded matrix solver remains in the library as an independent
test oracle. Rational RDEs add denominator reduction and canonical
obstructions around that polynomial core.

## Algebraic extensions

An `AlgebraicExtension` stores a monic irreducible polynomial

```text
m(y) in K[y]
```

and uses SymPy's `FiniteExtension`. The derivative of `y` is derived from the
relation:

```text
Dy = -D_K(m)(y) / (partial m/partial y)(y).
```

For `m(y) = y**2 - x`, this gives `Dy = 1/(2*y)`.

The implemented algebraic Hermite reducer currently assumes a one-variable
rational parent field and an integral basis. It can:

- use a caller-supplied integral basis;
- construct the explicit basis for a recognized simple radical; and
- lazily refine an incomplete basis when one of Bronstein's three refinement
  certificates exposes a missing integral element.

This is a reduction component, not yet a complete algebraic integrator.
Missing pieces include algebraic residues and logarithmic reconstruction,
places and divisors, the Risch equation on an algebraic curve, and recursive
composition with transcendental levels.

The proposed real algebraic-over-tangent direction is kept in the
[algebraic-hypertangent research note](algebraic-hypertangent.md).

## Automatic tower construction

A future expression front end must:

1. collect transcendental and algebraic generator candidates;
2. construct their dependency graph;
3. recognize exact, logarithmic, exponential, tangent, rational-power, and
   algebraic dependencies;
4. choose and topologically order a minimal generator set;
5. rewrite the input into the top exact field; and
6. retain back-substitutions, assumptions, and branch information.

Examples it should consolidate include:

```text
exp(x/2), exp(x)          one generator, with exp(x) = t**2
exp(x), exp(x + 1)        one generator plus a constant factor
log(x), log(x**2)         differential dependence with branch-sensitive output
tan(log(x))               primitive level before hypertangent level
```

The builder must prove constant-field and special-polynomial hypotheses rather
than silently setting their status flags.

## Real presentation

The differential-field core proves identities locally and up to constants.
A separate presentation layer must eventually:

- combine conjugate logarithmic terms into real logarithms and angles;
- use `log(Abs(...))` where appropriate on real intervals;
- retain tangent generators when they provide the natural real field;
- avoid simplifying `atan(tan(u))` globally without domain information;
- track discontinuities and branch corrections separately from the algebraic
  derivative identity; and
- use exact real root isolation for general LRT output.

The current `integrate_rational_real` function is an early, deliberately
limited piece of this layer.

## Roadmap

Each checkpoint ends with focused tests, the complete test suite, and a
separate commit.

The latest transcendental checkpoint extends scalar RDE dispatch through the
complete lower-coefficient/Laurent hyperexponential branch, including
recursive hyperexponential levels and an independent matrix oracle. The next
scalar checkpoint adds certified support bounds and leading/trailing
recurrences for Laurent coefficients that depend on the top generator. Weak
normalization and normal-denominator reduction follow that terminal solver.
The current algebraic priority remains branch-place handling before the
residue-field and divisor stages.

### Completed foundation

- [x] Polynomial and rational integration.
- [x] Independent rational Hermite and logarithmic-part oracles.
- [x] Immutable tower, exact derivation, and structural monomial
  classification.
- [x] Algebraic extension validation and implicit differentiation.
- [x] Algebraic Hermite reduction with simple-radical and lazy integral bases.
- [x] Normal/special splitting and generalized monomial Hermite reduction.
- [x] Generalized residue criterion and constructive logarithmic extraction.
- [x] Nonlinear no-special reduction.
- [x] Rational exact and logarithmic derivative decisions.
- [x] Recursive radical logarithmic-derivative decisions over primitive
  towers.
- [x] Parametric logarithmic-derivative result and exact witness contract.
- [x] Bronstein Section 7.3 parametric logarithmic-derivative fast path over
  `QQ(x)`.
- [x] Complete rational parametric logarithmic-derivative fallback.
- [x] Complete recursive parametric logarithmic-derivative fallback over
  primitive towers.
- [x] Polynomial and rational Risch differential equations.
- [x] Positive-degree primitive-field RDEs with rational right-hand sides.
- [x] Primitive-field RDE cancellation for lower-field coefficients.
- [x] Primitive-field weak normalization and full degree bounds for rational
  coefficients.
- [x] Primitive, hyperexponential, and hypertangent family reducers.
- [x] Recursive elementary dispatcher.
- [x] Recursively stacked primitive integration.
- [x] Hyperexponential integration over the supported primitive-lower-field
  branch.
- [x] Checked primitive-extension construction with exact dependency
  witnesses and conservative unknown outcomes.
- [x] Hyperexponential and hypertangent exact-derivative dispatch over the
  supported lower towers.
- [x] Hyperexponential Laurent RDEs with lower-field coefficients, recursive
  mode certificates, and an independent bounded matrix oracle.

### Next transcendental steps

- [ ] Extend hyperexponential RDEs to Laurent coefficients using valuation
  bounds and certified leading/trailing recurrences.
- [ ] Add weak normalization, normal-denominator reduction, and SPDE for
  general rational hyperexponential RDE coefficients.
- [ ] Generalize coupled RDE solving over further supported lower tower
  families.
- [ ] Enable the remaining nested hyperexponential and hypertangent levels.
- [ ] Extend automatic dependence and constant-field checks to the remaining
  monomial families.
- [ ] Add the missing general-special-polynomial branch for nonlinear
  monomials.

### Algebraic integration

Algebraic work has already begun: validation, implicit differentiation,
simple-radical and lazy integral bases, and algebraic Hermite reduction over
`QQ(x)` are implemented. For a Hermite-reduced differential with simple
finite poles, Trager first computes a polynomial whose roots are rational
multiples of all residues. Its splitting field is the minimal algebraic
constant extension needed to express the logarithmic coefficients. This
resultant computation comes before explicit divisor construction and can be
performed over the original coefficient field.

For a presentation `F(x, y) = 0` and a rationalized differential
`G(x, y)/D(x) dx`, Trager's unbranched formula is

```text
R(Z) = resultant_x(
           resultant_y(Z*D'(x) - G(x, y), F(x, y)),
           D(x)
       ).
```

The roots of `R` are the finite residues, with multiplicities from conjugate
places. At branch places the same computation contains rational branch-index
scalings, so the model check is part of the certificate rather than an
implicit genericity assumption.

- [x] Compute and certify Trager's algebraic residue polynomial for
  Hermite-reduced differentials with simple, unbranched finite poles over a
  polynomial model monic in the algebraic generator.
- [ ] Detect branch places at the poles and either prove that their residues
  vanish, as in the supported simple-radical case, or change/extend the model
  before applying the unbranched resultant formula.
- [ ] Compute a rational basis for the algebraic residues in their minimal
  constant extension.
- [ ] Construct the divisor associated with each residue-basis coefficient
  and solve the required principal/torsion divisor problem.
- [ ] Represent finite and infinite places, residue fields, ramification
  indices, uniformizers, and exact valuations needed by the divisor and
  infinity algorithms.
- [ ] Implement general divisor operations and exactness testing for
  differentials of the second kind.
- [ ] Solve the Risch differential equation on an algebraic curve.
- [ ] Add general integral-basis computation where lazy refinement is
  insufficient.
- [ ] Generalize algebraic Hermite reduction to monomial parent fields.
- [ ] Compose algebraic levels below, above, and between transcendental
  levels.
- [ ] Investigate the direct real algebraic-over-hypertangent branch.

### Front end and upstreaming

- [ ] Build and order towers from SymPy expressions.
- [ ] Preserve named presentations and branch assumptions.
- [ ] Complete general real logarithm and angle presentation.
- [ ] Stabilize public APIs around SymPy conventions.
- [ ] Split mature components into reviewable upstream proposals.

## Testing strategy

Every reducer is tested at three levels:

1. exact reconstruction and canonical postconditions;
2. adversarial and generated families, including rejected preconditions; and
3. an independent implementation or mathematical oracle where practical.

Negative certificates receive the same scrutiny as constructive answers.
Tests must distinguish an actual structure-theorem obstruction from
`NotImplementedError`.

See the [testing contract](testing.md) for the full gate and commands.

## Open design decisions

The following decisions are deferred until their first complete consumer:

- the constant-field oracle interface used by automatic construction;
- the representation of algebraic places and divisor classes;
- the internal real angle and branch-correction representation;
- whether future reducer dispatch uses a registry or explicit functions; and
- the boundary between SymPy domain infrastructure and integration-specific
  algebraic-curve code.

These are deliberate extension points, not descriptions of APIs that already
exist.

## Primary references

- Clemens G. Raab and Michael F. Singer, editors,
  *Integration in Finite Terms: Fundamental Sources*, Springer, 2022. A
  user-supplied local copy is indexed in
  [`references/README.md`](../references/README.md); it reprints Rosenlicht,
  Ritt, Risch's algebraic-operations paper, and Trager's thesis with modern
  commentary.
- Manuel Bronstein, *Symbolic Integration I: Transcendental Functions*.
- Manuel Bronstein,
  [*Symbolic Integration Tutorial*](https://www-sop.inria.fr/cafe/Manuel.Bronstein/publications/issac98.pdf),
  ISSAC 1998.
- Manuel Bronstein,
  [*The Lazy Hermite Reduction*](https://inria.hal.science/inria-00073121),
  INRIA RR-3562, 1998.
- Robert Risch, *The Problem of Integration in Finite Terms*.
- Daniel Lazard and Renaud Rioboo, *Integration of Rational Functions:
  Rational Computation of the Logarithmic Part*.
- James Davenport, work on integration over algebraic curves.
- David Jeffrey and collaborators, work on continuous real trigonometric
  antiderivatives and unwinding numbers.
- Clemens Raab, *Generalization of Risch's Algorithm to Special Functions*.
