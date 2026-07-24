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
- a hyperexponential level over a primitive lower field when its logarithmic
  derivative is polynomial of positive degree in the top primitive
  generator; and
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
- radical logarithmic-derivative signatures over one primitive level above
  `QQ(x)`, including complete repeated principal parts;
- rational and polynomial Risch differential equations;
- normal-denominator, SPDE, and polynomial RDE reduction over a primitive
  field for positive-degree polynomial coefficients;
- a recurrence implementation for polynomial RDEs and an independent matrix
  implementation used as a test oracle;
- rational coupled RDEs for the real hypertangent case;
- dependency tests for hyperexponential and hypertangent generators;
- an exact result and verification contract for Bronstein's parametric
  logarithmic-derivative equation, distinguishing a proved negative result
  from an inconclusive heuristic and an unsupported field;
- Bronstein's Section 7.3 degree and denominator heuristic for that equation
  over `QQ(x)`, with exact radical-logarithmic verification;
- a complete rational fallback that solves the heuristic's remaining constant
  and non-logarithmic residue constraints over `QQ`;
- validated finite algebraic extensions and implicit differentiation;
- algebraic Hermite reduction over `QQ(x)` with explicit or lazily refined
  integral bases; and
- simple-radical recognition and integral-basis construction.

The required exact ground-type backend is `python-flint`.

Recent implementation checkpoints, newest first, are:

```text
25e7c2c  Reduce nested primitive extensions
820f9bd  Add recursive elementary integration dispatcher
b5d3fef  Solve rational hypertangent extensions
023e465  Reduce hypertangent monomial extensions
d934cb3  Reduce hyperexponential monomial extensions
c09c9b9  Solve rational Risch differential equations
92ca519  Require python-flint arithmetic backend
1ade14d  Use Bronstein recurrence for polynomial RDEs
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
`ENLARGED`, or `UNKNOWN`. Manual towers must supply the evidence. Algorithms
that rely on the condition reject unproved cases.

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
verified solution or a proved negative result. Recursive primitive lower
fields currently return `UNSUPPORTED`; implementing their Chapter 9
fundamental-signature systems is the next checkpoint.

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

The end-to-end implementation solves these equations over the direct rational
lower field `QQ(x)`. It also handles a primitive lower field when `eta` is
polynomial of positive degree in the top primitive generator. In that case:

1. positive degree certifies that no positive multiple of `eta` is a
   logarithmic derivative in the primitive field;
2. normal-denominator reduction bounds rational solutions;
3. Rothstein SPDE reduces the numerator equation; and
4. a leading recurrence solves or obstructs the polynomial RDE.

The production recurrence is cross-checked with an independent bounded global
matrix ansatz over one primitive level above `QQ(x)`.

The cancellation case where the RDE coefficient lies in the lower field,
coefficients with top-generator poles, and other general lower differential
fields remain roadmap items.

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
- [x] Radical logarithmic-derivative signatures over one primitive level
  above `QQ(x)`.
- [x] Parametric logarithmic-derivative result and exact witness contract.
- [x] Bronstein Section 7.3 parametric logarithmic-derivative fast path over
  `QQ(x)`.
- [x] Complete rational parametric logarithmic-derivative fallback.
- [x] Polynomial and rational Risch differential equations.
- [x] Positive-degree primitive-field RDEs with rational right-hand sides.
- [x] Primitive, hyperexponential, and hypertangent family reducers.
- [x] Recursive elementary dispatcher.
- [x] Recursively stacked primitive integration.
- [x] Hyperexponential integration over the supported primitive-lower-field
  branch.

### Next transcendental steps

- [ ] Use the primitive signatures in the Chapter 9 parametric
  logarithmic-derivative fallback, first for one level and then recursively.
- [ ] Implement primitive RDE cancellation when the coefficient belongs to
  the next lower field.
- [ ] Add weak normalization for primitive RDE coefficients with
  top-generator poles.
- [ ] Generalize exact-derivative and RDE solving over further supported lower
  tower families.
- [ ] Enable the remaining nested hyperexponential and hypertangent levels.
- [ ] Complete automatic dependence checks needed to justify constant-field
  preservation.
- [ ] Add the missing general-special-polynomial branch for nonlinear
  monomials.

### Algebraic integration

- [ ] Generalize algebraic Hermite reduction to monomial parent fields.
- [ ] Compute algebraic residues and the logarithmic part.
- [ ] Implement local places, valuations, and divisor operations.
- [ ] Solve the Risch differential equation on an algebraic curve.
- [ ] Compose algebraic levels below, above, and between transcendental
  levels.
- [ ] Add general integral-basis computation where lazy refinement is
  insufficient.
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
