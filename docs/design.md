# Differential-tower design and implementation roadmap

This document records the working design for `sympy-risch`. It is intended to
preserve the mathematical and architectural decisions behind the implementation
and to define small, reversible development milestones.

The primary target is elementary integration by the Risch algorithm, including
real tangent/arctangent handling and algebraic function fields. The design also
leaves explicit extension points for selected non-elementary and differential-
algebraic function families.

## Current implementation

The repository currently provides:

- immutable differential towers with explicit level-bound derivations;
- monomial extensions ``K(t)`` with ``Dt`` in ``K[t]``;
- recursive coefficient, polynomial, and fraction differentiation over native
  SymPy domains;
- validated finite algebraic extensions with implicitly derived generator
  derivatives;
- algebraic Hermite reduction over `QQ(x)` with a caller-supplied integral
  basis;
- polynomial integration over `QQ[x]`;
- Horowitz--Ostrogradsky rational Hermite reduction;
- classical recursive rational Hermite reduction as an independent check;
- Rothstein--Trager residue polynomials;
- an independent Lazard--Rioboo--Trager logarithmic-part implementation;
- complex logarithmic rational integration through exact `RootSum` objects;
- real rational output for denominators whose factors over `QQ` have degree at
  most two;
- example-based and Hypothesis property tests.

The current checkpoints are:

```text
35bb248  Establish polynomial and rational integration baseline
0b0eb32  Implement Lazard-Rioboo-Trager logarithmic reduction
```

## Design principles

1. Use SymPy's exact `Domain`, `Poly`, fraction-field, and finite-extension
   elements directly. Do not introduce a parallel wrapper for field elements.
2. Separate expression parsing and presentation from differential-field
   computation.
3. Represent the tower and its levels immutably. Recursive algorithms must use
   an explicit level rather than mutating a shared "current level".
4. Named functions record useful construction and presentation semantics, but
   reduction algorithms dispatch on mathematical extension families.
5. Preserve real forms where possible. Do not make complex-exponential
   rewriting the only representation of trigonometric extensions.
6. Treat branch-sensitive equality separately from equality of derivatives.
7. Add each independently working milestone as a separate git commit.

## Three-layer architecture

```text
SymPy expression
      |
      v
TowerBuilder and presentation layer
      |  discovers generators, dependencies, and ordering
      v
DifferentialTower and exact SymPy domain elements
      |  differential-field algorithms
      v
Integration and reduction algorithms
      |
      v
Real/complex presentation and back-substitution
      |
      v
SymPy expression
```

The mathematical core must work with manually constructed towers before the
automatic `TowerBuilder` is implemented.

## Differential tower

The central container is an immutable tower:

```python
@dataclass(frozen=True)
class DifferentialTower:
    constants: Domain
    levels: tuple[DifferentialExtension, ...]

    def field(self, level: int | None = None) -> Domain: ...
    def derivation(self, level: int | None = None) -> Derivation: ...
    def extend(self, extension: DifferentialExtension) -> DifferentialTower: ...
```

The first level can represent `x`, with `Dx = 1`. Thus

```text
QQ
  < QQ(x)
  < QQ(x)(t1)
  < ...
```

uses the same level mechanism throughout.

Consecutive transcendental levels are stored physically in a flat native
SymPy fraction field. Thus the mathematical level `QQ(x)(t)` is represented by
`QQ.frac_field(x, t)`, while the tower metadata and each derivative polynomial
retain the coefficient-field boundary between `x` and `t`. This avoids known
coercion failures during normalization of deeply nested SymPy fraction fields
without introducing wrapper elements.

Algorithms operate at an explicit level. A lower-field operation receives a
lower-level `Derivation`; the tower itself never needs `increment_level()` or
`decrement_level()` state.

## Extension hierarchy

The hierarchy is open rather than a closed enumeration of named functions:

```text
DifferentialExtension
|-- TranscendentalExtension
|   |-- MonomialExtension                 Dt in K[t]
|   |   |-- PrimitiveExtension            Dt in K
|   |   |   |-- LogExtension
|   |   |   `-- NamedPrimitiveExtension   atan, erf, Ei, li, ...
|   |   |-- HyperexponentialExtension     Dt/t in K
|   |   |   `-- ExpExtension
|   |   |-- HypertangentExtension         Dt/(1+t^2) in K
|   |   |   `-- TanExtension
|   |   `-- PolynomialExtension           general Dt in K[t]
|   `-- RationalDifferentialExtension     Dt in K(t)
|-- AlgebraicExtension                    m(t) = 0
`-- DifferentialAlgebraicExtension        Q(t, Dt, ...) = 0
```

The elementary Risch implementation initially needs monomial and algebraic
extensions. `RationalDifferentialExtension` and
`DifferentialAlgebraicExtension` reserve clean boundaries for later work such
as Lambert-W-like and Weierstrass-like extensions.

### Common interface

Each level exposes at least:

```python
class DifferentialExtension(Protocol):
    generator: Symbol
    parent_field: Domain
    field: Domain

    @property
    def generator_derivative(self): ...
```

The concrete level is responsible for validating that its derivative and
defining relations belong to the expected domains.

### Monomial extensions

A monomial extension stores `Dt` as a `Poly` in `t` over the parent field. Its
mathematical family is determined exactly by the shape of that polynomial:

| Derivative law | Family |
| --- | --- |
| `Dt` is in `K` | primitive |
| `Dt = eta*t`, `eta` in `K` | hyperexponential |
| `Dt = eta*(1 + t**2)`, `eta` in `K` | hypertangent |
| otherwise `Dt` in `K[t]` | general polynomial/monomial |

Generic construction from a derivative law produces the appropriate
mathematical parent class. Named constructors additionally preserve the
original function.

For denominator reduction, square-free factors are classified by the total
tower derivative: `p` is normal when `gcd(p, Dp) = 1` and special when
`p | Dp`. The normal/special decomposition retains each factor's ordinary
square-free multiplicity and is independent of the named function provenance.
The monomial Hermite reducer then lowers repeated normal factors one at a time,
leaving a square-free normal denominator while retaining polynomial terms and
special factors for the family-specific reduction.

### Named elementary extensions

Named classes construct their derivative from their argument using the parent
derivation:

```python
LogExtension.from_argument(parent, t, u)
# Dt = Du/u

ExpExtension.from_exponent(parent, t, u)
# Dt = Du*t

TanExtension.from_argument(parent, t, u)
# Dt = Du*(1 + t**2)
```

Examples include:

```text
t = exp(x + 1)       -> Dt = t
t = log(x + 1)       -> Dt = 1/(x + 1)
t = tan(log(x))      -> Dt = (1 + t^2)/x
```

Named classes determine construction and preferred presentation. Their
mathematical parent classes determine which reduction algorithm applies.

### Algebraic extensions

An algebraic level stores a monic irreducible minimal polynomial

```text
m(t) in K[t]
```

and uses SymPy's `FiniteExtension` representation. The constructor must check
irreducibility and square-freeness because `FiniteExtension` does not itself
guarantee irreducibility.

The derivative of the generator is derived, not supplied independently:

```text
Dt = -D_K(m)(t) / partial_t(m)(t).
```

For `m(t) = t**2 - x`, this gives `Dt = 1/(2*t)`.

Algebraic constants should normally be adjoined to the constant domain rather
than represented as zero-derivative tower levels.

The first algebraic Hermite reducer operates over a one-variable rational
parent field and takes an explicit `IntegralBasis`. It verifies that the given
elements are a field basis, but the caller currently asserts that they span
the full integral closure. Lazy Hermite reduction can refine an incomplete
basis when one of Bronstein's three certificates exposes a missing integral
element; automatic full integral-basis computation remains a separate
milestone. `IntegralBasis.power_basis()` is valid only when the power basis is
already known to be integral and complete.

Simple radical extensions `y**n = a`, with `a` in the one-variable rational
parent field, have a specialized basis constructor. It removes `n`-th-power
factors from the radicand and applies the explicit integral-basis formula from
Bronstein's tutorial. This supplies the existing algebraic Hermite reducer
with a complete non-power basis without invoking general integral-basis or
lazy-refinement machinery.

## Derivation

A level-bound derivation is a callable object:

```python
@dataclass(frozen=True)
class Derivation:
    tower: DifferentialTower
    level: int

    def __call__(self, element): ...
    def polynomial(self, polynomial: Poly) -> Poly: ...
    def fraction(self, numerator: Poly, denominator: Poly): ...
    def coefficients(self, polynomial: Poly) -> Poly: ...
```

For

```text
p(t) = sum(a_i*t**i),  a_i in K,
```

the full derivative is

```text
D(p) = sum(D(a_i)*t**i) + partial_t(p)*Dt.
```

Coefficient differentiation omits the second term. Both operations are needed
by Risch reduction and by differentiation in algebraic extensions.

For a fraction:

```text
D(p/q) = (D(p)*q - p*D(q))/q**2.
```

The derivation must operate on existing SymPy domain elements rather than on a
new `DifferentialElement` wrapper.

## Classification versus named-function recognition

The derivative shape determines the mathematical extension family. Recovering
a named presentation requires solving an equation in the lower field.

| Family | Given | Named presentation requires |
| --- | --- | --- |
| primitive | `Dt = eta` | find `u` with `Du/u = eta` for `log(u)` |
| hyperexponential | `Dt = eta*t` | find `u` with `Du = eta` for `exp(u)` |
| hypertangent | `Dt = eta*(1+t**2)` | find `u` with `Du = eta` for `tan(u)` |

The APIs should remain separate:

```python
classify_extension(extension) -> ExtensionFamily
recognize_presentation(extension, tower) -> Presentation | None
```

### Ambiguous constants

Derivative laws recover named functions only up to constants:

```text
Dt = Du/u             -> t = log(u) + c
Dt = Du*t             -> t = c*exp(u)
Dt = Du*(1 + t**2)    -> t = tan(u + c)
```

The tangent phase corresponds to a constant Mobius transformation:

```text
tan(u + c) = (tan(u) + tan(c))/(1 - tan(u)*tan(c)).
```

Presentations inferred from derivative laws must record this ambiguity. A
presentation obtained from an input expression can retain the exact original
argument.

### Recognition subproblems

The recognition layer needs:

```python
exact_derivative(eta, tower)
# Find u in K such that Du = eta.

logarithmic_derivative(eta, tower)
# Find nonzero u in K such that Du/u = eta.

inverse_tangent_derivative(eta, tower)
# Find u in K such that Du/(1 + u**2) = eta.
```

The first two are standard recursive Risch subproblems. The inverse-tangent
problem is nonlinear and can be added after the primitive and exponential
machinery works.

### Dependency and constant-field checks

Candidate generators must be tested for dependence before being added.

For a primitive candidate `Dt = eta`:

- if `eta = Du` for `u` in `K`, then `t - u` is constant and no new generator
  is needed.

For a hyperexponential candidate `Dt = eta*t`:

- if `eta = Du/u`, then `t/u` is constant and the candidate belongs to the
  lower field up to a constant;
- if `n*eta = Du/u`, then `t**n/u` is constant and the candidate is algebraic
  over the lower field.

This detects cases such as:

```text
Dt = t/x                  -> t = c*x
exp(log(x)/2)             -> sqrt(x)
```

The constant field will eventually need effective zero tests and algebraic-
relation tests. Initially it is a SymPy `Domain`, beginning with `QQ`.
Unknown named constants can be represented as zero-derivative generators of
the constant domain, not as differential tower levels.

## Tower construction

`TowerBuilder` is a later front-end component:

```python
class TowerBuilder:
    def build(
        self,
        expression: Expr,
        variable: Symbol,
    ) -> tuple[Element, DifferentialTower, TowerPresentation]: ...
```

Its responsibilities are:

1. collect candidate function and algebraic generators;
2. construct a dependency graph;
3. detect exact, logarithmic, integer, rational-power, and algebraic
   dependencies;
4. choose a minimal generator set;
5. topologically order the generators;
6. rewrite the input into the top field;
7. retain exact back-substitutions and branch assumptions.

Examples the builder must consolidate include:

```text
exp(x/2), exp(x)          -> one generator t = exp(x/2), exp(x) = t**2
exp(x), exp(x + 1)        -> one generator plus a constant factor
log(x), log(x**2)         -> differential dependence, branch-sensitive output
tan(log(x))               -> log level before tangent level
```

The core algorithms will first be tested using manually constructed towers so
that builder errors cannot obscure differential-algebra errors.

## Real output and branches

The differential-field core proves derivative identities, which are local and
hold up to constants. The output layer decides global real presentation.

The real-output goals are:

- combine conjugate logarithmic terms into real logarithms and angle terms;
- use `log(Abs(...))` around real poles;
- retain tangent generators rather than forcing complex exponential rewrites;
- use an internal two-argument angle representation when `atan(y/x)` would
  introduce a spurious discontinuity;
- introduce `Piecewise` or explicit branch corrections only in the
  presentation layer;
- never simplify `atan(tan(u))` globally to `u` without domain information.

The current quadratic real rational integrator is the first limited version of
this layer. General real LRT output will require real algebraic root isolation
and conjugate-residue pairing.

## Extension points beyond elementary Risch

The initial implementation does not promise a decision procedure for arbitrary
special functions. The hierarchy leaves room for selected families with their
own structure theorems and reducers:

- named primitive extensions such as `erf`, `Ei`, and `li`;
- rational first-order extensions such as Lambert-W-like generators with
  `Dt` in `K(t)`;
- algebraic-curve function fields;
- coupled differential-algebraic extensions such as Weierstrass-like fields;
- higher-order differential systems, which may eventually require modules or
  differential ideals rather than simple one-generator levels.

Reducers for these families should be registered by mathematical extension
type. They must not require adding another named-function branch to every core
algorithm.

## Implementation roadmap

Each numbered milestone should end with focused tests, the full test suite, and
its own commit.

### 1. Differential tower foundation

- [ ] Add the open extension hierarchy through algebraic-extension descriptors.
- [x] Implement immutable `DifferentialTower` construction.
- [x] Implement level-bound coefficient, polynomial, and quotient derivations.
- [x] Use SymPy fraction fields directly.
- [x] Add classification of primitive, hyperexponential, hypertangent, and general
  polynomial derivative laws.
- [ ] Add named `LogExtension`, `ExpExtension`, and `TanExtension` factories.
- [x] Test manually constructed `x`, `log(x)`, and nested monomial levels.
- [ ] Test named `exp(x + 1)` and `tan(log(x))` levels.
- [x] Property-test linearity, Leibniz, quotient, and chain rules.

### 2. Rational-field recognition base cases

- Decide whether a rational function is the derivative of another rational
  function using Hermite reduction.
- Solve the rational logarithmic-derivative problem using factorization and
  residues.
- Detect dependent primitive and hyperexponential candidates.
- Recognize rational `log`, `exp`, and elementary `atan` presentations where
  possible.
- Property-test reconstructed derivative identities and negative cases.

### 3. Shared monomial and primitive-extension reduction

- [x] Generalize Hermite reduction from `QQ(x)` to `K(t)` using the tower
  derivation.
- [x] Implement normal/special denominator splitting using `gcd(p, Dp)`.
- [x] Implement Bronstein Section 5.4 polynomial reduction for nonlinear
  monomials, including the degree bound `deg(r) < deg(Dt)`.
- [x] Apply Liouville's theorem at normal poles through Bronstein's generalized
  Rothstein--Trager resultant and coefficient derivation `kappa`.
- [x] Return the square-free factors `R_i`, constant parts
  `gcd(R_i, kappa(R_i))`, and explicit nonconstant-residue obstructions.
- [x] Extract accepted constant-residue logarithmic terms with a generalized
  LRT subresultant computation, and reconstruct their derivative by a
  quotient-ring trace without adjoining algebraic constants explicitly.
- [x] Reduce the Section 5.7 residue at infinity for nonlinear monomials:
  certify nonconstant infinity residues, extract a logarithm using a known
  special polynomial, and explicitly defer the no-known-special case.
- Implement polynomial reduction in a primitive extension.
- Implement the recursive in-field exact-derivative problem.
- Prove nonexistence when the reduced obstruction is nonzero.

### 4. Hyperexponential extensions

- Implement normal and special reduction for `Dt = eta*t`.
- Implement the Risch differential equation `Dy + a*y = b` recursively.
- Detect logarithmic derivatives and radical/algebraic dependencies.
- Integrate examples such as `x*exp(x)` and prove standard negative cases.

### 5. Hypertangent and real inverse-tangent extensions

- Implement reduction for `Dt = eta*(1 + t**2)`.
- Implement the coupled lower-field differential equations used in the
  hypertangent case.
- Add `atan`/`atan2` recognition and real presentation.
- Add tangent half-angle rationalization as a front-end transformation.
- Test branch-sensitive results by intervals as well as by differentiation.

### 6. Automatic tower builder

- Discover log, exp, tangent, power, and algebraic candidates in expressions.
- Build and topologically sort the dependency graph.
- Consolidate rationally related exponentials and dependent primitives.
- Record exact substitutions and real assumptions.
- Keep construction deterministic.

### 7. Algebraic differential fields

- [x] Construct validated SymPy `FiniteExtension` domains over lower function
  fields.
- [x] Derive and test algebraic generator derivatives.
- [x] Implement algebraic Hermite reduction over `QQ(x)` for a supplied
  integral basis.
- [x] Recognize simple radical extensions and construct their explicit
  integral bases.
- [x] Lazily refine a basis when the Hermite-system solution denominator
  intersects the repeated factor (Bronstein's second refinement theorem).
- [x] Refine a basis from a primitive kernel relation when the Hermite system
  is dependent (Bronstein's first refinement theorem).
- [x] Refine a basis when a basis derivative has a nonsquarefree denominator
  (Bronstein's third refinement theorem).
- [ ] Compute full integral bases when later algorithms require them.
- [ ] Generalize algebraic Hermite reduction to monomial parent fields.
- [ ] Implement places, residue computation, and the algebraic Risch equation.
- [ ] Support mixed towers with algebraic levels below and between transcendental
  levels.

### 8. Complete real rational and algebraic presentation

- Realify general LRT output using real algebraic root isolation.
- Pair conjugate residues exactly.
- Remove spurious `atan` singularities using angle-polynomial
  transformations.
- Track intervals and branch corrections separately from the algebraic
  antiderivative.

### 9. Upstream integration

- Compare each core operation with the corresponding SymPy interfaces.
- Stabilize APIs and naming around SymPy domain conventions.
- Split changes into reviewable upstream patches.
- Retain this package as an experimentation and compatibility test bed until
  the relevant components are accepted upstream.

### 10. Optional extended families

- Add extension families only with a clear defining differential structure,
  applicable structure theorem, reduction algorithm, and test oracle.
- Prefer a reducer registry over central named-function dispatch.

## Testing strategy

Every layer should have three kinds of tests:

1. Hand-derived examples that expose the mathematical construction.
2. Algebraic invariants, such as derivative reconstruction, square-free
   remainders, field membership, and minimal-polynomial reduction.
3. Hypothesis-generated examples over small exact domains.

Property tests should generate mathematical objects satisfying the required
preconditions rather than generate arbitrary expressions and discard most of
them with `assume`.

For integration results, the primary oracle is exact differentiation in the
same tower. Comparison with SymPy is useful as a differential test but should
not require identical presentation or rely on the implementation being
replaced.

## Commit discipline

- Keep each mathematical capability in its own commit.
- Commit only after focused and full tests pass.
- Do not combine builder, representation, and reduction changes in one commit.
- Prefer additive migrations; avoid rewriting working rational algorithms while
  building the tower layer.
- Record any design change in this document in the same commit that introduces
  it.

## Open decisions

The following choices are intentionally deferred until their first consumer:

- the exact constant-field oracle interface;
- whether reducer dispatch uses `singledispatch`, an explicit registry, or a
  method on mathematical extension classes;
- the internal real angle representation;
- the exact public result type for elementary, nonelementary, and partially
  reduced integrals;
- how much of algebraic-curve infrastructure belongs in SymPy domains versus
  the integration layer.

These are extension points, not reasons to delay the differential-tower
foundation.

## Primary references

- Manuel Bronstein, *Symbolic Integration I: Transcendental Functions*.
- Manuel Bronstein, [*Symbolic Integration Tutorial*](https://www-sop.inria.fr/cafe/Manuel.Bronstein/publications/issac98.pdf),
  ISSAC 1998.
- Manuel Bronstein, [*The Lazy Hermite Reduction*](https://inria.hal.science/inria-00073121),
  INRIA RR-3562, 1998.
- Robert Risch, *The Problem of Integration in Finite Terms*.
- Daniel Lazard and Renaud Rioboo, *Integration of Rational Functions:
  Rational Computation of the Logarithmic Part*.
- James Davenport, work on integration over algebraic curves.
- David Jeffrey and collaborators, work on continuous real trigonometric
  antiderivatives and unwinding numbers.
- Clemens Raab, *Generalization of Risch's Algorithm to Special Functions*.
