# Algebraic function-field infrastructure

This document describes the exact algebraic-curve layer and its first
integration-facing consumer. It is software infrastructure, not a tutorial and
not a complete algebraic integrator.

The global layer lives in `src/sympy_risch/algebraic_geometry/`; its first
consumer is `src/sympy_risch/algebraic/logarithmic.py`. It reuses SymPy for
exact finite extensions, polynomial factorization, and `DomainMatrix`
arithmetic. It adds the function-field structures that SymPy does not currently
provide as a coherent exact API:

- certified polynomial orders and maximal orders;
- full-rank fractional ideals;
- ramified and unramified finite places;
- every place at infinity;
- exact valuations, residue maps, and uniformizers;
- finite local Laurent expansions;
- differents, divisors, principal divisors, and differentials.

It deliberately does **not** implement the general later Risch components:

- compatible algebraic constant descent through arbitrary composita;
- general principal- or torsion-divisor solving;
- complete exactness decisions for arbitrary second-kind differentials; or
- a general algebraic-curve Risch differential-equation solver.

Those algorithms can now consume a tested algebraic substrate rather than
embedding their own curve arithmetic.

## Coefficient contract

The current global model is one finite separable extension

```text
L = C(x)[y]/(m(y))
```

with these requirements:

1. `C` is an exact characteristic-zero field.
2. `m` is monic and irreducible over `C(x)`.
3. SymPy can factor the univariate residue polynomials created over `C` and
   its exact algebraic extensions.
4. The rational base has one generator `x`.

`QQ` and SymPy algebraic number fields are the primary tested constant
domains. Positive characteristic is rejected by maximal-order construction:
the trace-radical identity used here needs a separate inseparability and wild
ramification treatment there.

Order, ideal, and place arithmetic remains valid for a constant extension
such as `QQ(x)(sqrt(2))`. The reported genus assumes that `C` is the exact
constant field of `L`; the genus invariant rejects inconsistent negative
results rather than silently treating a larger constant field as exact.

## Constant-field boundary

Before an algebraic Risch decision uses a curve invariant, it must establish
that the displayed `C` is actually `Const(L)`. The public
`analyze_algebraic_constant_field(extension, derivation)` API is the first
theorem-backed gate for that question. It is intentionally narrow: the active
tower must be one algebraic extension directly over the standard rational
differential field `C(x)`, with `Dx = 1`. A relation whose coefficients all
lie in `C` is handled directly; otherwise the extension must be a recognized
simple radical.

After the existing radical normalization, the supported input has the form

```text
L = C(x)(z),    z**n = c * product(f_i(x)**e_i),
```

where the `f_i` are distinct nonconstant factors. Let
`d = gcd(n, e_1, ..., e_r)`. The analyzer has only these three outcomes:

- `PRESERVED` when `d = 1`. The binomial is then irreducible after extending
  `C` algebraically, so the function field is geometrically integral and
  `C` is algebraically closed in `L`.
- `REBASING_REQUIRED` when the relation is already constant, or when `d > 1`
  and the new constant can be represented by SymPy exactly. In the radical
  case the witness

  ```text
  w = z**(n/d) / product(f_i(x)**(e_i/d))
  ```

  satisfies `Dw = 0` and `w**d = c`. The result carries an exact
  `AlgebraicConstantEmbedding` of the proposed algebraic constant field into
  `L`; its `embed` method maps that field's primitive generator to `w`.
- `UNSUPPORTED` for every other presentation or unavailable exact field
  operation. It is never interpreted as preservation.

The embedding is deliberately not a call to `rebase_derivation_constants`.
Once the new constants contain `w`, the old algebraic minimal polynomial can
factor or its degree can drop. A correct rebase must construct a new
function-field presentation and explicit two-way maps; that is a later
constant-normalization checkpoint. The current certificate is enough to
let an algebraic integration entry point reject the presentation or hand it
off to a future rebaser before it invokes the place and divisor layer.
Low-level
`AlgebraicFunctionField` construction remains intentionally neutral and does
not inspect this analysis itself.

## Post-Hermite all-place residues

`collect_algebraic_all_place_residues(integrand, derivation, basis)` is the
integration-facing bridge from algebraic Hermite reduction to the global
place API. It first returns the exact decomposition

```text
integrand = D(exact_part) + remainder,
```

using the supplied basis (or its lazily refined replacement). It then runs
the constant-field analysis for a nonzero remainder. A zero remainder is an
`EXACT` result immediately: it is a complete exactness certificate with empty
pole support, so it needs neither a constant-field decision nor a global
model. Only a nonzero `PRESERVED` result reaches
`AlgebraicFunctionField`; `REBASING_REQUIRED` and `UNSUPPORTED` results keep
the verified Hermite data but return no model and no partial list of places.

For a nonzero collected remainder, the collector computes the complete
divisor of `remainder dx` and takes its negative support. Every record is an
`AlgebraicLocalDifferentialResidue` containing:

- the canonical finite or infinite place;
- its positive pole order;
- the place's valuation-one uniformizer; and
- the exact local residue in that place's residue field.

This route naturally includes ramified finite places and infinity. In
particular, it does not substitute a base polynomial for a branch-place
uniformizer. A pole with zero local residue is retained: later second-kind
exactness and Riemann--Roch algorithms need to distinguish it from the
absence of a pole.

`AlgebraicAllPlaceResidueResult` checks reconstruction and square-freeness of
the Hermite remainder in its retained basis. For a nonzero `COLLECTED` result,
it also checks the full differential divisor and the one-to-one correspondence
between negative divisor support and local records; each residue value must
belong to its record's exact residue field. It is a residue collection
boundary. The separate first logarithmic consumer below can use the special
case where every local residue is already rational; the collector itself does
not choose embeddings between non-rational local residue fields or decide
whether a nonzero zero-residue differential is exact.

## Closed-place residue profiles

`build_algebraic_residue_profile(collection, Z)` turns a complete all-place
collection into the canonical residue polynomial available *before* any
constant-field base change. For every pole place `P`, it builds the exact
matrix of multiplication by the local residue on `k(P)` over `C`, using that
place's native constant-coordinate basis. The resulting factor is

```text
charpoly_C(m_res_P) = Norm_{k(P)/C}(Z - res_P),
```

and their product is the closed-place profile `R_omega(Z)`. It contains every
finite, ramified, and infinity pole. A zero residue contributes its expected
power of `Z`, so the profile retains the information needed later to separate
logarithmic from second-kind work. `EXACT` gives the unit polynomial; an
unsupported all-place collection, or unavailable local coordinate operation,
gives an all-or-nothing `UNSUPPORTED` result rather than a partial product.
A unit profile from a nonzero `PROFILED` collection is not an exactness proof:
a holomorphic or second-kind differential can have no residues at all. The
status retains that distinction.

The profile generalizes the finite unramified polynomial computed by Trager's
plane-model resultant. Its
`finite_unramified_simple_pole_polynomial` property is the directly comparable
slice and is tested against that independent implementation. The full profile
also covers branch and infinity cases where the plane-model shortcut is
correctly inapplicable.

It is deliberately **not** itself a common algebraic constant field or a
logarithmic residue basis. The native coordinate systems of two closed places
do not choose compatible conjugates in a common algebraic closure. A first
narrow continuation now accepts a caller-selected factor and constructs the
compatible `QQ -> QQ(alpha)` target, but profile-driven factor planning and
general common-residue-field descent remain later work. This distinction
prevents a locally correct but globally inconsistent logarithmic
reconstruction.

## Displayed-constant logarithmic reconstruction

`integrate_algebraic` is the first integration-facing use of the global data.
It currently handles a narrow, useful cell: the displayed constant field is
`QQ` or a finite algebraic extension of `QQ`, every collected residue has
local coordinate form `(c, 0, ..., 0)`, and every resulting divisor has a
principal witness within a caller-selected finite lattice bound. It does not
guess compatible conjugates for a residue outside the displayed constant
field.

For a displayed constant field with exact `QQ` basis
`(b_0, ..., b_(d-1))`, write each local residue as
`r_P = sum_i b_i q_(i,P)`. For every nonzero rational coordinate family, it
takes positive rational content `lambda_i` and forms

```text
D_i = sum_P (q_(i,P) / lambda_i) P.
```

The coefficients are integral and the global residue theorem gives
`deg(D_i) = 0` coordinatewise. If the complete principality decision
constructs `div(f_i) = D_i`, the returned native logarithmic derivative includes
`b_i * lambda_i * D(f_i) / f_i`, with displayed term
`b_i * lambda_i * log(f_i)`. The result checks

```text
integrand = D(exact_part) + logarithmic_derivative + remainder.
```

`EXACT` and `ELEMENTARY` are constructive answers.
`NEEDS_CONSTANT_BASE_CHANGE` is the honest boundary for a residue outside the
displayed constants; and `RESIDUE_FREE_REMAINDER` carries the current
first-/second-kind exactness decision. None is a non-elementarity claim.

## Certified residue-free exactness

After logarithmic terms have been removed, `solve_algebraic_exactness` builds
a fresh differential from the **post-logarithmic** remainder. It does not
reuse the original all-place divisor: subtraction of logarithmic derivatives
can change both pole orders and residues.

For a nonzero differential `omega`, any algebraic primitive belongs to the
Riemann--Roch space `L(E)` with

```text
E = sum_(ord_P(omega) < -1) (-ord_P(omega) - 1) P.
```

The implementation first checks every fresh pole residue. A nonzero residue
proves that no algebraic primitive exists. When all residues vanish, it
computes the complete `L(E)` through the generic Riemann--Roch service. Only
then does it solve
`D(sum_i c_i b_i) = omega` as an exact linear system over the displayed
constant field. A failed solve in a complete space is a complete statement
that the current algebraic function field has no primitive; it is **not** an
elementary-integrability conclusion.

`integrate_algebraic` invokes this step automatically. On `y**2 = x`,
`dx / y = D(2*y)`, and `y / (x - 1)` is completed by the same `2*y` exact
correction after its logarithmic term is reconstructed. On an elliptic curve,
a holomorphic differential such as `dx / y` correctly remains a residue-free
remainder with a certified no-primitive exactness result.

## Integrating-factor algebraic RDEs

`find_algebraic_integrating_factor` recognizes the restricted but useful
coefficient shape

```text
a = D(h) / h,  h in L^*.
```

It requires every pole of `a*dx` to be simple with an integral residue,
decides the corresponding principal residue divisor, and verifies the field
identity after finding a witness. A coefficient that is not a logarithmic
derivative is a complete negative only for this factor-recognition question.

`solve_algebraic_integrating_factor_rde` then solves

```text
D(u) + a*u = b
```

by converting it to `D(h*u) = h*b` and invoking the certified exactness
solver. Thus it can return a constructive solution or a complete no-solution
certificate when the transformed exactness decision is complete. It does not
claim to solve a general algebraic RDE: arbitrary coefficients require local
resonance analysis and a global pole bound for `u`.

## Relative simple-radical RDE descent

The direct-curve algorithms above work over `C(x)`. A terminal algebraic
level above a transcendental lower field instead has the form

```text
L = K(theta),  theta**n = u in K.
```

It must be treated as an extension of `K`, not incorrectly as a new curve over
the constant field. The first mixed cell is implemented by
`solve_relative_simple_radical_rde`. When its coefficient `a` lies in `K`,
implicit differentiation gives

```text
D(theta) = eta*theta,  eta = D(u)/(n*u),
```

and the power-basis representation is unique. Thus an equation

```text
D(v) + a*v = b,  v = sum(c_j*theta**j),
```

splits into the independent lower-field equations

```text
D(c_j) + (a + j*eta)*c_j = b_j.
```

The implementation solves those equations with the existing scalar P/E/T RDE
dispatcher, recombines the verified coordinates, and checks the original
equation in `L`. `solve_exact_derivative` uses the same descent for the
special case `a = 0`.

For example, take `Dt = 1/x` and `theta**2 = t`. The lower coefficient
`a = 1/x` keeps the equations diagonal:

```python
from sympy import Poly, log, symbols
from sympy.polys.domains import QQ
from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_relative_simple_radical_rde,
)

x, t, theta = symbols("x t theta")
base = DifferentialTower(QQ).extend(
    MonomialExtension(Poly(1, x, domain=QQ), ConstantFieldStatus.PRESERVED)
)
lower = base.extend(
    MonomialExtension(
        Poly(1 / x, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
)
tower = lower.extend(
    AlgebraicExtension(Poly(theta**2 - t, theta, domain=lower.field()))
)
field = tower.field()
derivation = tower.derivation()

v = field.from_sympy((x + 1) + (t + 1) * theta)
a = field.from_sympy(1 / x)
decision = solve_relative_simple_radical_rde(a, derivation(v) + a * v, derivation)

assert decision.has_solution
assert decision.solution == v
assert decision.logarithmic_derivative == lower.field().from_sympy(1 / (2 * x * t))
```

This is deliberately a narrow first mixed-algebraic result. A coefficient
outside `K` mixes power-basis coordinates and therefore needs a coupled RDE;
general algebraic defining polynomials need a matrix/parametric RDE; and
relative Hermite reduction, place/residue computation, and logarithmic
reconstruction over `K` are later work.

## Terminal simple-radical elementary integration

`integrate_elementary` now uses this diagonal shape constructively.  For

```text
f = f_0 + sum(f_j*theta**j, j > 0),
```

it solves `D(c_j) + j*eta*c_j = f_j` for each nonzero upper coordinate.  If
all of those RDEs have solutions, it reduces the problem to `f_0` in the
lower tower.  The existing lower elementary integrator then supplies the
logarithmic terms, so exact algebraic modes and lower logarithms compose in a
single `ElementaryIntegralResult`.

For the tower above, this produces a mixed answer directly:

```python
from sympy_risch import integrate_elementary

integrand = derivation(field.from_sympy((x + 1) * theta)) + field.from_sympy(
    1 / (x * t)
)
result = integrate_elementary(integrand, derivation)

assert result.is_elementary
assert result.antiderivative == (x + 1) * theta + log(t)
assert result.reconstructs
```

This is a success-only route.  If an upper coordinate has no lower-field RDE
solution, or if the lower integrator finds an obstruction, the dispatcher
raises `NotImplementedError` rather than treating that fact as a top-level
non-elementarity proof.  Algebraic logarithms can mix power-basis coordinates;
relative Hermite/residue/logarithmic reconstruction is the next required
algorithmic layer.

## Relative normal-Hermite reduction

`hermite_reduce_relative_simple_radical` now supplies the first part of that
next layer.  It writes a terminal pure-radical integrand as `D(g) + h` by
working in the power basis and applying the scalar operator
`D + j*eta` to each `theta**j` coefficient.  Every repeated **normal** lower
denominator factor is removed, so the remaining normal denominator in each
coefficient of `h` is square-free.  Special factors are retained unchanged.

For example, with `Dt = 1/x` and `theta**2 = t`, the repeated normal pole in
`D(theta/t)` is removed exactly:

```python
from sympy_risch import hermite_reduce_relative_simple_radical

reduction = hermite_reduce_relative_simple_radical(
    derivation(field.from_sympy(theta / t)),
    derivation,
)

assert reduction.exact_part == field.from_sympy(theta / t)
assert reduction.remainder == field.zero
assert reduction.reconstructs
assert reduction.normal_denominators_are_square_free
```

The reducer uses the immediate `K0(t)` view of SymPy's flattened mixed field,
not the direct-curve `C(x)` integral-basis code.  A resonant leading equation
is an explicit `NotImplementedError` boundary; its treatment belongs to the
relative logarithmic/RDE stage rather than to Hermite reduction.

## Algebraic constant composita and iterative profile selection

`integrate_algebraic_with_constant_base_change` continues the displayed-
constant boundary with either one explicit irreducible factor `q(alpha)` or
an iterative profile plan. The automatic path factors the square-free
all-place residue profile, orders its supported rationally defined non-linear
factors by degree and exact coefficients, and tries them until a certified
target no longer needs another constant base change.

The underlying map is reusable beyond that one-step planner. It accepts
constants `C = QQ` or an existing SymPy algebraic number field and constructs
the exact compositum with the root field of a rational irreducible `q`.
Constants are embedded by an explicit number-field isomorphism; the map is
then lifted coefficientwise through `C(x)[y]/(f)`. Thus successive manual
changes such as
`QQ -> QQ(sqrt(2)) -> QQ(sqrt(2), sqrt(3))` do not rely on reparsing a whole
function-field expression. At every step the target relation is rebuilt, the
target constant field is checked again, and commutation with the derivation is
verified on the field generators.

After each successful step, the planner profiles the new target result rather
than assuming that source residue data survived unchanged. Every step strictly
enlarges the constant field, so an absorbed factor cannot be selected again.
Its current certified place layer supports finite-place splitting with
ramification index one. The preceding differential is embedded and collected
again at every target place. Recollecting is required: transported source
finite residues alone would miss target infinity and other descendants.

The result keeps both fields visible. Its `source_result` records why the
source needed a base change; when present, `target_result` and
`antiderivative` belong to the target field, never silently to the source.
For example, on `y**2 = x`, the source differential
`y / (x * (x - 2))` has residues outside `QQ`; its profile has the factor
`q(Z) = Z**2 - 1/2`, which gives an elementary target-field logarithmic
answer. The example `y / ((x - 2)*(x - 3))` requires two independent
quadratic residue fields and now finishes over their quartic compositum.

On a rational curve with a verified degree-one parameter `t`, the final
reconstruction uses the exact divisors
`div(t - t(P)) = P - P_infinity`. This keeps a large compositum from turning
small residue constants into high-power `QQ`-coordinate divisors. The general
complete Riemann--Roch principality algorithm remains the fallback.

## Riemann--Roch spaces

`compute_riemann_roch_space(model, divisor)` is a generic function-field
operation. It has no Risch admission or constant-field dependency: other
clients can reuse it for a presented algebraic function field. Risch callers
must establish `PRESERVED` constants before they use its basis in an
elementarity decision.

For a general divisor it uses the Hess finite/infinite ideal representation.
`hess_divisor_ideals` builds inverted ideals in the maximal finite order and
the reciprocal-chart maximal order; weak Popov reduction of their transition
matrix then computes their intersection. This replaces the former
expansion-per-candidate path and has no caller-selected degree bound. A
successful `RiemannRochSpace` rechecks every basis element through its full
principal divisor, so finite, ramified, and infinite valuation bounds are
exact.

The current complete cases are deliberately narrow:

- `deg(D) < 0`, where `L(D)` is zero;
- `D = 0`, where `L(D)` is the constant span of `1`;
- a degree-zero divisor supplied with a principal witness `f`, where
  `div(f) + D = 0` and `L(D) = C*f`;
- a caller-supplied basis in the nonspecial range
  `deg(D) > 2*g - 2`, whose exact membership, constant-field independence,
  and Riemann--Roch dimension prove completeness; and
- `D = m*(u)_infinity` for a supplied rational parameter `u` with a
  degree-one pole divisor, where the certified basis is
  `1, u, ..., u**m`.

If the exact order or ideal arithmetic needed by Hess is unavailable, the API
returns an empty `UNSUPPORTED` result rather than a partial basis. Computing
`L(-D)` is now enough to test a degree-zero divisor `D` for principality once
the corresponding principal-divisor consumer adopts it: a nonzero basis
member must have divisor exactly `D`. Divisor-class reduction and torsion
arithmetic are still needed for efficient relation discovery and for scaling
to high-degree examples.

### Finite-dimensional valuation kernels

`solve_algebraic_valuation_constraints` is the next reusable piece of the
general lattice-intersection problem. Given a finite constant-field subspace
`V` and a divisor `D`, it computes exactly `V ∩ L(D)`. It collects the full
principal-divisor support of every ambient basis element, then imposes the
Laurent-coefficient conditions required at each place. This captures genuine
cancellation between candidate functions, rather than merely filtering them
one at a time.

Every constructed closed residue field is flattened in its native finite basis
over the proved constant field. If the base point has root `alpha` and the
reduced fiber uses separator `beta`, the finite-place coordinate order is
`1, alpha, ..., alpha**(d - 1), beta, alpha*beta, ...`. A local Laurent
coefficient therefore supplies one scalar equation for each coordinate, so a
degree-`d` closed point is handled without pretending it is rational. The
coordinates are intentionally place-local: they do not yet identify values in
different residue fields inside a common algebraic constant extension. That
global embedding/residue-basis problem remains part of logarithmic
reconstruction. The result still requires preserved constants and rechecks
every returned basis element by its full principal divisor.

`finite_lattice_slice` supplies a bounded subspace from the exact finite ideal
associated with `-D_fin`. `search_nonspecial_lattice_slice` turns that bounded
search into a complete global space only after its dimension reaches the
nonspecial Riemann--Roch value. A smaller slice is `INCOMPLETE`, never a
negative integration or global-space decision.

### Complete principal-divisor decisions

`decide_algebraic_principality(model, D)` is the reusable global task needed
by logarithmic reconstruction. A nonzero-degree divisor is immediately
nonprincipal. For degree zero, it computes the complete space `L(-D)`. A
nonzero `f` in that space satisfies `div(f) - D >= 0`; this divisor has degree
zero and therefore vanishes, so `div(f) = D`. Conversely, an empty complete
space proves `D` nonprincipal. The result retains the Riemann--Roch certificate
or an explicit unsupported boundary.

Risch callers separately establish preserved differential constants before
using this geometric result in an elementary-integrability decision. The
underlying function-field operation has no Risch derivation or bound argument.

### Pell--Abel target example

The following genus-2 differential is a useful regression target for the
eventual global logarithmic-reconstruction algorithm:

```text
P(x) = x**6 + 4*x**5 + 6*x**4 - 12*x**3 + 33*x**2 - 16*x
y**2 = P(x)
omega = (29*x**2 + 18*x - 3) * dx / y
```

It has two simple poles at infinity, with residues `29` and `-29`.  It is
elementary, but not by the terminal simple-radical exact-coordinate route:
there are polynomials `A` and `B` of degrees 29 and 26 satisfying

```text
A**2 - P*B**2 = 2**54
A' = (29*x**2 + 18*x - 3)*B
```

Thus `v = A + B*y` is a Pell--Abel unit and
`D(v) / v = (29*x**2 + 18*x - 3) / y`; its integral is `log(v)`.
Equivalently, the degree-zero divisor of residues is a 29-torsion
principal-divisor relation between the two infinite places.

This example is intentionally not a normal test-suite input. The older
bounded search spent most of its time imposing local expansions. The Hess
route instead reaches a separate current bottleneck: high powers of a
reciprocal prime ideal trigger repeated Smith normal-form normalization in
SymPy's exact polynomial arithmetic. A reduced-ideal implementation should
remove that cost; a complete divisor-class or Pell--Abel solver should then
recognize and construct the relation directly. Until then, an impractically
slow search is an implementation boundary, never a non-elementarity claim.

## Public API

| Structure | Purpose |
| --- | --- |
| `AlgebraicOrder` | Certified full-rank `C[x]` subring of `L` |
| `AlgebraicFractionalIdeal` | Nonzero full-rank fractional ideal |
| `HessDivisorIdeals` | Finite and reciprocal-chart ideals representing one divisor |
| `IntegralClosureResult` | Maximal order plus all refinement evidence |
| `FiniteAlgebraicPlace` | Closed finite place, including ramified places and local `C` coordinates |
| `InfiniteAlgebraicPlace` | Closed place above infinity, with the reciprocal place's local coordinates |
| `LocalLaurentExpansion` | Certified finite Laurent segment |
| `AlgebraicDivisor` | Immutable formal sum of closed places |
| `AlgebraicDifferential` | Differential represented as `h dx` |
| `AlgebraicFunctionField` | Cached global owner for orders and places |
| `AlgebraicConstantFieldAnalysis` | Narrow, witnessed constant-field decision |
| `AlgebraicConstantEmbedding` | Exact proposed constant subfield map into `L` |
| `AlgebraicLocalDifferentialResidue` | One pole, uniformizer, and local residue |
| `AlgebraicAllPlaceResidueResult` | Hermite result plus complete all-place data |
| `collect_algebraic_all_place_residues` | Guarded global residue collector |
| `AlgebraicResidueProfileFactor` | `Norm_{k(P)/C}(Z-res_P)` at one closed pole |
| `AlgebraicResidueProfile` | All-place residue polynomial before base change |
| `build_algebraic_residue_profile` | Construct the exact closed-place profile |
| `RiemannRochSpace` | Immutable complete basis with its proving theorem |
| `compute_riemann_roch_space` | Certified first-scope global-space service |
| `AlgebraicValuationKernel` | Exact `V ∩ L(D)` in a declared finite subspace |
| `search_nonspecial_lattice_slice` | Bounded lattice search with a theorem-backed completion check |
| `AlgebraicPrincipalityDecision` | Verified principal witness, complete negative, or AFF boundary |
| `decide_algebraic_principality` | Decide principality using complete `L(-D)` |
| `AlgebraicIntegrationResult` | Native Hermite/logarithmic decomposition and an honest algebraic status |
| `AlgebraicLogarithmicTerm` | One coefficient, verified residue divisor, and log argument |
| `integrate_algebraic` | Current displayed-constant algebraic integration slice |
| `AlgebraicExactnessResult` | Verified primitive, complete in-field no-primitive decision, or unsupported AFF boundary |
| `solve_algebraic_exactness` | Low-level residue-free exactness decision in a direct algebraic function field |
| `AlgebraicConstantBaseChangeIntegrationResult` | Source boundary plus an explicit chain of target-field continuations |
| `integrate_algebraic_with_constant_base_change` | Recollect and integrate through explicit or iteratively profile-selected compatible factors |

The constructors `polynomial_order` and `integral_closure` are available for
lower-level consumers. Most code should start from
`AlgebraicFunctionField`.

Refinement-step, maximality-certificate, reciprocal-chart, and union-type
records are implementation-facing details in their
`sympy_risch.algebraic_geometry` owner modules; they are deliberately omitted
from both public facades. Likewise, the plane-model-only
`sympy_risch.algebraic.trager.unramified_finite_places` helper remains available to
the Trager resultant implementation, but it is not the canonical place API.
New code should use `AlgebraicFunctionField.finite_places`, which handles
ramification and singular displayed models after normalization.

## Minimal example

```python
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    MonomialExtension,
)

x, y = symbols("x y")
rational = DifferentialTower(QQ).extend(MonomialExtension(Poly(1, x, domain=QQ)))
extension = AlgebraicExtension(Poly(y**2 - x, y, domain=rational.field()))
curve = AlgebraicFunctionField(extension)

branch_place = curve.finite_places(Poly(x, x, domain=QQ))[0]
infinity = curve.infinite_places()[0]

assert branch_place.ramification_index == 2
assert branch_place.valuation(extension.field.generator) == 1
assert infinity.valuation(extension.field.from_sympy(x)) == -2
assert curve.principal_divisor(extension.field.from_sympy(x)).degree == 0
assert curve.genus == 0
```

## Orders and normalization

### Polynomial starting order

An `AlgebraicExtension` may have rational-function coefficients, so its
displayed generator need not be integral over `C[x]`.
`polynomial_order(extension)` takes one common denominator `d(x)` of the
nonleading coefficients and uses

```text
z = d(x) y.
```

The powers `1, z, ..., z^(n-1)` form a full-rank integral starting order.
`AlgebraicOrder` then certifies:

- the basis contains `1`;
- all basis products have polynomial coordinates; and
- the basis has full field rank.

Closure and finite generation imply integrality; no floating approximation or
root selection is involved.

### Maximal-order algorithm

`integral_closure` implements the characteristic-zero trace-radical and
idealizer algorithm described in Trager's thesis.

For a current order `A`:

1. compute the trace-pairing discriminant;
2. consider every irreducible `p(x)` for which `p**2` divides it;
3. compute the trace radical of `A/pA`;
4. compute the idealizer of that radical;
5. replace `A` when the idealizer is strictly larger; and
6. repeat until every candidate prime has trivial idealizer enlargement.

Every strict enlargement records the monic principal-ideal identity

```text
disc(A_old) = disc(A_new) * index(A_new : A_old)**2.
```

The final result records a separate maximality certificate for each remaining
square-discriminant prime. Verification recomputes every stored
discriminant, index, trace radical, and idealizer rather than trusting the
recorded evidence. Primes occurring only to the first power cannot divide an
order index because order indices enter discriminants as squares.

This algorithm finds normalization hidden by a singular plane presentation.
For example, the model

```text
y**2 = x**2 * (x + 1)
```

starts with the nonmaximal power order `C[x][y]`; normalization adjoins
`y/x`. The fiber over `x = 0` then has two unramified places rather than one
spurious branch point.

## Fractional ideals

`AlgebraicFractionalIdeal` stores a full-rank `C[x]` lattice in the field.
Generator matrices are reduced exactly with Smith decomposition over the PID
`C[x]`. Construction checks stability under multiplication by the owning
order.

Implemented operations are:

- containment and equality;
- sum, product, scaling, and integer powers;
- ideal quotients `(I : J)`;
- inverses;
- module intersections;
- determinant indices; and
- conversion between finite divisors and fractional ideals.

The intersection implementation is lattice-theoretic:

```text
I intersect J = dual(dual(I) + dual(J)).
```

The temporary duals use the coordinate pairing only; the final result is
validated again as an order ideal.

Ideal arithmetic requires the exact same `AlgebraicOrder` instance. The
global model owns one canonical maximal-order instance so place and divisor
operations satisfy this automatically.

## Finite places

`AlgebraicFunctionField.finite_places(p)` decomposes an irreducible base prime
in the maximal order, without relying on the singularities of the original
plane equation.

The residue algebra is

```text
A = O / pO.
```

In characteristic zero its nilradical is the kernel of the reduced trace
pairing. The implementation finds and certifies a separating element `a` of
the reduced algebra. If

```text
charpoly_A(a) = product(q_i**e_i),
```

then:

- each irreducible `q_i` identifies one closed place;
- `degree(q_i)` is its relative residue degree; and
- `e_i` is its ramification index.

The decomposition checks the fundamental identity

```text
sum(e_i * f_i) = [L : C(x)].
```

Each place stores:

- its exact nested residue field;
- the kernel prime ideal in the maximal order;
- residue images of the order basis;
- ramification and residue degrees; and
- a global element in `P \ P**2`, which is a local uniformizer.

This covers branch fibers, inert fibers, split fibers, and closed base points
of degree greater than one.

## Infinity

Every infinite place is represented on an exact reciprocal chart.

Set

```text
s = 1/x
z = s**k y,
```

where `k` is chosen so the transformed defining polynomial is integral at
`s = 0`. The reciprocal field receives its own certified maximal order, and
the finite-place machinery decomposes the prime `s`.

The internal reciprocal chart's forward and backward maps are exact field
isomorphisms. `InfiniteAlgebraicPlace` maps uniformizers, valuations, local
reductions, and Laurent expansions back to the original presentation.

## Valuations and local data

For a finite place `P`, valuation is normalized by `v_P(P) = 1`.

For an integral element, the implementation tests membership in successive
powers `P**k`. For a general field element, it first clears one common
polynomial denominator and subtracts

```text
e(P/p) * ord_p(denominator).
```

The same method factors arbitrary fractional ideals at a place.

`reduce_local` handles every element regular at the place, not only global
order elements. It computes a denominator ideal, selects a denominator
outside `P`, and reduces the resulting numerator/denominator pair.

`local_expansion(element, stop=N)` returns coefficients from the element's
valuation through `N - 1`. At each power it:

1. divides by the selected uniformizer power;
2. reduces the leading local unit;
3. lifts that residue exactly into the maximal order; and
4. subtracts the corresponding term.

The result certifies exact reconstruction and that the remainder has
valuation at least `N`. Coefficients below the starting valuation are known
zeros. A coefficient at or above `N` raises `IndexError` unless the exact
remainder is zero, because a truncated segment must not claim an uncomputed
tail coefficient is zero.

## Differents and differentials

For an order basis with trace matrix `T`, the codifferent coordinate lattice
is generated by the columns of `T**-1`. Its ideal inverse is the different.

At a finite place:

```text
ord_P(h dx) = v_P(h) + v_P(Different).
```

At infinity:

```text
ord_P(h dx) =
    v_P(h) + v_P(Different) - 2*e(P/infinity).
```

The second formula is the reciprocal identity `dx = -s**-2 ds`.

General residues are computed in a true local uniformizer. If `t` is the
place's uniformizer and `D = d/dx`, then

```text
h dx = (h / D(t)) dt.
```

The residue is the coefficient of `t**-1` in the exact local expansion. This
works at ramified places; it does not assume the base prime itself is a
uniformizer.

## Divisors

`AlgebraicDivisor` is an immutable formal sum with:

- addition, subtraction, and integer scaling;
- coefficient lookup and support;
- positive and negative parts;
- effectivity; and
- residue-degree-weighted degree.

Places are canonical objects owned and cached by one
`AlgebraicFunctionField`. Divisor arithmetic intentionally requires that
shared model identity.

`principal_divisor(f)` first clears a common polynomial denominator in the
maximal-order basis. It discovers finite support from the norm of that
integral numerator together with the clearing denominator, computes all place
valuations above those base primes, adds every infinite valuation, and
certifies total degree zero. This avoids losing a zero and pole above the same
base prime when their contributions cancel in `Norm(f)`.

`ideal_divisor(I)` similarly clears a common coordinate denominator and uses
the determinant of the resulting integral ideal together with that
denominator. Thus opposite exponents above one split base prime cannot cancel
out of support discovery. `ideal_divisor(I)` and `finite_ideal(D)` are inverse
conversions for divisors supported away from infinity.

The canonical divisor is the divisor of `dx`. Its degree yields

```text
genus = (degree(K) + 2) / 2.
```

No divisor-class or principal-divisor solver is hidden in these operations.

## Testing contract

`tests/test_algebraic_geometry.py` covers:

- already maximal and nonmaximal orders;
- hidden quadratic, cubic, and quartic normalization;
- every discriminant-index certificate;
- unstable-lattice rejection;
- ideal sum, product, inverse, quotient, intersection, and powers;
- codifferent/different inversion;
- ramified, split, inert, and algebraic-constant fibers;
- singular plane models whose normalized fibers split;
- finite and infinite uniformizers;
- exact local reduction with nontrivial denominators;
- negative-power Laurent expansions;
- rejection of uncomputed Laurent tail coefficients;
- generated valuation identities;
- branch-point and infinity residues;
- ideal/divisor round trips, including opposite split-prime exponents;
- principal-divisor degree zero and split-fiber norm cancellation;
- independently recomputed and adversarially forged closure certificates; and
- genus-zero and genus-one canonical divisors.

The broader test suite continues to cover the older plane-model Trager
resultant independently. The closed-place profile now cross-checks its finite
unramified slice while also exercising branch and infinity residues. The
plane-model route remains useful as an independent oracle; the canonical place
layer is the general local substrate for its future ramified extension.

## Literature mapping

The primary sources for this module were the locally available references plus
Brian L. Miller's 2012 dissertation:

- `references/books/integration-in-finite-terms-fundamental-sources-2022.pdf`,
  Trager thesis reprint, especially the integral-basis, idealizer, place, and
  divisor chapters;
- `references/papers/trager-1984-integration-of-algebraic-functions.pdf`;
- `references/papers/davenport-1984-integration-on-an-algebraic-curve.pdf`;
  and
- `references/papers/bronstein-1991-risch-differential-equation-algebraic-curve.pdf`.
- Brian L. Miller, [*On the Integration of Elementary Functions: Computing
  the Logarithmic Part*](https://ttu-ir.tdl.org/server/api/core/bitstreams/6a1324fa-b43d-4efe-9cfe-a95480f2dc86/content)
  (Texas Tech University, 2012), especially its residue polynomial and divisor
  reconstruction route.

See `references/README.md` for the locally available sources and provenance.

## Remaining algorithm boundary

The next algebraic Risch implementation may assume all of the following:

1. a certified finite maximal order;
2. exact decomposition of every requested finite prime;
3. complete infinity-place enumeration;
4. exact element and ideal valuations;
5. local residue fields, reduction, lifting, and Laurent coefficients;
6. principal and canonical divisor construction; and
7. finite divisor/fractional-ideal conversion.

It now also has a small set of certified global Riemann--Roch spaces and a
bounded direct-`QQ` logarithmic reconstruction cell, residue-free exactness,
and an integrating-factor RDE endpoint. It must still implement general
finite/infinite lattice intersection, compatible residue descent,
divisor-class/torsion solving, general logarithmic reconstruction, complete
second-kind exactness, and general algebraic RDEs. Those are intentionally not
infrastructure methods.
