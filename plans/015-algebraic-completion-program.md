# Plan 015: Complete the algebraic and mixed Risch program

> **Executor instructions**: This is a program plan, not one atomic change.
> Implement each numbered checkpoint as a separate plan and commit. Keep the
> algebraic-function-field layer independent of Risch derivations, and never
> turn an unavailable geometric operation into a non-elementarity result.

## Status

- **Priority**: P1
- **Effort**: XL, a multi-checkpoint research program
- **Risk**: HIGH
- **Depends on**: Plans 004-006 and 010-013; A11 also consumes Plan 014
- **Category**: algebraic function fields / algebraic and mixed integration
- **Planned**: 2026-07-27
- **Status**: IN PROGRESS

## Why this is a program rather than one Hess task

The current function-field layer implements the central finite/infinite ideal
intersection behind Hess's first Riemann--Roch algorithm. For an admitted
direct monogenic function field whose maximal orders are available,
`hess_parametric_basis` retains the reduced Theorem 7 generators and invariant
degrees, while `hess_riemann_roch_basis` remains the checked long-basis view.

That is not the entire computational framework assumed by Hess, and it is not
yet a complete algebraic Risch implementation. A practical general
implementation also needs:

- normalized function-field presentations and the exact constant field;
- total maximal-order, ideal, prime, and residue-field arithmetic;
- height-efficient divisor reduction and large-divisor Riemann--Roch spaces;
- differential spaces and scalable divisor-class operations;
- residue descent in a common algebraic constant field;
- the complete pure algebraic logarithmic and second-kind decisions; and
- algebraic RDEs and relative versions of those operations for mixed towers.

Hess is therefore the reusable geometry engine in the middle of the algebraic
program, not a substitute for the integration algorithms around it.

## Completion contract

The algebraic-function-field engine accepts an exact, explicitly supported
field `k`, a geometrically integral one-variable function field `F/k`, and a
certified separating element. For every accepted input it must provide:

1. a normalized presentation with the exact constant field identified;
2. maximal finite and infinite orders;
3. exact fractional-ideal, prime, place, valuation, and divisor arithmetic;
4. complete Riemann--Roch and differential spaces;
5. exact divisor-class equality, reduction, and witness reconstruction; and
6. explicit capability failures only at the admission boundary.

The pure algebraic integrator then accepts `f in F` and returns either:

- a reconstructible elementary antiderivative, possibly over a finite
  algebraic extension of `k`; or
- a theorem-backed Liouville obstruction.

The mixed integrator applies the same contract relative to an admitted
differential ground field. It must not assume that constants, factorization,
or residue-field embeddings are unchanged merely because the same syntax is
used.

## Layering

Keep the dependencies one-way:

```text
exact coefficient-field operations
              |
              v
function-field presentation and normalization
              |
              v
orders, ideals, primes, places, and divisors
              |
              v
Hess Riemann--Roch / differential / class services
              |
       +------+------+
       |             |
       v             v
pure algebraic    algebraic RDEs
integration          |
       +-------------+
              |
              v
relative and mixed-tower integration
              |
              v
automatic expression compilation
```

`sympy_risch.algebraic_geometry` must remain reusable by clients that know
nothing about a Risch tower. Integration-specific constant-field and
Liouville checks belong in `sympy_risch.algebraic` or the recursive solver.

## Ordered implementation checkpoints

### A0. Freeze the effective-field and presentation boundary

Define exactly which coefficient fields and function-field presentations the
geometry engine admits. The first complete cell should be characteristic-zero
number fields with exact arithmetic and univariate factorization, plus finite
fields used only for certified good reduction.

The generic effective constant-field capability record is shared with Plan
016 T0 and must be implemented once in a neutral domain layer. This checkpoint
adds the function-field-specific presentation and normalization admission
rather than creating a second capability hierarchy.

The admission result must distinguish:

- exact arithmetic unavailable;
- inseparable or reducible presentation;
- constant-field enlargement;
- normalization/maximal-order algorithm unavailable; and
- a valid field that a later integration backend does not yet support.

Do not spread ad hoc `domain == QQ` tests through the geometry algorithms.

**Exit gate:** every geometry entry point either receives a checked
presentation or returns one stable capability boundary.

### A1. Normalize general algebraic-function-field presentations

Replace the assumption that every useful curve arrives as one convenient
monogenic `k(x)[y]/(f)` model with a normalization pipeline:

1. choose or verify a separating transcendence element;
2. square-free factor and isolate the requested irreducible component;
3. detect zero divisors rather than continuing with a reducible coordinate
   ring;
4. compute primitive elements and explicit embeddings for algebraic
   composita;
5. identify the algebraic closure of `k` in `F` and rebase when constants
   enlarge;
6. support coordinate changes needed to obtain usable finite and infinity
   charts; and
7. retain bidirectional maps so every witness reconstructs in the source
   presentation.

This checkpoint is also the execution prerequisite for Plan 014's general
`RootOf`, independent-radical, and monomial-over-algebraic candidates.

**Exit gate:** equivalent admitted presentations normalize to an isomorphic
function-field model with checked forward and inverse maps.

### A2. Make maximal orders and ideal arithmetic total on the admitted cell

Complete the local algebra needed by Hess rather than relying on the current
trace-radical idealizer succeeding:

- certified `p`-maximal refinement, using a Round-2-style algorithm where
  appropriate;
- maximal finite and reciprocal-chart orders;
- ideal sum, product, quotient, inverse, intersection, and powers;
- prime-ideal decomposition with ramification and residue degree;
- residue-class fields and exact reduction/lifting;
- ideal factorization, norms, and valuations; and
- lossless conversion among ideals, places, and finite/infinite divisors.

The same operation should have one implementation used by Riemann--Roch,
principal divisors, good reduction, and integration. Do not add a
Risch-specific parallel ideal type.

**Exit gate:** randomized principal and fractional ideals reconstruct from
their prime factorization, and both charts give identical global divisors.

### A3. Implement Hess Algorithm 13 as a parametric space

**Progress:** complete for the direct function-field presentations admitted by
the existing Hess ideal arithmetic. `HessParametricBasis` retains the
weak-Popov lattice certificate, generators, and invariant degrees; dimensions
and shifted bases for `D + r*(x)_infinity` reuse that one reduction. The
long-basis API is a checked expansion view, and small spaces agree with the
independent local Laurent-valuation kernel.

Preserve the current verified long-basis API, but make the primary result the
short parametric data from Hess's Theorem 7:

```text
(v_i, d_i),  i = 1, ..., [F : k(x)]
```

so that

```text
{x^j v_i : 0 <= j <= d_i + r}
```

is a basis of `L(D + r*(x)_infinity)` whenever the upper bound is
nonnegative.

Required work:

- expose an immutable `HessParametricBasis`;
- retain the reduced transition matrix and invariant degrees;
- expand to a long basis only on demand;
- use a canonical, tested polynomial-lattice reduction;
- verify the dimension formula and membership without recomputing a complete
  principal divisor for every generated monomial; and
- cross-check small spaces against the independent local-valuation kernel.

The current `hess_riemann_roch_basis` is the behavioral oracle for its
supported direct presentations, not code to discard.

**Exit gate:** one reduced basis serves a family of shifted divisors and agrees
with the existing long basis on all current tests.

### A4. Implement Hess Algorithm 21 divisor reduction

**Progress:** complete for the direct function-field presentations supported
by the current Riemann--Roch engine. `reduce_hess_divisor` decomposes arbitrary
signed divisor coefficients in binary, reduces each signed-support digit, and
combines the results by Horner composition. The exact correcting function is
retained as a `HessPowerProduct`; expanding it is optional. Both Hess's degree
strategy and maximal reduction are available. With the default
`(x)_infinity` anchor, the implementation reads shifts from one
`HessParametricBasis` per elementary step; a general effective anchor uses
complete Riemann--Roch spaces and monotone search.

Replace linear repeated addition and repeated full Riemann--Roch construction
with Hess's height-efficient reduction:

- binary decomposition of large divisor coefficients;
- support reduction;
- exponent reduction;
- degree reduction for general positive-degree anchors;
- maximal reduction when a degree-one anchor permits a unique reduced
  representative; and
- power-product witnesses for the accumulated principal divisor.

The result must preserve both the reduced divisor and an exact source-field
element `f` satisfying the reported divisor identity. Large exponents should
cost logarithmically many composition steps rather than linearly many class
additions.

**Exit gate:** reduction of high-height divisors agrees with direct small
cases, reconstructs every witness, and has a regression showing logarithmic
growth in the exponent rather than linear growth. This gate is complete:
focused tests independently expand moderate witnesses, cover split infinity,
check maximality, and make the binary-layer bound observable without a timing
assertion.

### A5. Implement Hess Algorithm 24 and differential spaces

**Progress:** complete for the direct function-field presentations supported
by the current Hess engine.
`HessLargeRiemannRochSpace` combines Algorithm 21's compact correction with
the reduced divisor's `HessParametricBasis`. It stores the rank-sized
power-product generators and shifted invariant degrees, computes dimensions
without expansion, and expands or independently verifies the long basis only
on request. Small spaces agree with the original direct Hess construction,
while a regression keeps a 514-dimensional example in short form.
`HessDifferentialSpace` applies the exact correspondence
`Omega(D) = L(div(dx) - D)`, retaining the same compact representation for
coefficients. Holomorphic spaces have dimension equal to the genus, and tests
check Riemann--Roch duality for positive, negative, and mixed divisors.
For an effective pole bound `E`, `HessSecondKindDifferentialSpace` now
constructs the exact residue map on `Omega(-E)` and returns its complete
kernel for constant-certified direct models. A geometry-owned certificate
uses a degree-one place without depending on the Risch derivation analyzer;
an explicit adapter accepts an existing preserved Risch analysis. This
includes the nonradical `y**3 + y + x` regression. Closed places of arbitrary
degree contribute their full native constant-coordinate blocks.
`solve_hess_prescribed_residues` now solves the corresponding affine system
for third-kind differentials over a reusable second-kind residue operator. A
consistent target returns `HessPrescribedResidueSpace`, with an exact
particular differential and translation space; an inconsistent target returns
the distinct `HessPrescribedResidueInconsistency` carrying a left-kernel
obstruction. Tests cover operator reuse, the global residue obstruction,
nontrivial affine families, higher-degree closed residue fields, and invalid
public residue values.
`hess_genus_consistency` independently compares the genus obtained from the
different-derived canonical degree with `dim L(K)` computed by Algorithm 13.
The latter path does not consume `model.genus` or Algorithm 24, and both
identities require model-owned full-constant-field evidence. Regressions cover
genus zero through three, a nonradical cubic presentation, split infinity, and
rejection of an enlarged displayed constant field.

Use divisor reduction plus the parametric basis to compute large
Riemann--Roch spaces in short power-product form. Then expose the standard
function-field consequences:

- canonical divisors and the different;
- `Omega(D)` through the corresponding Riemann--Roch space;
- holomorphic differential bases;
- differentials of the second and third kind;
- Riemann--Roch duality checks; and
- genus consistency from independent formulas.

Expansion to individual field elements is a view requested by a caller, not
the internal representation used throughout class arithmetic.

**Exit gate:** complete for the admitted direct-model cell. Large divisor and
unrestricted differential spaces are
complete: dimensions satisfy Riemann--Roch, requested expansions have exact
membership and constant rank, holomorphic dimensions agree with genus, and
duality reconstructs. Bounded second-kind residue kernels are also complete
when the displayed constants have geometric or adapted Risch evidence. Models
with neither a supplied degree-one place nor a preserved Risch analysis remain
an explicit constant-field boundary. Prescribed-residue third-kind spaces are
complete over the same admitted models. The different genus and direct
Algorithm 13 canonical-space dimension agree independently after the full
constant field has been certified.

#### A5-P. Harden the differential-space path

**Progress:** complete. Residue matrices now expand each compact Hess
generator and the base parameter once per place, with direct per-basis
residues retained as an exact oracle. Differential membership reuses the
differential's cached divisor, zero-residue membership streams by place, and
sparse coordinate rows are allocated only when first encountered.

The prescribed-residue solver reuses the same ambient differential basis and
local expansions as the second-kind kernel. Its prerequisite work was:

- construct residue columns by place and compact Hess generator, propagating
  through its powers of the separating parameter instead of independently
  expanding every long-basis element;
- make repeated differential membership use the cached differential divisor
  (or an equivalent cached principal-divisor service);
- stream zero-residue membership checks so a nonzero place stops the
  computation immediately; and
- avoid eager dense-row allocation while assembling sparse Riemann--Roch
  constraint matrices.

These are representation fixes, not weakened validation. Publicly supplied
objects retain their exact checks; the optimized path must produce the same
residue matrix and membership decisions as the direct local-expansion oracle.

**Exit gate:** small spaces agree column-for-column with direct residue
expansion, repeated membership does not rebuild principal divisors, and work
counts for increasing pole bounds scale with compact generators and requested
precision rather than with independent long-basis expansions.

### A6. Complete scalable divisor-class and Jacobian operations

Build class arithmetic on A4-A5:

- canonical reduced representatives when a degree-one anchor exists;
- deterministic representatives with a documented anchor otherwise;
- equality, addition, negation, and double-and-add scalar multiplication;
- principality with a reconstructed function;
- exact order of a specified class using suitable group algorithms rather
  than scanning every integer through a Weil bound; and
- certified good reduction over rational and algebraic number fields.

Computing an entire class group, unit group, or `S`-unit group is useful
function-field infrastructure but is not on the first Risch exit path unless a
downstream proof actually needs it.

**Exit gate:** the existing Pell--Abel/torsion regression no longer depends on
linear class iteration, and positive decisions still reconstruct in
characteristic zero.

The first performance slice of A6 is explicit:

- replace the current linear scan through the Weil bound with a certified
  baby-step/giant-step or factor-refinement order algorithm; and
- keep full validation for public ideals while providing a proof-carrying
  internal construction path for multiplication, powers, inverses, and
  colons, so each result does not repeat the same module-stability proof.

The internal path must retain enough source evidence to replay validation at a
public boundary. It is not permission to make arbitrary unvalidated ideals.

**Progress:** complete for the current exact class-arithmetic cell. A
degree-one anchor supplies canonical representatives for certified
baby-step/giant-step order search and prime-factor refinement; higher-degree
anchors retain an explicit bounded search but are not automatically scanned
through an exponential Weil bound. Internally derived ideal sums,
products, powers, inverses, colons, and intersections skip the redundant
module-stability replay, while the public ideal constructor still performs
the full check.

### A7. Complete pure algebraic logarithmic reconstruction

**Progress:** the admitted rational-factor cell constructs one exact
number-field compositum for all currently visible residue factors, retains the
source embedding through its primitive-element chain, and rebuilds/recollects
the curve once. Displayed residues are expanded in a rational constant basis
and their coordinate divisors are combined. Exact prime-ideal residue maps
now reduce integral `QQ(alpha)` constants to `GF(p^f)`; supported split primes
drive positive-genus divisor-class torsion automatically. A sparse prepared
batch caches the exact orders for each divisor at its own pair of good
characteristics, while reusing reduced curves wherever possible. The
genus-one order-two regression over `QQ(sqrt(2))` reconstructs
`log(x - sqrt(2))/2` without an order hint. Conversely, an elliptic
rational-coordinate divisor certified nontorsion now produces a sealed
Liouville `NON_ELEMENTARY` obstruction; bounded exhaustion remains
unsupported. Curve arithmetic over non-prime
`GF(p^f)` and the complete positive-genus combination solver remain open.
General splitting fields over `QQ` or displayed algebraic number fields are
constructed by successive exact root adjunctions; every step retains the
source embedding and its new root through a resultant primitive element, and
the final certificate rechecks complete splitting.

Generalize the current displayed-constant path:

1. perform algebraic Hermite reduction in normalized integral bases;
2. collect every finite, branch, and infinite residue;
3. construct one compatible algebraic constant extension containing the
   residues;
4. express residues in a rational basis before solving relations;
5. form the required *linear combinations* of residue divisors;
6. decide principality/torsion of those combinations; and
7. reconstruct logarithmic arguments in the common target field.

The combination step is essential: individual residue divisors can be
nontorsion even when the constant-linear combination required by Liouville's
theorem is torsion.

Use Trager's thesis, Miller's logarithmic-part treatment, and Schultz's
curve-based reconstruction as mutually checking formulations. Independent
resultant and divisor routes should remain separate where they provide a
useful oracle.

**Exit gate:** genus-zero, elliptic, and positive-genus examples with
ramification, infinity residues, independent residue fields, and nontrivial
divisor combinations all receive reconstructible logarithmic answers or a
proved obstruction.

### A8. Complete second-kind exactness and pure algebraic dispatch

**Progress:** complete on the admitted direct-model integration cell. After
all logarithmic residues are removed, the existing certified primitive-pole
Riemann--Roch solve now returns an exact primitive or a distinct
`AlgebraicExactnessLiouvilleObstruction`; unavailable complete spaces remain
`RESIDUE_FREE_REMAINDER`. Automatic expression dispatch and the broader A7
residue/constant-field cells remain open.

After logarithmic residues are removed:

- regularize the remaining differential;
- compute its class in the exact-differential quotient using A5;
- solve for an algebraic primitive when the class vanishes; and
- return a certified non-elementarity obstruction when it does not.

This replaces the current direct in-field exactness slice with a total
decision over the admitted pure algebraic cell. Wire the result into the
ordinary expression and tower dispatcher; users must not construct an
`AlgebraicFunctionField` or integral basis by hand.

**Exit gate:** every admitted pure algebraic expression is decided
automatically, including high-degree nonradical relations and positive-genus
curves.

### A9. Solve scalar and parametric RDEs on algebraic curves

Use the geometry engine to implement the algebraic-curve Risch differential
equation:

```text
D(y) + a*y = b
```

and its parametric form. Include:

- local valuation bounds at all places;
- infinity bounds;
- reduction to a finite Riemann--Roch space;
- the residual constant-field linear system;
- integrating-factor and logarithmic-derivative cases; and
- exact reconstruction and obstruction certificates.

The current `a = 0` exactness solver and verified integrating-factor route are
base cases and oracles, not evidence that the general RDE is complete.

**Exit gate:** the scalar and parametric algebraic RDEs are total on the
admitted curve cell and satisfy substitution-based reconstruction tests.

### A10. Generalize geometry and integration to mixed towers

Lift A1-A9 from `k = C(x)` to the relative setting used by the full Risch
algorithm:

- an algebraic extension above a primitive, hyperexponential, or
  hypertangent level;
- a monomial extension whose coefficient field is algebraic;
- repeated interleavings, collapsed by primitive-element normalization where
  useful;
- relative integral bases, Hermite reduction, residues, and infinity data;
- scalar, parametric, and coupled RDE recursion through algebraic levels; and
- trace, norm, logarithmic derivative, and no-new-constants structure
  decisions across the interleaving.

The reusable absolute function-field engine remains independent. Relative
adapters supply the effective operations of the lower differential field
rather than forking the geometry algorithms.

**Exit gate:** `integrate_elementary` is total on every explicit, admitted
finite interleaving of algebraic, primitive, exponential, and real
tangent/arctangent extensions.

### A11. Finish automatic algebraic and mixed expression execution

Resume Plan 014 after A1 and A10:

- normalize `RootOf`, radicals, and independent algebraic candidates into
  composita;
- identify and minimize algebraic dependencies;
- execute monomial-over-algebraic towers;
- build the normalized function field and integral bases automatically;
- preserve source-to-tower and tower-to-source maps; and
- keep local complex identities separate from branch-aware real
  presentation.

**Exit gate:** accepted SymPy expressions need no caller-supplied tower,
function-field model, basis, divisor, or torsion order.

## Testing strategy

Use mathematical families rather than one showcase curve:

| Layer | Required families |
| --- | --- |
| Presentation | reducible input, nonmonic input, independent radicals, nontrivial composita, constant enlargement |
| Orders/places | ramified, split, inert, singular plane model with nonsingular normalization, several infinity places |
| Riemann--Roch | genus 0, 1, and at least 2; special and nonspecial divisors; negative and large positive degree |
| Divisor classes | principal, finite torsion, nontorsion, large-height representatives, no degree-one rational place |
| Logarithmic part | rational and algebraic residues, several residue fields, branch residues, torsion visible only after a residue-linear combination |
| Second kind | exact and nonexact residue-free differentials |
| Algebraic RDE | `a = 0`, verified integrating factor, genuine nontrivial `a`, parametric solution and obstruction |
| Mixed | algebraic-over-P/E/T, P/E/T-over-algebraic, and at least one repeated interleaving |
| Frontend | radicals, `RootOf`, algebraic composita, dependent candidates, automatic reconstruction |

For each constructive algorithm, differentiate or take the relevant divisor
to reconstruct the input. For each negative result, test the theorem
hypotheses and the finite-dimensional obstruction, not merely the result
enum. Keep bounded searches explicitly inconclusive.

Independent checks should be used where they test mathematics rather than
architecture:

- current long Hess basis versus the new parametric basis;
- local valuation constraints versus Hess spaces on small divisors;
- direct class iteration versus fast scalar multiplication on small orders;
- Trager resultant residues versus local all-place residues; and
- direct differentiation of every returned antiderivative.

## Performance gates

Track at least:

- extension degree;
- genus;
- maximal-order discriminant degree;
- divisor support and coefficient bit length;
- reduced-basis matrix size/degree;
- residue constant-field degree; and
- class-order bit length.

Add benchmark families for a high-degree nonradical relation, several
positive-genus curves, a large-height divisor, and a residue compositum.
Avoid performance gates tied to one machine; assert algorithmic work counts
where possible and retain wall-clock benchmarks for regression reporting.

Also track the external SymPy algebraic-field `Field.gcd` fallback separately.
Its repeated failed `get_ring()` conversion is an upstream domain-layer issue:
fix and test it in SymPy, then consume the released or pinned correction here
rather than adding a Risch-local monkey patch.

## Source map

- F. Hess, *Computing Riemann--Roch Spaces in Algebraic Function Fields and
  Related Topics*, especially Theorem 7 and Algorithms 13, 21, and 24.
- Barry M. Trager, *Integration of Algebraic Functions*.
- Brian L. Miller, *On the Integration of Elementary Functions: Computing
  the Logarithmic Part*.
- Daniel Schultz, *Trager's Algorithm for Integration of Algebraic Functions
  Revisited*.
- Manuel Bronstein, *The Risch Differential Equation on an Algebraic Curve*.
- J. H. Davenport, *Intégration algorithmique des fonctions élémentairement
  transcendantes sur une courbe algébrique*.

## Program exit gate

This plan is complete only when:

1. the Hess/function-field layer is total over its declared exact input cell;
2. pure algebraic integration is an automatic total decision on that cell;
3. algebraic scalar and parametric RDEs are complete there;
4. relative mixed towers are total over the declared Risch input class; and
5. the expression frontend constructs every supported tower and basis without
   caller-supplied algebraic geometry.

Until then, the missing branch is `UNSUPPORTED` or `UNKNOWN`; it is never a
non-elementarity certificate.
