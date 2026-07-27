# Plan 001: Make the Risch implementation complete over its declared input class

> **Executor instructions**: This is a program roadmap, not one atomic coding
> task. Before implementing any milestone below, write a checkpoint plan that
> narrows the milestone to a reviewable mathematical capability, its exact
> source and test files, and one separate commit. Run every verification gate
> for that checkpoint. Do not claim a negative integration decision from an
> incomplete branch.
>
> **Drift check (run first)**:
> `git diff --stat "$(git log -1 --format=%H -- plans/001-full-risch-roadmap.md)"..HEAD -- src tests docs README.md pyproject.toml`
> If relevant code has changed, reconcile the "Current state" and coverage
> ledger before selecting the next checkpoint.

## Status

- **Priority**: P1
- **Effort**: XL, a multi-stage research and implementation program
- **Risk**: HIGH, because a false negative certificate is a mathematical bug
- **Depends on**: none
- **Category**: direction
- **Planned at**: commit `bf9a424`, 2026-07-24
- **Last implementation reconciliation**: Plans 006 finite algebraic
  valuation constraints, 007 narrow P/E/T expression frontend, 008 certified
  Riccati Möbius normalization, 009 local linear-special-pole reduction, and
  010 finite algebraic constant bases plus the first one-step profile-selected
  algebraic-constant target integration cell, 2026-07-26

## Why this matters

The existing roadmap identifies nearby missing branches but does not define a
testable endpoint for "the full Risch algorithm." In particular, it does not
yet place Bronstein's parametric and structure-theorem algorithms, the
effective constant-field assumptions, the coupled differential system, or the
complete mixed algebraic/transcendental algorithm into the dependency graph.

This plan defines completion precisely and orders the work so that every
negative answer is justified. It follows Bronstein's *Symbolic Integration I:
Transcendental Functions* for the transcendental kernel, then uses the
Trager-Davenport-Bronstein algebraic and mixed algorithms that lie outside the
scope of that volume.

## What "complete" means

### Mathematical input class

The core completion target is:

1. A characteristic-zero computable constant field `C` with effective:
   - exact arithmetic and equality;
   - polynomial factorization;
   - algebraic extensions and embeddings;
   - rational linear algebra; and
   - the constant relations required by the structure algorithms.
2. The rational differential base `C(x)` with `Dx = 1`.
3. A finite, regular elementary tower obtained by adjoining, in any order:
   - finite separable algebraic elements;
   - primitive/logarithmic monomials;
   - hyperexponential/exponential monomials; and
   - the real tangent/arctangent representations equivalent to elementary
     complex extensions.
4. General polynomial monomials already admitted by this library, including
   the branch with nonconstant special polynomials, as an explicit extension
   of the classical elementary-input target.

The first required constant fields are `QQ`, algebraic number fields over
`QQ`, and finitely generated formal constant extensions for which the required
relations are supplied or computable. "Every SymPy constant" is not a coherent
completion criterion: arbitrary constant identities and arbitrary elementary
zero equivalence are not generally decidable.

### Required result

For every accepted input in that class, the public decision procedure must
terminate with exactly one of:

- a constructive elementary antiderivative represented in an explicit
  elementary extension, together with an exact reconstruction certificate; or
- a proved Liouville obstruction showing that no elementary antiderivative
  exists.

`NotImplementedError`, `unknown`, and a heuristic failure are forbidden for an
accepted input. They remain valid only at the admission boundary, where the
system explains which effective-field requirement or expression construct is
unsupported.

### Local identity versus global presentation

The differential-field algorithm proves a local identity on a domain where the
input generators and branches are valid. A globally continuous real
antiderivative is a separate presentation problem. Completion therefore has
two explicit outputs:

- a branch-independent differential-field certificate; and
- when real assumptions are requested, interval/branch metadata and a real
  presentation whose discontinuities are not hidden by unsafe simplification.

### Explicit exclusions from the classical completion gate

These may be added later, but are not part of "full Risch":

- nonelementary special-function output such as `erf`, `Ei`, gamma,
  hypergeometric, or elliptic functions;
- definite integration and contour integration;
- arbitrary simplification or identity decisions outside the effective field;
- distributional derivatives of `Abs`, `sign`, or unrestricted `Piecewise`;
  and
- numerical branch tracking without symbolic assumptions.

## Current state

The current implementation already has a strong exact-arithmetic kernel:

- `docs/design.md` records end-to-end rational, recursive primitive, direct
  hyperexponential/hypertangent, recursive primitive-lower
  hyperexponential branches,
  generalized monomial reduction, RDE components, and algebraic Hermite
  reduction.
- `src/sympy_risch/tower.py` classifies monomial families, records
  constant-field and special-polynomial evidence, represents finite
  algebraic extensions with SymPy's `FiniteExtension`, and differentiates
  algebraic generators implicitly, and classifies which algebraic/monomial
  prefixes the current native backend can actually represent.
- `src/sympy_risch/dependency.py` provides parametric
  logarithmic-derivative decisions for rational and recursive primitive paths,
  plus implemented mixed relation-system and hypertangent Gaussian fallbacks.
  Some P/E/T orderings still reach the `QQ(x)`-only primitive-PLogD boundary.
- `src/sympy_risch/monomial.py` provides recursive radical logarithmic
  decisions, primitive reduction, and direct P/E/T family reducers. Its
  nonlinear reducer is complete only under a certified absence-of-special
  hypothesis; a separate positive-evidence API now cancels a currently
  exposed principal part at one supplied linear special factor when its lower
  RDE is solvable, without widening that theorem.
- `src/sympy_risch/rde.py` implements scalar and coupled RDEs for documented
  normalized P/E/T solver shapes, including rational frontends, weak
  normalization, denominator/degree bounds, SPDE, and direct Chapter 8
  cancellation recurrences. Its shared scalar admission checks the complete
  recursive prefix before a local linear-special-pole consumer invokes it;
  general polynomial special-factor RDEs and parametric RDEs remain open.
- `src/sympy_risch/exact.py` dispatches exact derivatives at rational and
  normalized P/E/T levels.
- `src/sympy_risch/elementary.py` recursively integrates documented explicit
  P/E/T shapes with certified preserved constants. Structurally admitted
  Some PTE primitive-PLogD examples retain the explicit boundary. It
  deliberately rejects algebraic levels and
  general polynomial monomials whose special structure has not been decided.
- `src/sympy_risch/expression.py` compiles a conservative one-variable
  `QQ`-constant grammar of rational arithmetic and nested P/E/T calls into
  that certified explicit tower, returning either a local presented answer, a
  native obstruction, or a structured compiler boundary. It intentionally
  does not minimize general expression generators or resolve global branches.
- `src/sympy_risch/algebraic.py` validates algebraic extensions and implements
  algebraic Hermite reduction over `QQ(x)` with explicit, simple-radical, or
  lazily refined integral bases, plus Trager's finite-residue polynomial for
  supported unbranched poles. `src/sympy_risch/algebraic_geometry.py` now adds
  certified maximal orders, ramified and infinite places, residue fields,
  local expansions, divisors, and differential residues. A narrow
  constant-field analyzer and all-place collector are implemented for admitted
  direct `C(x)` simple-radical presentations. A theorem-limited global-space
  kernel now certifies explicit complete Riemann--Roch cases and exact
  finite-dimensional valuation intersections. A first direct-
  `C(x)`, `C/QQ` finite algebraic, displayed-constant logarithmic slice is
  constructive; general lattice intersection, exact presentation
  normalization, compatible residue descent, and second-kind exactness remain
  open.

The implementation checkpoints since this roadmap was created are:

```text
92599b0  Define parametric logarithmic derivative results
0a48352  Implement parametric logarithmic derivative heuristic
444aebf  Complete rational parametric logarithmic derivatives
f47439e  Add independent parametric logarithmic oracle
5556d95  Add primitive logarithmic signatures
6e8e693  Complete primitive parametric logarithmic derivatives
90f2533  Reconcile the full Risch roadmap
30f280d  Correct the algebraic residue roadmap
71fe12e  Compute algebraic finite residues
a9e62ff  Generalize primitive logarithmic signatures recursively
c3225a2  Generalize primitive parametric decisions recursively
889e4f6  Complete primitive RDE cancellation
bc763f1  Reconcile recursive primitive checkpoint
4d732c6  Complete primitive RDE weak normalization
```

At the last reconciliation, the full parallel suite passed using:

```console
uv run --no-sync pytest -q
```

The existing design conventions to preserve are:

1. Native SymPy `Domain`, `Poly`, fraction-field, and finite-extension
   elements; no parallel element hierarchy.
2. Immutable `DifferentialTower` and extension metadata.
3. Structural classification from derivative laws.
4. Exact reconstruction in the native field for every constructive result.
5. Explicit evidence for no-new-constants and special-polynomial hypotheses.
6. Separate success, proved obstruction, and unsupported-input outcomes.
7. Independent algorithmic oracles where practical.
8. Small checkpoint commits that leave the suite passing.

### Milestone ledger

| Milestone | Current status |
| --- | --- |
| 0. Completeness contract and coverage ledger | Structural P/E/T admission and coverage ledger complete: the public admission result and declared-cell matrix distinguish unsupported towers from theorem-backed obstructions without claiming every admitted solver path is implemented. |
| 1. Effective field and tower kernel | Partial: exact `QQ` towers, native finite algebraic number-field constants at a standard rational base, and finite extensions exist; a checked feasibility result maps the current mixed-prefix backend boundary, while general constant capabilities and canonical mixed algebraic towers remain. |
| 2. Dependency and structure decisions | Partial: the normalized P/E/T relation, fundamental signatures, Gaussian fallbacks, an explicit two-linear-factor Riccati normalizer, and positive-evidence local linear-special-pole reduction are checked over `QQ(x)`. Native finite `QQ`-algebraic constants now support the base relation kernel and direct P/E factory proposals; proof-carrying recursive constant evidence, general effective constants, and automatic non-generator special-factor decisions remain. |
| 3. Scalar, parametric, and coupled RDEs | Partial: scalar and coupled RDEs cover documented normalized P/E/T solver shapes, and scalar lower RDEs drive one explicit local special-pole consumer; parametric RDE/limited integration and general special-factor branches remain. |
| 4. Transcendental integration | Partial: explicit P/E/T integration covers documented shapes rather than every family ordering, alongside a narrow checked one-variable P/E/T expression grammar, an explicit two-factor Riccati pullback route, and local certified linear-special-pole reduction; general polynomial special branches and general expression-to-tower construction are not. |
| 5. Algebraic-function-field kernel | Partial: maximal orders, all finite/infinite places, local expansions, divisor arithmetic, all-place residue collection, theorem-certified first global spaces, exact finite valuation kernels, and a bounded principal-divisor semialgorithm now exist; constant normalization and general global-space/divisor-class decisions remain. |
| 6. Pure algebraic integration | Partial: direct `C(x)`, for `C/QQ` finite algebraic, now reconstructs displayed-constant residues with bounded-search principal witnesses; compatible constant descent, torsion/divisor-class work, and residue-free exactness remain. |
| 7. Mixed algebraic/transcendental towers | Not started. |
| 8. Automatic expression towers | Partial: a conservative reversible P/E/T frontend is complete over its declared grammar; algebraic/mixed discovery, canonical generator minimization, and branch-aware presentation remain. |
| 9. Real/complex presentation | Partial exploratory support for rational quadratic factors only. |
| 10. Completeness evidence | Partial: the declared P/E/T cell matrix, testing conventions, and independent oracles exist; algebraic, mixed, and automatic-tower coverage remains. |
| 11. SymPy upstreaming | Not started. |

## Reference map

| Capability | Primary reference |
| --- | --- |
| Rational integration | Bronstein, Chapter 2 |
| Differential fields and monomials | Bronstein, Chapters 3-4 |
| Transcendental reduction and Liouville tests | Bronstein, Chapter 5 |
| Scalar Risch differential equation | Bronstein, Chapter 6 |
| Parametric RDE, limited integration, parametric logarithmic derivative | Bronstein, Chapter 7 |
| Coupled differential systems, including real tangent | Bronstein, Chapter 8 |
| Complex and real Risch structure theorems | Bronstein, Chapter 9 |
| General algebraic-function integration | Trager, *Integration of Algebraic Functions*; Davenport, *On the Integration of Algebraic Functions* |
| Mixed algebraic/logarithmic/exponential extensions | Bronstein, "Integration of elementary functions," JSC 9 (1990), 117-173 |
| Lazy integral-basis refinement | Bronstein, *The Lazy Hermite Reduction* |

Bronstein's monograph is the principal implementation guide for the
transcendental track, but it is not by itself the complete algebraic/mixed
roadmap.

Raab and Singer's 2022 *Integration in Finite Terms: Fundamental Sources* is
the local consolidated source for the foundations and algebraic track. It
reprints Rosenlicht's theorem, Risch's algebraic-operations paper, and Trager's
thesis, together with modern commentary. See `references/README.md` for exact
PDF-page pointers.

## Program dependency graph

There are two tracks after the shared decision-oracle foundation:

```text
Shared effective-field and certificate contracts
                |
                +-----------------------------+
                |                             |
                v                             v
   Transcendental structure/RDE       Algebraic-function-field kernel
                |                             |
                v                             v
   Complete transcendental reducer     Complete algebraic integrator
                |                             |
                +--------------+--------------+
                               |
                               v
             Arbitrarily interleaved mixed towers
                               |
                               v
                Expression-to-tower construction
                               |
                               v
                  Real and complex presentation
                               |
                               v
                    Completeness/upstream gates
```

The algebraic track can be developed in parallel conceptually, but mixed-tower
completion waits for both tracks.

## Milestone 0: Establish the completeness contract and coverage ledger

### Deliverables

- Add a machine-readable or table-driven inventory of accepted:
  - constant domains;
  - tower-level families;
  - parent-field families;
  - decision problems; and
  - result outcomes.
- Publish `assess_elementary_admission`, `TowerAdmission`,
  `TowerAdmissionOutcome`, and `TowerAdmissionReason` for the declared
  elementary-dispatch boundary. The result must carry the exact failed
  admission requirement, where applicable, and must not be a mathematical
  no-solution result.
- Record the current certified P/E/T contract, including the exact hyperbolic
  Möbius normalization exception at the active top level. Keep the
  experimental nonlinear no-special-polynomial branch outside that total
  public contract.
  The user-facing boundary is documented in the
  [mixed-tower admission contract](../docs/mixed-transcendental-towers.md).
- Define certificate protocols for:
  - exact derivative;
  - logarithmic derivative and radical logarithmic derivative;
  - scalar and parametric RDE;
  - elementary integration; and
  - constant-field/dependency decisions.
- Record the theorem hypotheses used by every obstruction certificate.
- Add a test which enumerates all declared supported cells and fails if a cell
  reaches `NotImplementedError`.

### Exit gate

It is possible to answer mechanically whether a tower belongs to the promised
input class, every declared accepted cell reaches reconstruction or a
structured obstruction rather than `NotImplementedError`, and every negative
result names the theorem and executable hypotheses that justify it. Do not
mark this milestone complete until the public admission result, cell matrix,
documentation, and CI gate are all present.

## Milestone 1: Generalize the effective field and tower kernel

### 1.1 Constant-field capabilities

Introduce a narrow capability interface backed by SymPy domains:

- exact equality and zero testing;
- factorization and square-free factorization;
- rational and constant-field linear algebra;
- algebraic closure operations needed for residues;
- embeddings/conjugation;
- a finite basis over `QQ` when Chapter 9 rational projection needs one; and
- explicit refusal when a required constant relation is not decidable.

Do not force `QQ` assumptions into algorithms whose mathematics only needs
these capabilities.

### 1.2 Canonical mixed towers

- Support field coercion and canonical representation for:
  - monomial over monomial;
  - algebraic over monomial;
  - monomial over algebraic; and
  - repeated algebraic levels collapsed through a primitive element when
    appropriate.
- Preserve immutable conceptual levels even if SymPy uses a flattened physical
  representation.
- Add exact trace, norm, minimal-polynomial, embedding, and conjugation
  operations needed by later algebraic and real algorithms.
- Cache only immutable derived data such as generator derivatives, bases,
  factorization data, and local expansions.

### Exit gate

Every supported mixed field has exact coercion, equality, differentiation,
factorization, trace/norm, and stable canonical representation independent of
the integration algorithms.

## Milestone 2: Complete dependency and structure-theorem decisions

This transcendental track is active. The normalized P/E/T relation system has
implemented certified directions over `QQ(x)`, and its rational base kernel now admits native finite
algebraic constant fields over `QQ`; this milestone extends that theorem
machinery to proof-carrying recursive effective fields and general polynomial
monomials. It can proceed in parallel with the algebraic-function-field
kernel.

### 2.1 Parametric logarithmic derivative over current towers

Implement Bronstein §7.3 for the existing rational and primitive towers:

```text
n*f = D(v)/v + m*D(theta)/theta
```

Status: complete for the rational and recursive primitive paths plus the
implemented mixed relation-system and Gaussian fallback shapes, including the fast
hypertangent Gaussian normalization, integer normalization, exact witness
reconstruction, generated planted witnesses, and independent global
linear-system oracles. Plan 003 exposes those already-certified P/E/T
directions as immutable signatures. Plan 010 also gives a standard
`C(x)` base for native finite algebraic `C/QQ`, but deliberately does not
trust caller metadata to recurse through such towers. The remaining work is
proof-carrying general-constant recursion and non-generator special factors.

- Keep the fast denominator/degree/split-factor algorithm.
- When it returns "failed," invoke the Chapter 9 structure-theorem solver.
- Solve the resulting exact rational linear system.
- Normalize integer solutions and construct a radical witness.
- Verify the defining equation exactly.
- Include Bronstein Example 7.3.2 as a regression: the fast path fails but the
  structure fallback finds the obvious solution.

### 2.2 Complex Risch structure theorem

Status: the P/E/T relation compiler supplies the specialized implementation;
Plan 003 exposes its certified source directions as immutable signatures
rather than duplicating its P/E/T logic. Plan 010 expands a finite algebraic
constant at its standard base into exact `QQ` power-basis coordinates, so the
valuation lattice remains rational. The signature is deliberately not a
general Darboux-polynomial decision.

- Compute and retain the ordered logarithmic and exponential index sets for
  the tower.
- Build the fundamental signatures:

  ```text
  Dt_i                         for logarithmic generators
  Dt_i/t_i                     for exponential generators
  ```

- Solve Corollary 9.3.1's rational linear systems over the represented
  constant field.
- Decide:
  - exact derivatives arising from logarithmic data;
  - logarithmic derivatives of radicals;
  - exponential algebraic dependence; and
  - no-new-constants for proposed generators.

### 2.3 Real structure theorem

Status: hypertangent P/E/T inputs use an immutable Cayley Gaussian normal
form. Plan 003 records the real source coefficient `eta` in the signature and
keeps the Cayley image `2*i*eta` as an independently checked adapter. Direct
real rational relations for general special polynomials and a branch-aware
public presentation remain future work.

- Add the tangent and arctangent index sets from Theorem 9.3.2.
- Implement the real rational relations of Corollary 9.3.2.
- Keep the complex Cayley transform as an independent oracle, not the only
  production presentation.

### 2.4 Special-polynomial decision

Status: primitive levels with certified preserved constants now have a
complete no-special theorem; hyperexponential and hypertangent special
behavior has family-specific positive evidence. A caller-supplied monic
linear certificate can now drive local special-pole reduction through a
scalar lower RDE, but general polynomial monomials still lack a complete
automatic Darboux/no-special decision.

- Extend derivation-bound special-factor analysis wherever the structure
  theorems make it effective.
- Return explicit special factors and their multiplicative relations, not only
  `PRESENT`.
- Feed those factors into the general nonlinear and RDE special-denominator
  branches.

### Exit gate

For every explicitly listed current solver cell, dependency,
radical logarithmic derivative, and no-new-constants questions reach a
decision or documented boundary. General special-polynomial absence remains a
future algorithmic decision; `UNKNOWN` is possible when a declared
effective-field prerequisite or that future decision is absent.

## Milestone 3: Complete scalar, parametric, and coupled RDEs

Status: scalar and coupled RDE dispatch is implemented for documented
normalized P/E/T solver shapes. The work below applies to general special polynomial monomials and
to the public parametric/limited-integration layer; it must not reopen or
replace the P/E/T production recurrences.

### 3.1 Scalar RDE over arbitrary supported lower towers

Complete:

```text
Dy + a*y = b
```

using Bronstein Chapter 6:

- weak normalization;
- normal denominator;
- every family-specific special denominator;
- degree bounds at finite places and infinity;
- SPDE;
- polynomial no-cancellation cases;
- primitive cancellation;
- hyperexponential cancellation;
- hypertangent/nonlinear cases; and
- recursion into any supported lower tower.

The current Bronstein recurrence remains the production polynomial solver and
bounded exact matrix ansatzes remain test oracles.

### 3.2 Parametric RDE and limited integration

Implement Chapter 7:

```text
Dy + a*y = sum(c_i*b_i),       c_i in Const(K)
```

- Return a basis for all solutions, not one arbitrary vector.
- Implement limited integration as a consumer.
- Use the parametric logarithmic-derivative solver from Milestone 2 for degree
  bounds and cancellation.

### 3.3 Coupled differential systems

Implement Chapter 8 for:

- primitive;
- hyperexponential;
- nonlinear;
- hypertangent; and
- recursive lower-field cases.

Generalize the existing two-component rational rotation solver. Verify real
solutions directly and through conjugate complex scalar systems.

### Exit gate

Scalar RDE, parametric RDE, limited integration, and required coupled systems
are total over every supported explicit transcendental tower.

## Milestone 4: Complete transcendental integration

Status: the primitive, hyperexponential, and hypertangent branches are
complete for explicit certified towers. The remaining integration branch is
the general polynomial monomial with nonconstant special factors.

Using the completed RDE and dependency engines, finish every branch in
Bronstein Chapter 5:

- canonical/Hermite reduction;
- polynomial reduction;
- generalized residue criterion;
- integration of reduced functions;
- primitive case;
- hyperexponential case;
- hypertangent case;
- nonlinear case both with and without special polynomials; and
- in-field integration.

Remove assumptions tied to the direct `QQ(x)` lower field. Support arbitrary
finite nestings and orderings of the supported transcendental families.

### Exit gate

For an explicit same-constants transcendental tower over a supported constant
field, `integrate_elementary` never reaches an implementation gap and returns
either an exact elementary answer or a theorem-backed obstruction.

This is the first major release target: **complete explicit transcendental
Risch**, before automatic expression discovery.

## Milestone 5: Build the algebraic-function-field kernel

This milestone supplies the geometry currently absent from
`src/sympy_risch/algebraic.py`.

Status: the local geometry substrate is implemented. Finite extensions,
implicit differentiation, integral-basis coordinates, simple-radical bases,
lazy refinement, algebraic Hermite reduction over `QQ(x)`, and Trager's
nested-resultant finite-residue polynomial are implemented. Certified maximal
orders, ramified/infinite places, exact residue fields, uniformizers,
valuations, local expansions, differents, and divisor arithmetic now exist in
`algebraic_geometry.py`. A narrow constant-field gate and an all-place residue
collector already consume that canonical geometry API for admitted preserved
presentations. The immediate gate is now global divisor/cohomology work, plus
actual presentation normalization when constants enlarge.

### 5.1 Integral closure and bases

- General integral-basis computation over supported rational and monomial
  parents.
- Lazy refinement retained as the preferred fast path.
- Integral bases at finite places and infinity.
- Different/discriminant and normalization data.

### 5.2 Places and local computation

- Finite and infinite places;
- ramification indices, residue fields, and uniformizers;
- exact valuations;
- Puiseux/local expansions to a certified order; and
- residues of algebraic differentials.

### 5.3 Divisors and global functions

- Divisor arithmetic and principal-divisor testing;
- Riemann-Roch or an equivalent effective global-space algorithm;
- trace, norm, and logarithmic derivative behavior;
- torsion-divisor handling required for elementary logarithmic terms; and
- exactness testing for differentials of the second kind.

Use SymPy arithmetic where it provides the necessary exact operation; add
integration-specific geometry beside it rather than creating a duplicate
general-purpose computer-algebra system.

### Exit gate

The library can compute the local and global invariants needed by Trager's
algebraic integration algorithm for a nonsingular algebraic function field.

## Milestone 6: Complete pure algebraic integration

For `K/C(x)` finite algebraic:

1. Perform general algebraic Hermite reduction.
2. Compute all finite and infinite residues.
3. Prove residues are constants in the necessary constant extension.
4. Reconstruct logarithmic terms by solving the associated principal/torsion
   divisor problems.
5. Test whether the remaining differential of the second kind is exact.
6. Return the algebraic exact part, logarithmic terms, or a precise
   obstruction.

Required cases include:

- simple radicals and high-degree nonradical defining polynomials;
- ramification at finite points and infinity;
- nontrivial integral bases;
- algebraic constants and residue-field extensions;
- genus zero, elliptic, and positive-genus curves; and
- elementary pseudo-elliptic cases whose logarithmic part is easy to miss.

### Exit gate

Every element of a supported algebraic function field over `C(x)` receives a
complete elementary-integrability decision.

## Milestone 7: Complete mixed algebraic/transcendental towers

Implement the generalization in Bronstein's 1990 "Integration of elementary
functions":

- treat an algebraic extension of `k(delta)`, where `delta` is logarithmic or
  exponential over `k`, as an algebraic function field over `k`;
- generalize Hermite reduction and local expansions to a Liouvillian ground
  field;
- solve scalar and parametric RDEs in algebraic extensions;
- compute generalized residues and infinity expansions;
- propagate trace/norm and logarithmic data through algebraic levels; and
- recurse for algebraic levels below, above, and between transcendental
  levels.

Extend the Chapter 9 structure decisions so algebraic layers do not reintroduce
heuristic `unknown` outcomes.

### Exit gate

`integrate_elementary` is total over every explicit tower in the mathematical
input class, including arbitrary finite interleavings of algebraic,
logarithmic, exponential, and real tangent/arctangent levels.

This is the core mathematical definition of a **complete explicit full Risch
implementation**.

## Milestone 8: Construct minimal towers from SymPy expressions

### 8.1 Accepted expression grammar

Define and test an explicit grammar including:

- arithmetic and integer/rational powers;
- algebraic `RootOf`/radical expressions;
- `exp`, `log`;
- trigonometric, hyperbolic, and inverse forms reducible to elementary
  extensions; and
- constant parameters admitted by the constant-field capability layer.

### 8.2 Generator discovery and ordering

- Collect algebraic and transcendental candidates.
- Normalize multiplicative/additive constant shifts.
- Build a dependency graph.
- Use the complete exact/logarithmic/structure solvers to merge dependent
  generators.
- Recognize rational powers as algebraic extensions rather than falsely
  transcendental generators.
- Select a minimal generator set and topologically order it.
- Prove no-new-constants and special-polynomial facts during construction.

The builder must handle at least:

```text
exp(x/2), exp(x)          -> one generator with an integer-power relation
exp(x), exp(x + 1)        -> one generator and a constant multiplier
log(x), log(x**2)         -> one differential generator plus branch metadata
tan(log(x))               -> primitive then hypertangent
exp(log(x)/2)             -> algebraic, not transcendental
algebraic(exp(log(x)))     -> simplified dependency before tower admission
```

### 8.3 Exact rewrite and back-substitution

- Rewrite the input into a native exact top-field element.
- Retain an invertible map between field generators and source expressions.
- Verify round-trip reconstruction before integration.
- Keep branch-sensitive presentation facts out of the algebraic field element.

### Exit gate

Every expression in the accepted grammar either constructs and verifies a
minimal supported tower or returns a precise unsupported-input reason. No
manual `ConstantFieldStatus.PRESERVED` assertion is required for constructed
towers.

## Milestone 9: Complete real and complex presentation

- Back-substitute formal generators into source expressions.
- Present algebraic numbers and functions without losing exactness.
- Combine conjugate logarithms into real logarithms and angles.
- Use exact real-root isolation for general rational and algebraic residues.
- Preserve tangent/arctangent forms when they give the natural real answer.
- Track intervals, singularities, branch cuts, phase/unwinding corrections,
  and additive constants explicitly.
- Avoid global identities such as `atan(tan(u)) = u` without domain evidence.
- Offer a complex-local answer even when a global real presentation is not
  requested.

### Exit gate

Every constructive field certificate can be converted into an exact SymPy
expression whose derivative reconstructs the input under the recorded
assumptions. Real mode additionally states the domains on which its
presentation is valid.

## Milestone 10: Establish completeness evidence

### Coverage matrix

The test suite must cover the cross-product of:

- top extension family;
- lower-field family;
- algebraic placement;
- denominator type: normal, special, mixed, and infinity;
- RDE case: cancellation and every no-cancellation branch;
- result: constructive, proved obstruction, and rejected admission with a
  stable `TowerAdmissionReason`; and
- complex versus real presentation.

### Required test layers

For every algorithm:

1. exact reconstruction;
2. canonical postconditions;
3. idempotence where applicable;
4. independent formulation or oracle;
5. generated planted solutions and obstructions;
6. adversarial rejected preconditions;
7. smallest regression for every false certificate;
8. examples from the primary literature; and
9. bounded performance tests for degree, height, and tower depth.

Maintain production algorithms and structurally independent test oracles where
possible. Do not use `sympy.integrate` as the sole oracle.

### Completeness gate

Add a public test corpus which:

- constructs every supported explicit tower-shape cell;
- checks each cell with `assess_elementary_admission` before dispatch;
- runs every decision problem required by the integration dispatcher;
- asserts that no cell raises `NotImplementedError`;
- distinguishes `TowerAdmissionOutcome.UNSUPPORTED` from a mathematical
  non-elementarity result;
- verifies every positive certificate by exact differentiation;
- verifies every negative certificate's hypotheses; and
- exercises automatic construction and back-substitution for every expression
  grammar rule.

## Milestone 11: Integrate with and upstream to SymPy

Upstream incrementally rather than as one replacement:

1. exact domain/tower fixes that are independently useful;
2. rational and algebraic reduction improvements;
3. RDE and structure-decision components;
4. algebraic-function-field infrastructure;
5. expression tower construction;
6. the complete dispatcher and result certificates; and
7. real presentation.

Each proposal must:

- use SymPy naming and public-domain conventions;
- include standalone tests and documentation;
- avoid depending on this package's private compatibility layer;
- preserve existing `integrate` fallback behavior until the new branch is
  demonstrably total on its declared input class; and
- distinguish non-elementary proof from unimplemented input in the public API.

### Exit gate

The upstream-facing Risch entry point accepts the declared elementary
expression grammar and is total over its effective field class. The standalone
package may then become a development harness or be retired.

## Immediate execution order

The following order reflects the current committed state. A checkpoint may run
in parallel only where the dependency graph says so; every item remains a
separate commit after focused and full-suite verification.

### Shared contract gate

1. [x] **Plan 002**: made the current `QQ(x)` P/E/T admission
   boundary, including the exact hyperbolic Möbius normalization exception at
   the active top level,
   and coverage cells executable. The documented type/test gates are now in
   CI.

### Parallel first mathematical gates

2. [x] **Plan 003**: extracted the current P/E/T relation logic into
   immutable complex and real fundamental signatures. The existing compiler
   and Gaussian normal form remain behavioral oracles.
3. [x] **Plan 004**: decide or explicitly reject algebraic constant-field
   enlargement, then bridge nonzero admitted Hermite remainders to canonical
   all-place residue data. Exact Hermite successes do not need global geometry.
   Do not extend the unramified plane-model shortcut by guess.
4. [x] Add a minimal native-SymPy mixed-prefix feasibility analyzer. It
   certifies a monomial prefix with an optional terminal algebraic level by
   exercising native coercion and differentiation, and reports
   monomial-over-algebraic as an explicit backend limitation. Nested algebraic
   extension construction remains rejected before a proposal can exist. It is
   a narrow boundary map, not a broad element-hierarchy rewrite.

### Transcendental completion path

5. [x] **Plan 008**: add a certified two-linear-factor Riccati Möbius
   normalizer and explicit integration route. It records, rather than
   searches for, the special-factor proof and leaves ordinary general-
   polynomial dispatch unchanged.
6. [x] **Plan 009**: consume one caller-supplied monic linear special-factor
   certificate in a local exact principal-part reduction. It records solved,
   unsolved, and unsupported lower RDEs without treating a local failure as a
   non-elementarity result.
7. [ ] Extend fundamental signatures to the declared effective constant fields
   and use them for automatic no-new-constants/dependency certification.
   - [x] **Plan 010**: admit native finite algebraic `C/QQ` at the standard
     `C(x)` base, retain rational-coordinate relation constraints, and enable
     direct checked primitive/hyperexponential proposals.
   - [ ] Replace caller metadata with replayed or proof-carrying
     no-new-constants evidence before recursively certifying a general-`C`
     P/E tower.
8. [ ] Solve the complete special-polynomial/Darboux decision for each
   explicitly admitted general-monomial family. In particular, a linear factor
   `t-a` requires the Riccati equation `D(a) = Dt|_(t=a)`; failed search is
   never an `ABSENT` proof.
9. [ ] Extend the local factor consumer to complete special-denominator
   Hermite/RDE reduction, public parametric RDE, limited integration, coupled
   systems, and logarithmic reconstruction. This completes explicit
   transcendental Risch beyond the normalized P/E/T core.

### Algebraic completion path

10. [x] **Plan 005**: establish a first certified Riemann--Roch/global-space
   service on admitted preserved presentations. It proves only theorem-backed
   complete cases and returns no partial basis outside them; general
   finite/infinite lattice intersection remains the next extension.
   **Plan 006** adds exact all-place valuation constraints for any declared
   finite constant subspace and a bounded nonspecial ideal-lattice search;
   only a reached Riemann--Roch dimension certifies a complete global space.
11. [ ] Use it for residue-basis descent and principal/torsion divisor solving
   (algebraic logarithmic reconstruction) and, in parallel, exactness of
   second-kind differentials. Test genus-zero and positive-genus cases.
   - [x] Build the all-place closed-residue profile
     `product_P Norm_{k(P)/C}(Z-res_P)`, including branch and infinity places,
     without pretending independent local residue fields are already a common
     constant extension.
   - [x] Add a bounded, globally verified principal-divisor semialgorithm for
     degree-zero divisors; exhaustion is incomplete, while nonzero degree is
     the complete obstruction.
   - [x] Integrate direct displayed-constant residues over `QQ` or a finite
     algebraic extension of `QQ` when every coordinate divisor has a verified
     bounded-search principal witness.
   - [x] Construct the first compatible base change
     `QQ(x, y) -> QQ(alpha)(x, y)`, split finite places, recollect all target
     residues, and reconstruct a target-field logarithmic answer. An explicit
     factor or the first compatible irreducible factor from the square-free
     all-place residue profile may select the target. This is deliberately
     limited to one factor that splits in its root field.
   - [x] Decide direct residue-free exactness whenever `L(E)` is the certified
     constant space or a bounded nonspecial lattice search reaches its
     Riemann--Roch dimension. A complete failed solve proves only no
     primitive in the current algebraic function field.
   - [ ] Generalize compatible constant descent and place splitting to
     composita and iterative extensions, and solve the remaining
     principal/torsion divisor cases.
12. [ ] Generalize the scalar and parametric RDE to algebraic curves, reusing
    the exactness result as its `a=0` slice.
13. [ ] Compose the completed algebraic and transcendental solvers over the
    proven mixed-tower representation. This is the exit gate for a complete
    *explicit* Risch tower.

### User-facing and upstream path

14. [x] **Plan 007**: build a narrow reversible P/E/T expression compiler for
    one `QQ`-constant symbol, rational arithmetic, integer powers, and nested
    `log`/`exp`/`tan`. It replays checked extension factories and returns only
    local back-substituted presentations; dependent and branch-sensitive
    forms are structured boundaries rather than rewrites.
15. [ ] Expand expression construction to algebraic/mixed expressions only
    after the corresponding mathematical cells are total.
16. [ ] Broaden back-substitution to complex-local versus
    assumption-scoped-real presentation, branch metadata, and a small stable
    SymPy-facing facade.
17. [ ] Expand the coverage matrix to every declared mixed-tower and expression
    cell, then upstream independently mature domain, reduction, RDE, and
    frontend components.

## Commands required for every checkpoint

| Purpose | Command | Expected on success |
| --- | --- | --- |
| Focused tests | `uv run --no-sync pytest -q tests/test_relevant_module.py` | exit 0 |
| Ruff lint | `uv run --no-sync ruff check .` | exit 0 |
| Ruff format | `uv run --no-sync ruff format --check .` | exit 0 |
| Type check | `uv run --no-sync ty check` | exit 0 |
| Complete suite | `uv run --no-sync pytest -q -n auto --dist worksteal` | all pass |
| Patch hygiene | `git diff --check` | no output, exit 0 |
| Changed files | `git status --short` | only checkpoint files and pre-existing user changes |

Documentation examples added by a checkpoint must also run in an installed
interpreter.

## Git workflow

- Use a `codex/` branch when a branch is needed.
- Keep one mathematical checkpoint per commit.
- Match the existing imperative commit style, for example:
  `Reduce hyperexponentials over primitive fields`.
- Do not push or open a pull request unless explicitly requested.

## Done criteria

All conditions must hold:

- [ ] The accepted constant fields and expression grammar are documented and
      mechanically checked.
- [ ] Every accepted expression constructs a verified minimal tower.
- [ ] Exact derivative, radical logarithmic derivative, RDE, PRDE, limited
      integration, and structure decisions are total over that tower class.
- [ ] Pure transcendental, pure algebraic, and arbitrarily interleaved mixed
      towers all receive complete elementary-integrability decisions.
- [ ] Real tangent/arctangent cases are decided without requiring lossy
      complex output.
- [ ] Every constructive result reconstructs exactly.
- [ ] Every negative result carries executable Liouville/structure hypotheses.
- [ ] Every public admission result distinguishes unsupported towers from
      theorem-backed non-elementarity, with stable reasons for rejected cells.
- [ ] No accepted-input test reaches `NotImplementedError` or `unknown`.
- [ ] The coverage matrix contains a test for every declared algorithmic cell.
- [ ] User and developer documentation distinguish local differential
      correctness from global branch presentation.
- [ ] The mature implementation is partitioned into reviewable SymPy upstream
      contributions.

## STOP conditions

Stop the current checkpoint and report instead of improvising if:

- the required theorem assumes an operation the declared constant field cannot
  decide;
- a proposed negative certificate relies on an unproved no-new-constants,
  irreducibility, or special-polynomial claim;
- SymPy's domain cannot represent an intermediate exactly;
- the implementation would silently replace an algebraic dependency with a
  transcendental generator;
- a real rewrite loses branch or conjugation information;
- the checkpoint grows across two roadmap milestones; or
- a failure suggests the cited literature algorithm has additional hypotheses
  not yet represented in the tower metadata.

## Maintenance notes

- Treat this document as the stable program map and `docs/design.md` as the
  current-state document. Update both after each completed milestone, but do
  not mark a roadmap milestone complete from the existence of low-level
  components alone.
- Structure-theorem and effective-field assumptions must remain visible in
  result certificates even after automatic tower construction makes them
  mostly invisible to ordinary users.
- Algebraic integration is not merely "more Hermite reduction." Residue
  fields, infinity places, divisor/principal tests, and exact differentials are
  separate completion gates.
- A richer special-function integrator can consume the elementary obstruction
  later, but must not blur the classical Risch completeness claim.
