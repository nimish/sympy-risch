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
  010 finite algebraic constant bases, plus Plan 011 reusable algebraic
  constant composita, Plan 012 general divisor-class reduction, and Plan 013
  good-prime finite Jacobian torsion decisions; Plan 014 mixed expression
  discovery; Plan 015's reconciled algebraic-completion program; Plan 016's
  explicit transcendental-completion program; and Plan 017's automatic
  construction, presentation, completeness, and upstream program, 2026-07-27

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
- `src/sympy_risch/dependency/` provides total parametric
  logarithmic-derivative decisions on recursively checked P/E/T towers over
  admitted finite constants. Tangent relations use the direct real relation
  system, with the Gaussian path retained as an oracle.
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
  normalized P/E/T levels, plus a terminal pure-radical algebraic level when
  its power-basis coordinates reduce to implemented scalar lower RDEs.
- `src/sympy_risch/elementary.py` recursively integrates documented explicit
  P/E/T shapes with certified preserved constants and has a success-only
  terminal pure-radical composition route: verified exact upper coordinates
  descend to scalar lower RDEs, then the lower coordinate recurses. Remaining
  explicit-tower boundaries are scalar/parametric RDE cells, not PLogD family
  orderings. General algebraic levels and polynomial monomials whose special
  structure has not been decided remain unsupported.
- `src/sympy_risch/expression_discovery.py` discovers a one-variable candidate
  DAG for rational powers and nested P/E/T calls, including both mixed
  algebraic/transcendental orders. The native expression compiler constructs
  rational-power and terminal algebraic-over-P/E/T levels and reclassifies
  certified finite-power exponentials as algebraic. Algebraic composita,
  `RootOf`, monomial-over-algebraic execution, and global branches remain open.
- `src/sympy_risch/algebraic/` validates algebraic extensions and implements
  algebraic Hermite reduction, all-place residues, displayed-constant
  logarithmic reconstruction, and complete in-field exactness for supported
  direct curves. `src/sympy_risch/algebraic_geometry/` provides certified
  maximal orders, ramified and infinite places, residue fields, divisors,
  Hess ideal arithmetic, reusable Theorem 7 parametric Riemann--Roch families
  for admitted direct models, Algorithm 21 divisor reduction, Algorithm 24
  large-space and differential infrastructure, and principality.
  Constant base change now lifts exact number-field embeddings coefficientwise
  through `C(x)[y]/(f)` and iterates through compatible residue factors over
  `QQ` or the displayed algebraic number field, rebuilding and rechecking
  every target. Independent quadratic residue fields and relative factors such
  as `Z**2 - sqrt(2)` can therefore be combined in a certified compositum.
  General reduced divisor-class arithmetic reconstructs positive
  torsion relations, while two supported good-prime reductions over `QQ`
  select the only characteristic-zero candidate or prove nontorsion before
  exact replay. A proved nontorsion rational-coordinate residue divisor now
  yields a sealed Liouville non-elementarity obstruction rather than an
  unsupported result. General splitting fields, the remaining positive-genus
  residue-combination cases, and arbitrary algebraic-curve RDEs remain open.

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
| 0. Completeness contract and coverage ledger | Partial, completed by Plans 016 T0 and 017 F0/F7: the executable operation matrix guards every total classical P/E/T solver cell and distinguishes unsupported towers from theorem-backed obstructions; the full algebraic/mixed expression-cell corpus remains. |
| 1. Effective field and tower kernel | Partial, shared by Plans 015 A0-A2 and 016 T0-T1: exact `QQ` towers, recursively checked P/E/T towers over native finite algebraic number-field constants, exact `C(i)` tangent normalization for provably real `C`, and finite extensions exist. General effective-field capabilities, algebraic-function-field proof transport, presentation normalization, and canonical mixed towers remain. |
| 2. Dependency and structure decisions | Complete on the classical finite-constant P/E/T cell through Plan 016 T1-T2: every family ordering has direct real/complex relations, fundamental signatures, checked dependencies, and Gaussian oracles. General effective constants, Riccati/Darboux discovery, and automatic non-family special-polynomial decisions remain. |
| 3. Scalar, parametric, and coupled RDEs | Complete for the finite-constant P/E/T cell under Plan 016 T3-T5: scalar and parametric RDE, limited integration, and every real-integration coupled system return reconstruction-checked solutions or theorem-backed obstructions. General polynomial monomials and algebraic lower fields remain separate milestones. |
| 4. Transcendental integration | Complete on the classical finite-constant P/E/T cell through Plan 016 T8: the executable operation matrix guards total exact, PLogD, RDE, limited, coupled, and elementary dispatch, and accepted integration returns a verified answer or Liouville obstruction. General-polynomial monomial families additionally need T6-T7 special-factor and special-denominator work. |
| 5. Algebraic-function-field kernel | Partial through Plan 015 A3: direct admitted models have maximal orders, places, local expansions, divisors, reusable Hess parametric spaces, principality, class reduction, and good-prime torsion decisions. General presentation normalization and exact constants, total maximal-order arithmetic, Hess Algorithms 21/24, scalable class operations, and algebraic-number-field good reduction remain. |
| 6. Pure algebraic integration | Partial, expanded in Plan 015 A7-A8: direct `C(x)`, for `C/QQ` finite algebraic, reconstructs displayed-constant residues through complete principality and verified `QQ` torsion decisions, continues through compatible constant composita, and solves an in-field residue-free exactness problem. General compatible residue descent and divisor combinations, full second-kind cohomology, normalization, and automatic dispatch remain. |
| 7. Mixed algebraic/transcendental towers | Started, expanded in Plan 015 A9-A10: terminal pure-radical exact/RDE descent, positive elementary composition, and normal-pole Hermite reduction reduce exact upper coordinates to scalar P/E/T RDEs, then recurse on the lower coordinate. General algebraic scalar/parametric RDEs, relative geometry, residue/logarithmic reduction, and arbitrary interleavings remain. |
| 8. Automatic expression towers | Active through Plan 014; completion is split across Plan 015 A1/A11 and Plan 017 F0-F3. A backend-independent candidate DAG covers rational powers and P/E/T calls in either mixed order; the native compiler executes rational-power and terminal algebraic-over-P/E/T towers and finite-power exponential reclassification. General algebraic relation objects/composita, monomial-over-algebraic execution, canonical minimization, and exact round trips remain. |
| 9. Real/complex presentation | Partial, expanded in Plan 017 F4-F5: rational real logarithmic output and internal Gaussian tangent normalization exist; exact source back-substitution, algebraic root isolation, interval validity, and branch/unwinding metadata remain. |
| 10. Completeness evidence | Partial, expanded in Plan 017 F7-F8: operation-specific totality is executable for the declared P/E/T cell, with testing conventions and independent oracles; algebraic/mixed/frontend coverage and end-to-end performance families remain. |
| 11. SymPy upstreaming | Planned in Plan 017 F9; no upstream extraction has started. |

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
| Function-field spaces and divisor reduction | Hess, *Computing Riemann--Roch Spaces in Algebraic Function Fields and Related Topics* |
| General algebraic-function integration | Trager, *Integration of Algebraic Functions*; Davenport, *On the Integration of Algebraic Functions* |
| Algebraic logarithmic part | Trager; Miller, *Computing the Logarithmic Part*; Schultz, *Trager's Algorithm Revisited* |
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

Plan 016 owns the complete explicit transcendental branch of this graph.
Plan 015 owns the algebraic-function-field, pure algebraic, algebraic-RDE, and
mixed-explicit branches. Plans 014 and 017 own automatic construction,
presentation, completeness evidence, and the upstream-facing path.

## Milestone 0: Establish the completeness contract and coverage ledger

### Deliverables

- Add a machine-readable or table-driven inventory of accepted:
  - constant domains;
  - tower-level families;
  - parent-field families;
  - decision problems; and
  - result outcomes.
- Publish `assess_risch_operation`, `RischOperation`,
  `assess_elementary_admission`, `TowerAdmission`,
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

Plan 016 T0-T1 owns the effective constant-field and checked transcendental
tower portion of this milestone. Plan 015 A0-A2 owns algebraic presentation,
normalization, orders, and mixed-field representation.

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

Status: complete on recursively checked P/E/T towers over the finite constant
fields admitted by T0. The Chapter 9 relation compiler handles every family
ordering, including the direct real divisor `1+t**2`; Smith saturation gives
minimal radical powers and the exact `C(i)` Cayley implementation remains an
independent oracle. Plan 003 supplies immutable source signatures, while the
constant-profile replay ensures caller metadata is not the proof. The
remaining work is general effective constants and special factors outside the
explicit P/E/T families.

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

Status: complete for the declared P/E/T cell. Hypertangent inputs use the
direct real divisor `1+t**2` (or its factors when `i` is already present);
Plan 003 records the real coefficient `eta`, and the Cayley image `2*i*eta`
is retained as an independent oracle. General special-polynomial discovery
and branch-aware public presentation remain future work.

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

For every accepted solver cell, dependency, radical logarithmic derivative,
and no-new-constants questions reach a complete decision. Every advertised
general-monomial family also has a complete special-polynomial decision;
otherwise that family is rejected at admission. See Plan 016 T1-T2 and T6.

## Milestone 3: Complete scalar, parametric, and coupled RDEs

Status: complete for the finite-constant P/E/T cell. Scalar and parametric RDE,
limited integration, and the general real two-coordinate system are total
there. Direct Chapter 8 recurrences cover their documented normalized shapes;
an exact `C(i)` scalar reduction supplies the remaining coupled cases. Work
below applies only to general special-polynomial monomials and algebraic lower
fields; it must not reopen or replace the P/E/T production recurrences.

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

Status: complete across the certified finite-constant P/E/T cell. The solver
returns the complete parameter space, and limited integration computes its
affine section with integrand coefficient one.

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
are total over every supported explicit transcendental tower. See Plan 016
T3-T5.

## Milestone 4: Complete transcendental integration

Status: primitive, hyperexponential, and hypertangent integration is
implemented for the documented explicit certified cells. T8 must still make
the operation-specific cell matrix executable and remove any accepted-cell
implementation gap before this milestone is complete. General polynomial
monomials with nonconstant special factors separately require T6-T7.

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
Risch**, before automatic expression discovery. Plan 016 T6-T8 gives its
remaining checkpoint order.

## Milestone 5: Build the algebraic-function-field kernel

This milestone supplies the geometry currently absent from
`src/sympy_risch/algebraic.py`.

Status: the local geometry substrate is substantially implemented. Finite
extensions, implicit differentiation, integral-basis coordinates,
simple-radical bases, lazy refinement, algebraic Hermite reduction over
`QQ(x)`, and Trager's nested-resultant finite-residue polynomial are
implemented. Certified maximal orders, ramified/infinite places, exact residue
fields, uniformizers, valuations, local expansions, differents, and divisor
arithmetic now exist in `algebraic_geometry/`. Hess Algorithm 13 now retains
its short Theorem 7 generators and invariant degrees, so a single
finite/infinite ideal reduction expands `L(D + r*(x)_infinity)` for many
shifts. The original verified long basis remains available as an on-demand
view. A narrow constant-field gate and an all-place residue collector already
consume that canonical geometry API for admitted preserved presentations.

Plan 015 records the larger remaining Hess/function-field program. Algorithm
21's logarithmic-height divisor reduction, Algorithm 24's short large-divisor
spaces, general presentation normalization, total maximal-order arithmetic,
differential spaces, and scalable class arithmetic remain. Those services
precede the complete algebraic logarithmic, second-kind, algebraic-RDE, and
mixed-tower branches.

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
algebraic integration algorithm for every admitted nonsingular algebraic
function field. See Plan 015 A0-A6 for the executable checkpoint order.

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
complete elementary-integrability decision. See Plan 015 A7-A8.

## Milestone 7: Complete mixed algebraic/transcendental towers

### Implemented first cell

For a representable terminal extension `K(theta)` with
`theta**n = u in K`, a lower-field coefficient `a in K`, and an implemented
scalar RDE on `K`, the library now decomposes the RDE in the power basis:

```text
D(c_j) + (a + j*D(u)/(n*u))*c_j = b_j.
```

This gives certified terminal simple-radical exactness (`a = 0`) and scalar
RDE solutions over primitive, hyperexponential, or hypertangent lower towers.
When every nonconstant coordinate of an integrand is exact by this descent,
`integrate_elementary` removes those coordinates and recursively integrates
the lower-field coordinate. It is deliberately a positive route, not a claim
that the generic scalar dispatcher or `integrate_elementary` supports
arbitrary algebraic levels or can lift lower non-elementarity obstructions.

The first relative normal-Hermite step is also implemented.  In the immediate
lower monomial field, it removes repeated normal denominator factors from the
scalar operators `D + j*eta` in each power-basis coordinate.  It leaves
special poles and resonant leading equations for later relative
residue/logarithmic reduction.

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

Plan 015 A9-A10 makes the algebraic RDE and relative-geometry dependencies of
this milestone explicit.

## Milestone 8: Construct minimal towers from SymPy expressions

### 8.1 Accepted expression grammar

Define and test an explicit grammar including:

- arithmetic and integer/rational powers;
- radicals, algebraic-number `RootOf`, and an explicit selected-root/relation
  object for general algebraic functions;
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
towers. Plans 014 and 015 A11 supply algebraic/mixed discovery; Plan 017
F0-F3 completes the grammar, minimization, and exact round trip.

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
presentation is valid. See Plan 017 F4-F5.

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

Plan 017 F7-F8 turns this matrix into the final end-to-end correctness and
performance gate without requiring tests for incidental internal architecture.

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
package may then become a development harness or be retired. See Plan 017
F6 and F9.

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
7. [x] **Plan 016 T0-T2**: make effective constant operations and the
   per-operation solver matrix executable, then complete proof-carrying
   recursive P/E/T constant, dependency, and structure decisions.
   - [x] **Plan 010**: admit native finite algebraic `C/QQ` at the standard
     `C(x)` base, retain rational-coordinate relation constraints, and enable
     direct checked primitive/hyperexponential proposals.
8. [x] **Plan 016 T3-T5**: filled every P/E/T scalar RDE ordering and
   implemented complete parametric RDE, limited integration, and required real
   coupled systems over supported finite algebraic constants.
9. [ ] **Plan 016 T6-T8**: T8 is complete for the classical P/E/T cell.
   Decide special polynomials for each additionally advertised general-
   monomial family and complete its special-denominator and logarithmic
   branches before extending the total matrix. A failed Darboux search
   without a proved bound remains an admission boundary.

### Algebraic completion path

10. [x] **Plan 005**: establish a generic algebraic-function-field
   Riemann--Roch/global-space service. The Hess finite/infinite ideal
   representation and weak-Popov intersection now compute complete spaces
   without local-expansion enumeration; Risch admission remains a separate
   consumer-side constant-field proof.
   **Plan 006** adds exact all-place valuation constraints for any declared
   finite constant subspace and a bounded nonspecial ideal-lattice search;
   only a reached Riemann--Roch dimension certifies a complete global space.
11. [x] **Plan 006**: consume complete Riemann--Roch spaces for generic
   principal-divisor decisions. Algebraic logarithmic reconstruction and
   integrating-factor recognition now use this AFF service after their
   consumer-side preserved-constant check; a degree-zero empty `L(-D)` is a
   complete nonprincipality result, not a bounded-search miss. Plan 012 later
   replaces large direct spaces with reduced class arithmetic.
12. [ ] Implement residue-basis descent, torsion/divisor-class solving, and
   exactness of
   second-kind differentials. Test genus-zero and positive-genus cases.
   - [x] Build the all-place closed-residue profile
     `product_P Norm_{k(P)/C}(Z-res_P)`, including branch and infinity places,
     without pretending independent local residue fields are already a common
     constant extension.
   - [x] Decide degree-zero principality through complete `L(-D)` spaces;
     an empty certified space proves nonprincipality while unavailable AFF
     arithmetic remains an explicit boundary.
   - [x] Integrate direct displayed-constant residues over `QQ` or a finite
     algebraic extension of `QQ` when every coordinate divisor is principal.
   - [x] Construct the first compatible base change
     `QQ(x, y) -> QQ(alpha)(x, y)`, split finite places, recollect all target
     residues, and reconstruct a target-field logarithmic answer. An explicit
     factor or the first compatible irreducible factor from the square-free
     all-place residue profile may select the target. This is deliberately
     limited to one factor that splits in its root field.
   - [x] Decide direct residue-free exactness using the complete `L(E)` from
     the generic Riemann--Roch service. A complete failed solve proves only no
     primitive in the current algebraic function field.
   - [x] **Plan 011**: factor the one-step `QQ -> QQ(alpha)` base-change
     presentation into a reusable algebraic-constant extension map, then add
     certified composita and iterative target-field integration. Each step
     must rebuild the function-field relation, re-prove preserved constants,
     and recollect all target places; it must not transport a source residue
     list through an assumed embedding. This unlocks compatible residue-basis
     descent for multiple independent factors.
   - [x] **Plan 012**: implement general Riemann--Roch divisor-class
     reduction, bounded positive torsion search, witness reconstruction, and
     content-bounded complete principality. The degree-29 genus-2 integral is
     now an end-to-end regression without a Pell-specific algorithm.
   - [x] **Plan 013**: reduce `QQ` curves, maximal orders, and divisors at
     supported good primes; determine finite class orders within the Weil
     bound; reconcile two primes; and verify the unique candidate over
     characteristic zero. The degree-29 regression no longer supplies its
     order.
13. [ ] **Plan 015 A0-A6**: complete the reusable Hess/function-field
    engine: normalize presentations and exact constants, totalize maximal
    orders and ideal arithmetic, retain Algorithm 13 as a short parametric
    basis, add Algorithms 21 and 24, differential spaces, and scalable
    divisor-class operations.
14. [ ] **Plan 015 A7-A8**: complete pure algebraic logarithmic reconstruction
    and second-kind exactness, including compatible residue fields and
    constant-linear combinations of residue divisors.
15. [ ] **Plan 015 A9**: generalize the scalar and parametric RDE to algebraic
    curves, reusing the exactness result as its `a=0` slice.
    - [x] Add the direct-curve integrating-factor cell: when a verified
      `h` satisfies `a = D(h)/h`, reduce `D(u) + a*u = b` to exactness of
      `h*b`. This is not a general algebraic RDE solver.
16. [ ] **Plan 015 A10**: compose the completed algebraic and transcendental
    solvers over the proven mixed-tower representation. This is the exit gate
    for a complete *explicit* Risch tower.

### User-facing and upstream path

17. [x] **Plan 007**: build a narrow reversible P/E/T expression compiler for
    one `QQ`-constant symbol, rational arithmetic, integer powers, and nested
    `log`/`exp`/`tan`. It replays checked extension factories and returns only
    local back-substituted presentations; dependent and branch-sensitive
    forms are structured boundaries rather than rewrites.
18. [ ] **Plan 014**: expand expression construction to algebraic/mixed
    expressions while keeping discovery separate from integration support.
    - [x] Discover rational-power and P/E/T candidates as a dependency DAG in
      either mixed order.
    - [x] Construct rational-power and terminal algebraic-over-P/E/T towers.
    - [x] Reclassify a certified finite-power exponential dependency as an
      algebraic level.
    - [ ] Add `RootOf`/general algebraic relations and algebraic composita.
    - [ ] Execute monomial-over-algebraic and arbitrarily interleaved native
      towers after Plan 015 A1/A10 provide their normalization and relative
      backends.
19. [ ] **Plan 015 A11**: finish automatic algebraic/mixed execution, including
    source-model isomorphisms and automatic integral-basis construction.
20. [ ] **Plan 017 F0-F3**: make the full elementary expression grammar
    executable, complete candidate minimization, and require exact
    source-to-tower round trips.
21. [ ] **Plan 017 F4-F6**: implement canonical complex-local and
    assumption-scoped real back-substitution, then expose one stable
    SymPy-facing decision API.
22. [ ] **Plan 017 F7-F8**: expand the completeness matrix to every declared
    tower/expression cell and add end-to-end performance families.
23. [ ] **Plan 017 F9**: upstream independently mature domain, reduction, RDE,
    function-field, frontend, dispatcher, and presentation components.

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
