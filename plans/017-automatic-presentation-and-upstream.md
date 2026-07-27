# Plan 017: Complete automatic construction, presentation, and upstreaming

> **Executor instructions**: This is a program plan, not one atomic change.
> It consumes the completed explicit algorithms from Plans 015 and 016; it
> must not hide an incomplete solver behind expression rewriting. Implement
> each checkpoint through a smaller plan and separate commit.

## Status

- **Priority**: P1
- **Effort**: XL, a multi-checkpoint engineering and mathematics program
- **Risk**: HIGH
- **Depends on**: Plans 014-016
- **Category**: expression frontend / presentation / completeness / upstream
- **Planned**: 2026-07-27
- **Status**: TODO

## Why this is separate from the explicit algorithm

A total explicit-tower decision procedure is the mathematical core, but it is
not yet a useful SymPy integrator. A user starts with an expression, not:

- an ordered differential tower;
- constant-field and dependency certificates;
- an algebraic-function-field presentation;
- integral bases or divisor data; or
- a choice of complex logarithmic branches.

The frontend must construct those objects automatically and reversibly. The
presentation layer must then convert a field certificate into an exact SymPy
expression without confusing a local differential identity with a globally
continuous real antiderivative.

## Completion contract

For every expression in the declared elementary grammar, the public entry
point must:

1. decide whether the constant field and expression operations are supported;
2. discover and order a sufficient set of generators;
3. prove dependencies and constant preservation;
4. construct an exact explicit tower and compile the integrand into it;
5. invoke the total explicit algorithm;
6. back-substitute a verified complex-local result;
7. optionally construct a branch-aware real presentation; and
8. return a stable unsupported reason only at the declared grammar or
   effective-operation boundary.

The frontend is not required to solve arbitrary SymPy simplification,
identity, or special-function problems.

## Declared expression grammar

The first complete grammar contains:

- symbols declared as the integration variable or effective constants;
- exact arithmetic and integer powers;
- rational powers, algebraic-number `RootOf` objects, and a declared
  representation for selected roots of general algebraic-function relations;
- `exp`, `log`, and forms normalized to them;
- `tan`, `atan`, and elementary trigonometric/hyperbolic forms whose field
  representation is certified;
- finite compositions of these operations; and
- assumptions needed to select a real branch when real output is requested.

Functions outside elementary differential fields, unrestricted `Piecewise`,
distributional derivatives, and arbitrary identities among opaque constants
remain explicit exclusions.

Current SymPy `CRootOf` does not represent a polynomial root over `QQ(x)`.
F0 must therefore choose an explicit algebraic-function root/relation object
with branch identity, or retain a separate relation-binding input until an
appropriate object is upstream. It must not pretend that an algebraic-number
`RootOf` carries this information.

## Ordered implementation checkpoints

### F0. Make the public grammar and effective assumptions executable

Represent the grammar as data rather than documentation alone. For every
expression node, record:

- its field operation;
- required constant-field capabilities;
- branch/assumption requirements;
- candidate generator kind; and
- the solver cell ultimately required.

Reject unsupported functions and undecidable constant relations before tower
construction. Preserve the distinction between:

- unsupported syntax;
- unsupported effective constant operations;
- a representable tower whose solver backend is incomplete; and
- a proved non-elementary integrand.

**Exit gate:** every input expression receives a deterministic grammar and
capability classification.

### F1. Complete candidate discovery and algebraic normalization

Extend Plan 014's candidate DAG to all grammar nodes:

- rational powers, radicals, and `RootOf`;
- the selected-root/relation representation chosen in F0;
- logarithmic, exponential, tangent, and inverse candidates;
- algebraic composita and primitive elements from Plan 015 A1;
- repeated or syntactically different candidates for the same field element;
- candidates containing other algebraic or transcendental candidates; and
- exact constant multipliers and shifts.

Discovery remains syntax-driven only at the proposal stage. Minimal
polynomials, exact derivatives, logarithmic derivatives, and structure
relations decide the actual generator type.

**Exit gate:** every accepted expression produces a complete acyclic candidate
graph with no integration decisions performed during traversal.

### F2. Construct and minimize the verified tower

Process the candidate graph bottom-up:

- reuse a candidate when an exact field expression already represents it;
- merge logarithmically or exponentially dependent generators;
- reclassify finite-power exponential relations as algebraic;
- normalize several algebraic candidates into a certified compositum;
- choose a stable topological ordering compatible with all dependencies;
- construct no-new-constants and special-polynomial evidence through Plans
  015 and 016; and
- preserve immutable maps from source expressions to tower elements.

Minimality means no admitted generator is provably redundant under the
available complete relation decisions. It does not license unsafe
branch-sensitive identities such as replacing `log(x**2)` globally by
`2*log(x)`.

**Exit gate:** equivalent source candidates compile to one verified tower
generator whenever the relevant structure theorem proves the relation.

### F3. Compile exactly and verify round trips

Compile the full integrand into the native top field:

- use exact SymPy domains and extension elements;
- retain source-to-field and field-to-source maps;
- handle algebraic normalization coordinate changes;
- avoid string or expression-pattern coefficient extraction;
- re-evaluate the source expression under the generator map; and
- require exact round-trip equality before integration.

For local branch choices, the field element records the algebraic identity
while presentation metadata records which source branch supplied the
generator.

**Exit gate:** every successfully compiled input reconstructs exactly before
the solver sees it.

### F4. Back-substitute a canonical complex-local result

Convert explicit-tower certificates back to SymPy:

- substitute exact and algebraic terms through inverse presentation maps;
- normalize logarithmic arguments without discarding constants or
  conjugation;
- combine only identities proved in the current field;
- retain additive integration constants outside certificate equality; and
- differentiate the final expression through the same source map.

The canonical core result is a local complex differential identity. It does
not claim continuity across branch cuts.

**Exit gate:** every constructive explicit result has a source-expression
presentation whose derivative reconstructs the original integrand locally.

### F5. Add branch-aware real presentation

Build real output as a separate transformation of the verified complex-local
certificate:

- combine conjugate logarithms into real logarithms and angles;
- use exact real-root isolation for rational and algebraic residue factors;
- preserve tangent/arctangent generators when natural;
- record intervals separated by poles, zeros, and branch points;
- use `log(Abs(...))` only with the required real-domain evidence;
- track phase and unwinding corrections explicitly; and
- avoid global inverse identities without range assumptions.

The output should state its validity domain. Where no single continuous real
formula exists, return interval presentations or the valid complex-local
answer rather than apply an unsafe simplification.

**Exit gate:** real mode differentiates correctly on every reported interval
and exposes all excluded singular or branch points.

### F6. Define the stable public decision API

Provide one narrow SymPy-facing entry point with explicit results for:

- elementary antiderivative;
- certified non-elementarity;
- unsupported grammar;
- unavailable effective constant operation; and
- an internal invariant failure.

The public result should expose reconstruction and theorem data without
requiring users to understand internal reduction classes. Decide how it
cooperates with `sympy.integrate`: during development, unsupported input falls
through to existing SymPy behavior, while a certified non-elementary result
does not masquerade as an implementation miss.

**Exit gate:** ordinary users can integrate every accepted expression without
specifying a tower, basis, extension family, or solver.

### F7. Establish completeness and regression evidence

Build a maintained corpus from the executable cross-product of:

- expression grammar node;
- tower ordering;
- constant field;
- pure transcendental, pure algebraic, and mixed placement;
- normal, special, ramified, and infinity behavior;
- constructive and obstructed outcomes; and
- complex-local and real presentation.

Testing priorities are mathematical:

1. exact source-to-field round trip;
2. exact differentiation of positive answers;
3. replay of negative theorem hypotheses;
4. generated planted solutions across tower families;
5. independent algorithms already present in the repository;
6. adversarial expressions at grammar and branch boundaries; and
7. literature examples, including high-degree and positive-genus cases.

Keep architecture tests small and focused on concrete regressions. Do not
multiply tests merely to encode internal class names or dependency direction.

**Exit gate:** every declared grammar/solver cell is represented, and no
accepted cell reaches `NotImplementedError`, `UNKNOWN`, or an unbounded
heuristic.

### F8. Control performance and resource growth

Profile complete expression families rather than isolated helper calls:

- tower candidate count and dependency-solving cost;
- polynomial degree and coefficient height;
- algebraic extension degree and genus;
- RDE dimension and recursion depth;
- divisor height and residue-field degree; and
- final expression size.

Add:

- structural memoization scoped to one immutable integration request;
- common-subexpression reuse in discovery and back-substitution;
- short algebraic witnesses and Hess power products where available;
- output-size controls that do not alter mathematical decisions; and
- documented performance tiers for expensive but complete cases.

Timeouts and caller-selected degree bounds may limit a convenience API, but
exhausting them returns an interrupted/inconclusive result, never a
non-elementarity proof.

**Exit gate:** representative accepted families terminate with controlled
growth, and expensive cases report their cost without weakening correctness.

### F9. Prepare and upstream independent SymPy components

Upstream in reviewable dependency order:

1. exact domain, finite-extension, and coercion improvements;
2. rational integration and independent LRT-related improvements;
3. reusable polynomial, RDE, and relation solvers;
4. algebraic-function-field orders, ideals, places, and Riemann--Roch
   services;
5. tower construction and expression compilation;
6. the elementary decision API and certificates; and
7. branch-aware real presentation.

For each contribution:

- follow SymPy public-domain and naming conventions;
- isolate a generally useful mathematical capability;
- include focused user documentation and tests;
- avoid depending on this package's private compatibility layer;
- preserve existing `integrate` fallbacks until the new admitted cell is
  demonstrably total; and
- keep unsupported input distinct from a proof of non-elementarity.

The standalone package remains a research and integration harness until the
upstream dispatcher has equivalent coverage.

**Exit gate:** SymPy's public integration path can invoke the complete
elementary decision procedure over its declared effective grammar.

## Acceptance families

The final frontend must automatically handle at least:

```text
exp(x/2) + exp(x)
exp(x) + exp(x + 1)
log(x) + log(x**2)
tan(log(x))
sqrt(x + log(x))
exp(sqrt(x))
log(1 + sqrt(x))
RootOf(z**5 - z + 1, z, 0) * exp(x)
```

plus a selected root of `y**3 - y - x`, independent algebraic generators,
positive-genus algebraic examples, and arbitrary finite P/E/T/algebraic
interleavings admitted by Plans 015 and 016. These are family representatives,
not hard-coded rewrite targets.

## Source map

- Bronstein, *Symbolic Integration I*, for expression-to-tower normalization
  and real rational logarithmic output.
- Bronstein, *Simplification of Real Elementary Functions*, for the real
  structure and presentation boundary.
- Corless and Jeffrey, *The Unwinding Number*, for branch corrections in
  logarithmic identities.
- Current SymPy domain, `RootOf`, assumptions, and integration APIs, treated
  as the upstream compatibility target rather than reimplemented privately.

## Program exit gate

Plan 017 is complete when:

1. the accepted expression grammar and capability boundary are executable;
2. every accepted expression automatically constructs a verified sufficient
   tower;
3. compilation and back-substitution round-trip exactly;
4. complex-local and requested real presentations state the correct domains;
5. every accepted expression receives a constructive answer or proved
   non-elementarity result;
6. coverage and performance gates protect the declared cell; and
7. the implementation is partitioned into independently upstreamable SymPy
   contributions.
