# Plan 016: Complete the explicit transcendental Risch algorithm

> **Executor instructions**: This is a program plan, not one atomic change.
> Implement each checkpoint through a smaller file-scoped plan and a separate
> commit. Preserve the working P/E/T recurrences as production algorithms and
> use independent formulations as tests where they already exist. An
> unsupported solver branch is never a non-elementarity result.

## Status

- **Priority**: P1
- **Effort**: XL, a multi-checkpoint research program
- **Risk**: HIGH
- **Depends on**: Plans 002, 003, and 008-010
- **Category**: structure theorems / RDE / transcendental integration
- **Planned**: 2026-07-27
- **Status**: TODO

## Current boundary

The repository already has a substantial transcendental kernel:

- exact polynomial and rational integration;
- normal/special splitting and generalized monomial Hermite reduction;
- rational and recursive primitive exact/logarithmic-derivative decisions;
- fundamental P/E/T signatures and Gaussian tangent normalization;
- scalar and coupled RDE algorithms for documented P/E/T shapes;
- primitive, hyperexponential, and hypertangent reducers; and
- recursive elementary integration for the implemented family orderings.

Structural admission is currently broader than algorithmic totality. A finite
P/E/T tower can pass `assess_elementary_admission` and still reach a recursive
logarithmic-derivative or RDE boundary. General polynomial monomials also lack
an exhaustive derivation-bound special-polynomial decision and the complete
special-denominator branch.

This plan closes those gaps without rewriting the working rational and P/E/T
algorithms.

## Required and extended input cells

The first exit gate is the classical explicit transcendental cell:

```text
C(x)(t_1)...(t_n)
```

where `C` is an admitted effective constant field, `Dx = 1`, every extension
preserves constants, and each `t_i` is:

- primitive/logarithmic: `Dt_i in K`;
- hyperexponential: `Dt_i/t_i in K`; or
- hypertangent: `Dt_i = eta_i*(1 + t_i**2)`, retained for exact real output.

Finite algebraic levels and mixed towers are owned by Plan 015.

General polynomial monomials are a second, explicitly extended cell. They are
admitted only when the required special-polynomial/Darboux decision is
effective. Failure to implement that oracle is an admission boundary, not a
negative integration decision.

For every accepted explicit transcendental tower, the following decisions
must be total:

1. exact derivative;
2. radical logarithmic derivative and parametric logarithmic derivative;
3. scalar RDE;
4. parametric RDE and limited integration;
5. coupled RDE where required by the real algorithm; and
6. elementary integration.

## Layering

```text
effective constant-field operations
                 |
                 v
checked tower construction and constant replay
                 |
                 v
structure / dependency / special-polynomial decisions
                 |
                 v
scalar RDE ------+------ parametric and coupled RDE
                 |
                 v
monomial reduction and logarithmic reconstruction
                 |
                 v
total explicit-tower integration
```

The composition engine coordinates these services. Leaf reduction and theorem
modules should depend on narrow operation protocols or explicit callbacks,
not import the public dispatcher or a global engine to recurse.

## Ordered implementation checkpoints

### T0. Define the effective constant-field and solver matrix

**Progress:** the finite-number-field half now has one shared
`FiniteConstantFieldProfile` for exact arithmetic, rational coordinates,
degree, reality, and the presence of `i`. The full per-operation solver matrix
and the algebraic-function-field capability half remain open.

Replace scattered domain checks with one small capability boundary shared with
Plan 015. It must state whether the current constant field supplies:

- exact arithmetic and equality;
- polynomial square-free factorization and factorization;
- exact linear algebra;
- rational coordinates and integer-relation normalization where required;
- finite algebraic extensions and compatible embeddings; and
- real closure/root-isolation operations requested only by presentation.

This capability record is implemented once in a neutral domain layer. Plan
015 A0 consumes the same record and adds function-field-presentation checks;
neither program should define a parallel notion of an effective constant
field.

Separately publish a decision matrix indexed by tower shape and operation:

```text
exact | PLogD | scalar RDE | PRDE | coupled RDE | integrate
```

Structural tower admission and per-operation support must remain distinct.

**Exit gate:** callers can determine before dispatch whether an input belongs
to a total solver cell, and every refusal identifies the missing effective
operation.

### T1. Complete checked recursive tower construction

**Progress:** complete for recursively checked P/E/T prefixes over `QQ`,
`QQ_I`, and native finite algebraic number fields over `QQ`. Tangent replay
uses the exact `C(i)` compositum when `C` is provably real. General effective
constant fields and mixed algebraic-function/transcendental levels remain
owned by Plan 015.

Generalize constant-field preservation and dependency replay beyond the
current `QQ(x)`-centered recursive cases:

- certify each proposed primitive generator by exact-derivative failure;
- certify hyperexponential dependency through radical logarithmic
  derivatives;
- certify hypertangent levels directly and through the Gaussian oracle;
- transport exact evidence through finite effective constant fields;
- detect dependencies among several existing generators; and
- retain immutable proof data bound to the exact tower position.

The library should construct these proofs itself. Caller metadata may describe
an extension but must not be the reason a theorem assumes no new constants.

**Exit gate:** every admitted P/E/T prefix has replayable constant-field and
dependency evidence over every constant field admitted by T0.

### T2. Complete the complex and real structure-theorem solvers

Turn the existing fundamental signatures into a total recursive structure
decision:

- collect all primitive, logarithmic, exponential, and tangent directions in
  tower order;
- solve the exact rational/integer relation systems over the effective
  constant field;
- finish parametric logarithmic-derivative recursion for every P/E/T
  ordering;
- invoke the complete structure fallback whenever a fast Chapter 7 heuristic
  is inconclusive;
- normalize radical witnesses and verify their defining equations; and
- implement the direct real relations, retaining the Cayley/Gaussian
  formulation as an independent oracle.

Bronstein's Example 7.3.2 remains a required regression because it
distinguishes a complete structure fallback from the incomplete fast
heuristic.

**Exit gate:** exact, radical-logarithmic, exponential-dependency, and
no-new-constants decisions are total on the classical P/E/T cell.

### T3. Complete scalar RDE recursion on P/E/T towers

Audit every current scalar RDE `NotImplementedError` against Bronstein
Chapter 6 and fill the missing family/lower-field combinations:

```text
D(y) + a*y = b.
```

Each route must implement or reuse:

- weak normalization;
- normal and family-specific special denominator bounds;
- finite and infinity degree bounds;
- SPDE reduction;
- polynomial no-cancellation and cancellation recurrences;
- recursive lower-field equations; and
- exact reconstruction of a returned solution.

The current Bronstein recurrences remain the production method. Bounded matrix
ansatzes are useful small-degree oracles, not a replacement that silently
introduces a search bound.

**Exit gate:** scalar RDE is total for every family ordering in the declared
P/E/T matrix.

### T4. Implement parametric RDE and limited integration

Implement the Chapter 7 problem:

```text
D(y) + a*y = c_1*b_1 + ... + c_m*b_m,
    c_i in Const(K).
```

The result must describe the full constant-vector solution space, not select
one convenient solution. Required pieces are:

- common denominator and degree bounds;
- recursive cancellation with parameter constraints;
- exact constant-field nullspaces;
- canonical independent solution bases;
- reconstruction of every basis vector; and
- limited integration as a direct consumer.

Do not infer nonexistence from an exhausted bounded ansatz. A negative result
must be the nullspace obstruction of the complete finite problem.

**Exit gate:** PRDE and limited integration are total on the same explicit
P/E/T cell as scalar RDE.

### T5. Complete coupled systems required by real integration

Finish the Chapter 8 systems with diagonal and rotation coefficients, not only
the public zero-diagonal convenience case. Cover:

- primitive recurrences;
- hyperexponential Laurent and cancellation cases;
- hypertangent polynomial and rational cases;
- recursive lower-field systems; and
- every system generated by the real tangent/arctangent algorithm.

Construct real solutions directly. For small generated systems, compare
against the conjugate complex scalar formulation and the independent block
matrix oracle.

**Exit gate:** no accepted real P/E/T integration path reaches an unsupported
coupled equation.

### T6. Decide special polynomials for the extended monomial cell

For a general monomial derivation, compute the derivation-bound irreducible
polynomials `p` satisfying:

```text
p divides D(p).
```

Required outcomes are:

- a complete list with verified cofactors and multiplicative relations;
- a theorem-backed proof that no nonconstant special polynomial exists; or
- an admission refusal because the selected monomial family lacks an
  effective decision.

The implementation must:

- derive degree and denominator bounds from the structure theorem;
- solve the resulting Riccati/Darboux equations recursively;
- factor over the declared effective constant field;
- recognize known coordinate normalizations such as tangent and Riccati
  Möbius transforms; and
- verify every positive or absence certificate independently.

An empty enumeration under an unproved bound remains inconclusive.

**Exit gate:** every general-monomial family advertised by the package has a
total special-polynomial decision. The classical P/E/T gate does not wait for
unadvertised families.

### T7. Complete special-denominator reduction and logarithmic reconstruction

Using T3-T6, finish the general monomial integration branch:

- remove all repeated normal and special denominator powers;
- solve local principal-part RDEs at every special factor;
- handle interactions among several factors rather than reducing one exposed
  pole in isolation;
- compute the generalized residue criterion;
- solve the polynomial reduced part;
- reconstruct exact and logarithmic terms; and
- retain a certified reduced obstruction when Liouville's theorem excludes an
  elementary primitive.

The existing one-linear-factor consumer remains a focused local oracle and
useful fast path.

**Exit gate:** the extended general-monomial cell is total whenever T6 admits
its special-polynomial family.

### T8. Make explicit transcendental integration total

Unify the completed services through the recursive composition engine:

- dispatch by a checked operation cell rather than broad structural
  admission alone;
- make every recursive call preserve the exact lower-tower position;
- distinguish constructive, proved obstruction, and unsupported input;
- attach the finite theorem data needed to replay every negative result;
- reconstruct every antiderivative in the original explicit tower; and
- remove accepted-cell `NotImplementedError` paths.

This checkpoint concerns explicit towers only. Automatic expression discovery
and branch-aware SymPy output belong to Plan 017.

**Exit gate:** every accepted explicit transcendental tower returns a verified
elementary antiderivative or a theorem-backed Liouville obstruction.

## Test strategy

Use a table of natural mathematical cells:

| Axis | Values |
| --- | --- |
| top family | primitive, hyperexponential, hypertangent, admitted general monomial |
| lower family | rational, primitive, hyperexponential, hypertangent |
| denominator | polynomial, normal, special, mixed, infinity |
| equation | exact, logarithmic, scalar RDE, PRDE, coupled |
| result | constructive, proved obstruction, unsupported admission |
| constants | `QQ`, Gaussian rationals, admitted number field |

The suite should contain:

- literature examples for every theorem branch;
- generated planted exact/logarithmic/RDE solutions;
- generated denominators realizing each reduction case;
- independent rational, matrix, or Gaussian checks where already natural;
- exact differentiation or equation reconstruction for every positive result;
  and
- theorem-hypothesis checks for negative results.

Do not test implementation architecture for its own sake. Import-cycle and
layering checks should be small regression guards only where a concrete cycle
has previously broken isolated imports.

## Performance and maintainability gates

- Keep theorem-specific algorithms in focused modules rather than rebuilding
  a giant `rde.py` or `monomial.py`.
- Reuse normalized coefficients, denominator factorizations, signatures, and
  lower-field solutions during one recursive decision.
- Measure degree, coefficient height, and tower-depth growth on generated
  families.
- Retain fast family-specific paths before general structure fallbacks.
- Keep matrix or exhaustive methods bounded to independent tests unless their
  completeness bound is proved.

## Source map

- Bronstein, *Symbolic Integration I*, Chapters 3-9.
- Risch, *The Problem of Integration in Finite Terms*.
- Rosenlicht, *Integration in Finite Terms*.
- Bronstein, *Integration of Elementary Functions*.
- Bronstein, *Simplification of Real Elementary Functions*.

## Program exit gate

Plan 016 is complete when:

1. the effective constant and operation matrices are executable;
2. structure and dependency decisions are total on the declared P/E/T cell;
3. scalar RDE, PRDE, limited integration, and required coupled systems are
   total there;
4. explicit P/E/T integration has no algorithmic gaps;
5. each advertised general-monomial family has a complete special-polynomial
   and special-denominator algorithm; and
6. every positive or negative result carries exact replayable mathematics.
