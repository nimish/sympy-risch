# Plan 005: Establish certified algebraic global spaces

> **Executor instructions**: Build the first global algebraic service without
> pretending that a bounded ansatz or a successful local calculation is a
> complete Riemann--Roch algorithm. Every complete result must carry an exact
> valuation and dimension certificate. Return an explicit boundary whenever
> the current geometry layer lacks the required lattice-intersection theorem.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/004-algebraic-constants-and-all-place-residues.md`
- **Category**: correctness / algebraic integration
- **Started**: 2026-07-25
- **Status**: DONE

## Delivered

- Immutable `RiemannRochSpace` results with complete/unsupported separation
  and constructor-level certificate checks.
- Exact complete cases for negative and zero divisors, degree-zero principal
  witnesses, verified nonspecial candidate bases, and degree-one-pole rational
  parameter spaces.
- Full-place valuation membership and exact constant-field independence tests
  for every nontrivial certified basis.
- Explicit empty `UNSUPPORTED` results outside those theorems; no bounded
  search is presented as a general Riemann--Roch basis algorithm.

## Why this matters

After algebraic Hermite reduction, local residues identify candidate
logarithmic and second-kind obstructions. They do not decide whether those
local data arise from a global function. The common global object is

```text
L(D) = { f in K : div(f) + D >= 0 }.
```

Its exact constant-field vector-space basis is the shared input required by
principal/torsion divisor solving, logarithmic reconstruction, and
second-kind exactness. It must be based on the canonical maximal-order and
infinity-place APIs, never on a plane-model shortcut.

## Narrow first checkpoint

Introduce an immutable `RiemannRochSpace` result with explicit outcomes:

- a complete basis only when all valuation and dimension facts are proved;
- an exact certificate for universal cases such as `L(0) = C`; and
- `UNSUPPORTED` when the missing operation is the general finite/infinite
  lattice intersection or reduction algorithm.

The first complete nontrivial family is a genus-zero rational parameter `u`
whose polar divisor has degree one. For `D = m*(u)_infinity`, the theorem gives
`L(D) = span_C(1, u, ..., u**m)`. A separately supplied candidate basis can
also be certified in the nonspecial range `deg(D) > 2*g - 2`, after exact
membership, linear-independence, and Riemann--Roch dimension checks.

## Scope

**In scope**:

- a new global-space module over existing `AlgebraicFunctionField` and
  `AlgebraicDivisor` objects;
- immutable basis/result certificates and exhaustive constructor invariants;
- exact valuation, independence, and dimension verification;
- genus-zero parameter examples, `L(0)`, adversarial forged candidates, and
  explicit unsupported results; and
- package exports, documentation, and focused tests.

**Out of scope**:

- a general lattice-intersection/reduced-basis implementation at finite and
  infinite places;
- principal/torsion divisor solving or logarithmic reconstruction;
- second-kind exactness and algebraic RDEs; and
- actual function-field presentation rebasing after an enlarged constant
  field.

## Mathematical contract

For a complete result, each basis element must satisfy

```text
div(f) + D >= 0,
```

the basis must be independent over the displayed exact constant field, and
the reported dimension must be justified by a theorem valid for the supplied
divisor. In the nonspecial range this is Riemann--Roch:

```text
dim_C L(D) = deg(D) + 1 - g,    deg(D) > 2*g - 2.
```

No failed search, sampled local expansion, or incomplete candidate list may
produce a complete result.

## Verification

- `L(0)` has basis `(1,)` over every admitted model.
- Rational parameter spaces reconstruct the expected powers and satisfy every
  finite and infinite valuation bound.
- Candidate bases in the nonspecial range are accepted only when their exact
  dimension equals Riemann--Roch's value.
- Forged basis entries, wrong divisor ownership, dependence, and a basis of
  the wrong dimension are rejected.
- Positive-genus low-degree spaces without a global construction return
  `UNSUPPORTED`, not an invented basis.

## Follow-on

Once a general finite/infinite lattice-intersection algorithm makes global
spaces total, split two consumers into separate checkpoints:

1. residue-basis descent plus principal/torsion divisor solving for algebraic
   logarithms; and
2. exactness of nonzero second-kind differentials.

Only then should the algebraic RDE reuse exactness as its `a = 0` case.
