# Plan 006: Add exact finite algebraic valuation constraints

> **Executor instructions**: Extend global algebraic infrastructure only by
> proving the result for the finite space actually searched. Do not turn a
> bounded lattice enumeration into a negative answer or a general
> Riemann--Roch claim.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: HIGH
- **Depends on**: `plans/005-certified-global-spaces.md`
- **Category**: correctness / algebraic integration
- **Completed**: 2026-07-25
- **Status**: DONE

## Delivered contract

For a preserved-constant algebraic function field, divisor `D`, and finite
constant subspace `V`, `solve_algebraic_valuation_constraints` computes
exactly:

```text
V intersect L(D).
```

It derives all local conditions from the full principal-divisor support of
the ambient basis, solves the resulting exact Laurent-coefficient equations,
and independently rechecks the returned basis through full principal
divisors. Zero and dependent ambient generators are normalized away.

A place with an active constraint must presently have residue field equal to
the certified constant field. The result is explicitly `UNSUPPORTED` when
the missing operation is higher-degree residue-coordinate flattening, and
contains no partial candidate basis.

`finite_lattice_slice` supplies a finite prefix from the fractional ideal for
the finite part of `D`. `search_nonspecial_lattice_slice` reports
`INCOMPLETE` until that prefix reaches the nonspecial Riemann--Roch dimension;
only then does it delegate to `RiemannRochSpace` for a complete certificate.

## Deliberate boundary

This checkpoint does not implement a terminating degree-bound theorem for an
arbitrary divisor, higher-degree residue coordinates, principal/torsion
divisor solving, logarithmic reconstruction, or second-kind exactness. Those
remain distinct consumers of the eventual general finite/infinite lattice
intersection algorithm.

## Verification

- Ramified finite and infinite local constraints, including cancellation that
  creates a regular combination.
- Algebraic constant fields, higher-degree residue refusal only when an
  equation is needed, and rebasing-required constants.
- Forged coordinates/result metadata, zero/dependent ambient generators, and
  generated planted cancellations.
- Bounded nonspecial exhaustion versus theorem-backed completed bases.
