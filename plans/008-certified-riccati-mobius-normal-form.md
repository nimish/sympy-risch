# Plan 008: Add certified Riccati Möbius normalization

> **Executor instructions**: Consume concrete positive special-factor evidence
> without turning an incomplete factor search into a structure theorem. Keep
> the ordinary P/E/T admission boundary unchanged, and make the coordinate
> change inspectable and reversible.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: HIGH
- **Depends on**: `plans/002-executable-admission-matrix.md`, concrete
  `SpecialPolynomialCertificate` support
- **Category**: transcendental structure theorem / explicit integration
- **Completed**: 2026-07-26
- **Status**: DONE

## Delivered contract

For an active preserved-constants quadratic monomial `K(t)` with

```text
Dt = A*t**2 + B*t + C,  A != 0,
```

callers may supply two checked, distinct, monic linear special-polynomial
certificates for `t-a` and `t-b`.  The immutable
`RiccatiMobiusNormalForm` retains both certificates and constructs

```text
u = (t-a)/(t-b),      Du = A*(a-b)*u.
```

It verifies both inverse coordinate maps and their intertwining of the full
active tower derivation. Reversing the certificates is meaningful: it uses
the reciprocal coordinate and negates the hyperexponential direction.

`integrate_riccati_mobius_normal_form` is an explicit proof-carrying route.
It transports an integrand to the resulting hyperexponential target, requires
that target to be structurally admitted, invokes the normal recursive
integrator, propagates any documented solver boundary, and transports exact
parts, logarithmic terms, and Liouville obstructions back to the source field.

The exact `Dt = eta*(1-t**2)` hyperbolic normalizer shares target construction
internally and retains its automatic dispatcher path. Its pure coordinate map
can retain unknown constant-field metadata; it neither copies a source
special-polynomial status nor treats missing factor evidence as absence.

## Deliberate boundary

This checkpoint does not solve the Riccati equation

```text
D(a) = Dt evaluated at t=a
```

and does not search for Darboux/special factors. A missing supplied
certificate is therefore `UNSUPPORTED` input, never evidence that no such
factor exists or that an integral is non-elementary. The ordinary
`integrate_elementary` dispatcher continues to reject an arbitrary quadratic
polynomial monomial: it cannot infer either factor identities or their chosen
orientation from metadata or an incomplete factor search.

The normalizer applies only at the active top level. Rewriting a lower
quadratic generator would require transporting every higher extension and is
not part of this checkpoint.

## Verification

- A non-hyperbolic asymmetric Riccati law with a factor not dividing `Dt`.
- Exact source/target round trips and derivation intertwining on rational
  functions.
- Generated quadratic laws from varied moving roots and leading coefficients.
- Factor-order reversal, fresh-symbol, nonlinear-factor, duplicate-factor,
  foreign-certificate, contradictory-status, and constant-field boundaries.
- End-to-end planted elementary derivatives, a pulled-back logarithmic term,
  a non-elementary hyperexponential obstruction, and rejection of a target
  outside the P/E/T admission contract.
