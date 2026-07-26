# Plan 009: Consume a certified linear special factor locally

> **Executor instructions**: Build only a positive-evidence consumer. A
> failed local Risch differential equation is not a Liouville obstruction:
> retain it as a structured boundary and do not widen the elementary
> dispatcher.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: HIGH, because a local no-solution result can be mistaken for a
  global non-elementarity proof
- **Depends on**: `SpecialPolynomialCertificate`, recursive scalar RDE
  dispatch
- **Category**: transcendental special-denominator reduction
- **Completed**: 2026-07-26
- **Status**: DONE

## Delivered contract

`reduce_certified_linear_special_poles(integrand, derivation, certificate)`
consumes one caller-supplied monic *linear*
`SpecialPolynomialCertificate` for

```text
p = t-a,     Dp = h*p.
```

At an apparent pole of order `m`, it obtains the principal coefficient `r` in
the lower field and reduces the cofactor modulo `p`, giving `hbar`. It solves
the scalar lower-field RDE

```text
Dq - m*hbar*q = r.
```

When solved, `q/p**m` is added to the exact part, its **full** derivative is
subtracted, and the next pole order is recomputed. The full derivative is
essential: if `h` depends on `t`, `(h-hbar)/p` contributes to the next lower
principal part.

The immutable `LinearSpecialPoleReduction` records the exact part, remaining
remainder, every local RDE, and one of these outcomes:

- `REDUCED`: the supplied factor no longer occurs in the denominator;
- `UNRESOLVED_RDE`: the supported lower RDE had no solution; or
- `UNSUPPORTED_LOWER_RDE`: the lower RDE dispatcher does not cover that
  tower.

Every result verifies `integrand = D(exact_part) + remainder` and the actual
remaining pole order. A checked factor certificate is positive evidence;
nonlinear factors and foreign certificates are rejected.

Before dispatch, the reducer uses the same full-prefix scalar-RDE structural
admission check as `solve_risch_differential_equation`: certified P/E/T over
`QQ(x)`, or certified P/E over `QQ(i)(x)`. This prevents an uncertified or
unsupported ancestor from surfacing as an incidental recursive solver
exception; it does not make every admitted prefix a total lower-RDE solver.

## Mathematical boundary

This is a local constructive reduction, **not** a complete special-denominator
algorithm and not an elementary-integration decision. In particular, the
apparent leading pole may have arisen by cancellation from a higher-pole exact
antiderivative. For example, over `QQ(x)(t)` with

```text
Dt = t/x + t**2/x**2,   p = t,
D(x**2/t**2) = -2/t,
```

the order-one local RDE has no rational solution even though the integrand is
exact. `UNRESOLVED_RDE` must therefore never become a non-elementarity
certificate.

The checkpoint also intentionally excludes:

- special factors of degree greater than one, whose residue equation lives in
  `K[t]/(p)` rather than the scalar lower field;
- automatic Darboux/special-factor discovery;
- special RDE denominator bounds, parametric RDEs, coupled systems, and
  logarithmic reconstruction; and
- changes to `hermite_reduce_monomial`, `solve_exact_derivative`, or
  `integrate_elementary`.

## Verification

- A moving Riccati factor `p=t-x` with cofactor `t+1/x`, proving that the
  algorithm uses `h mod p = x+1/x` rather than the raw constant coefficient.
- Repeated poles that require recomputation after the full derivative.
- A mixed denominator preserving non-target poles.
- Supported no-solution and unsupported lower-RDE boundaries.
- An unsupported ancestor beneath otherwise supported lower levels, plus
  direct scalar-RDE admission tests for `QQ(i)(x)` and its hypertangent
  boundary.
- The higher-order-resonance adversary above.
- Foreign, nonlinear, contradictory-status, and no-op inputs.
- Generated planted linear-special derivatives with varied moving roots,
  cofactors, numerators, and pole orders.
