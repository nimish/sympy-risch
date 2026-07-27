# Plan 012: General divisor-class torsion

**Status:** completed 2026-07-27

## Goal

Replace high-degree direct principal-divisor searches with general arithmetic
in the degree-zero divisor class group of an algebraic function field.  The
implementation must use the existing complete Riemann--Roch service, make no
hyperelliptic or Pell--Abel assumptions, and recover a field element
certifying every positive torsion relation it finds.

## Checkpoints

1. Reduce an arbitrary degree-zero divisor class relative to an effective
   anchor divisor.  For the least `n >= 0` with
   `L(n*A - D) != 0`, choose `f` in that complete space and return
   `D - div(f)`.  Riemann--Roch guarantees such an `n` once
   `n*deg(A) >= genus`.
2. Iterate reduced class addition to search for the first positive `m` with
   `m*D = div(f)`.  Retain the individual reductions as the certificate, so
   verification does not refactor the final high-degree function.
3. Use coefficient content in the complete principality service.  If
   `D = c*D0`, only the first `c` multiples of the primitive divisor `D0`
   are relevant: the first torsion order divides `c` exactly when `D` is
   principal.  This gives a terminating, general decision for the supplied
   divisor while keeping every intermediate Riemann--Roch problem bounded.
4. Connect the service to algebraic logarithmic reconstruction and verify the
   genus-2 degree-29 example end to end.

## Explicit boundary

The positive torsion search by itself is a semidecision: exhausting a
caller-selected order bound does not prove that a divisor class has infinite
order.  The principality application above is nevertheless complete because
the coefficient content supplies the exact finite bound relevant to the
given divisor.

A total standalone torsion decision requires Trager's next layer: reduce the
curve and divisor class at suitable finite primes, compute the finite
Jacobian orders, reconcile the candidate characteristic-zero order, and
verify it with the characteristic-zero class arithmetic implemented here.
That finite-field/Jacobian checkpoint remains separate from this one.
