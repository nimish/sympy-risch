# Plan 013: Finite-prime torsion decisions

**Status:** completed in the degree-one-anchor cell

## Goal

Turn the general positive divisor-class search from Plan 012 into an automatic
torsion decision for algebraic curves over `QQ`. Follow Trager's
good-reduction strategy: compute the class order in suitable finite
Jacobians, reconcile the finite orders into a characteristic-zero candidate,
and verify that candidate with the existing characteristic-zero reduced-class
arithmetic.

## Checkpoints

1. Decouple exact algebraic-function-field presentations from differential
   towers. `AlgebraicExtension` and the neutral geometry layer may use exact
   finite fields; `DifferentialTower` and derivations remain characteristic
   zero. In positive characteristic, certify polynomial maximal orders when
   their discriminant is square-free and reject unresolved bad models.
2. Reduce a `QQ(x)[y]` curve, its finite/reciprocal maximal orders, and a
   degree-zero divisor modulo a good rational prime. Verify that the reduced
   order bases and divisor degrees agree rather than assuming good reduction.
3. Compute the reduced divisor-class order using the finite Jacobian and a
   Weil upper bound. Record the prime, finite order, genus, reduced model, and
   exact reduced-class witness.
4. Reconcile results at multiple good primes, accounting for the possible
   residue-characteristic part of a characteristic-zero torsion order.
   Verify every proposed order over characteristic zero. Incompatible finite
   data is a nontorsion certificate; unavailable reductions remain unknown.
5. Feed the automatic result into algebraic logarithmic reconstruction, so
   the degree-29 genus-2 regression no longer supplies
   `torsion_order_bound=29`.
6. Require proof that the displayed source constants are full and independently
   certify the reduced target through its degree-one anchor. Cache exact finite
   orders sparsely per divisor, rather than requiring one pair of primes to be
   good for every divisor in a reconstruction batch.

## Initial scope

The reduction map is for curves and divisor classes defined over `QQ`.
Reduction from algebraic number fields requires prime ideals of the constant
field and is a later generalization. Singular reductions, inseparable models,
and models whose finite or reciprocal polynomial order is not certified
maximal are skipped as bad primes.

Supported split prime ideals of integral `QQ(alpha)` constants have since been
added. General `GF(p^f)(x)` curve factorization remains open. Automatic
Weil-bound order computation requires a degree-one anchor; higher-degree
anchors retain only the honest explicitly bounded search, not the former
exponential linear fallback.
