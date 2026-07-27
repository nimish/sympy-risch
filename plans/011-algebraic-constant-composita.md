# Plan 011: Certified algebraic constant composita

## Goal

Replace the one-off `QQ -> QQ(alpha)` integration path with a reusable exact
map between algebraic constant fields, then use a chain of those maps to
continue algebraic integration through every compatible residue factor.

## Checkpoints

1. Lift a certified constant-field embedding
   `C -> C(alpha)` coefficientwise through `C(x)[y]/(f)`.  Support `C = QQ`
   and an existing SymPy algebraic number field, rebuild `f` over the
   compositum, re-prove that the rebuilt function field has constants
   `C(alpha)`, and verify commutation with the derivation.
2. Replace the one-step integration result with an explicit chain.  At each
   unresolved residue boundary, factor the newly collected all-place profile,
   select a factor that strictly enlarges the constants, rebuild the target
   field, and integrate the embedded differential again.
3. Add examples with two independent residue fields.  A successful answer
   belongs to the final compositum; an unavailable embedding, nonsplitting
   factor, or unsupported rebuilt relation remains an explicit unknown
   boundary, never a non-elementarity claim.

## Non-goals

This checkpoint does not solve divisor-class torsion, arbitrary extensions
defined over non-rational algebraic coefficients, or mixed
algebraic/transcendental RDEs.
