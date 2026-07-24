# Mathematical testing contract

Every algorithmic checkpoint in `sympy-risch` must test more than a worked
example.  Tests should make the mathematical hypotheses and promised
postconditions executable.

## Required layers

For each reduction or decision procedure, cover the applicable layers below.

1. **Exact reconstruction**
   Verify the defining identity in the native SymPy domain.  For a reduction
   `f = D(g) + h`, test that identity exactly rather than numerically.
2. **Canonical postcondition**
   Check the property established by the algorithm: degree bounds,
   square-free normal denominators, special factors, reduced representatives,
   or constant residues.
3. **Idempotence**
   Reapplying a completed reduction to its remainder should return zero exact
   part and the same remainder.
4. **Independent oracle**
   Where practical, compare with a different algorithm or formulation:
   recursive versus global Hermite reduction, implicit differentiation,
   direct pole residues, or SymPy differentiation of the presented result.
5. **Generated adversarial families**
   Generate planted exact parts and remainders across degrees,
   multiplicities, nonunit leading coefficients, mixed normal/special factors,
   and algebraic relations.  Include zero and cancellation cases.
6. **Rejected preconditions**
   Test wrong generators, domains, repeated or noncoprime inputs, colliding
   auxiliary symbols, and unsupported extension families.
7. **Structure-theorem hypotheses**
   A non-elementarity certificate must require executable evidence for every
   hypothesis it uses.  Include a concrete counterexample showing what becomes
   elementary when a hypothesis is removed.
8. **Required arithmetic backend**
   Run the complete suite with `python-flint`.  A backend guard test must fail
   if SymPy falls back to its pure-Python ground types; backend parity is not a
   project requirement.

## Test design rules

- Prefer exact domain equality or `cancel(...) == 0`; floating-point sampling
  is not evidence for a symbolic identity.
- A planted derivative alone is not an independent oracle.  Pair it with a
  normal-form assertion or an independently derived formula.
- Keep generated cases bounded enough for the ordinary suite.  Put only truly
  expensive stress families behind the `slow` marker.
- When a regression exposes a false certificate, retain both the smallest
  counterexample and a generated family containing it.
- Tests for partial algorithms must distinguish “proved impossible,”
  “successfully reduced,” and “not yet decided.”

## Checkpoint gate

Before each incremental commit:

```text
focused tests
full python-flint suite
git diff --check
```
