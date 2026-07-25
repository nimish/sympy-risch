# Mathematical testing contract

Every algorithmic checkpoint in `sympy-risch` must test more than a worked
example. Tests should make the mathematical hypotheses and promised
postconditions executable.

## Running the suite

Install the required development and `python-flint` dependencies from a
checkout:

```console
uv sync --no-editable
```

Run the ordinary suite without the generated rational stress test:

```console
uv run --no-sync pytest -q -m "not slow"
```

Run every test before a checkpoint commit:

```console
uv run --no-sync pytest -q
```

Focused files mirror the implementation modules. For example,
`tests/test_nested_primitive.py` exercises recursive primitive descent,
`tests/test_primitive_rde.py` cross-checks the primitive-field recurrence
against an independent bounded matrix oracle,
`tests/test_hypertangent.py` exercises the direct tangent reducer, and
`tests/test_algebraic.py` exercises algebraic Hermite reduction and integral
bases. `tests/test_parametric_logarithmic.py` checks Bronstein's equation
(7.37), the Section 7.3 rational fast path, and exact witnesses over rational
and primitive lower fields. It exercises both coefficient-comparison
branches, generated planted solutions, necessary-condition failures, and
Example 7.3.2. The complete rational solver is checked against an independently
derived global logarithmic-derivative ansatz over generated polynomial parts,
repeated poles, and irreducible factors of degrees two and three. The oracle
does not consume the solver's certificate or locally constructed constraints.
The tests enforce the distinction between a proved obstruction, an
inconclusive fast path, and an unsupported field.
`tests/test_primitive_logarithmic.py` checks recursive primitive signatures
through three tower levels. It compares both one-level and nested decisions
with separate global linear ansatzes. Generated inputs mix inherited
lower-field obstructions, positive-degree polynomial parts, repeated top
poles, and logarithmic and nonlogarithmic residues at
coefficient-dependent irreducible factors.
`tests/test_primitive_parametric_logarithmic.py` compares the local
structure-theorem recurrence with a separate global parametric ansatz. It
includes cancellations that occur only after selecting `m/n`, conflicting
top and lower constraints, and generated quadratic and cubic factors.
`tests/test_algebraic_residues.py` checks Trager's nested resultant against
directly specialized residues, the rational Rothstein--Trager degeneration,
and a high-degree explicit local oracle. Generated quadratic models vary the
pole, branch locus, and numerator. Adversarial tests reject repeated poles,
branch poles, nonpolynomial models and numerators, reused symbols, mismatched
fields, and nonstandard rational derivations.

## Required layers

For each reduction or decision procedure, cover the applicable layers below.

1. **Exact reconstruction**
   Verify the defining identity in the native SymPy domain. For a reduction
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
   and algebraic relations. Include zero and cancellation cases.
6. **Rejected preconditions**
   Test wrong generators, domains, repeated or noncoprime inputs, colliding
   auxiliary symbols, and unsupported extension families.
7. **Structure-theorem hypotheses**
   A non-elementarity certificate must require executable evidence for every
   hypothesis it uses. Include a concrete counterexample showing what becomes
   elementary when a hypothesis is removed.
8. **Required arithmetic backend**
   Run the complete suite with `python-flint`. A backend guard test must fail
   if SymPy falls back to its pure-Python ground types; backend parity is not a
   project requirement.

## Test design rules

- Prefer exact domain equality or `cancel(...) == 0`; floating-point sampling
  is not evidence for a symbolic identity.
- A planted derivative alone is not an independent oracle. Pair it with a
  normal-form assertion or an independently derived formula.
- Keep generated cases bounded enough for the ordinary suite. Put only truly
  expensive stress families behind the `slow` marker.
- When a regression exposes a false certificate, retain both the smallest
  counterexample and a generated family containing it.
- Tests for partial algorithms must distinguish “proved impossible,”
  “successfully reduced,” and “not yet decided.”

## Checkpoint gate

Before each incremental commit:

```console
uv run --no-sync pytest -q tests/test_relevant_module.py
uv run --no-sync pytest -q
git diff --check
```

Replace `tests/test_relevant_module.py` with the focused test file or node ID
for the change. Documentation examples must also be executed in an installed
interpreter; prose that merely resembles code is not considered verified.
