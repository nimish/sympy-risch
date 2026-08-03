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

If the environment previously installed the project in editable mode, add
`--reinstall-package sympy-risch` when switching to a non-editable install.

Run the static quality gates:

```console
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
```

Ruff checks both production and test code, and its formatter checks Python and
fenced Python examples in Markdown. Production functions require annotations;
test functions are exempt from annotation-style rules because pytest and
Hypothesis supply many values dynamically. Ty checks `src` and therefore
verifies the public implementation and all recursive solver paths without
treating the untyped test harness as part of the library API.

The `Quality` workflow runs these three static gates and the parallel pytest
suite on pull requests and pushes to `master`. Python 3.13 measures statement
and branch coverage with an 84% ratchet, while Python 3.14 checks compatibility
without duplicating coverage collection. It first performs the same locked
`uv sync --locked --all-groups` install, so CI and a checkpoint commit use one
declared dependency set rather than separate package managers. Separate
packaging jobs smoke-test an installed wheel on Ubuntu, macOS, and Windows;
only Ubuntu also checks the source distribution.

Run every test before a checkpoint commit:

```console
uv run --no-sync pytest -q -n auto --dist worksteal
```

The complete gate uses one worker per detected CPU and xdist's work-stealing
scheduler. The largest generated oracle comparisons are split into independent
pytest items without reducing their total Hypothesis example count, so they can
be shared between workers. Use the serial form when debugging ordering,
tracebacks, or a single failing example:

```console
uv run --no-sync pytest -q
```

Run the Python 3.13 CI coverage gate locally with:

```console
uv run --no-sync pytest -q -n auto --dist worksteal \
  --cov=sympy_risch --cov-branch --cov-report=term:skip-covered \
  --cov-fail-under=84
```

The floor is deliberately below the measured baseline so small
platform-dependent branch differences do not make the gate flaky, while a
material regression still fails CI.

Focused files mirror the implementation modules. For example,
`tests/test_admission.py` declares the certified elementary-dispatch cells and
the separate scalar-RDE prefix contract. The latter scans every active lower
level, including the Gaussian `QQ(i)(x)` P/E boundary, so an uncertified
ancestor cannot leak a solver precondition error during recursion.
Every explicitly declared primitive/hyperexponential/hypertangent solver
cell, including the exact hyperbolic Möbius normalization exception at the
active top level, must either reconstruct or return a structured mathematical
obstruction; an unexpected `NotImplementedError` fails that cell.
Structurally admitted but unimplemented cells assert their documented
boundary. Rejected towers instead assert their stable `TowerAdmissionReason`
and are never used as evidence of non-elementarity. General nonlinear
no-special experiments remain outside that matrix.
`tests/test_hyperbolic_mobius.py` additionally checks the explicit
two-factor quadratic/Riccati normalizer: asymmetric factors that do not divide
`Dt`, coordinate orientation, generated moving-root laws, exact map and
derivation identities, and foreign/duplicate/nonlinear/status boundaries.
`tests/test_special_poles.py` checks the separate positive-evidence consumer
for one linear special factor: moving Riccati cofactors reduced modulo the
factor, repeated-pole descent, mixed denominators, generated planted cases,
unsupported lower RDEs (including uncertified ancestors), immutable
certificate replay, and a higher-order resonance where an unresolved local
RDE is deliberately *not* treated as non-elementarity.
`tests/test_riccati_mobius_integration.py` checks its separate public
integration route with planted derivatives, a pulled-back non-elementary
hyperexponential obstruction, a pulled-back logarithmic term, and rejection
when the transformed target is outside the declared P/E/T contract.
`tests/test_nested_primitive.py` exercises recursive primitive descent,
`tests/test_primitive_rde.py` cross-checks the primitive-field recurrence
against an independent bounded matrix oracle. Its cancellation tests cover
logarithmic-coefficient exact transformations, recursively solved leading
coefficients, weak normalization of top-generator poles, rational solution
denominators, certified failures, nested primitive towers, and generated
positive and negative oracle comparisons. The oracle accepts an independent
top-denominator ansatz, so it does not reuse the production normalizer or
normal-denominator theorem.
`tests/test_hypertangent.py` exercises the direct tangent reducer, and
`tests/test_special.py` verifies concrete Darboux certificates together with
the primitive no-special theorem: a primitive level can certify absence only
after its constant field is certified preserved; unknown and enlarged cases
remain nonnegative evidence only.
`tests/test_algebraic.py` exercises algebraic Hermite reduction and integral
bases. `tests/test_algebraic_geometry.py` exercises certified maximal orders,
fractional ideals, ramified and infinite places, exact valuations and local
expansions, differents, divisors, and differential residues. Its adversarial
families include singular plane models with hidden normalization, deep
quadratic/cubic/quartic index factors, closed points with algebraic residue
fields, and generated finite/infinite valuation identities.
`tests/test_algebraic_all_places.py` connects that global geometry layer to
algebraic Hermite reduction. It checks a ramified branch pole and infinity,
an unramified finite cross-check against Trager's independent resultant
route, a singular displayed model after normalization, and a genus-one
zero-residue differential that is intentionally not treated as exact. It also
checks that an exact Hermite result bypasses constant-field/global-geometry
work, that enlarged or unproved constant fields return no partial place list,
that each local residue belongs to its own residue field, and that the
collector reuses its already verified divisor and local-residue computations.
`tests/test_algebraic_residue_profile.py` builds the all-place
characteristic-polynomial profile from that data. It exercises a ramified
branch pole together with infinity, compares the unramified finite slice to
Trager's independent resultant, keeps degree-two and degree-four closed-place
factors exact (including a degree-two infinity factor), distinguishes a unit
profile from exactness, and preserves exact versus unsupported collection
outcomes.
`tests/test_riemann_roch.py` checks the first certified global-space cases,
including a genus-zero parameter basis, a genus-one nonspecial basis, an exact
algebraic constant field, degree-zero principal witnesses, and forged or
dependent candidates. Positive-genus low-degree and rebasing-required cases
must return an empty `UNSUPPORTED` result rather than a partial basis.
`tests/test_algebraic_constraints.py` checks exact finite-subspace valuation
intersections, including finite and infinite ramification, cancellation that
removes a pole, algebraic constant fields, and degree-four closed-place
coordinate flattening. Its nonspecial bounded search is allowed to certify a
global basis only by delegating to the existing Riemann--Roch certificate
after reaching the theorem dimension.
`tests/test_algebraic_principal.py` checks the separate bounded
principal-divisor semialgorithm: it reconstructs genus-zero and
positive-genus witnesses through their complete divisors, never turns an
exhausted degree-zero slice into a negative decision, and treats nonzero
degree as the complete elementary obstruction.
`tests/test_algebraic_logarithmic.py` exercises the first end-to-end algebraic
integration cell: ramified and split rational-residue logarithms, independent
logarithmic directions over an algebraic constant field, a retained
residue-free remainder, residues requiring compatible base change, explicit
and profile-selected target-field reconstruction over algebraic constants, and a
bounded principal-divisor miss that is not misreported as a non-elementarity
decision.
`tests/test_algebraic_exactness.py` covers the residue-free exactness step:
genus-zero primitives, first- and second-kind elliptic no-primitive decisions,
and a bounded primitive-space search that remains incomplete until its
Riemann--Roch dimension is reached.
`tests/test_algebraic_rde.py` covers the direct integrating-factor RDE cell:
it reduces a genus-zero algebraic equation to exactness, preserves a bounded
exactness miss, proves a transformed no-solution case, and does not mistake a
nonintegral logarithmic residue for a general-RDE negative result.
`tests/test_relative_algebraic.py` covers the first mixed cells: terminal pure
radicals over primitive, hyperexponential, and hypertangent lower towers;
quadratic and cubic power bases; nonzero lower coefficients; exact radical
modes composed with lower logarithms; zero upper coordinates; and boundaries
for unsupported lower RDEs, non-lower coupled coefficients, nonexact upper
coordinates, and lower obstructions that cannot yet be lifted safely.
`tests/test_relative_hermite.py` plants repeated normal poles in primitive,
hyperexponential, and hypertangent lower towers, checks exact reconstruction
and square-free normal remainders, retains special poles, and exercises a
cubic power-basis coordinate plus a resonant leading-equation boundary.
`tests/test_tower.py` also exercises the immutable mixed-prefix feasibility
result: a native monomial prefix plus a terminal algebraic extension must
coerce and differentiate at its top, while monomial-over-algebraic and nested
algebraic orderings remain structured backend limitations rather than partial
towers.
`tests/test_parametric_logarithmic.py` checks Bronstein's equation
(7.37), the Section 7.3 rational fast path, and exact witnesses over rational
and primitive lower fields. It exercises both coefficient-comparison
branches, generated planted solutions, necessary-condition failures, and
Example 7.3.2. The complete rational solver is checked against an independently
derived global logarithmic-derivative ansatz over generated polynomial parts,
repeated poles, and irreducible factors of degrees two and three. The oracle
does not consume the solver's certificate or locally constructed constraints.
The tests enforce the distinction between a proved obstruction, an
inconclusive fast path, and an unsupported field.
`tests/test_dependency.py` verifies rational and radical hyperexponential
dependencies over both `QQ(x)` and primitive lower towers, including exact
named-exponential presentations and exact reconstruction of the resulting
constant relation.
`tests/test_finite_algebraic_constants.py` checks the finite-algebraic
constant boundary: exact degree-two and degree-four native power-basis
coordinates, rational (not algebraic-scalar) relation kernels, generated
residue cancellations, direct checked P/E proposals over `QQ(sqrt(2))(x)`,
and explicit provenance, hypertangent, and nonconstant-field boundaries.
`tests/test_expression.py` is the public frontend contract. It checks source
round trips, native and source-tree derivatives, nested primitive and tangent
levels, positive and negative integration results, and constructor resistance
to forged tower metadata or presentations. Its generated rational and
one-level P/E/T cases prove that coercion, factory construction, and
back-substitution survive varied exact input; adversarial cases ensure that
non-rational constants, unsupported algebraic powers, misdeclared algebraic
parameters, singular calls, branch-sensitive functions, and dependent logarithmic,
exponential, or tangent calls remain structured `UNSUPPORTED` outcomes rather
than accidental mathematical conclusions.
`tests/test_algebraic_presentation.py` checks the reusable primitive-element
normalizer independently: full tensor degree, exact source recovery, foreign
parents, forged images, reducible tensor boundaries, and one global
characteristic-polynomial computation.
`tests/test_exact.py` checks family dispatch with planted derivatives,
generated hyperexponential and hypertangent examples, normal and logarithmic
residues, unsolved special-mode RDEs, recursive lower-field exactness, and
idempotent obstructions. The family RDE decisions are independently covered
by the bounded matrix-oracle comparisons in the corresponding reducer tests.
`tests/test_hyperexponential_rde.py` checks positive, zero, and negative
Laurent modes; recursive dispatch through two hyperexponential levels;
certified support bounds; descending and ascending recurrences; ordinary and
merely radical boundary relations; high-degree inputs; partial-solution and
obstruction invariants; and constant-field preconditions. Generated arbitrary
Laurent equations, planted solutions, and single-mode nonsolutions are
cross-checked against a global bivariate matrix oracle that neither separates
Laurent modes nor calls either production recurrence.
`tests/test_hyperexponential_rational_rde.py` covers resultant weak
normalization, repeated and nonintegral residues, normal and special
denominator bounds, degree and divisibility obstructions, recursive primitive
lower fields, and generated rational solutions. Its matrix oracle accepts an
independent top denominator and broad Laurent support, so it does not reuse
the production normalizer, denominator theorem, valuation bound, or SPDE.
`tests/test_recursive_coupled_rde.py` checks the Gaussian scalar construction
over one and two primitive levels, direct recursion into a hyperexponential
level, denominator obstructions, and generated solution and
arbitrary-right-hand-side families. Its oracle builds the two real equations
directly as a block matrix over `QQ`, so it does not share the production
tower rebasing or scalar RDE dispatcher.
`tests/test_coupled_cancellation.py` reproduces Bronstein Example 8.4.1 and
checks direct hyperexponential Laurent modes, hypertangent no-cancellation and
`CoupledDECancelTan` branches, recursive lower-field obstructions, degree and
shape errors, and noncanonical `QQ_I` fraction units. Generated arbitrary
systems are compared with an independent global real block-matrix ansatz that
does not split Laurent modes, evaluate at `i`, or divide through
`t**2 + 1`.
`tests/test_nested_hypertangent.py` checks Gaussian dependency analysis,
dependent-generator rejection, special and full reductions, exact recursive
coupled certificates, elementary dispatch, two primitive lower levels, and
generated Cayley-mode families.
`tests/test_structure_theorem_totality.py` checks positive radical and
parametric relations plus a certified negative decision in every three-family
P/E/T ordering. A repeated-tangent example ensures the direct real relation
system is the production path, while focused hypertangent tests compare it
with the independent Gaussian oracle.
`tests/test_primitive_logarithmic.py` checks recursive primitive signatures
through three tower levels. It compares both one-level and nested decisions
with separate global linear ansatzes. Generated inputs mix inherited
lower-field obstructions, positive-degree polynomial parts, repeated top
poles, and logarithmic and nonlogarithmic residues at
coefficient-dependent irreducible factors.
`tests/test_primitive_parametric_logarithmic.py` compares the local
structure-theorem recurrence with separate one-level and nested global
parametric ansatzes. It includes cancellations that occur only after selecting
`m/n`, ratios determined only after recursive descent, conflicting top and
lower constraints, and generated quadratic and cubic factors.
`tests/test_algebraic_residues.py` checks Trager's nested resultant against
directly specialized residues, the rational Rothstein--Trager degeneration,
and a high-degree explicit local oracle. Generated quadratic models vary the
pole, branch locus, and numerator. Adversarial tests reject repeated poles,
branch poles, nonpolynomial models and numerators, reused symbols, mismatched
fields, and nonstandard rational derivations.
`tests/test_documentation.py` executes the README quick-start block through
the public package import, so documented construction and result assertions
cannot silently drift from the API.

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
8. **Admission contract**
   Test public support classification separately from integration. Every
   accepted cell must reach either exact reconstruction or a structured
   obstruction, while every unsupported cell must have a stable
   `TowerAdmissionReason` and must not be described as non-elementary.
9. **Required arithmetic backend**
   Run the complete suite with `python-flint`. A backend guard test must fail
   if SymPy falls back to its pure-Python ground types; backend parity is not a
   project requirement.

## Test design rules

- Prefer exact domain equality or `cancel(...) == 0`; floating-point sampling
  is not evidence for a symbolic identity.
- A planted derivative alone is not an independent oracle. Pair it with a
  normal-form assertion or an independently derived formula.
- Keep generated cases bounded enough for the complete suite. Split or
  optimize a stress family that cannot meet that bar rather than excluding it
  from the standard checkpoint gate.
- When a regression exposes a false certificate, retain both the smallest
  counterexample and a generated family containing it.
- Tests for partial algorithms must distinguish “proved impossible,”
  “successfully reduced,” and “not yet decided.”
- Do not use an unsupported `TowerAdmission` as an integration oracle. The
  admission matrix guards the dispatcher contract; it does not decide whether
  an integral in an unadmitted tower is elementary.

## Checkpoint gate

Before each incremental commit:

```console
uv run --no-sync pytest -q tests/test_relevant_module.py
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest -q -n auto --dist worksteal
git diff --check
```

Replace `tests/test_relevant_module.py` with the focused test file or node ID
for the change. Documentation examples must also be executed through the
public package import; prose that merely resembles code is not considered
verified.
