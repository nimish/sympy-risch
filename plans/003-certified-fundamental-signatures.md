# Plan 003: Extract certified fundamental signatures from the P/E/T relation engine

> **Executor instructions**: Implement only the first structure-theorem
> checkpoint: make the existing certified P/E/T relation information
> explicit and independently testable. Do not claim general effective
> constants, a Darboux decision, or a real branch presentation in this plan.
> Preserve every current constructive and negative result exactly.
>
> **Drift check (run first)**:
> `git diff --stat 97fccfc..HEAD -- src/sympy_risch/{relations.py,dependency.py,gaussian.py,tower.py} tests docs plans`
> If the current P/E/T relation path has changed, stop and reconcile the
> theorem mapping before extracting an API.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/002-executable-admission-matrix.md`
- **Category**: direction / architecture / correctness
- **Planned at**: commit `97fccfc`, 2026-07-25
- **Implementation status**: DONE, 2026-07-25

## Implemented-contract note

The implementation is intentionally narrower than a general Chapter 9
solver.  A supported `FundamentalSignature` describes one active immutable
tower prefix over the standard `QQ(x)` or `QQ_I(x)` base with `Dx = 1`.  It
contains exactly one native-field direction for every active level, in tower
order: `Dt` for primitive levels, `Dt/t` for hyperexponential levels, and the
real coefficient `eta` for hypertangent levels.  The rational base variable
is retained as the initial primitive direction.  Every non-base direction
requires derivation-bound replay evidence of constant preservation.

For a hypertangent source level, the signature deliberately retains the real
law `Dt = eta*(1 + t**2)`.  Its Cayley image, whose logarithmic derivative is
`2*i*eta`, is checked separately by the immutable Gaussian normal form.  The
signature must not make the Gaussian conversion self-validating, and it does
not claim a direct real structure theorem for arbitrary special polynomials.

`UNSUPPORTED` means only that this narrow representation cannot be certified;
it is never a negative relation, dependency, or integrability conclusion.
The existing normal-factor valuation logic remains a dynamic part of relation
solving, so the signature does not enumerate all Darboux or non-generator
special factors.  Checked extension factories may retain signature evidence
only when it is already available; lack of a signature must remain
`UNKNOWN`/`UNSUPPORTED`, never an inferred preserved-constants result.

## Why this matters

The current P/E/T relation system has implemented certified directions within
its declared `QQ(x)` inputs, but its Chapter 9 information is embedded in specialized relation
compilation and Gaussian fallback paths. A full Risch implementation needs an
explicit, immutable fundamental signature—ordered logarithmic, exponential,
and real tangent directions—before it can safely generalize constants,
no-new-constants decisions, or special-factor relations. This checkpoint
freezes the proven subset behind an inspectable object and makes current
solvers its behavioral oracle.

## Current state

- `src/sympy_risch/relations.py:194-221` restricts relation construction to
  `QQ`/`QQ_I` rational towers and rejects algebraic levels.
- `src/sympy_risch/relations.py:358-415` compiles normal-factor valuations and
  canonical hyperexponential directions, but rejects non-generator special
  factors.
- `src/sympy_risch/dependency.py:1827-1940` dispatches the complete
  parametric logarithmic-derivative decisions for supported towers.
- `src/sympy_risch/gaussian.py` provides immutable Cayley normal forms for
  hypertangent levels; it must remain an independent representation check.
- `plans/001-full-risch-roadmap.md:361-408` maps the target to Bronstein
  Chapter 9 and deliberately leaves general constants and special factors
  open.

The signature must use native `Domain` elements and immutable tower metadata.
Do not create a parallel element hierarchy or make an incomplete search look
like a negative relation theorem.

## Commands you will need

| Purpose | Command | Expected on success |
| --- | --- | --- |
| Focused tests | `uv run --no-sync pytest -q tests/test_logarithmic_relations.py tests/test_dependency.py tests/test_parametric_logarithmic.py` | all pass |
| Lint | `uv run --no-sync ruff check .` | exit 0 |
| Format | `uv run --no-sync ruff format --check .` | exit 0 |
| Types | `uv run --no-sync ty check` | exit 0 |
| Complete suite | `uv run --no-sync pytest -q -n auto --dist worksteal` | all pass |
| Patch hygiene | `git diff --check` | no output |

## Scope

**In scope**:

- `src/sympy_risch/relations.py`
- `src/sympy_risch/dependency.py`
- `src/sympy_risch/gaussian.py` (only adapter/certificate integration)
- `src/sympy_risch/__init__.py`
- `tests/test_logarithmic_relations.py`
- `tests/test_dependency.py`
- `tests/test_parametric_logarithmic.py`
- `tests/test_hypertangent_gaussian.py`
- `docs/design.md`, `docs/mixed-transcendental-towers.md`, and the roadmap

**Out of scope**:

- Algebraic levels or algebraic number fields beyond the already supported
  Gaussian constants.
- New RDE recurrences or a public parametric-RDE solver.
- General polynomial/Darboux factor discovery.
- Expression parsing and source presentation.

## Git workflow

- Branch: `codex/003-certified-fundamental-signatures`.
- Keep one mathematical checkpoint commit, for example
  `Extract certified P/E/T fundamental signatures`.
- Do not push or open a pull request unless instructed.

## Steps

### Step 1: Specify the signature invariants before refactoring

Add immutable signature dataclasses in `relations.py` (or a narrowly named new
module if that prevents a dependency cycle). The signature must retain:

- the exact source derivation and active tower prefix;
- ordered primitive/logarithmic, hyperexponential, and hypertangent/Cayley
  directions;
- the exact native field elements representing their logarithmic derivatives;
- replay evidence that each active level preserves constants; and
- a status that is `UNSUPPORTED`, not negative, outside the current P/E/T
  domain.

Document the invariant: it represents the directions supplied by the current
Chapter 9 relation construction, not all Darboux polynomials of a general
monomial.

**Verify**: a unit test can inspect ordered directions for a primitive,
hyperexponential, and hypertangent-containing tower without running a
parametric query.

### Step 2: Compile the existing relation system from the signature

Refactor the current relation compiler so the signature is the single source
for canonical generator directions. Keep the existing normal-factor valuation
logic and exact rational linear-system solve; do not change its equations.
For hypertangent levels, record the real source direction and retain the
Gaussian normal form as the independently verified coordinate translation.

**Verify**: all existing parametric logarithmic-derivative tests are unchanged
in outcome and witness reconstruction; add equality tests comparing the old
relation vectors (captured in test helpers) to those emitted by the signature.

### Step 3: Connect certified extension factories conservatively

Make primitive, hyperexponential, and hypertangent checked factories retain or
consume the relevant signature evidence only inside the already accepted P/E/T
scope. Any lower tower for which a signature cannot be compiled must retain
`UNKNOWN`/`UNSUPPORTED`; it must not be marked `PRESERVED` by default.

**Verify**: regressions cover dependent generators such as two logarithms with
linearly dependent derivatives and dependent exponentials, plus a rejected
algebraic lower level.

### Step 4: Add equivalence and adversarial tests

Add generated small P/E/T towers with planted integer relations. Compare:

- signature-derived relation results;
- the existing direct relation-system path; and
- Gaussian-coordinate calculations for hypertangent cases.

Add adversarial tests for forged or unknown constant metadata, non-generator special
factor attempts, and coerced/mismatched parent fields.

**Verify**: all three descriptions agree on a normalized witness or on the
same supported negative certificate.

### Step 5: Record the remaining theorem boundary

Document that the next follow-up adds a capability protocol for algebraic
number constants and a complete non-generator special-factor decision. Do not
relax `relations.py` guards until that protocol and proof obligations exist.

**Verify**: docs describe the implemented P/E/T relation scope without
claiming every recursive solver ordering or a full Chapter 9 implementation.

## Test plan

- Explicit signature shape for every P/E/T family and at least two mixed
  orderings.
- Generated planted relation and no-relation cases.
- Exact witness reconstruction for every positive relation.
- Agreement with the existing matrix/global relation oracles.
- Rejected algebraic, unknown-constant, and non-generator-special cases.

## Done criteria

- [x] A public immutable signature object describes every accepted P/E/T
  tower prefix.
- [x] Existing relation and dependency results preserve their outcomes and
  exact witness checks.
- [x] Gaussian hypertangent conversion remains independently checked.
- [x] No unsupported lower family receives a false `PRESERVED` status.
- [x] Focused and full quality gates pass.

## STOP conditions

Stop and report if:

- extraction changes an existing negative relation outcome;
- a required direction cannot be represented as a native exact field element;
- the work requires a claim about a non-generator special polynomial; or
- supporting algebraic constants needs a capability not yet specified.

## Maintenance notes

The next structure checkpoint must extend this object rather than duplicate
the current relation compiler. Reviewers should scrutinize direction ordering,
constant-field provenance, and every `UNKNOWN`/`UNSUPPORTED` transition.
