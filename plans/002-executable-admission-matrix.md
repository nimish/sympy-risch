# Plan 002: Make the current P/E/T contract executable

> **Executor instructions**: Follow this plan step by step. The goal is to
> make the *existing* certified P/E/T contract mechanically visible; it is not
> to widen that contract or replace mathematical solver exceptions wholesale.
> Run every verification command before committing. If a proposed supported
> cell reaches an algorithmic gap, stop and report it rather than changing the
> advertised boundary.
>
> **Drift check (run first)**:
> `git diff --stat 97fccfc..HEAD -- src/sympy_risch tests docs .github pyproject.toml`
> If any scoped code differs, compare this plan's current-state statements to
> the live implementation before proceeding.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: `plans/001-full-risch-roadmap.md`
- **Category**: direction / tests / DX
- **Planned at**: commit `97fccfc`, 2026-07-25

## Why this matters

The library is total only for a deliberately narrow class: explicit immutable
towers over `QQ(x)` with certified preserved constants and primitive,
hyperexponential, or hypertangent levels. The public dispatcher documents that
boundary but currently callers infer it from a mixture of tower metadata and
`NotImplementedError`. A table-driven admission result and generated coverage
matrix make the claim reviewable, prevent accidental contract expansion, and
give later front-end work one authoritative boundary.

## Current state

- `src/sympy_risch/elementary.py:634-715` dispatches a supplied tower and
  deliberately raises `NotImplementedError` for algebraic and unsupported
  monomial shapes.
- `src/sympy_risch/tower.py` stores immutable `ConstantFieldStatus` and
  `SpecialPolynomialStatus`, but has no public admission decision object.
- `docs/mixed-transcendental-towers.md:1-93` states the accepted P/E/T scope.
- `plans/001-full-risch-roadmap.md:264-290` already defines the required
  machine-checkable coverage gate.
- Existing tests use exact native-field reconstruction and theorem-backed
  obstruction objects; follow `tests/test_mixed_transcendental_integration.py`
  and `tests/test_elementary.py` rather than using `sympy.integrate` as an
  oracle.

The design constraints are non-negotiable: keep native SymPy domain elements,
immutable tower metadata, and the distinction between a proved obstruction and
an unsupported admission boundary.

## Commands you will need

| Purpose | Command | Expected on success |
| --- | --- | --- |
| Focused tests | `uv run --no-sync pytest -q tests/test_admission.py tests/test_mixed_transcendental_integration.py` | all pass |
| Lint | `uv run --no-sync ruff check .` | exit 0 |
| Format | `uv run --no-sync ruff format --check .` | exit 0 |
| Types | `uv run --no-sync ty check` | exit 0 |
| Complete suite | `uv run --no-sync pytest -q -n auto --dist worksteal` | all pass |
| Patch hygiene | `git diff --check` | no output |

## Scope

**In scope**:

- `src/sympy_risch/admission.py` (new immutable admission/coverage types)
- `src/sympy_risch/__init__.py` (only exports for the new public types)
- `tests/test_admission.py` (new generated coverage matrix)
- `tests/test_package.py` (only if public-export assertions need extension)
- `docs/mixed-transcendental-towers.md`
- `docs/getting-started.md`
- `docs/testing.md`
- `.github/workflows/ruff.yml` (rename/extend the quality job if necessary)
- `plans/001-full-risch-roadmap.md` and `plans/README.md`

**Out of scope**:

- New RDE, relation, or integration algorithms.
- Algebraic levels, general polynomial special-factor levels, expression
  parsing, or real branch presentation.
- Converting all internal `NotImplementedError` paths into one exception.

## Git workflow

- Branch: `codex/002-executable-admission-matrix`.
- Keep one checkpoint commit with an imperative message such as
  `Make the P/E/T admission contract executable`.
- Do not push or open a pull request unless instructed.

## Steps

### Step 1: Define an immutable admission vocabulary

Create `src/sympy_risch/admission.py` with a small, typed result protocol:

- an enum distinguishing `ACCEPTED` from `UNSUPPORTED` only (never a
  mathematical negative result);
- a stable, machine-readable reason enum for failed requirements such as base
  field, constant-field status, algebraic level, unsupported monomial family,
  and undecided special-polynomial hypothesis;
- an immutable `TowerAdmission` containing the problem name, tower, outcome,
  failed requirement when any, and human-readable detail; and
- one assessor for the present public elementary-integration P/E/T contract.

Use exact `isinstance`/metadata checks already used by the dispatcher. The
assessor must accept a finite P/E/T tower precisely when every non-base level
has `ConstantFieldStatus.PRESERVED`, and must reject rather than guess about
any algebraic or general nonlinear level. It must not call a solver or turn a
failed solver into an admission result.

**Verify**: focused unit tests construct accepted primitive, hyperexponential,
hypertangent, and mixed P/E/T towers plus rejected unknown/enlarged constants,
algebraic levels, and nonlinear unknown-special levels.

### Step 2: Build the declared-cell matrix

In `tests/test_admission.py`, declare the current support cells explicitly:

- each of the three top families;
- each permitted lower P/E/T family and at least one nested ordering;
- constructive exact cases and known theorem-backed non-elementary cases; and
- every relevant admission failure category.

For accepted cells, invoke `integrate_elementary` and assert either exact
reconstruction or a structured obstruction; an unexpected
`NotImplementedError` fails the test. For rejected cells, assert the exact
admission reason and do not invoke the integration dispatcher as evidence of
non-elementarity. Use bounded generated variants only where their construction
has an independent reconstruction check.

**Verify**: `uv run --no-sync pytest -q tests/test_admission.py` exits 0 and
its parametrization names every declared cell.

### Step 3: Align public documentation and CI with the executable contract

Add a concise support table to `docs/mixed-transcendental-towers.md` that
identifies the admission API and says it guards support, not mathematical
integrability. Link it from `docs/getting-started.md` and the roadmap.

Extend the current `.github/workflows/ruff.yml` quality workflow so CI runs the
same non-mutating gates documented in `docs/testing.md`: Ruff lint, Ruff
format check, `ty check`, and the parallel pytest suite. Preserve `uv`'s
locked install and avoid adding a separate package manager.

**Verify**: documentation examples remain executable; inspect the workflow to
confirm every documented required gate has a CI step.

### Step 4: Reconcile the roadmap

Mark Milestone 0 as complete only if the admission table, cell test, and
unsupported-vs-obstruction protocol are all present. Otherwise record exactly
which result or cell remains missing. Update `plans/README.md` after the
checkpoint lands.

**Verify**: `git diff --check` has no output and the full quality gate passes.

## Test plan

- One accepted base rational cell and one accepted tower for each P/E/T family.
- At least one accepted nested mixed ordering.
- One constructive and one non-elementary/obstruction result per accepted top
  family.
- Regressions for `UNKNOWN` and `ENLARGED` constants, algebraic levels, and
  general nonlinear special status.
- A package export test for every newly public admission symbol.

## Done criteria

- [ ] The present P/E/T scope has one programmatic assessor and stable reason
  codes.
- [ ] Every declared accepted cell either reconstructs or returns a
  theorem-backed obstruction; none raises `NotImplementedError`.
- [ ] Rejected shapes have an explicit unsupported reason and are never
  presented as non-elementary.
- [ ] CI runs lint, format, type, and test gates.
- [ ] Focused and full test suites, Ruff, ty, and `git diff --check` pass.

## STOP conditions

Stop and report if:

- an accepted P/E/T cell reaches `NotImplementedError`;
- the only way to classify a cell is to run a solver and reinterpret a failure;
- an existing obstruction lacks its theorem hypothesis; or
- the proposed work requires changes to algebraic or special-factor algorithms.

## Maintenance notes

Every later accepted tower family must add a matrix cell and admission reason
before its user-facing documentation claims support. Keep the matrix small and
declarative; it is a contract test, not a duplicate implementation of the
dispatcher.
