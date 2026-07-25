# Plan 004: Specify algebraic constants and bridge Hermite remainders to all-place residues

> **Executor instructions**: This is a design-and-characterization checkpoint
> for the first missing algebraic integration gate. Do not silently treat the
> displayed coefficient field as the constant field, and do not extend the old
> unramified plane-model shortcut to branch places by guesswork. Produce a
> theorem-backed contract and only wire code paths that its tests certify.
>
> **Drift check (run first)**:
> `git diff --stat 97fccfc..HEAD -- src/sympy_risch/{tower.py,algebraic.py,algebraic_geometry.py} tests/test_algebraic* docs/algebraic-infrastructure.md docs/design.md plans`
> If the field or place API changed, stop and update the mathematical model
> before using this plan.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/002-executable-admission-matrix.md`
- **Category**: direction / correctness / tests
- **Planned at**: commit `97fccfc`, 2026-07-25

## Why this matters

The repository now has maximal orders, ramified and infinite places, local
expansions, and divisor arithmetic. It is still not a pure algebraic
integrator: a finite extension can add constants, and the current Trager path
rejects branch poles and omits infinity. Before logarithmic reconstruction or
second-kind exactness can be sound, the implementation must know the true
constant field and collect residues at every place of a Hermite remainder.

## Current state

- `src/sympy_risch/tower.py:672-724` can rebase an existing derivation over a
  supplied larger exact constant domain, but it does not decide whether such a
  rebase is required.
- `src/sympy_risch/algebraic.py:420-489` implements Trager's finite resultant
  only for the standard `Dx = 1` rational parent and rejects poles meeting the
  plane-model discriminant.
- `src/sympy_risch/algebraic.py:1002-1045` performs algebraic Hermite
  reduction only over the standard rational parent.
- `src/sympy_risch/algebraic_geometry.py` already represents maximal orders,
  finite and infinite places, uniformizers, valuations, local Laurent
  expansions, and local differential residues.
- `docs/design.md:875-885` identifies constant normalization and all-place
  residue collection as the first missing algebraic gates.

Use the canonical `AlgebraicFunctionField` place APIs rather than growing the
legacy `UnramifiedFinitePlace` shortcut. Preserve the latter as an independent
Trager oracle on its certified unbranched subcase.

## Commands you will need

| Purpose | Command | Expected on success |
| --- | --- | --- |
| Focused tests | `uv run --no-sync pytest -q tests/test_algebraic.py tests/test_algebraic_geometry.py tests/test_algebraic_places.py tests/test_algebraic_residues.py` | all pass |
| Lint | `uv run --no-sync ruff check .` | exit 0 |
| Format | `uv run --no-sync ruff format --check .` | exit 0 |
| Types | `uv run --no-sync ty check` | exit 0 |
| Complete suite | `uv run --no-sync pytest -q -n auto --dist worksteal` | all pass |
| Patch hygiene | `git diff --check` | no output |

## Scope

**In scope**:

- `src/sympy_risch/tower.py`
- `src/sympy_risch/algebraic.py`
- `src/sympy_risch/algebraic_geometry.py`
- `tests/test_algebraic.py`
- `tests/test_algebraic_geometry.py`
- `tests/test_algebraic_places.py`
- `tests/test_algebraic_residues.py`
- `docs/algebraic-infrastructure.md`, `docs/design.md`, and the roadmap

**Out of scope**:

- Principal/torsion divisor solving, Riemann--Roch spaces, or second-kind
  exactness.
- General algebraic RDEs, monomial parent fields, or mixed towers.
- Replacing native SymPy finite extensions with a private element hierarchy.

## Git workflow

- Branch: `codex/004-algebraic-constants-and-residues`.
- Commit the characterization/specification and any certified narrow code path
  separately from a later full all-place collector.
- Do not push or open a pull request unless instructed.

## Steps

### Step 1: Write and test the algebraic constant-field contract

Add an immutable algebraic-constant analysis result with only sound outcomes:

- `PRESERVED` when the algorithm proves the function field is geometric over
  its declared constants;
- `REBASING_REQUIRED` with an exact larger constant domain and embedding when
  a finite algebraic constant extension is proved; and
- `UNSUPPORTED` when the current backend cannot decide the required constant
  operation.

Before selecting an implementation algorithm, add characterization tests for a
geometric curve, a constant extension such as a relation independent of `x`,
and an algebraic-number constant base. State the selected theorem and exact
operation requirements in `docs/algebraic-infrastructure.md`. A failed
heuristic or failed factorization must never mean `PRESERVED`.

**Verify**: every test has an exact rebase or preservation witness; the current
genus safeguard remains a regression rather than being bypassed.

### Step 2: Define the all-place residue collection result

Specify an immutable post-Hermite result containing:

- the original differential, Hermite exact part, and square-free remainder;
- the certified constant-field analysis;
- every finite and infinite pole place, with its uniformizer and exact local
  residue; and
- exact reconstruction and no-double-counting invariants.

The result may be `UNSUPPORTED` until the chosen constant-field analysis
supports a field, but it must not omit a branch or infinity place silently.

**Verify**: unit tests enumerate the collector's place support and assert the
unramified finite subset agrees with `trager_algebraic_place_residues` and its
aggregate resultant where that routine applies.

### Step 3: Characterize branch and infinity cases with local expansions

Use the existing canonical place/local-expansion APIs to add adversarial
examples for:

- a branch pole whose base polynomial is not a uniformizer;
- a pole at infinity;
- a singular displayed plane model normalized by the maximal-order route; and
- a positive-genus curve with a zero-residue differential.

For this checkpoint, it is valid to return `UNSUPPORTED` for a case only if
the result names the missing theorem operation. It is not valid to reuse the
unramified residue formula at a ramified place.

**Verify**: local Laurent residue extraction agrees with hand-derived small
examples and remains exact in native residue fields.

### Step 4: Choose the next global algorithm boundary

Record the precise inputs the future Riemann--Roch/principal-divisor service
will receive: normalized constants, all-place residues, divisor conventions,
and exact local witnesses. Update the roadmap so the next algebraic
implementation plan is global logarithmic reconstruction plus second-kind
exactness, not more unconnected place primitives.

**Verify**: the docs distinguish the completed substrate from the still-open
global integration decisions.

## Test plan

- Geometric and non-geometric finite extensions over `QQ(x)`.
- Exact constant rebasing over an algebraic-number field.
- Unramified finite residues cross-checked against the existing Trager oracle.
- Ramified finite and infinity residue examples using local uniformizers.
- A zero-residue, non-exact positive-genus differential retained as a
  regression for the later second-kind decision.

## Done criteria

- [ ] The constant-field status for every admitted algebraic residue case is
  explicit and witnessed.
- [ ] A collector cannot silently omit branch or infinity poles.
- [ ] The old unramified resultant route remains an independent oracle.
- [ ] Documentation names the next global divisor/cohomology gate exactly.
- [ ] Focused and full quality gates pass.

## STOP conditions

Stop and report if:

- the required constant subfield cannot be computed with the declared SymPy
  domain operations;
- a branch-place computation lacks a certified uniformizer or local expansion;
- support enumeration requires a non-maximal order without a normalization
  witness; or
- a proposed implementation needs principal-divisor solving to classify a
  residue.

## Maintenance notes

This plan intentionally stops before logarithmic reconstruction. Reviewers
should require exact field embeddings, explicit constant provenance, and an
all-place support proof. The subsequent global Riemann--Roch service should
consume this result rather than rediscover places from a plane-model shortcut.
