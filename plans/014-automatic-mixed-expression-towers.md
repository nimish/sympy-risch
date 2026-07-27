# Plan 014: Discover algebraic and mixed expression towers

> **Executor instructions**: Separate structural discovery from integration
> support. A successfully identified tower may still reach an explicit
> integration-backend boundary; neither outcome is a non-elementarity proof.
> Do not encode individual showcase integrals as special cases.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: Plans 007, 010, 011, 012, and 013
- **Category**: automatic tower construction / mixed integration
- **Started**: 2026-07-27
- **Status**: IN PROGRESS

## Goal

Replace the syntax-directed P/E/T-only frontend with a staged expression
compiler that:

1. discovers every non-rational subexpression that may generate a field
   extension;
2. orders candidates by their actual expression dependencies;
3. classifies each candidate as algebraic, primitive, hyperexponential, or
   hypertangent;
4. asks the existing relation and constant-field algorithms whether the
   candidate is independent, dependent, or requires reclassification;
5. constructs an immutable native differential tower when the SymPy domain
   backend represents the resulting ordering; and
6. reports integration support independently from successful discovery.

The construction is systematic in the differential-algebra sense: syntax
proposes generators, while minimal-polynomial, exact-derivative, and
logarithmic-derivative decisions determine what extension is actually needed.
The final tower is not required to preserve the source traversal order.

## Checkpoint slices

### 1. Candidate graph

Walk the expression once and produce an immutable dependency DAG. Initial
candidate forms are:

- nonintegral rational powers `u**(p/q)`, proposed with
  `Y**q - u**p = 0`;
- `log(u)`, proposed primitive with `Dt = Du/u`;
- `exp(u)`, proposed hyperexponential with `Dt/t = Du`; and
- `tan(u)`, proposed hypertangent with `Dt = Du*(1+t**2)`.

Rational arithmetic and integer powers introduce no generator. `RootOf` and
general algebraic-function objects are a later candidate parser that feeds the
same algebraic node type.

### 2. Relation-aware construction

Process the DAG bottom-up. At each node:

- embed its dependencies in the current field;
- normalize and factor the proposed algebraic polynomial;
- replay the appropriate no-new-constants or dependency decision;
- reuse an existing field element when a relation supplies one;
- reclassify a finite-power exponential relation as algebraic; and
- adjoin a generator only when the candidate remains independent.

Branch-sensitive identities are not generic rewrite rules. The compiler binds
an algebraic generator to the particular local branch represented by the
source expression and records that source substitution for presentation.

### 3. Mixed native towers

The representation must cover both directions:

- algebraic over transcendental, such as `sqrt(x + log(x))` and
  `sqrt(exp(x) + 1)`; and
- transcendental over algebraic, such as `exp(sqrt(x))` and
  `log(1 + sqrt(x))`.

If SymPy's domain coercion cannot represent an otherwise valid ordering, the
discovery result remains successful and records an
`INTEGRATION_BACKEND`/representation boundary. It must not pretend that tower
identification failed.

Multiple algebraic candidates require a compositum or primitive-element
normalization rather than indefinitely nesting unrelated finite-extension
wrappers.

### 4. Integration dispatch

Once compilation succeeds:

- rational and P/E/T towers use the elementary dispatcher;
- a direct algebraic `C(x)[y]/(m)` tower constructs its
  `AlgebraicFunctionField` and integral basis automatically;
- a terminal simple radical over a supported lower tower uses relative
  algebraic composition; and
- every incomplete mixed or general-algebraic branch returns a precise
  backend boundary.

The expression caller never supplies a tower or integral basis.

## Acceptance examples

The test matrix must use families, not one privileged example:

| Expression family | Expected discovered order |
| --- | --- |
| `sqrt(x)` | rational, algebraic |
| `(x + 1)**(2/3)` | rational, algebraic |
| `sqrt(x + log(x))` | rational, primitive, algebraic |
| `sqrt(exp(x) + 1)` | rational, hyperexponential, algebraic |
| `exp(sqrt(x))` | rational, algebraic, hyperexponential |
| `tan(log(1 + sqrt(x)))` | rational, algebraic, primitive, hypertangent |
| `exp(log(x)/2)` | relation-classified algebraic element |
| repeated identical calls/powers | one shared candidate |
| independent radicals | algebraic compositum/primitive element |

The high-degree Pell--Abel example remains a useful end-to-end regression, but
it does not determine the architecture or receive a dedicated rewrite.

## Verification

- Unit tests for candidate discovery and topological ordering.
- Native round-trip and derivative reconstruction for every representable
  pure and mixed family above.
- Integration examples for every dispatcher cell that is already complete.
- Explicit backend-boundary tests for correctly discovered but not yet
  integrable cells.
- Focused Ruff and type checks, then the full parallel suite.
