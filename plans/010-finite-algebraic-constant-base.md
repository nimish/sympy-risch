# Plan 010: Admit finite algebraic constants at the rational P/E boundary

> **Executor instructions**: Keep this a constant-field boundary checkpoint.
> Do not treat a caller-set `ConstantFieldStatus.PRESERVED` as proof for a
> recursive finite-constant tower.  The current result covers the standard
> base and direct checked extension proposals only.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: HIGH, because allowing algebraic-valued valuation coefficients
  would produce false logarithmic relations
- **Depends on**: Plan 003 fundamental signatures; native SymPy
  `AlgebraicField` arithmetic
- **Category**: effective constant fields
- **Completed**: 2026-07-26
- **Status**: DONE

## Delivered contract

For a standard rational base

```text
C(x),  Dx = 1,
```

the rational exact-derivative and radical-logarithmic decisions, fundamental
base signature, and logarithmic-relation compiler now admit:

```text
C = QQ, QQ_I, or a native finite AlgebraicField over QQ.
```

`constant_fields.rational_constant_coordinates` expands a native algebraic
constant in the fixed power basis

```text
(1, theta, ..., theta**(d - 1)),  theta = C.ext.
```

It uses SymPy's exact `ANP.to_list()` representation, left-pads its descending
coefficient list to `d = degree(C.mod)`, then reverses it.  Consequently a
rational relation is compiled as

```text
sum(q_i*c_i) = 0  <=>  every QQ power-basis coordinate is zero,
```

not as an invalid `C`-linear valuation relation.  The existing `QQ_I`
real/imaginary-coordinate case is the degree-two special case retained by
the same adapter.

Direct checked factories can now decide the first primitive or
hyperexponential proposal over such a base.  In particular, the factory can
certify `Dt = 1/x` as preserving constants and can detect
`Dtheta = theta/x` as dependent.

## Deliberate boundary

This checkpoint does **not** make finite-algebraic-constant P/E towers a
proof-safe recursive signature.  Existing non-base
`ConstantFieldStatus.PRESERVED` metadata has no provenance field, so a
general-C non-base signature returns
`CONSTANT_FIELD_PROVENANCE` rather than trusting a caller assertion.  A later
admission refactor must carry or replay no-new-constants evidence before that
recursive path is opened.

Also excluded:

- hypertangent/Gaussian conversion over general `C`;
- scalar RDE, parametric LogD, and recursive primitive/hyperexponential
  reductions over general `C`;
- algebraic tower levels, automatic Darboux discovery, and expression
  construction; and
- arbitrary SymPy domains or function fields as constants.

## Verification

- Exact degree-two and degree-four native power-basis coordinate round trips.
- A relation family in which algebraic residues cancel only in the rational
  source direction `(1, 1)`, including generated offsets.
- Direct exact, primitive, and hyperexponential factory outcomes over
  `QQ(sqrt(2))(x)`.
- Explicit rejection of a nonconstant coefficient field, finite-constant
  recursive signature without proof-carrying evidence, and hypertangent
  directions over general `C`.
- Regression checks for existing `QQ`/`QQ_I` relations, dependencies,
  extension factories, and admission results.
