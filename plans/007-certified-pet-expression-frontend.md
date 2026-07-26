# Plan 007: Add a certified narrow P/E/T expression frontend

> **Executor instructions**: Expose automatic construction only for the
> grammar whose extension and constant-field decisions are already certified.
> A compiler boundary must never be reported as non-elementarity, and
> back-substitution must not assert unproved global branch identities.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: HIGH
- **Depends on**: `plans/002-executable-admission-matrix.md`,
  `plans/003-certified-fundamental-signatures.md`
- **Category**: user-facing correctness / transcendental integration
- **Completed**: 2026-07-25
- **Status**: DONE

## Delivered contract

`compile_expression` accepts a single integration symbol, `QQ` constants,
addition, multiplication, integer powers, and nested unary `log`, `exp`, and
`tan` calls. It creates the standard certified `QQ(x)` base and uses the
existing primitive, hyperexponential, and hypertangent factories for every
source call. Only `PRESERVED` extensions enter the immutable tower; a
dependent, uncertified, singular, branch-sensitive, or otherwise unsupported
construct is returned as a stable `UNSUPPORTED` result with no partial tower.

`integrate_expression` maps a successfully compiled expression through the
existing elementary dispatcher. It distinguishes an elementary result, a
native Liouville obstruction, and a compiler/admission/backend boundary. Its
back-substitution is intentionally a local differential presentation: it
retains the source function calls and never applies global logarithmic or
trigonometric branch identities.

The public immutable certificates independently replay the standard base and
every source extension, check native/source chain rules, source round trips,
native integration reconstruction, and the returned elementary presentation.
This keeps result metadata from being a caller assertion.

## Deliberate boundary

The grammar excludes algebraic powers and extensions, non-rational effective
constants, extra independent symbols, arbitrary SymPy functions, global
principal-branch simplification, and general generator minimization. Related
calls such as `log(exp(x))`, `exp(x)` with `exp(2*x)`, and dependent tangent
calls are declined rather than rewritten. Extending the frontend must wait for
the relevant dependency, algebraic, mixed-tower, and presentation cells to
become total.

## Verification

- Rational, primitive, hyperexponential, hypertangent, and nested P/E/T
  compilation and integration examples.
- Certified non-elementarity after successful compilation.
- Generated rational source trees and nonconstant one-level P/E/T calls with
  native/source round trips and chain-rule checks.
- Grammar, constant, singularity, and branch boundaries; repeated calls;
  logarithmic, exponential, and tangent dependence adversaries.
- Forged source mappings, derivative laws, and public answer presentations.
