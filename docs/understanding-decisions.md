# Understanding solver decisions

`sympy-risch` is a decision procedure, not a black-box antiderivative
printer.  Its most useful output is often *why* an integral has an answer—or
why the current field cannot contain one.  This guide explains how to read
that answer without first learning the full Risch algorithm.

Start with the [getting-started guide](getting-started.md) to construct an
explicit differential tower.  Return here when you want to interpret a
result.

## The four outcomes

Keep these outcomes separate:

| Outcome | What it means | What to do next |
| --- | --- | --- |
| Elementary answer | The library constructed an exact tower identity for an antiderivative. | Check `result.reconstructs`; then apply any presentation substitutions you chose. |
| Proved obstruction | For this supported tower, a Risch/Liouville criterion proves that no elementary antiderivative exists. | Inspect `result.obstruction` and its certificate. This is a mathematical negative result, not a failed simplification. |
| Unsupported input | The requested tower or algorithm is outside the implemented domain. | The call raises `NotImplementedError`. It says nothing about whether an elementary antiderivative exists. |
| Inconclusive lower-level query | A low-level exploratory algorithm could not yet make a complete decision. | Do not turn it into a negative conclusion; inspect its explicit outcome and boundary. |

The top-level `integrate_elementary` API has the first three outcomes.  Some
lower-level tools—especially parametric logarithmic-derivative solvers—also
expose an explicit `UNSUPPORTED` or `INCONCLUSIVE` outcome so callers can
preserve the fourth distinction.

`integrate_expression` adds the same distinction before dispatching to the
native tower API. Its `UNSUPPORTED` result means either that the narrow
one-variable P/E/T grammar declined the source expression or that it could
not certify the proposed tower. In particular, a dependent `log`, `exp`, or
`tan` call is not rewritten through a branch-sensitive identity. Its
`NON_ELEMENTARY` result is only produced after successful compilation and is
backed by the contained `ElementaryIntegralResult` obstruction.

The explicit Riccati API has the same discipline. A
`RiccatiMobiusNormalForm` can be built only from two supplied checked linear
special-factor certificates; the library does not interpret an unsuccessful
factor search as their absence. `integrate_riccati_mobius_normal_form` then
checks the transformed hyperexponential target before integrating it. If that
target is outside the P/E/T contract, its `NotImplementedError` is an
unsupported-input boundary, not a conclusion about the original quadratic
integrand.

## Reading a top-level integration result

`integrate_elementary` returns an immutable `ElementaryIntegralResult`.
The usual pattern is:

```python
result = integrate_elementary(integrand, derivation)

assert result.reconstructs
if result.is_elementary:
    answer = result.antiderivative
else:
    proof = result.obstruction
```

`reconstructs` is the primary safety check: it differentiates the stored
exact part and logarithmic terms with the *tower derivation*, then verifies
that they reproduce the input.  Ordinary `sympy.diff` does not know that a
formal generator such as `t` represents `log(x)` or `exp(x)`.

On success, an antiderivative is assembled from two kinds of terms:

- `total_exact_part`: elements already in the current differential field;
- `all_logarithmic_terms`: Liouville logarithms whose derivatives are known
  exactly.

On a proved obstruction, `unintegrated_remainder` is the part that cannot be
removed.  `obstruction` records both the tower level and the mathematical
reason, for example a nonconstant residue.  The field assumptions used by
that proof are retained in the associated reduction certificate.

## Why the tower matters

The same printed expression can mean different things in different fields.
For example, the symbol `t` is not automatically `log(x)`; it becomes that
representation only after placing it in a tower with `Dt = 1/x`.

Two recorded hypotheses are especially important:

- **Preserved constants** means adjoining a generator did not introduce a
  new constant.  Risch structure theorems rely on this fact.
- **No special polynomials** is the extra hypothesis required by the general
  nonlinear-monomial branch.

The checked extension factories certify these conditions where the current
algorithms can decide them.  Otherwise the status is deliberately
`UNKNOWN`, and an algorithm that needs the missing theorem refuses to issue a
certificate.

## A small vocabulary

| Term | Plain-language meaning |
| --- | --- |
| Differential tower | The ordered list of fields and rules defining each generator's derivative. |
| Exact part | Something already of the form `D(g)`. |
| Logarithmic part | A term of the form `c*D(v)/v`, whose integral is `c*log(v)`. |
| Residue | The local simple-pole coefficient that determines whether a logarithm can remove a pole. |
| RDE | A Risch differential equation, the structured equation used to decide the remaining coefficient problem. |
| Certificate | Immutable data that records the reduction, assumptions, and witness or obstruction. |

For algebraic functions, the current library has low-level supporting
infrastructure, but it does not yet make a top-level algebraic integration
decision. That is an unsupported top-level input, not a claim of
non-elementarity.
The current SymPy domain backend also cannot represent a monomial level above
such a function-field algebraic level, or a second algebraic function-field
level. Those compositions are rejected immediately for the same reason.

## A good debugging order

When an example surprises you, ask these questions in order:

1. Is the intended expression represented in the correct tower?
2. Does the result reconstruct under `derivation`, rather than under ordinary
   symbolic differentiation?
3. Is the result an answer, a proved obstruction, or an unsupported input?
4. If it is a proof, which certificate and tower hypothesis does it use?
5. If it is unsupported, is the missing piece an algebraic level, a
   generalized monomial family, automatic tower construction, or real branch
   presentation?

The [architecture and roadmap](design.md) answers the last question in
algorithmic detail.  The [testing contract](testing.md) explains how each
claim is verified before it becomes part of the supported surface.
