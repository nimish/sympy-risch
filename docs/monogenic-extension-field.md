# Checked monogenic extension fields

`sympy_risch.domains.MonogenicExtensionField` is a reusable exact coefficient
domain for a field

```text
K[y]/(m(y))
```

where `K` is an exact field and the available SymPy backend proves that `m` is
irreducible. It is intentionally narrower than SymPy's `FiniteExtension`,
which also represents quotient rings with reducible moduli.

The field currently provides:

- native power-basis arithmetic and reduction modulo `m`;
- exact division through polynomial inversion modulo `m`;
- provenance-preserving embedding of elements of `K`;
- polynomial and rational-function domains over the extension;
- field `gcd`, `gcdex`, `lcm`, division, and unit operations.

Construction distinguishes two failures. A modulus proved reducible raises
`ValueError`. If irreducibility cannot be decided over the requested parent,
construction raises `NotImplementedError`; this does not assert reducibility.

This is an independent infrastructure checkpoint. Risch towers and algebraic
geometry still use SymPy's `FiniteExtension` until their call sites are
migrated and their required factorization capabilities are checked. In
particular, a working field representation does not by itself provide
univariate factorization over every mixed algebraic-transcendental parent.
The arithmetic and coercion design is recursive, but checked construction over
another `MonogenicExtensionField` remains unsupported until an irreducibility
certificate or equivalent factorization service is available.
