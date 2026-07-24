# Direct real algebraic-hypertangent integration

This is a research note, not a description of an implemented API. It records a
possible future Risch branch for algebraic extensions above a real
hypertangent generator. The goal is to preserve real structure instead of
rewriting tangent generators as complex exponentials.

For implemented behavior and the main roadmap, see
[architecture and implementation roadmap](design.md).

## Problem

Let `K` be a real differential field and let

```text
Dt = a * (1 + t**2),  a in K,
E = K(t)[y] / (F(t, y)),
```

where `t` is transcendental over `K` and `y` is algebraic over `K(t)`.
The desired algorithm decides elementary integrability in `E` while keeping
the computation and result over the real field whenever possible.

This is distinct from:

- a hypertangent extension over an algebraic lower field, where the tower is
  `K < K(y) < K(y)(t)`; and
- an algebraic element which happens to satisfy a tangent-shaped derivative
  law.

Here the algebraic level is above the hypertangent level:

```text
K < K(t) < E.
```

## Project literature assessment

The project review has not identified a complete published algorithm under the
name “direct real algebraic-hypertangent integration.” The closest sources
provide the necessary pieces but do not assemble this tower order:

1. Bronstein's ISSAC 1998 tutorial treats the direct real transcendental
   tangent case in Section 3.7. It then treats algebraic extensions above
   logarithmic and exponential generators in Sections 3.8 and 3.9, but has no
   corresponding algebraic tangent section.
2. Bronstein's 1990 paper *Integration of Elementary Functions* gives the
   mixed algebraic-transcendental algorithm for algebraic extensions of a
   logarithmic or exponential extension.
3. Bronstein's *A Unification of Liouvillian Extensions* includes
   trigonometric extensions in a common framework, but states the rational
   integration treatment for the transcendental case.
4. Davenport's 1984 paper treats transcendental towers over an algebraic
   function field. That is the opposite tower order from the problem here.
5. Bronstein's 1991 algorithm for the Risch differential equation on an
   algebraic curve supplies an important subroutine, but not the complete
   algebraic-over-tangent integration algorithm.
6. Bronstein's work on real elementary-function simplification and the real
   structure theorem supplies guidance for real representations, but is not a
   full integration algorithm for this case.

Accordingly, any implementation should be presented as a proposed real-form
synthesis of established algorithms, not attributed wholesale to one
reference.

## Relation to the algebraic-exponential case

After adjoining `i`, use the Cayley transformation

```text
u = (1 + i*t) / (1 - i*t).
```

Then

```text
Du/u = 2*i*Dt/(1 + t**2) = 2*i*a,
```

so `u` is hyperexponential over `K(i)`, and

```text
K(i)(t) = K(i)(u).
```

Thus `E(i)` is an algebraic extension of a hyperexponential field and can in
principle be handled by the algebraic-exponential algorithm. The direct real
algorithm should be derived by restricting that computation from `K(i)` back
to `K`.

Under the Cayley transformation,

```text
t =  i  <->  u = 0,
t = -i  <->  u = infinity.
```

The two conjugate exponential special places therefore become the single
irreducible quadratic special place

```text
q = t**2 + 1
```

over the real field.

## Real representation

Computations at the special place can use the quadratic residue algebra

```text
K[t] / (t**2 + 1)
```

without introducing complex SymPy expressions. Represent a residue `r + s*t`
as the pair `(r, s)`. Multiplication by `t` is the real linear map

```text
(r, s) -> (-s, r).
```

This is the restriction-of-scalars explanation for the 2-by-2 coupled
differential system in the ordinary transcendental tangent algorithm. For an
algebraic extension of degree `n`, the corresponding direct real system will
generally have dimension `2*n`.

## Expected algorithmic components

1. Perform algebraic Hermite reduction at ordinary finite places.
2. Apply the algebraic residue criterion and extract constant-residue
   logarithmic terms.
3. Compute local integral bases, valuations, and ramification at all places of
   `E` above `t**2 + 1`.
4. Compute local or Puiseux expansions at those special places and at all
   places above infinity.
5. Derive and solve coupled Risch differential systems over `K`, generalizing
   the transcendental tangent system.
6. Construct logarithmic divisors invariant under real conjugation and test
   their torsion/principality conditions.
7. Reconstruct real logarithmic and angle terms. Conjugate complex logarithms
   should descend to combinations of norms and an internal `atan2`-like angle
   representation.
8. Return explicit obstruction certificates when residue, differential
   equation, or divisor conditions fail.

The likely hardest step is conjugation-equivariant divisor reconstruction, not
the coupled linear algebra.

## Implementation strategy

Develop the branch through reviewable checkpoints:

1. Generalize the existing direct transcendental hypertangent reducer from
   `QQ(x)` to a supported arbitrary lower differential field.
2. Generalize algebraic Hermite reduction to monomial parent fields.
3. Implement local arithmetic at the quadratic special polynomial
   `t**2 + 1` for a simple radical extension.
4. Derive the degree-`2*n` coupled system and verify it against the result of
   complexification on generated examples.
5. Add conjugation-invariant logarithmic divisor reconstruction.
6. Generalize from simple radicals to arbitrary finite separable algebraic
   extensions.

Complexification remains a valuable independent test oracle even though it is
not the desired production representation.

## Open questions

- Which hypotheses on `K` are needed for effective real descent and
  constant-field decisions?
- Should the quadratic residue algebra be a small internal type or represented
  directly using an existing SymPy algebraic field?
- How should places above `t**2 + 1` be represented without explicitly
  splitting the quadratic?
- Can lazy integral-basis refinement be localized at the quadratic special
  place?
- What is the minimal real divisor/principality interface needed before a
  general Jacobian implementation?
- How should angle terms and their branch data be represented separately from
  differential-field identities?

## Primary references

- Manuel Bronstein, [*Symbolic Integration Tutorial*](https://www-sop.inria.fr/cafe/Manuel.Bronstein/publications/issac98.pdf),
  ISSAC 1998, especially Sections 3.7-3.9.
- Manuel Bronstein, [*Integration of Elementary Functions*](https://doi.org/10.1016/S0747-7171(08)80027-2),
  Journal of Symbolic Computation 9 (1990), 117-173.
- Manuel Bronstein, [*A Unification of Liouvillian Extensions*](https://research.ibm.com/publications/a-unification-of-liouvillian-extensions),
  Applicable Algebra in Engineering, Communication and Computing 1 (1990),
  5-24.
- J. H. Davenport,
  [*Integration algorithmique des fonctions elementairement transcendantes sur
  une courbe algebrique*](https://doi.org/10.5802/aif.973),
  Annales de l'Institut Fourier 34 (1984), 271-276.
- Manuel Bronstein,
  [*The Risch Differential Equation on an Algebraic Curve*](https://doi.org/10.1145/120694.120731),
  ISSAC 1991, 241-246.
- Manuel Bronstein,
  [*Simplification of Real Elementary Functions*](https://doi.org/10.1145/74540.74566),
  ISSAC 1989, 207-211.
