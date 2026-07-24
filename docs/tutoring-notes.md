# Risch algorithm tutoring notes

These notes keep the tutoring sequence coherent across sessions. They track
the syllabus, checkpoint answers, and concepts that need revisiting. They are
a learning record, not part of the library API; see the
[getting-started guide](getting-started.md) for current usage.

## Teaching agreement

- Introduce one conceptual layer at a time.
- Connect every abstraction to a worked integral.
- Ask for the student's reasoning at each checkpoint.
- Correct misconceptions before advancing.
- Relate the mathematics to this repository only after the underlying idea is
  clear.

## Syllabus

- [x] 1. The decision problem
  - elementary integrands versus elementary antiderivatives
  - why a decision procedure is different from pattern matching
  - the informal Liouville form
- [x] 2. Differential fields
  - fields, constants, and derivations
  - representing functions by generators and derivative laws
  - differential towers
- [x] 3. Rational-function integration
  - polynomial parts and partial fractions
  - Hermite reduction
  - square-free remainders and logarithmic terms
  - Rothstein--Trager and Lazard--Rioboo--Trager at a conceptual level
- [ ] 4. Liouville's theorem
  - a precise working statement
  - exact derivatives plus logarithmic derivatives
  - how non-elementarity becomes an algebraic obstruction
- [ ] 5. Transcendental extensions
  - primitive and logarithmic extensions
  - hyperexponential and exponential extensions
  - hypertangent extensions
  - normal and special denominator factors
- [ ] 6. The Risch differential equation
  - where it comes from
  - polynomial, rational, and monomial cases
  - degree bounds and coefficient equations
- [ ] 7. Algebraic extensions
  - minimal polynomials and implicit differentiation
  - traces, residues, and integral bases
  - algebraic Hermite reduction
- [ ] 8. The complete algorithm
  - recursive descent through a tower
  - proof certificates and failure modes
  - assumptions and practical implementation limits
- [ ] 9. Repository walkthrough
  - map each mathematical stage to `src/sympy_risch`
  - trace representative examples through the code
  - identify implemented and future stages

## Current position

Lessons 1 through 3 are complete. We are beginning Lesson 4: Liouville's
theorem.

Covered:

- the decision problem;
- an elementary function can lack an elementary antiderivative;
- the distinction between nonexistence of an elementary form and nonexistence
  of an antiderivative;
- the first differential-field example
  `QQ(x) < QQ(x)(t)`, where `t = exp(x)` and `Dt = t`;
- the informal Liouville decomposition
  `f = D(v0) + sum(c_i D(v_i)/v_i)`.

## Checkpoint record

### Lesson 1

1. `exp(x**2)` is elementary: correct.
2. No elementary antiderivative does not mean no antiderivative exists:
   correct.
3. Replacing `exp(x)` by `t` lets us work with polynomials and rational
   functions: correct. Important refinement: the derivative law `Dt = t`
   preserves the differential meaning of the otherwise algebraic-looking
   generator.

### Lesson 2

With `t = exp(x**2)` and `Dt = 2*x*t`:

1. `t` is not a constant because `Dt != 0`: correct.
2. `D(t**2) = 2*t*Dt = 4*x*t**2`: correct.
3. `D(t/(x + 1)) = -t/(x + 1)**2 + 2*x*t/(x + 1)`:
   correct.

The student correctly applies the generator derivative law inside otherwise
rational-function algebra.

### Lesson 3: Hermite-reduction checkpoint

For
`f = 2*x + 1/(x - 2)**2 + 3/(x - 2)`, the student found

- `g = x**2 - 1/(x - 2)`;
- `h = 3/(x - 2)`;
- the antiderivative `g + 3*log(x - 2)`.

This is correct because `f = D(g) + h`. Over a real interval crossing neither
side of the pole, the branch-neutral presentation is
`g + 3*log(abs(x - 2))`.

### Lesson 3: residue reconstruction discussion

The student questioned why values of the numerator only at denominator roots
justify writing the complete antiderivative. This exposed the missing
interpolation step. They identified Lagrange interpolation as the reason.

Geometric interpretation to retain: polynomials of degree below `n` form an
`n`-dimensional vector space, and evaluation at `n` distinct points gives an
invertible coordinate system. Square-freeness makes the points distinct;
properness provides the degree bound.

### Lesson 3: polynomial-division follow-up

The student correctly identified polynomial division as the step that restores
the degree bound needed before interpolation. They then asked how repeated
roots change the argument. Key follow-up:

- at a repeated root `alpha`, `q'(alpha) = 0`, so the simple-residue formula is
  undefined;
- values alone must be replaced by values plus derivatives (confluent/Hermite
  interpolation);
- algorithmically, rational integration instead uses Hermite reduction to
  extract higher-order poles into an exact rational derivative, leaving a
  square-free denominator for the logarithmic residue step.

### Lesson 3: student's Hermite-reduction model

The student correctly summarized the rational case:

- a proper fraction with square-free denominator is ready for the logarithmic
  residue step;
- repeated factors represent higher-order poles;
- Hermite reduction subtracts derivatives of rational ansatz terms to lower
  those pole orders.

Their next question is how termination is detected. Answer: in lowest terms,
stop when `gcd(q, q') = 1`. Termination follows because every successful
reduction lowers a positive integer pole multiplicity, or equivalently lowers
the repeated-denominator measure.

The student correctly predicted that reducing a denominator
`(x - 1)**3 * (x + 2)**2` leaves at most its square-free kernel
`(x - 1)*(x + 2)`. The residual denominator may be a proper divisor if a
simple-pole coefficient vanishes.

### Lesson 3: choosing reduction denominators

The student asked whether the algorithm must choose `v = x - 1` and then
`v = x + 2`. Clarify two equivalent organizational choices:

- recursive Hermite reduction uses the square-free decomposition
  `q = product(v_i**i)`, grouping all factors with the same multiplicity;
- Horowitz--Ostrogradsky uses the global ansatz determined by
  `d = gcd(q, q')` and `s = q/d`, avoiding sequential selection of individual
  factors.

For `q = (x - 1)**3 * (x + 2)**2`,
`d = (x - 1)**2 * (x + 2)` and `s = (x - 1)*(x + 2)`.

### Lesson 3: why Horowitz--Ostrogradsky works

The student correctly identified the computational motivation: naive
factor-by-factor reduction requires factorization, whereas H--O uses
`d = gcd(q, q')`, `s = q/d`, and linear algebra over the base field.

Proof structure to retain:

1. A pole of multiplicity `m` needs at most multiplicity `m - 1` in the
   rational antiderivative, so all rational terms fit over `d`.
2. The logarithmic obstruction has only simple poles, so it fits over the
   square-free polynomial `s`.
3. With `deg(A) < deg(d)` and `deg(B) < deg(s)`, the map
   `(A, B) -> D(A/d) + B/s` has `deg(q)` unknown coefficients.
4. It is injective: a rational derivative has no simple-pole residues, so it
   cannot cancel a nonzero proper `B/s`; then `D(A/d) = 0` forces `A = 0`.
5. Equal finite dimensions make the map bijective, proving existence and
   uniqueness for every proper numerator.

The implementation clears denominators to solve
`p = A'*s - A*(d'*s/d) + B*d`, where `d'*s/d` is polynomial because `s`
contains one copy of every repeated factor.

The final understanding check was answered correctly: the derivative of a
rational function cannot have a simple pole. Locally, differentiating a pole
of order `r >= 1` produces a pole of order `r + 1`, so every rational
derivative has zero residue. This establishes the uniqueness separation
between `D(A/d)` and `B/s`.

## Pending checkpoint

Lesson 4 will generalize the decomposition into an exact derivative plus
logarithmic derivatives.

## Concepts to revisit

No unresolved Lesson 3 misconception. Reuse the connection between Lagrange
interpolation, simple poles, and logarithmic derivatives when introducing
Liouville's theorem.

### Lesson 4: example-quality correction

The student correctly flagged
`2*x*exp(x**2) + 3/(x - 1)` as a contrived Liouville example constructed
backward from its answer. Replace it pedagogically with two natural examples:

- `integral(x*exp(x), x)`, whose exact part follows from solving
  `r' + r = x` for `r` in `QQ(x)`;
- `integral(exp(x)/(1 + exp(x)), x)`, which is directly the logarithmic
  derivative `D(1 + exp(x))/(1 + exp(x))`.

Use separate examples for the two Liouville building blocks before combining
them abstractly.

### Lesson 4: theorem and proof route

The student wants to understand why Liouville's theorem is true, not merely
use its conclusion. Use Rosenlicht's descent formulation:

- if `f` belongs to a differential field `K` and `DY = f` in an elementary
  extension with no new constants, then
  `f = D(v) + sum(c_i*D(u_i)/u_i)` with `v, u_i` back in `K` and constant
  `c_i`;
- the theorem's strength is descent to `K`, not the tautology `f = DY` in the
  larger field;
- prove it inductively down an elementary tower;
- algebraic adjunctions descend through normalized trace and norm;
- a newly adjoined logarithm can survive only linearly, becoming a
  logarithmic derivative of its argument;
- a newly adjoined exponential contributes no new primitive type for an
  integrand already in the lower field; pole and coefficient comparison force
  its dependence to disappear.

Emphasize the no-new-constants hypothesis and use local pole analysis as the
conceptual continuation of rational Hermite reduction.

The student correctly identified the nontrivial conclusion: a primitive found
in a larger elementary tower can be reorganized into an exact derivative and
constant-weighted logarithmic derivatives with data back in the original
field `K`. Next make the transcendental logarithm descent lemma concrete via
pole orders in the new generator.

### Lesson 4: only-if logic

The student identified a possible circularity: the descent argument appeared
to assume that the primitive already had Liouville form. Resolve it with the
inductive invariant:

- if `Y` is any primitive in the top elementary field `E`, then
  `f = D(Y)` is trivially Liouville form in `E`, with an empty logarithmic sum;
- each one-step descent lemma transforms a Liouville representation in
  `K_i` into one in `K_{i-1}`;
- induction down the finite tower produces the required representation in the
  original field `K`.

The converse direction is immediate by adjoining the logarithms of the
arguments.

The student's current formulation is essentially correct. Preserve two
precision corrections:

- descent constrains the derivative `f`, not necessarily the original syntax
  of the primitive `Y`;
- on the primitive side the normal form is a base-field element plus finitely
  many constant-weighted logarithms, while on the derivative side it is an
  exact derivative plus logarithmic derivatives.

### Lesson 4: contrapositive and non-elementarity

The student identified the algorithmic payoff correctly:

- Liouville form is necessary for an elementary primitive;
- therefore, a proof that no Liouville decomposition exists is a proof that
  the integral is non-elementary.

Keep emphasizing that failure to guess or compute a form with an incomplete
method is not proof. The Risch algorithm supplies exhaustive reductions,
degree and pole bounds, and solvability tests so that a nonzero obstruction is
a certificate rather than a heuristic failure.

### Lesson 4: quantification over elementary extensions

The student asked whether `exp(x**2)` might integrate in a different
elementary tower `E`. Make the quantifiers explicit:

- fix the differential field `K = C(x)(t)` with `t = exp(x**2)`;
- Liouville's theorem says that existence in any elementary extension `E/K`
  with the same constants implies a Liouville decomposition already over
  `K`;
- proving no such decomposition exists therefore rules out every elementary
  choice of `E`, not only one attempted tower.

Distinguish this from changing the permitted class: adjoining a new generator
`y` with `Dy = t` creates a primitive/special-function extension (the `erfi`
direction), but that generator is not elementary over `K`. Likewise, putting
the answer into the base field makes integration tautological but changes the
problem.
