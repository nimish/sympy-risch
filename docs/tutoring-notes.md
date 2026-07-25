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
- [x] 4. Liouville's theorem
  - a precise working statement
  - exact derivatives plus logarithmic derivatives
  - how non-elementarity becomes an algebraic obstruction
- [x] 5. Transcendental extensions
  - primitive and logarithmic extensions
  - hyperexponential and exponential extensions
  - hypertangent extensions
  - normal and special denominator factors
- [x] 6. The Risch differential equation
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

Lessons 1 through 6 are complete. The next topic is Lesson 7: algebraic
extensions, beginning with how a derivation extends through a minimal
polynomial and then moving to algebraic Hermite reduction.

Covered:

- the decision problem, differential fields, and Liouville's theorem;
- rational Hermite and logarithmic-residue reduction;
- primitive, exponential, and hypertangent top generators;
- normal versus special denominator factors;
- rational Risch differential equations, including finite-pole and infinity
  bounds;
- triangular primitive equations, diagonal exponential equations, and
  Cayley-paired hypertangent equations.

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

Lesson 5 will classify the top transcendental generator and reduce
Liouville's condition to finite equations in the lower field.

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

### Lesson 4: exponential one-step descent

For a transcendental exponential adjunction `K(t)` with
`Dt/t = Da`, and a Liouville representation of an element already in `K`:

- pole analysis leaves `v` in `K[t]`;
- each logarithmic argument has the form `u_i = a_i*t**m_i`, since `t = 0`
  is the one exceptional divisor whose logarithmic derivative is already in
  `K`;
- `D(u_i)/u_i = D(a_i)/a_i + m_i*Da`, so the power of `t` becomes an exact
  derivative and can be absorbed downstairs;
- writing `v = sum(b_j*t**j)`, coefficient comparison gives
  `D(b_j*t**j) = 0` for every `j > 0`;
- the no-new-constants condition and transcendence of `t` force every such
  `b_j` to vanish, leaving `v in K`.

Contrast with logarithmic descent: a new logarithm can survive linearly as a
logarithmic derivative of its argument; a new exponential above the integrand
field contributes no new integration atom.

For the checkpoint `t = exp(x)` and `u = (x**2 + 1)*t**3`, the student
correctly computed
`D(u)/u = 2*x/(x**2 + 1) + 3`. The first term is
`D(x**2 + 1)/(x**2 + 1)` and the second is the exact derivative `D(3*x)`.

### Transition to the algorithm: what counts as an obstruction

The student asked whether any failed requirement, such as one attempted
coefficient satisfying
`D(b_j) + j*D(a)*b_j != 0`, proves non-elementarity. Correct distinction:

- failure of one guessed coefficient only rejects that candidate;
- the homogeneous descent equation always permits `b_j = 0` and ordinarily
  removes unnecessary top-generator dependence;
- non-elementarity follows only when the algorithm proves that a required
  inhomogeneous equation has no solution in the allowed field, after all
  exact and logarithmic alternatives have been exhausted;
- failure of a theorem precondition, such as a valid tower or unchanged
  constant field, makes the test inapplicable rather than proving
  non-elementarity.

Use the contrast `t = exp(x**2)`: integrating `t` requires
`D(r) + 2*x*r = 1`, which has no rational solution, while integrating `x*t`
requires `D(r) + 2*x*r = x`, solved by `r = 1/2`.

### Lesson 5: algorithm walkthrough route

The student is ready to move from Liouville's existence theorem to the actual
recursive algorithm. Teach the control flow before individual family formulas:

1. build and validate a differential tower for the integrand;
2. at the top level, view the integrand as a rational function in the top
   generator with lower-field coefficients;
3. Hermite-reduce repeated normal denominator factors using the total tower
   derivation;
4. test the square-free normal remainder for constant logarithmic residues;
5. reduce polynomial and special parts using family-specific degree bounds and
   Risch differential equations over the lower field;
6. recurse on those lower-field equations;
7. assemble and back-substitute, or return a proved obstruction.

Use three separate natural examples rather than one reverse-engineered sum:
`exp(x)/(1 + exp(x))` for the logarithmic branch, `x*exp(x)` for the exact/RDE
branch, and `exp(x**2)` for a proved obstruction.

For the first routing checkpoint, with `t = exp(x)` and
`f = t/(1 + t)**2`, the student correctly chose Hermite reduction because
`1 + t` is a repeated normal factor. Indeed,
`D(-1/(1 + t)) = t/(1 + t)**2`, so the reduced remainder is zero.

Next explain why special factors cannot be multiplicity-lowered by ordinary
Hermite reduction. For `Dt = D(a)*t`, the factor `t` is special and
`D(b*t**j) = (D(b) + j*D(a)*b)*t**j`; its exponent is preserved, turning each
special-factor coefficient into a lower-field Risch differential equation.

For `t = exp(x**2)` and `f = x/t`, the student initially proposed
`D(b) - 2*x*t = 0`. Reinforce the coefficient rule: differentiating
`b*t**(-1)` gives `(D(b) - 2*x*b)*t**(-1)`, and matching `x*t**(-1)` requires
`D(b) - 2*x*b = x`. The constant solution is `b = -1/2`.

### Lesson 5: logarithmic residues at normal factors

After normal Hermite reduction, teach the local test for a square-free
irreducible normal factor `p(t)`. For a partial fraction `A/p`, a logarithmic
term exists at that factor only if a constant `c` satisfies
`A = c*D(p) mod p`. Equivalently, the generalized residue
`A/D(p)` in the residue field `F[t]/(p)` must be constant. Exact derivatives
cannot cancel a simple normal pole.

Use `t = exp(x)` and `1/(1 + t)`:
`p = 1 + t`, `D(p) = t = -1 mod p`, so the residue is `c = -1`; subtracting
`-D(p)/p` leaves `1 = D(x)`. This yields
`x - log(1 + exp(x))`.

For `x*t/(1 + t)`, the student correctly computed
`D(p) = -1 mod p`, `A = -x mod p`, and hence
`rho = A/D(p) = x mod p`. Clarify notation: `rho` is the uniquely computed
candidate residue in `F[t]/(p)`; `c` is an allowed element of the constant
field. The logarithmic test passes exactly when `rho` is the image of such a
`c`. Here `D(x) = 1`, so `x` is not a constant and the simple normal pole is
an elementary-integrability obstruction.

The student then asked why equality modulo `p` can be lifted and whether
constants mean elements of `K` or `C`. Precision:

- the current field is `K = F(t)` and its constant field is
  `C = Const(K) = Const(F)` under the no-new-constants hypothesis;
- `rho` lives in the residue field `kappa(p) = F[t]/(p)`, not initially in
  `K`;
- constants reduce through an injective map `C -> kappa(p)` because
  `(p) intersect F = 0` for nonconstant `p`;
- the test is membership of `rho` in the image of `C`, not membership in all
  of `K`;
- with the unique representative of degree below `deg(p)`, equality
  `rho = c mod p` is equivalent to literal equality of that remainder with
  the constant polynomial `c`.

Mention that practical algorithms may enlarge or explicitly manage algebraic
constant fields; the allowed residue coefficients are still constants, never
arbitrary lower-field functions.

### Lesson 6A: rational Risch differential equations

Begin the solver for
`D(b) + a*b = r` over `C(x)`. Its finite-search structure is:

1. local pole analysis bounds the possible denominator of `b`;
2. multiply out that denominator to obtain a polynomial unknown;
3. valuation at infinity bounds its degree;
4. coefficient comparison gives a finite linear system.

Use the hyperexponential coefficient equation
`D(b) + j*D(alpha)*b = f_j`.

- For `t = exp(x**2)` and `f = t`,
  `b' + 2*x*b = 1`: no finite poles are possible, and polynomial degree at
  infinity rules out every nonzero `b`.
- For `f = x*t`, `b' + 2*x*b = x`: the same analysis bounds `b` to degree
  zero and yields `b = 1/2`.
- For `t = exp(x)` and `f = t/x`, `b' + b = 1/x`: a pole of `b` at zero
  would make `b'` have order at least two, while no pole gives no way to match
  the simple pole on the right. This is the `Ei(x)` obstruction.

For the pole-bound checkpoint
`b' + b = (x - 1)/x**2`, the student correctly tried `b = c/x` and found
`c = 1`. Clarify that although `b = 1/x = D(log(x))`, `b` is only the
lower-field coefficient. The actual exact part is
`g = b*t = exp(x)/x`, whose derivative is the integrand.

The student then correctly challenged why the exact-part search was restricted
to `g = b*t`. Supply the missing normal-form proof:

- after normal-pole Hermite reduction, a rational exact part with only the
  exponential special factor `t` is a Laurent polynomial
  `g = sum_k b_k*t**k`;
- `D(t)/t = eta in F` makes the derivation diagonal in powers of `t`:
  `D(b_k*t**k) = (D(b_k) + k*eta*b_k)*t**k`;
- transcendence makes distinct powers of `t` linearly independent over `F`;
- if the integrand contains only `f_j*t**j`, every `k != j` coefficient
  satisfies the homogeneous equation, making `b_k*t**k` a constant;
- unchanged constants and transcendence force those `b_k` to vanish for
  `k != 0`, while `k = 0` contributes only an irrelevant additive constant.

Logarithmic derivatives cannot supply a nonzero Laurent power: at infinity
they have `t`-degree zero, and finite normal factors produce only simple normal
poles.

### Lesson 6B: local denominator bounds

For `D(y) + A*y = B` over `C(x)`, teach pole bounds using a local parameter
`z = x - alpha`, with `y ~ c*z**(-m)`, `A` having pole order `s`, and `B`
having pole order `r`:

- if `A` is regular, `D(y)` dominates and any pole must have
  `m = r - 1`;
- if `A` has pole order `s >= 2`, `A*y` dominates and any pole must have
  `m = r - s`;
- if `A` has a simple pole with residue `lambda`, the leading coefficient is
  `(lambda - m)*c`; normally `m = r - 1`, but a positive integer
  `m = lambda` is an exceptional cancellation candidate.

These candidates give a denominator bound, not a solution. Substitution and
coefficient comparison still decide existence. Use
`y' + 2*y/x = 1/x**2`: the ordinary candidate is `m = 1`, the exceptional
homogeneous candidate is `m = 2`, and the general rational solution is
`y = 1/x + C/x**2`.

### Lesson 6C: degree bounds at infinity

After fixing a universal denominator `Q`, write `y = P/Q`. If
`deg(P) = N` and `deg(Q) = q`, then `y ~ c*x**n` with `n = N - q`.
Compare this with `A ~ a*x**s` and `B ~ b*x**r`:

- if `s >= 0`, `A*y` dominates and normally `n = r - s`;
- if `s <= -2`, `D(y)` dominates and normally `n = r + 1`;
- if `s = -1` and `A ~ lambda/x`, then
  `D(y) + A*y ~ (n + lambda)c*x**(n - 1)`;
  normally `n = r + 1`, with exceptional cancellation at
  `n = -lambda`.

Translate candidate growth exponents back to numerator degrees
`N = n + q`, take the finite bound, substitute a polynomial ansatz, and solve
the resulting linear system.

For `y' + 2*y/x = 1/x**2`, the finite-pole bound gives `Q = x**2`.
At infinity, the ordinary exponent is `n = -1`, hence `N = 1`; the exceptional
exponent is `n = -2`, hence `N = 0`. Thus `P = c1*x + c0`, yielding
`c1 = 1` and free `c0`.

For the checkpoint `y' + y/x = 1`, finite-pole and infinity analysis give
`Q = x` and `deg(P) <= 2`. The student correctly predicted that the constant
coefficient `a0` remains free. Substitution gives
`a2 = 1/2`, `a1 = 0`, and arbitrary `a0`, so
`y = x/2 + a0/x`. The free term is the exceptional homogeneous solution
because `D(1/x) + (1/x)*(1/x) = 0`.

### Lesson 6D: primitive and logarithmic top generators

For `K = F(t)` with `D(t) = eta in F`, differentiation is triangular:
`D(b_k*t**k) = D(b_k)*t**k + k*b_k*eta*t**(k - 1)`.
If `f = sum_{k=0}^n f_k*t**k`, allow
`v = sum_{k=0}^{n+1} b_k*t**k`; the coefficient equations are

`D(b_k) + (k + 1)*eta*b_(k + 1) = f_k`,

with `D(b_(n+1)) = 0`, so the possible top extra coefficient is constant.
Solve downward, retaining parameter freedom when it affects lower equations.

Natural examples:

- `t = log(x)`, `D(t) = 1/x`, and `f = t/x`:
  choose `b_2 = 1/2`, giving `D(t**2/2) = t/x`;
- `f = t`: the top equation forces `b_1 = x` and excludes a `t**2` term;
  the next equation gives `b_0 = -x`, hence
  `integral(log(x), x) = x*log(x) - x`;
- `f = 1/t`: the square-free normal factor `p = t` has
  `D(p) = 1/x`, so its generalized residue is `x`, not a constant; this
  proves the logarithmic integral `li(x)` is non-elementary.

For the checkpoint `t = log(x)` and `f = t**2/x`, the student correctly
identified the degree bound `deg(v) <= 3` and the cubic shape. The normalized
term is `v = t**3/3`, since `D(t**3) = 3*t**2/x`.

### Lesson 6E: hypertangent top generators

For `K = F(t)` with

`D(t) = eta*(1 + t**2)`,

the quadratic `q = 1 + t**2` is special:

`D(q) = 2*eta*t*q`.

Thus `q | D(q)`, so powers of `q` cannot be handled by ordinary normal-factor
Hermite reduction. The polynomial part is also coupled rather than triangular:

`D(b_j*t**j) = D(b_j)*t**j + j*eta*b_j*(t**(j-1) + t**(j+1))`.

Consequently, if `v = sum_j b_j*t**j`, its coefficient at `t**k` is

`D(b_k) + (k + 1)*eta*b_(k + 1) + (k - 1)*eta*b_(k - 1)`.

The special part is converted to exponential form through the Cayley
coordinate

`u = (1 + i*t)/(1 - i*t)`.

It satisfies `D(u)/u = 2*i*eta`, while

`1 + t**2 = 4*u/(1 + u)**2`.

After the common nonlinear reductions, a rational function supported only at
the special place becomes a Laurent polynomial in `u`. Real inputs have
conjugate paired modes. A pair

`(a + i*b)*u**n + (a - i*b)*u**(-n)`

is an exact derivative precisely when there are rational lower-field
coefficients `P,Q` satisfying

`D(P) - 2*n*eta*Q = a`,

`D(Q) + 2*n*eta*P = b`.

These are the real coupled Risch differential equations. Pairing the modes
keeps the reconstructed answer real.

For `eta = 1`,

`1/(1 + t**2) = (u + 2 + u**(-1))/4`.

The nonzero paired mode has `a = 1/4`, `b = 0`; the coupled system has the
constant solution `P = 0`, `Q = -1/8`, reconstructing

`t/(2*(1 + t**2))`.

The zero mode `1/2` descends to the lower field and integrates to `x/2`.
Therefore

`integral(1/(1 + t**2), x) = x/2 + t/(2*(1 + t**2))`

when `D(t) = 1 + t**2`.

Conceptual distinction:

- every exponential monomial `D(t) = eta*t` has the special polynomial `t`,
  since `t | D(t)`;
- the existence of a special polynomial does not by itself make the chosen
  generator exponential over the lower field;
- in the hypertangent case, `q = 1 + t**2` is special but
  `D(q)/q = 2*eta*t` is not in the lower field;
- after adjoining `i` and making the invertible Cayley change
  `u = (1 + i*t)/(1 - i*t)`, however,
  `D(u)/u = 2*i*eta` is in the lower field extended by `i`.

Thus a hypertangent extension is birationally equivalent to an exponential
extension after a controlled algebraic extension of the constants. Over a
constant field already containing `i`, no additional constant extension is
needed.

The Cayley transformation is the same identity underlying the tangent
half-angle substitution. If `t = tan(a)`, then

`(1 + i*t)/(1 - i*t) = exp(2*i*a)`.

More conventionally, if `z = tan(a/2)`, then

`exp(i*a) = (1 + i*z)/(1 - i*z)`,

whose inverse gives the Weierstrass formulas

`sin(a) = 2*z/(1 + z**2)` and
`cos(a) = (1 - z**2)/(1 + z**2)`.

Thus the half-angle substitution converts trigonometric functions to rational
functions, while the Cayley step used here converts the rational tangent
generator into an exponential generator. They are inverse views of the same
birational parametrization of the circle.

### Lesson 7A: extending the derivation algebraically

If an algebraic generator `alpha` has minimal polynomial

`m(alpha) = 0`,

then differentiating the relation forces

`D(alpha) = -D_F(m)(alpha)/(partial_m/partial_y)(alpha)`.

For the checkpoint `alpha**3 = x`, the student correctly computed

`3*alpha**2*D(alpha) - 1 = 0`,

so

`D(alpha) = 1/(3*alpha**2) = alpha/(3*x)`.

Unlike a transcendental generator, the powers of an algebraic generator are
not independent. If `deg(m) = n`, every member of `F(alpha)` has a unique
representation

`a_0 + a_1*alpha + ... + a_(n-1)*alpha**(n-1)`.

Irreducibility of `m` makes `F[y]/(m)` a field, so inverses are computed by
the extended Euclidean algorithm modulo `m`.

For `alpha**3 = x`, the student correctly reduced

`1/(1 + alpha) = (1 - alpha + alpha**2)/(1 + x)`,

using

`(1 + alpha)*(1 - alpha + alpha**2) = 1 + alpha**3 = 1 + x`.

In the basis `1, alpha, alpha**2`, differentiation gives

`D(A + B*alpha + C*alpha**2)`

`= D(A) + (D(B) + B/(3*x))*alpha`

`  + (D(C) + 2*C/(3*x))*alpha**2`.

Thus exact algebraic integration becomes a finite system of lower-field RDEs.
For the checkpoint `f = alpha**2`, the student correctly used `C = k*x`:

`k + 2*k/3 = 1`,

so `k = 3/5` and

`integral(alpha**2, x) = 3*x*alpha**2/5`.

### Lesson 7B: local parameters and algebraic poles

Pole reduction applies to the differential `f*dx`, not to `f` in isolation.
At a ramified place, `x` need not be a local parameter. For
`alpha**2 = x`, the local parameter at the place over `x = 0` is `alpha`,
and

`dx = 2*alpha*dalpha`.

Thus

`dx/alpha**2 = 2*dalpha/alpha`

is logarithmic, while

`dx/alpha**3 = 2*dalpha/alpha**2 = d(-2/alpha)`

is exact. For the checkpoint `f = 1/alpha**4`, the student correctly found

`f*dx = 2*dalpha/alpha**3 = d(-1/alpha**2)`,

so the algebraic antiderivative is `-1/alpha**2 = -1/x`.

### Lesson 7C: integral bases

An element is integral over `C[x]` if it satisfies a monic polynomial with
coefficients in `C[x]`. The integral closure of `C[x]` in a finite algebraic
function field is a free `C[x]`-module of the extension degree; a module basis
is an integral basis. It supplies coordinates that do not create spurious
finite poles.

For `alpha**2 = x**3`, the power-basis expression

`beta = alpha/x`

appears to have a pole at `x = 0`. The student correctly computed

`beta**2 = x`,

so `beta` is integral. In fact `x = beta**2` and `alpha = beta**3`; hence
`beta` is a local parameter that vanishes at the point over `x = 0`.
The apparent coefficient denominator in the basis `1, alpha` is spurious.
An integral basis for this field over `C[x]` is `1, beta`.

For the first algebraic Hermite ansatz

`f = beta/x**2`, `g = c*beta/x`,

the student chose the correct denominator and basis shape but initially chose
`c = 2`. Combining both derivative terms gives

`D(c*beta/x) = -c*beta/(2*x**2)`,

so matching the positive integrand requires `c = -2`.

### Lesson 7D: algebraic residues

After algebraic Hermite reduction, the remaining finite poles of the
differential `omega = f*dx` are simple. At a place `P` with local parameter
`z`,

`omega = (r_P/z + regular terms)*dz`,

and `r_P` is its residue.

Exact differentials have zero residue: termwise differentiation of a Laurent
series cannot produce a nonzero `dz/z` coefficient. A logarithmic
differential satisfies

`res_P(dv/v) = ord_P(v)`,

so Liouville logarithms account for simple-pole residues.

An `x`-place can split into multiple places on the algebraic curve. For
`beta**2 = x`, the denominator `x - 1` splits as

`(beta - 1)*(beta + 1)`.

Thus `dx/(x - 1)` has separate simple poles at `beta = 1` and `beta = -1`;
their local residues determine its logarithmic decomposition.

The student correctly found

`2*beta/(beta**2 - 1) = 1/(beta - 1) + 1/(beta + 1)`,

so both finite residues are `1`. Their logarithmic terms combine as

`dlog(beta - 1) + dlog(beta + 1)`

`= dlog(beta**2 - 1) = dlog(x - 1)`.

This example motivates the norm: products over algebraic conjugates package
families of algebraic logarithms without choosing each conjugate separately.

For a finite algebraic extension, the identities

`Tr(Dg) = D(Tr(g))`

and

`Tr(Dv/v) = D(N(v))/N(v)`

explain how exact and logarithmic terms descend. For `beta**2 = x` and
`v = beta - 2`, the student correctly found

`N(v) = (beta - 2)*(-beta - 2) = 4 - x`

and

`D(N(v))/N(v) = 1/(x - 4)`.

### Lesson 7E: holomorphic obstructions

After higher-order poles and logarithmic residues have been reduced, a
nonzero pole-free algebraic differential can remain on a positive-genus
curve. Such a holomorphic differential is a Liouville obstruction: it cannot
be a nonzero exact differential or a nonzero sum of logarithmic
differentials. This phenomenon has no rational-curve analogue because a
genus-zero curve has no nonzero holomorphic differentials.

For `y**2 = x**3 - x`, the differential `dx/y` is regular at each finite
branch point. Differentiating the curve gives

`dx/y = 2*dy/(3*x**2 - 1)`.

At `(x,y) = (0,0)`, the denominator has nonzero value `-1`, so `y` is a valid
local parameter and the differential starts as `-2*dy`. Equivalently, the
curve equation gives `x = -y**2 + higher terms`, hence
`3*x**2 - 1 = -1 + O(y**4)`. A finite coefficient of `dy` means there is no
pole.
