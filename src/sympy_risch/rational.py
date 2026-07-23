r"""Rational-function integration over :math:`\mathbb{Q}`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly, cancel, invert, resultant
from sympy.polys.rootoftools import RootSum

from .lrt import lazard_rioboo_trager
from .polynomial import integrate_polynomial


@dataclass(frozen=True, slots=True)
class RationalExactDerivativeResult:
    r"""Certified result of the exact-derivative problem in ``QQ(x)``.

    The fields satisfy

    ``f = derivative(exact_part, x) + obstruction``.

    Hermite reduction makes ``obstruction`` canonical: it is zero exactly
    when ``f`` is the derivative of another rational function.  Otherwise its
    square-free denominator exposes the simple-pole obstruction.
    """

    exact_part: Expr
    obstruction: Expr

    @property
    def has_solution(self) -> bool:
        """Whether a rational antiderivative exists."""
        return self.obstruction == 0

    @property
    def antiderivative(self) -> Expr | None:
        """Return the rational antiderivative, or ``None`` if none exists."""
        return self.exact_part if self.has_solution else None


@dataclass(frozen=True, slots=True)
class RationalLogDerivativeFactor:
    r"""One irreducible denominator factor in a logarithmic-derivative test.

    For a square-free denominator factor ``p``, ``partial_numerator`` is the
    numerator of the corresponding partial fraction.  A rational logarithmic
    derivative must have the form

    ``multiplicity*p.diff()/p``

    at that factor.  ``multiplicity`` is ``None`` when the partial numerator
    is not a constant multiple of ``p.diff()``.
    """

    factor: Poly
    partial_numerator: Poly
    multiplicity: Any | None

    @property
    def has_logarithmic_residue(self) -> bool:
        """Whether the residue has the required multiple-of-``p'`` shape."""
        return self.multiplicity is not None

    @property
    def has_integral_multiplicity(self) -> bool:
        """Whether the residue is the valuation of a rational function."""
        return (
            self.multiplicity is not None
            and QQ.denom(QQ.convert(self.multiplicity)) == 1
        )


@dataclass(frozen=True, slots=True)
class RationalLogDerivativeResult:
    r"""Certified solution of ``Du/u = f`` in :math:`\mathbb{Q}(x)`.

    A rational logarithmic derivative has no polynomial part, only simple
    poles, and an integer valuation at every irreducible denominator factor.
    ``argument`` is the canonical product of those monic factors; its
    arbitrary nonzero rational constant multiplier is normalized to one.
    """

    argument: Expr | None
    polynomial_part: Poly
    denominator_factors: tuple[tuple[Poly, int], ...]
    factors: tuple[RationalLogDerivativeFactor, ...]

    @property
    def has_solution(self) -> bool:
        """Whether a nonzero rational ``u`` satisfies ``Du/u = f``."""
        return self.argument is not None

    @property
    def has_polynomial_obstruction(self) -> bool:
        """Whether a nonzero polynomial part prevents a solution."""
        return not self.polynomial_part.is_zero

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether some finite pole has order greater than one."""
        return any(
            multiplicity > 1
            for _, multiplicity in self.denominator_factors
        )

    @property
    def has_nonlogarithmic_residue_obstruction(self) -> bool:
        """Whether some residue is not a constant multiple of ``p'``."""
        return any(not factor.has_logarithmic_residue for factor in self.factors)

    @property
    def has_nonintegral_residue_obstruction(self) -> bool:
        """Whether some logarithmic residue is rational but not integral."""
        return any(
            factor.has_logarithmic_residue
            and not factor.has_integral_multiplicity
            for factor in self.factors
        )


def _as_reduced_fraction(f: Expr, x: Symbol) -> tuple[Poly, Poly]:
    """Return coprime numerator and denominator polynomials in ``QQ[x]``."""
    numerator, denominator = cancel(f).as_numer_denom()
    numerator_poly = Poly(numerator, x, domain=QQ)
    denominator_poly = Poly(denominator, x, domain=QQ)
    coefficient, numerator_poly, denominator_poly = numerator_poly.cancel(
        denominator_poly,
    )
    numerator_poly = numerator_poly.mul_ground(coefficient)
    return numerator_poly, denominator_poly


def hermite_reduce_rational(f: Expr, x: Symbol) -> tuple[Expr, Expr]:
    r"""Write ``f`` as ``D(g) + h`` with a square-free denominator for ``h``.

    This uses the Horowitz--Ostrogradsky form of rational Hermite reduction.
    Both the polynomial part of the integral and the rational function ``g``
    are included in the first return value.
    """
    numerator, denominator = _as_reduced_fraction(f, x)
    polynomial_part, proper_numerator = numerator.div(denominator)

    exact_part = integrate_polynomial(polynomial_part, x)

    if proper_numerator.is_zero:
        return exact_part, S.Zero

    repeated_part, square_free_part, _ = denominator.cofactors(denominator.diff())

    rational_degree = repeated_part.degree()
    remainder_degree = square_free_part.degree()
    correction = (repeated_part.diff()*square_free_part).quo(repeated_part)
    rational_columns = []
    for degree in range(rational_degree):
        monomial = Poly(x**degree, x, domain=QQ)
        rational_columns.append(
            monomial.diff()*square_free_part - monomial*correction,
        )
    remainder_columns = [
        Poly(x**degree, x, domain=QQ)*repeated_part
        for degree in range(remainder_degree)
    ]
    columns = rational_columns + remainder_columns
    unknown_count = len(columns)
    system = DomainMatrix(
        [
            [QQ.convert(column.nth(degree)) for column in columns]
            for degree in range(unknown_count)
        ],
        (unknown_count, unknown_count),
        QQ,
    )
    right_hand_side = DomainMatrix(
        [
            [QQ.convert(proper_numerator.nth(degree))]
            for degree in range(unknown_count)
        ],
        (unknown_count, 1),
        QQ,
    )
    try:
        solution_numerators, solution_denominator = system.solve_den(
            right_hand_side,
        )
    except DMNonInvertibleMatrixError as exc:
        raise ValueError(
            "Hermite reduction did not produce a unique solution",
        ) from exc
    solution = [
        row[0]/solution_denominator
        for row in solution_numerators.to_list()
    ]
    rational_numerator = Poly.from_dict(
        {
            (degree,): coefficient
            for degree, coefficient in enumerate(solution[:rational_degree])
        },
        x,
        domain=QQ,
    ).as_expr()
    remainder_numerator = Poly.from_dict(
        {
            (degree,): coefficient
            for degree, coefficient in enumerate(solution[rational_degree:])
        },
        x,
        domain=QQ,
    ).as_expr()

    rational_part = cancel(rational_numerator/repeated_part.as_expr())
    reduced_remainder = cancel(remainder_numerator/square_free_part.as_expr())

    return exact_part + rational_part, reduced_remainder


def hermite_reduce_rational_recursive(
    f: Expr,
    x: Symbol,
) -> tuple[Expr, Expr]:
    r"""Classically Hermite-reduce ``f`` one repeated factor at a time.

    If the current denominator is ``D = U*V**m`` with ``V`` square-free and
    ``m > 1``, extended Euclidean reduction computes ``B`` such that
    subtracting ``D(B/V**(m - 1))`` lowers the multiplicity of ``V``.  The
    process terminates when the remaining denominator is square-free.

    This is an independent recursive counterpart to the global
    Horowitz--Ostrogradsky system in :func:`hermite_reduce_rational`.
    """
    numerator, denominator = _as_reduced_fraction(f, x)
    polynomial_part, numerator = numerator.div(denominator)
    exact_part = integrate_polynomial(polynomial_part, x)

    if numerator.is_zero:
        return exact_part, S.Zero

    while denominator.gcd(denominator.diff()).degree() > 0:
        repeated_factors = [
            (factor, multiplicity)
            for factor, multiplicity in denominator.sqf_list()[1]
            if multiplicity > 1
        ]
        factor, multiplicity = max(
            repeated_factors,
            key=lambda item: item[1],
        )
        cofactor = denominator.exquo(factor**multiplicity)

        # Bronstein, ISSAC'98 section 1.2: solve
        # A/(1-m) = B*U*V' + C*V with deg(B) < deg(V).
        scaled_numerator = numerator.mul_ground(QQ(1, 1 - multiplicity))
        modular_coefficient = cofactor*factor.diff()
        correction_numerator = (
            scaled_numerator*invert(modular_coefficient, factor)
        ).rem(factor)
        quotient_numerator = (
            scaled_numerator - correction_numerator*modular_coefficient
        ).exquo(factor)

        exact_part += cancel(
            correction_numerator.as_expr()
            / factor.as_expr()**(multiplicity - 1),
        )
        numerator = (
            quotient_numerator.mul_ground(1 - multiplicity)
            - cofactor*correction_numerator.diff()
        )
        denominator = cofactor*factor**(multiplicity - 1)
        coefficient, numerator, denominator = numerator.cancel(denominator)
        numerator = numerator.mul_ground(coefficient)

    remainder = cancel(numerator.as_expr()/denominator.as_expr())
    return exact_part, remainder


def solve_rational_exact_derivative(
    f: Expr,
    x: Symbol,
) -> RationalExactDerivativeResult:
    r"""Decide whether ``f`` is a derivative in :math:`\mathbb{Q}(x)`.

    Rational Hermite reduction writes

    .. math::

        f = Dv + r,

    where ``r`` has a square-free denominator.  Derivatives of rational
    functions have no simple poles, so a rational ``v`` with ``Dv = f``
    exists if and only if ``r = 0``.  A nonzero ``r`` is therefore both the
    reduced remainder and a certificate that the exact-derivative equation
    has no rational solution.
    """
    exact_part, obstruction = hermite_reduce_rational(f, x)
    return RationalExactDerivativeResult(exact_part, obstruction)


def solve_rational_logarithmic_derivative(
    f: Expr,
    x: Symbol,
) -> RationalLogDerivativeResult:
    r"""Decide whether ``f = Du/u`` for a nonzero ``u`` in ``QQ(x)``.

    Write the proper part of ``f`` over its factored denominator.  At an
    irreducible factor ``p``, the logarithmic derivative of a rational
    function has partial numerator ``n*p'``, where the integer ``n`` is the
    order of a zero or pole of ``u`` along ``p``.  Thus a solution exists
    exactly when:

    1. the polynomial part of ``f`` vanishes;
    2. its denominator is square-free;
    3. every partial numerator is an integer multiple of ``p'``.

    The factor certificates retain rational noninteger multiples.  For
    example, ``1/(2*x)`` has multiplicity ``1/2``: it has no rational
    logarithmic argument, but the certificate records why squaring a
    hyperexponential candidate can reveal an algebraic dependency.
    """
    numerator, denominator = _as_reduced_fraction(f, x)
    polynomial_part, proper_numerator = numerator.div(denominator)
    denominator_factors = tuple(denominator.factor_list()[1])

    if proper_numerator.is_zero:
        argument = S.One if polynomial_part.is_zero else None
        return RationalLogDerivativeResult(
            argument,
            polynomial_part,
            denominator_factors,
            (),
        )

    if any(multiplicity > 1 for _, multiplicity in denominator_factors):
        return RationalLogDerivativeResult(
            None,
            polynomial_part,
            denominator_factors,
            (),
        )

    factor_certificates: list[RationalLogDerivativeFactor] = []
    argument: Expr = S.One
    for factor, _ in denominator_factors:
        cofactor = denominator.exquo(factor)
        partial_numerator = (
            proper_numerator*invert(cofactor, factor)
        ).rem(factor)
        quotient, remainder = partial_numerator.div(factor.diff())
        multiplicity = None
        if remainder.is_zero and quotient.degree() <= 0:
            multiplicity = QQ.convert(quotient.nth(0))
            if QQ.denom(multiplicity) == 1:
                argument *= factor.as_expr()**int(QQ.numer(multiplicity))

        factor_certificates.append(
            RationalLogDerivativeFactor(
                factor,
                partial_numerator,
                multiplicity,
            )
        )

    factors = tuple(factor_certificates)
    has_solution = (
        polynomial_part.is_zero
        and all(factor.has_integral_multiplicity for factor in factors)
    )
    if not has_solution:
        argument = None
    elif cancel(argument.diff(x)/argument - f) != 0:
        raise ValueError(
            "rational logarithmic-derivative reconstruction failed"
        )

    return RationalLogDerivativeResult(
        argument,
        polynomial_part,
        denominator_factors,
        factors,
    )


def rothstein_trager_residue_polynomial(
    f: Expr,
    x: Symbol,
    residue: Symbol | None = None,
) -> Poly:
    r"""Return the polynomial whose roots are the residues of reduced ``f``.

    ``f`` must be a proper rational function with a square-free denominator.
    For ``f = A/D``, the returned polynomial is

    ``resultant_x(D, A - residue*D')``.
    """
    numerator, denominator = _as_reduced_fraction(f, x)
    residue = Dummy("z") if residue is None else residue
    if not isinstance(residue, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")
    if residue == x:
        raise ValueError("the residue symbol must differ from the variable")

    if numerator.is_zero:
        return Poly(1, residue, domain=QQ)

    if numerator.degree() >= denominator.degree():
        raise ValueError("Rothstein--Trager requires a proper rational function")

    if denominator.gcd(denominator.diff()).degree() > 0:
        raise ValueError("Rothstein--Trager requires a square-free denominator")

    residue_polynomial = resultant(
        denominator,
        numerator - Poly(residue, x)*denominator.diff(),
        x,
    )
    return Poly(residue_polynomial, residue)


def integrate_rational(f: Expr, x: Symbol) -> Expr:
    r"""Integrate a rational function in ``QQ(x)``.

    Hermite reduction extracts the derivative of a rational function. The
    Lazard--Rioboo--Trager form of Rothstein--Trager then constructs the
    logarithmic part without splitting the denominator over an algebraic
    closure explicitly.
    """
    exact_part, reduced_remainder = hermite_reduce_rational(f, x)

    if reduced_remainder == 0:
        return exact_part

    numerator, denominator = _as_reduced_fraction(reduced_remainder, x)
    residue = Dummy("z")
    logarithmic_terms = []

    for logand, residue_factor in lazard_rioboo_trager(
        numerator,
        denominator,
        x,
        residue,
    ):
        logarithmic_terms.append(RootSum(
            residue_factor,
            Lambda(residue, residue*log(logand.as_expr())),
            quadratic=True,
        ))

    return exact_part + Add(*logarithmic_terms)


def _integrate_real_linear_or_quadratic(
    numerator: Poly,
    denominator: Poly,
    x: Symbol,
) -> Expr:
    """Integrate ``numerator / denominator`` in real elementary form."""
    degree = denominator.degree()
    denominator_expr = denominator.as_expr()

    if degree == 1:
        coefficient = numerator.nth(0) / denominator.nth(1)
        return coefficient*log(Abs(denominator_expr))

    if degree != 2:
        raise NotImplementedError(
            "real rational integration currently requires denominator "
            "factors of degree at most two over QQ",
        )

    a = denominator.nth(2)
    b = denominator.nth(1)
    c = denominator.nth(0)
    m = numerator.nth(1)
    n = numerator.nth(0)

    logarithm_coefficient = m / (2*a)
    constant_remainder = n - logarithm_coefficient*b
    result = logarithm_coefficient*log(Abs(denominator_expr))

    if constant_remainder == 0:
        return result

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        scale = sqrt(-discriminant)
        angle = atan((2*a*x + b) / scale)
        return result + 2*constant_remainder*angle/scale

    if discriminant > 0:
        scale = sqrt(discriminant)
        linear_part = 2*a*x + b
        logarithms = (
            log(Abs(linear_part - scale))
            - log(Abs(linear_part + scale))
        )
        return result + constant_remainder*logarithms/scale

    raise ValueError("a square-free quadratic cannot have zero discriminant")


def integrate_rational_real(f: Expr, x: Symbol) -> Expr:
    r"""Integrate a rational function in ``QQ(x)`` in real elementary form.

    Hermite reduction first extracts the rational derivative. The remaining
    square-free denominator is factored over ``QQ`` and decomposed by modular
    inverses. Linear factors produce real logarithms, while irreducible
    quadratics produce either arctangents or real logarithms depending on the
    sign of their discriminant.

    This first real-form implementation supports factors of degree at most two.
    Its result is an antiderivative on every real interval avoiding poles.
    """
    exact_part, reduced_remainder = hermite_reduce_rational(f, x)

    if reduced_remainder == 0:
        return exact_part

    numerator, denominator = _as_reduced_fraction(reduced_remainder, x)
    real_terms = []

    for factor, multiplicity in denominator.factor_list()[1]:
        if multiplicity != 1:
            raise ValueError("Hermite reduction returned a repeated factor")

        cofactor = denominator.exquo(factor)
        inverse = invert(cofactor, factor)
        partial_numerator = (numerator*inverse).rem(factor)
        real_terms.append(_integrate_real_linear_or_quadratic(
            partial_numerator,
            factor,
            x,
        ))

    return exact_part + Add(*real_terms)
