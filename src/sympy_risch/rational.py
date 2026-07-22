r"""Rational-function integration over :math:`\mathbb{Q}`."""

from __future__ import annotations

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
            [column.nth(degree) for column in columns]
            for degree in range(unknown_count)
        ],
        (unknown_count, unknown_count),
        QQ,
    )
    right_hand_side = DomainMatrix(
        [[proper_numerator.nth(degree)] for degree in range(unknown_count)],
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

    if numerator.is_zero:
        residue = residue or Dummy("z")
        return Poly(1, residue, domain=QQ)

    if numerator.degree() >= denominator.degree():
        raise ValueError("Rothstein--Trager requires a proper rational function")

    if denominator.gcd(denominator.diff()).degree() > 0:
        raise ValueError("Rothstein--Trager requires a square-free denominator")

    residue = residue or Dummy("z")
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
