"""Shared exact coercion for multivariate RDE test oracles."""

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly, cancel


def as_qq_bivariate_fraction(
    expression: Expr,
    lower_generator: Symbol,
    top_generator: Symbol,
) -> tuple[Poly, Poly]:
    """Return a reduced fraction in ``QQ[lower_generator, top_generator]``."""
    numerator_expression, denominator_expression = cancel(expression).as_numer_denom()
    numerator = Poly(
        numerator_expression,
        lower_generator,
        top_generator,
        domain=QQ,
    )
    denominator = Poly(
        denominator_expression,
        lower_generator,
        top_generator,
        domain=QQ,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(QQ.convert(coefficient))
    leading_inverse = QQ.one / denominator.LC()
    return (
        numerator.mul_ground(leading_inverse),
        denominator.monic(),
    )
