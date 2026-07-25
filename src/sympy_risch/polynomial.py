r"""Polynomial integration over :math:`\mathbb{Q}`."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from ._typing import FieldElement


def integrate_polynomial(f: Expr | Poly, x: Symbol) -> Expr:
    r"""Integrate a polynomial in ``QQ[x]`` coefficient by coefficient.

    The returned antiderivative has integration constant zero. Inputs outside
    ``QQ[x]`` are rejected by SymPy's exact domain conversion rather than being
    silently moved to the expression domain.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_risch.polynomial import integrate_polynomial
    >>> x = symbols("x")
    >>> integrate_polynomial(3*x**2 + 2*x + 1, x)
    x**3 + x**2 + x
    """
    polynomial = Poly(f, x, domain=QQ)

    integrated_terms: dict[tuple[int], FieldElement] = {}

    for (degree,), coefficient in polynomial.terms():
        new_degree = degree + 1
        integrated_terms[(new_degree,)] = coefficient / QQ(new_degree)

    return Poly.from_dict(integrated_terms, x, domain=QQ).as_expr()
