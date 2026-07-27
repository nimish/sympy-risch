r"""Czichowski's Gröbner-basis rational logarithmic-part algorithm."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polytools import Poly, groebner

from .lrt import _normalize_lrt_inputs


def czichowski_groebner(
    numerator: Expr | Poly,
    denominator: Expr | Poly,
    x: Symbol,
    residue: Symbol | None = None,
) -> list[tuple[Poly, Poly]]:
    r"""Compute rational logarithmic parts with a lexicographic Gröbner basis.

    For coprime ``A, D`` with square-free ``D``, the numerator is first
    reduced modulo ``D`` because its polynomial quotient contributes no
    logarithmic terms.  Czichowski's algorithm then computes the reduced
    Gröbner basis of

    .. math::

        \langle A-zD', D\rangle \subset K[x,z]

    in pure lexicographic order ``x > z``.  If its elements, ordered by
    increasing ``x`` degree, are ``P_k = R_k(z) S_k(x, z)``, then
    ``Q_k = R_k/R_{k+1}`` is a square-free residue factor and ``S_{k+1}``
    is the corresponding logand.  The returned ``(logand, residue_factor)``
    pairs describe the same logarithmic part as
    :func:`lazard_rioboo_trager`:

    .. math::

        \operatorname{LogPart}\left(\int A/D\,dx\right)
        = \sum_i \sum_{q_i(c)=0} c\log(s_i(c, x)).

    This implementation intentionally does not call a resultant or a
    subresultant routine, so it provides an independent check of LRT.
    """
    numerator, denominator = _normalize_lrt_inputs(numerator, denominator, x)

    if denominator.is_zero:
        raise ZeroDivisionError("the denominator polynomial cannot be zero")
    if denominator.degree() <= 0:
        raise ValueError("Czichowski's algorithm requires a nonconstant denominator")
    if numerator.gcd(denominator).degree() > 0:
        raise ValueError(
            "Czichowski's algorithm requires coprime numerator and denominator"
        )
    if denominator.gcd(denominator.diff()).degree() > 0:
        raise ValueError("Czichowski's algorithm requires a square-free denominator")
    numerator = numerator.rem(denominator)
    if numerator.is_zero:
        return []

    residue = Dummy("z") if residue is None else residue
    if not isinstance(residue, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")
    if residue == x:
        raise ValueError("the residue symbol must differ from the variable")
    coefficient_symbols = (
        numerator.as_expr().free_symbols | denominator.as_expr().free_symbols
    ) - {x}
    if residue in coefficient_symbols:
        raise ValueError("the residue symbol must be new over the coefficient field")

    coefficient_field = denominator.domain
    leading_coefficient = denominator.LC()
    numerator = Poly(
        numerator.as_expr() / leading_coefficient,
        x,
        domain=coefficient_field,
    )
    denominator = denominator.monic()

    basis = groebner(
        (
            numerator.as_expr() - residue * denominator.diff().as_expr(),
            denominator.as_expr(),
        ),
        x,
        residue,
        order="lex",
        domain=coefficient_field,
    )
    ordered_basis = sorted(basis.polys, key=lambda polynomial: polynomial.degree(x))
    if len(ordered_basis) < 2 or ordered_basis[0].degree(x) != 0:
        raise ValueError("the Czichowski Gröbner basis has an unexpected shape")

    coefficient_ring = coefficient_field.poly_ring(residue)
    residue_contents: list[Poly] = []
    primitive_parts: list[Poly] = []
    for polynomial in ordered_basis:
        polynomial_in_x = Poly(
            polynomial.as_expr(),
            x,
            domain=coefficient_ring,
        )
        content, primitive_part = polynomial_in_x.primitive()
        residue_content = Poly(content, residue, domain=coefficient_field).monic()
        residue_contents.append(residue_content)
        primitive_parts.append(
            Poly(
                primitive_part.as_expr(),
                x,
                domain=coefficient_field.frac_field(residue),
            ).monic()
        )

    logarithmic_parts = []
    for index in range(len(ordered_basis) - 1):
        try:
            residue_factor = residue_contents[index].exquo(residue_contents[index + 1])
        except ExactQuotientFailed as exc:
            raise ValueError(
                "the Czichowski Gröbner-basis contents do not form a divisibility chain"
            ) from exc
        if residue_factor.degree() <= 0:
            continue
        residue_factor = residue_factor.monic()
        if residue_factor.gcd(residue_factor.diff()).degree() > 0:
            raise ValueError(
                "the Czichowski residue factor is unexpectedly not square-free"
            )
        logarithmic_parts.append((primitive_parts[index + 1], residue_factor))

    return logarithmic_parts
