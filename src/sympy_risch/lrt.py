r"""Lazard--Rioboo--Trager rational logarithmic-part computation."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.polytools import Poly, resultant


def _remove_vanishing_leading_factors(
    subresultant: Poly,
    residue_factor: Poly,
    residue: Symbol,
    x: Symbol,
) -> Poly:
    """Remove content that would make the specialization drop in degree."""
    leading_coefficient = Poly(subresultant.LC(), residue, field=True)
    _, square_free_factors = leading_coefficient.sqf_list(all=True)

    for factor, multiplicity in square_free_factors:
        common_factor = factor.gcd(residue_factor)

        if common_factor.degree() > 0:
            divisor = Poly(common_factor.as_expr()**multiplicity, x)
            subresultant = subresultant.exquo(divisor)

    return subresultant


def _make_monic_modulo(
    polynomial: Poly,
    modulus: Poly,
    residue: Symbol,
    x: Symbol,
) -> Poly:
    """Normalize a polynomial in ``x`` to be monic modulo ``modulus``."""
    leading_coefficient = Poly(polynomial.LC(), residue, field=True)
    inverse = leading_coefficient.invert(modulus)
    normalized_terms = {}

    for monomial, coefficient in polynomial.terms():
        coefficient = Poly(coefficient, residue, field=True)
        normalized = (inverse*coefficient).rem(modulus)
        normalized_terms[monomial] = normalized.as_expr()

    return Poly.from_dict(normalized_terms, x)


def _differential_lazard_rioboo_trager(
    numerator: Poly,
    denominator: Poly,
    denominator_derivative: Poly,
    residue_factors: tuple[tuple[Poly, int], ...],
    x: Symbol,
    residue: Symbol,
) -> list[tuple[Poly, Poly]]:
    r"""Recover selected generalized RT logands with a subresultant PRS.

    ``residue_factors`` contains square-free factors of the generalized
    resultant together with their multiplicities.  Callers use Bronstein's
    coefficient derivation first and pass only factors whose roots are
    constants.  Keeping this helper separate preserves the simpler public
    rational LRT interface below.
    """
    if not (
        numerator.gens
        == denominator.gens
        == denominator_derivative.gens
        == (x,)
    ):
        raise ValueError("generalized LRT polynomials must share their generator")
    if not (
        numerator.domain
        == denominator.domain
        == denominator_derivative.domain
    ):
        raise ValueError("generalized LRT polynomials must share their domain")

    coefficient_field = denominator.domain
    if isinstance(coefficient_field, FractionField):
        residue_field = coefficient_field.domain.frac_field(
            *coefficient_field.symbols,
            residue,
        )
    else:
        residue_field = coefficient_field.frac_field(residue)

    denominator_with_residue = Poly(
        denominator.as_expr(),
        x,
        domain=residue_field,
    )
    numerator_with_residue = Poly(
        numerator.as_expr() - residue*denominator_derivative.as_expr(),
        x,
        domain=residue_field,
    )
    _, remainder_sequence = resultant(
        denominator_with_residue,
        numerator_with_residue,
        x,
        includePRS=True,
    )
    subresultants_by_degree = {
        polynomial.degree(): polynomial
        for polynomial in remainder_sequence
    }

    logarithmic_parts = []
    for residue_factor, multiplicity in residue_factors:
        if residue_factor.degree() <= 0:
            continue
        if multiplicity == denominator.degree():
            logand = denominator.monic()
        else:
            try:
                logand = subresultants_by_degree[multiplicity]
            except KeyError as exc:
                raise ValueError(
                    "the generalized LRT subresultant degree is missing"
                ) from exc
            logand = _remove_vanishing_leading_factors(
                logand,
                residue_factor,
                residue,
                x,
            )
            logand = _make_monic_modulo(
                logand,
                residue_factor,
                residue,
                x,
            )
        logarithmic_parts.append((logand, residue_factor))

    return logarithmic_parts


def lazard_rioboo_trager(
    numerator: Expr | Poly,
    denominator: Expr | Poly,
    x: Symbol,
    residue: Symbol | None = None,
) -> list[tuple[Poly, Poly]]:
    r"""Compute the logarithmic part of a reduced rational function.

    For coprime ``A, D`` with ``deg(A) < deg(D)`` and square-free ``D``, the
    returned pairs ``(s_i, q_i)`` represent

    .. math::

        \int A/D\,dx
        = \sum_i \sum_{q_i(c)=0} c\log(s_i(c, x)).

    The computation uses the resultant of ``D`` and ``A - z*D'`` to find the
    residues and the polynomial remainder sequence to recover the pole factors
    belonging to each residue.
    """
    numerator = Poly(numerator, x)
    denominator = Poly(denominator, x)

    if denominator.is_zero:
        raise ZeroDivisionError("the denominator polynomial cannot be zero")

    if denominator.degree() <= 0:
        raise ValueError("LRT requires a nonconstant denominator")

    if numerator.is_zero:
        return []

    if numerator.degree() >= denominator.degree():
        raise ValueError("LRT requires a proper rational function")

    if numerator.gcd(denominator).degree() > 0:
        raise ValueError("LRT requires coprime numerator and denominator")

    if denominator.gcd(denominator.diff()).degree() > 0:
        raise ValueError("LRT requires a square-free denominator")

    residue = residue or Dummy("z")

    if not isinstance(residue, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")
    if residue == x:
        raise ValueError("the residue symbol must differ from the variable")
    coefficient_symbols = (
        numerator.as_expr().free_symbols
        | denominator.as_expr().free_symbols
    ) - {x}
    if residue in coefficient_symbols:
        raise ValueError("the residue symbol must be new over the coefficient field")

    residue_polynomial, remainder_sequence = resultant(
        denominator,
        numerator - denominator.diff()*Poly(residue, x),
        x,
        includePRS=True,
    )
    residue_polynomial = Poly(residue_polynomial, residue, composite=False)

    if residue_polynomial.is_zero:
        raise ValueError("the LRT residue resultant unexpectedly vanished")

    subresultants_by_degree = {
        polynomial.degree(): polynomial
        for polynomial in remainder_sequence
    }
    _, square_free_factors = residue_polynomial.sqf_list()
    logarithmic_parts = []

    for residue_factor, multiplicity in square_free_factors:
        _, residue_factor = residue_factor.primitive()

        if multiplicity == denominator.degree():
            logand = denominator.monic()
        else:
            logand = subresultants_by_degree[multiplicity]
            logand = _remove_vanishing_leading_factors(
                logand,
                residue_factor,
                residue,
                x,
            )
            logand = _make_monic_modulo(
                logand,
                residue_factor,
                residue,
                x,
            )

        logarithmic_parts.append((logand, residue_factor))

    return logarithmic_parts
