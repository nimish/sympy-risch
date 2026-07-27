r"""Shared principal-part helpers for monomial logarithmic signatures."""

from __future__ import annotations

from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..tower.core import Derivation


def _qq_constant_polynomial_multiple(
    numerator: Poly,
    direction: Poly,
    lower_derivation: Derivation,
) -> FieldElement | None:
    r"""Return ``c in QQ`` when ``numerator = c*direction``.

    The quotient first has to be a differential constant in the lower field.
    Under the preserved-constants hypotheses of the recursive primitive
    tower, that constant then belongs to ``QQ``.
    """
    if numerator.is_zero:
        return QQ.zero
    if direction.is_zero or numerator.degree() != direction.degree():
        return None

    parent = numerator.domain
    candidate = parent.exquo(
        parent.convert(numerator.LC()),
        parent.convert(direction.LC()),
    )
    if lower_derivation(candidate) != lower_derivation.field.zero:
        return None
    try:
        constant = QQ.from_sympy(parent.to_sympy(candidate))
    except CoercionFailed:
        return None
    if numerator != direction.mul_ground(parent.convert(constant)):
        return None
    return constant


def _factor_principal_numerators(
    proper_numerator: Poly,
    denominator: Poly,
    factor: Poly,
    multiplicity: int,
) -> tuple[Poly, ...]:
    r"""Return ``A_j`` in the local expansion ``sum(A_j/p**j)``."""
    factor_power = factor**multiplicity
    cofactor = denominator.exquo(factor_power)
    local_numerator = (proper_numerator * cofactor.invert(factor_power)).rem(
        factor_power
    )
    principal_numerators = [factor.zero] * multiplicity
    working = local_numerator
    for pole_order in range(multiplicity, 0, -1):
        coefficient = working.rem(factor)
        principal_numerators[pole_order - 1] = coefficient
        working = (working - coefficient).exquo(factor)
    if not working.is_zero:
        raise ValueError("principal-part expansion did not terminate")
    return tuple(principal_numerators)
