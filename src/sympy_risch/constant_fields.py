r"""Exact finite constant fields used by the rational Risch kernels.

The implemented rational algorithms work over a standard base ``C(x)`` with
``Dx = 1`` when ``C`` is ``QQ``, Gaussian rationals, or a native finite
algebraic number field over ``QQ``.  The relation algorithms still seek
relations with *rational* coefficients, so they need an exact fixed
``QQ``-basis for elements of ``C``.
"""

from __future__ import annotations

from typing import Protocol, cast

from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain

from ._typing import FieldElement


class _PowerBasisElement(Protocol):
    """Native algebraic-number element with canonical power coordinates."""

    def to_list(self) -> list[object]: ...


def is_supported_finite_constant_field(constants: Domain) -> bool:
    r"""Whether ``constants`` is an effective finite extension of ``QQ``.

    ``QQ_I`` has a dedicated native representation rather than SymPy's
    :class:`~sympy.polys.domains.algebraicfield.AlgebraicField`, so it is
    intentionally listed separately.  Restricting ``AlgebraicField`` to
    ground field ``QQ`` excludes algebraic *function* fields from this
    constant-field admission rule.
    """
    return constants in {QQ, QQ_I} or (
        isinstance(constants, AlgebraicField) and constants.domain == QQ
    )


def rational_constant_coordinates(
    constants: Domain,
    value: FieldElement,
) -> tuple[FieldElement, ...]:
    r"""Return exact ``QQ`` coordinates of ``value`` in a fixed basis of ``C``.

    For an ``AlgebraicField`` the basis is ``(1, theta, ..., theta**(d - 1))``
    for SymPy's native primitive element ``theta = constants.ext``.  Native
    ``ANP.to_list()`` values are in descending powers with leading zero powers
    omitted, hence the left-padding and reversal below.
    """
    if constants == QQ:
        return (QQ.convert(value),)
    if constants == QQ_I:
        gaussian_value = QQ_I.convert(value)
        return (
            QQ.convert(gaussian_value.x),
            QQ.convert(gaussian_value.y),
        )
    if not isinstance(constants, AlgebraicField) or constants.domain != QQ:
        raise NotImplementedError(
            "rational coordinates require QQ, QQ_I, or a finite algebraic "
            "constant field over QQ",
        )

    native_value = cast(_PowerBasisElement, constants.convert(value))
    descending = native_value.to_list()
    degree = int(constants.mod.degree())
    if len(descending) > degree:
        raise ValueError("native algebraic constant exceeds its power-basis degree")
    padded = [QQ.zero] * (degree - len(descending)) + descending
    return tuple(QQ.convert(coefficient) for coefficient in reversed(padded))
