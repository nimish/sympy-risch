r"""Exact finite constant fields used by the rational Risch kernels.

The implemented rational algorithms work over a standard base ``C(x)`` with
``Dx = 1`` when ``C`` is ``QQ``, Gaussian rationals, or a native finite
algebraic number field over ``QQ``.  The relation algorithms still seek
relations with *rational* coefficients, so they need an exact fixed
``QQ``-basis for elements of ``C``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from sympy.core.numbers import I
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed

from ._typing import FieldElement


class _PowerBasisElement(Protocol):
    """Native algebraic-number element with canonical power coordinates."""

    def to_list(self) -> list[object]: ...


@dataclass(frozen=True, slots=True)
class FiniteConstantFieldProfile:
    """Effective data shared by the finite-constant Risch kernels."""

    constants: Domain
    degree_over_qq: int
    is_real: bool
    contains_i: bool


def finite_constant_field_profile(
    constants: Domain,
) -> FiniteConstantFieldProfile | None:
    r"""Describe an effective finite extension of ``QQ``, if supported."""
    if constants == QQ:
        return FiniteConstantFieldProfile(constants, 1, True, False)
    if constants == QQ_I:
        return FiniteConstantFieldProfile(constants, 2, False, True)
    if not isinstance(constants, AlgebraicField) or constants.domain != QQ:
        return None

    try:
        constants.from_sympy(I)
    except CoercionFailed:
        contains_i = False
    else:
        contains_i = True
    is_real = all(extension.is_real is True for extension in constants.orig_ext)
    return FiniteConstantFieldProfile(
        constants,
        int(constants.mod.degree()),
        is_real,
        contains_i,
    )


def is_supported_finite_constant_field(constants: Domain) -> bool:
    r"""Whether ``constants`` is an effective finite extension of ``QQ``.

    ``QQ_I`` has a dedicated native representation rather than SymPy's
    :class:`~sympy.polys.domains.algebraicfield.AlgebraicField`, so it is
    intentionally listed separately.  Restricting ``AlgebraicField`` to
    ground field ``QQ`` excludes algebraic *function* fields from this
    constant-field admission rule.
    """
    return finite_constant_field_profile(constants) is not None


def gaussian_constant_field(constants: Domain) -> Domain:
    r"""Return the exact compositum ``C(i)`` for a supported real field ``C``."""
    profile = finite_constant_field_profile(constants)
    if profile is None:
        raise NotImplementedError(
            "Gaussian constants require QQ, QQ_I, or an algebraic number field over QQ",
        )
    if profile.contains_i:
        return constants
    if constants == QQ:
        return QQ_I
    if not profile.is_real or not isinstance(constants, AlgebraicField):
        raise NotImplementedError(
            "Gaussian tangent conversion requires a provably real algebraic "
            "constant field or a field already containing i",
        )

    target = QQ.algebraic_field(*constants.orig_ext, I)
    if constants.unify(target) != target:
        raise ValueError("the Gaussian compositum lost the source constant field")
    return target


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
