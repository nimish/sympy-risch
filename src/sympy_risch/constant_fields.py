r"""Exact constant fields used by the rational Risch kernels.

The implemented rational algorithms work over a standard base ``C(x)`` with
``Dx = 1`` when ``C`` is a supported finite algebraic number field over
``QQ`` or a purely transcendental rational-function extension of one. The
relation algorithms still seek relations with *rational* coefficients; they
expand finite constants in a fixed ``QQ`` basis and clear transcendental
constant denominators one variable at a time before comparing coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from sympy.core.numbers import I
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly

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


def is_supported_rational_constant_field(constants: Domain) -> bool:
    r"""Whether rational kernels support exact arithmetic over ``constants``.

    Besides the finite fields above, this admits one rational-function field
    over such a field. Its generators are differential constants; no
    rational-coordinate, ordering, or reality claim is implied.
    """
    return is_supported_finite_constant_field(constants) or (
        isinstance(constants, FractionField)
        and is_supported_finite_constant_field(cast(Domain, constants.domain))
    )


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


def rational_constant_relation_rows(
    constants: Domain,
    values: tuple[FieldElement, ...],
) -> tuple[tuple[FieldElement, ...], ...]:
    r"""Return exact rational rows equivalent to ``sum(q_i*values[i]) = 0``."""
    if not values:
        return ()
    if not isinstance(constants, FractionField):
        columns = tuple(
            rational_constant_coordinates(constants, value) for value in values
        )
        return tuple(
            row
            for row in zip(*columns, strict=True)
            if any(entry != QQ.zero for entry in row)
        )

    kernel = _rational_constant_relation_kernel(constants, values)
    dimension = len(values)
    if not kernel:
        return tuple(
            tuple(QQ.one if row == column else QQ.zero for column in range(dimension))
            for row in range(dimension)
        )
    matrix = DomainMatrix(
        [list(row) for row in kernel],
        (len(kernel), dimension),
        QQ,
    )
    return tuple(
        tuple(QQ.convert(entry) for entry in row)
        for row in matrix.nullspace(divide_last=True).to_list()
    )


def _rational_constant_relation_kernel(
    constants: Domain,
    values: tuple[FieldElement, ...],
) -> tuple[tuple[FieldElement, ...], ...]:
    """Return the rational kernel of one finite tuple in ``constants``.

    A rational-function field is handled recursively as ``K(a)``. Coefficient
    equations in ``a`` restrict the candidate kernel before descending into
    ``K``, so unrelated parameter denominators never form a Cartesian product.
    """
    dimension = len(values)
    basis = tuple(
        tuple(QQ.one if row == column else QQ.zero for column in range(dimension))
        for row in range(dimension)
    )
    if all(constants.convert(value) == constants.zero for value in values):
        return basis
    if not isinstance(constants, FractionField):
        coordinate_columns = tuple(
            rational_constant_coordinates(constants, value) for value in values
        )
        rows = tuple(
            tuple(column[index] for column in coordinate_columns)
            for index in range(len(coordinate_columns[0]))
        )
        matrix = DomainMatrix(
            [list(row) for row in rows],
            (len(rows), dimension),
            QQ,
        )
        return tuple(
            tuple(QQ.convert(entry) for entry in row)
            for row in matrix.nullspace(divide_last=True).to_list()
        )

    ground = cast(Domain, constants.domain)
    symbols = cast(tuple[Symbol, ...], constants.symbols)
    if not symbols or not is_supported_finite_constant_field(ground):
        raise NotImplementedError(
            "rational relation rows require a rational-function field over "
            "supported finite constants"
        )
    lower = ground if len(symbols) == 1 else ground.frac_field(*symbols[:-1])
    variable = symbols[-1]
    fractions = tuple(
        constants.to_sympy(constants.convert(value)).as_numer_denom()
        for value in values
    )
    polynomial_fractions = tuple(
        (
            Poly(numerator, variable, domain=lower),
            Poly(denominator, variable, domain=lower),
        )
        for numerator, denominator in fractions
    )
    denominator = polynomial_fractions[0][1]
    for _, item_denominator in polynomial_fractions[1:]:
        denominator = denominator.lcm(item_denominator)
    numerators = tuple(
        numerator * denominator.exquo(item_denominator)
        for numerator, item_denominator in polynomial_fractions
    )
    maximum_degree = max(
        (numerator.degree() for numerator in numerators if not numerator.is_zero),
        default=-1,
    )
    rows = tuple(
        tuple(lower.convert(numerator.nth(degree)) for numerator in numerators)
        for degree in range(maximum_degree + 1)
    )
    return _intersect_rational_kernel(basis, rows, lower)


def _intersect_rational_kernel(
    basis: tuple[tuple[FieldElement, ...], ...],
    rows: tuple[tuple[FieldElement, ...], ...],
    field: Domain,
) -> tuple[tuple[FieldElement, ...], ...]:
    """Restrict a rational candidate basis by field-valued row equations."""
    current = basis
    for row in rows:
        if not current:
            break
        projected = tuple(
            sum(
                (
                    field.convert(coefficient) * field.convert(row[index])
                    for index, coefficient in enumerate(candidate)
                ),
                field.zero,
            )
            for candidate in current
        )
        constraint_kernel = _rational_constant_relation_kernel(field, projected)
        current = tuple(
            tuple(
                sum(
                    (
                        QQ.convert(coefficient) * QQ.convert(current[index][column])
                        for index, coefficient in enumerate(combination)
                    ),
                    QQ.zero,
                )
                for column in range(len(current[0]))
            )
            for combination in constraint_kernel
        )
    return current
