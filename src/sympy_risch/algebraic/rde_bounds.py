r"""Certified pole bounds for first-order algebraic Risch equations.

Write the scalar equation invariantly as

``du + A*u = B``, where ``A = a*dx`` and ``B = b*dx``.

At a place ``P`` with uniformizer ``t``, let ``m = ord_P(u)``,
``alpha = ord_P(A)``, and ``beta = ord_P(B)``.  For a pole of ``u``,
``ord_P(du) = m - 1``.  Thus the derivative dominates when
``alpha > -1``, multiplication dominates when ``alpha < -1``, and only a
simple pole of ``A`` can be resonant.  In that last case the leading
coefficient is ``m + res_P(A)``.  Cancellation is therefore possible only
when the residue is a positive integer and ``m`` is its negative.

These local alternatives give a finite pole divisor containing every global
solution.  The construction uses exact closed-place residue fields; it never
chooses embeddings into an algebraic closure or truncates a search.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.numbers import Integer
from sympy.polys.agca.extensions import ExtensionElement

from ..algebraic_geometry.function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)
from ..tower import Derivation
from ._field import _base_polynomial_ring, _extension_element


class AlgebraicRDELocalCase(Enum):
    """The dominant-balance theorem used at one closed place."""

    ZERO_COEFFICIENT = "zero_coefficient"
    DERIVATIVE_DOMINANT = "derivative_dominant"
    MULTIPLICATION_DOMINANT = "multiplication_dominant"
    SIMPLE_POLE_RESONANCE = "simple_pole_resonance"


def _positive_integer_residue(
    model: AlgebraicFunctionField,
    place: AlgebraicPlace,
    residue: ExtensionElement,
) -> int | None:
    """Return a positive rational integer represented by a local residue."""
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    coordinates = place.constant_coordinates(residue)
    if any(constants.convert(value) != constants.zero for value in coordinates[1:]):
        return None
    expression = constants.to_sympy(constants.convert(coordinates[0]))
    if not isinstance(expression, Integer) or expression <= 0:
        return None
    return int(expression)


@dataclass(frozen=True, slots=True)
class AlgebraicRDELocalBound:
    r"""A lower bound for the valuation of every scalar RDE solution at ``P``.

    ``coefficient_order`` and ``right_hand_side_order`` are differential
    orders: the orders of ``a*dx`` and ``b*dx``, not merely of ``a`` and
    ``b``.  ``coefficient_order`` is ``None`` exactly when ``a = 0``.
    """

    model: AlgebraicFunctionField
    place: AlgebraicPlace
    coefficient_order: int | None
    right_hand_side_order: int
    lower_bound: int
    case: AlgebraicRDELocalCase
    residue: ExtensionElement | None
    resonant_order: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a local RDE bound needs a function field")
        # Divisor construction performs the complete model/place ownership check.
        AlgebraicDivisor(self.model, ((self.place, 1),))
        if not isinstance(self.right_hand_side_order, int):
            raise TypeError("a local RDE right-hand-side order must be an integer")
        if self.coefficient_order is not None and not isinstance(
            self.coefficient_order,
            int,
        ):
            raise TypeError("a local RDE coefficient order must be an integer")
        if not isinstance(self.lower_bound, int):
            raise TypeError("a local RDE valuation bound must be an integer")
        if not isinstance(self.case, AlgebraicRDELocalCase):
            raise TypeError("a local RDE bound has an unknown dominant case")

        beta = self.right_hand_side_order
        alpha = self.coefficient_order
        if alpha is None:
            if self.case is not AlgebraicRDELocalCase.ZERO_COEFFICIENT:
                raise ValueError("a zero RDE coefficient has the wrong local case")
            expected = min(0, beta + 1)
            if self.residue is not None or self.resonant_order is not None:
                raise ValueError("a zero RDE coefficient cannot carry resonance data")
        elif alpha < -1:
            if self.case is not AlgebraicRDELocalCase.MULTIPLICATION_DOMINANT:
                raise ValueError("a higher coefficient pole has the wrong local case")
            expected = min(0, beta - alpha)
            if self.residue is not None or self.resonant_order is not None:
                raise ValueError("a dominant coefficient cannot carry resonance data")
        elif alpha > -1:
            if self.case is not AlgebraicRDELocalCase.DERIVATIVE_DOMINANT:
                raise ValueError("a regular coefficient has the wrong local case")
            expected = min(0, beta + 1)
            if self.residue is not None or self.resonant_order is not None:
                raise ValueError("a regular coefficient cannot carry resonance data")
        else:
            if self.case is not AlgebraicRDELocalCase.SIMPLE_POLE_RESONANCE:
                raise ValueError("a simple coefficient pole has the wrong local case")
            if not isinstance(self.residue, ExtensionElement):
                raise TypeError("a resonant local case needs its exact residue")
            if self.residue.ext is not self.place.residue_field:
                raise ValueError("a local RDE residue belongs to another place")
            positive_residue = _positive_integer_residue(
                self.model,
                self.place,
                self.residue,
            )
            expected_resonance = None if positive_residue is None else -positive_residue
            if self.resonant_order != expected_resonance:
                raise ValueError("the local RDE resonant order is inconsistent")
            expected = min(0, beta + 1, self.resonant_order or 0)

        if self.lower_bound != expected:
            raise ValueError("the local RDE valuation bound is inconsistent")

    @property
    def pole_order(self) -> int:
        """Maximum pole order allowed by this local theorem."""
        return max(0, -self.lower_bound)


def _validate_rde_context(
    model: AlgebraicFunctionField,
    derivation: Derivation,
) -> None:
    """Require the direct ``C(x)``, ``Dx = 1`` curve cell used by this theorem."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("an algebraic RDE pole bound needs a function field")
    if not isinstance(derivation, Derivation):
        raise TypeError("an algebraic RDE pole bound needs a derivation")
    if derivation.field != model.extension.field:
        raise ValueError("the algebraic RDE derivation belongs to another field")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not model.extension
    ):
        raise ValueError("the algebraic RDE derivation is not bound to the model")
    _parent, constants, variable = _base_polynomial_ring(model.extension)
    if not constants.is_Exact or constants.characteristic() != 0:
        raise ValueError(
            "algebraic RDE pole bounds require exact characteristic-zero constants"
        )
    if derivation(_extension_element(model.extension, variable)) != (
        model.extension.field.one
    ):
        raise NotImplementedError(
            "algebraic RDE pole bounds currently require the standard derivative Dx = 1"
        )


def _local_bound(
    model: AlgebraicFunctionField,
    place: AlgebraicPlace,
    coefficient_differential: AlgebraicDifferential | None,
    right_hand_side_differential: AlgebraicDifferential,
) -> AlgebraicRDELocalBound:
    """Apply the exact dominant-balance alternatives at one relevant place."""
    beta = right_hand_side_differential.order_at(place)
    if not isinstance(beta, int):
        raise ValueError("a nonzero RDE right-hand side had infinite local order")
    if coefficient_differential is None:
        return AlgebraicRDELocalBound(
            model,
            place,
            None,
            beta,
            min(0, beta + 1),
            AlgebraicRDELocalCase.ZERO_COEFFICIENT,
            None,
            None,
        )

    alpha = coefficient_differential.order_at(place)
    if not isinstance(alpha, int):
        raise ValueError("a nonzero RDE coefficient had infinite local order")
    if alpha < -1:
        return AlgebraicRDELocalBound(
            model,
            place,
            alpha,
            beta,
            min(0, beta - alpha),
            AlgebraicRDELocalCase.MULTIPLICATION_DOMINANT,
            None,
            None,
        )
    if alpha > -1:
        return AlgebraicRDELocalBound(
            model,
            place,
            alpha,
            beta,
            min(0, beta + 1),
            AlgebraicRDELocalCase.DERIVATIVE_DOMINANT,
            None,
            None,
        )

    residue = coefficient_differential.residue(place)
    positive_residue = _positive_integer_residue(model, place, residue)
    resonant_order = None if positive_residue is None else -positive_residue
    return AlgebraicRDELocalBound(
        model,
        place,
        alpha,
        beta,
        min(0, beta + 1, resonant_order or 0),
        AlgebraicRDELocalCase.SIMPLE_POLE_RESONANCE,
        residue,
        resonant_order,
    )


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicRDEPoleBoundCertificate:
    r"""Factory-owned proof that every solution belongs to ``L(pole_divisor)``."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    right_hand_side: ExtensionElement
    coefficient_differential: AlgebraicDifferential | None
    right_hand_side_differential: AlgebraicDifferential
    local_bounds: tuple[AlgebraicRDELocalBound, ...]
    pole_divisor: AlgebraicDivisor
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("algebraic RDE pole bounds must be created by the factory")

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        derivation: Derivation,
        coefficient: ExtensionElement,
        right_hand_side: ExtensionElement,
        coefficient_differential: AlgebraicDifferential | None,
        right_hand_side_differential: AlgebraicDifferential,
        local_bounds: tuple[AlgebraicRDELocalBound, ...],
        pole_divisor: AlgebraicDivisor,
        detail: str,
    ) -> AlgebraicRDEPoleBoundCertificate:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "derivation", derivation)
        object.__setattr__(result, "coefficient", coefficient)
        object.__setattr__(result, "right_hand_side", right_hand_side)
        object.__setattr__(result, "coefficient_differential", coefficient_differential)
        object.__setattr__(
            result,
            "right_hand_side_differential",
            right_hand_side_differential,
        )
        object.__setattr__(result, "local_bounds", local_bounds)
        object.__setattr__(result, "pole_divisor", pole_divisor)
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        _validate_rde_context(self.model, self.derivation)
        zero = self.model.extension.field.zero
        if self.right_hand_side == zero:
            raise ValueError("a homogeneous RDE does not need a pole-bound certificate")
        if self.coefficient == zero:
            if self.coefficient_differential is not None:
                raise ValueError("a zero RDE coefficient cannot carry a differential")
            coefficient_divisor = None
        else:
            if self.coefficient_differential is None:
                raise ValueError("a nonzero RDE coefficient needs its differential")
            if (
                self.coefficient_differential.model is not self.model
                or self.coefficient_differential.coefficient != self.coefficient
            ):
                raise ValueError("the RDE coefficient differential is inconsistent")
            coefficient_divisor = self.coefficient_differential.divisor()
        if (
            self.right_hand_side_differential.model is not self.model
            or self.right_hand_side_differential.coefficient != self.right_hand_side
        ):
            raise ValueError("the RDE right-hand-side differential is inconsistent")
        right_hand_side_divisor = self.right_hand_side_differential.divisor()

        support = dict.fromkeys(
            (() if coefficient_divisor is None else coefficient_divisor.support)
            + right_hand_side_divisor.support
        )
        if tuple(bound.place for bound in self.local_bounds) != tuple(support):
            raise ValueError(
                "the algebraic RDE local bounds do not cover exact support"
            )
        for bound in self.local_bounds:
            if bound.model is not self.model:
                raise ValueError(
                    "an algebraic RDE local bound belongs to another model"
                )
            coefficient_order = (
                None
                if coefficient_divisor is None
                else coefficient_divisor.coefficient(bound.place)
            )
            if bound.coefficient_order != coefficient_order or (
                bound.right_hand_side_order
                != right_hand_side_divisor.coefficient(bound.place)
            ):
                raise ValueError("an algebraic RDE local bound has inconsistent orders")

        expected_divisor = AlgebraicDivisor(
            self.model,
            tuple(
                (bound.place, bound.pole_order)
                for bound in self.local_bounds
                if bound.pole_order
            ),
        )
        if self.pole_divisor != expected_divisor:
            raise ValueError("the algebraic RDE global pole divisor is inconsistent")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic RDE pole bound needs a nonempty detail")


def certify_algebraic_rde_pole_bound(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
) -> AlgebraicRDEPoleBoundCertificate:
    r"""Return a complete pole bound for ``D(u) + coefficient*u = rhs``.

    The right-hand side must be nonzero.  The homogeneous equation has the
    unconditional solution zero and is handled before this factory is called.
    """
    _validate_rde_context(model, derivation)
    extension = model.extension
    a = _extension_element(extension, coefficient)
    b = _extension_element(extension, right_hand_side)
    if b == extension.field.zero:
        raise ValueError("a homogeneous algebraic RDE does not need a pole bound")

    coefficient_differential = (
        None if a == extension.field.zero else model.differential(a)
    )
    right_hand_side_differential = model.differential(b)
    coefficient_divisor = (
        None if coefficient_differential is None else coefficient_differential.divisor()
    )
    right_hand_side_divisor = right_hand_side_differential.divisor()
    places = tuple(
        dict.fromkeys(
            (() if coefficient_divisor is None else coefficient_divisor.support)
            + right_hand_side_divisor.support
        )
    )
    local_bounds = tuple(
        _local_bound(
            model,
            place,
            coefficient_differential,
            right_hand_side_differential,
        )
        for place in places
    )
    pole_divisor = AlgebraicDivisor(
        model,
        tuple(
            (bound.place, bound.pole_order)
            for bound in local_bounds
            if bound.pole_order
        ),
    )
    return AlgebraicRDEPoleBoundCertificate._create(
        model,
        derivation,
        a,
        b,
        coefficient_differential,
        right_hand_side_differential,
        local_bounds,
        pole_divisor,
        "dominant balance at every divisor-support place bounds all poles; "
        "outside that support the right-hand differential has order zero and "
        "the coefficient differential is either zero or has order zero",
    )


__all__ = [
    "AlgebraicRDELocalBound",
    "AlgebraicRDELocalCase",
    "AlgebraicRDEPoleBoundCertificate",
    "certify_algebraic_rde_pole_bound",
]
