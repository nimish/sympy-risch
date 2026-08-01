r"""All-place residue collection after algebraic Hermite reduction."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, TypeAlias

from sympy.polys.agca.extensions import ExtensionElement

from .._typing import FieldElement
from ..tower import Derivation
from ._field import _extension_element
from .basis import IntegralBasis
from .constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    analyze_algebraic_constant_field,
)
from .hermite import (
    hermite_reduce_algebraic,
    lazy_hermite_reduce_algebraic,
)

if TYPE_CHECKING:
    from ..algebraic_geometry.function_field import (
        AlgebraicDifferential,
        AlgebraicDivisor,
        AlgebraicFunctionField,
        AlgebraicPlace,
    )


@dataclass(frozen=True, slots=True)
class AlgebraicLocalDifferentialResidue:
    r"""One pole of an algebraic differential with exact local data.

    ``pole_order`` is the positive integer ``-ord_P(omega)``. ``value`` is
    intentionally retained even when it is zero: a zero residue at a pole is
    essential later for the second-kind exactness decision.
    """

    place: AlgebraicPlace
    pole_order: int
    uniformizer: ExtensionElement
    value: ExtensionElement

    def __post_init__(self) -> None:
        from ..algebraic_geometry.finite_places import FiniteAlgebraicPlace
        from ..algebraic_geometry.infinity import InfiniteAlgebraicPlace

        if not isinstance(self.place, (FiniteAlgebraicPlace, InfiniteAlgebraicPlace)):
            raise TypeError("a local residue needs a finite or infinite place")
        if not isinstance(self.pole_order, int) or self.pole_order < 1:
            raise ValueError("a local residue needs a positive pole order")
        uniformizer = _extension_element(self.place.extension, self.uniformizer)
        if uniformizer != self.place.uniformizer:
            raise ValueError("the local residue must retain the place uniformizer")
        if self.place.valuation(uniformizer) != 1:
            raise ValueError("the local residue uniformizer must have valuation one")
        if not isinstance(self.value, ExtensionElement):
            raise TypeError("a local residue value must be an ExtensionElement")
        if self.value.ext is not self.place.residue_field:
            raise ValueError(
                "a local residue value must belong to the place residue field"
            )
        object.__setattr__(self, "uniformizer", uniformizer)

    @property
    def constant_coordinates(self) -> tuple[FieldElement, ...]:
        r"""Return the exact place-local coordinates of :attr:`value` over ``C``.

        These coordinates make one closed residue field into a finite
        constant-vector space.  They do not identify residue fields at
        distinct places; constructing such a common algebraic constant field
        is a later logarithmic-reconstruction step.
        """
        return self.place.constant_coordinates(self.value)


class _AlgebraicAllPlaceResidueResult:
    r"""The certified all-place local data of one algebraic Hermite remainder.

    :class:`AlgebraicHermiteExact` proves that the Hermite
    remainder vanishes, so its complete pole support is empty without any
    constant-field or global-geometry work. For
    :class:`AlgebraicAllPlacesCollected`, ``residues`` is exactly
    the negative support of ``differential_divisor`` and includes branch and
    infinity places. An unsupported result still retains the exact Hermite
    decomposition, but has no global model or partial residue list.
    """

    derivation: Derivation
    integrand: ExtensionElement
    basis: IntegralBasis
    exact_part: ExtensionElement
    remainder: ExtensionElement
    detail: str

    def _validate_common(self) -> ExtensionElement:
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an all-place result needs a Derivation")
        if not isinstance(self.basis, IntegralBasis):
            raise TypeError("an all-place result needs an integral basis")
        analysis = getattr(self, "constant_field_analysis", None)
        if analysis is not None and not isinstance(
            analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError("an all-place result needs constant-field analysis")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an all-place result needs a detail")

        extension = self.basis.extension
        if self.derivation.field != extension.field:
            raise ValueError("the derivation and integral basis use different fields")
        if analysis is not None:
            if analysis.extension is not extension:
                raise ValueError("the constant analysis belongs to another extension")
            if not analysis.derivation.has_same_position(self.derivation):
                raise ValueError("the constant analysis belongs to another derivation")

        integrand = _extension_element(extension, self.integrand)
        exact_part = _extension_element(extension, self.exact_part)
        remainder = _extension_element(extension, self.remainder)
        object.__setattr__(self, "integrand", integrand)
        object.__setattr__(self, "exact_part", exact_part)
        object.__setattr__(self, "remainder", remainder)

        if self.derivation(exact_part) + remainder != integrand:
            raise ValueError("the all-place result does not reconstruct its integrand")
        if not self.basis.fraction(remainder).is_square_free:
            raise ValueError(
                "the all-place result needs a square-free Hermite remainder"
            )

        return remainder

    @property
    def reconstructs_input(self) -> bool:
        """Whether the retained Hermite decomposition reconstructs the input."""
        return self.derivation(self.exact_part) + self.remainder == self.integrand


@dataclass(frozen=True, slots=True)
class AlgebraicHermiteExact(_AlgebraicAllPlaceResidueResult):
    """A Hermite decomposition with a certified zero remainder."""

    derivation: Derivation
    integrand: ExtensionElement
    basis: IntegralBasis
    exact_part: ExtensionElement
    remainder: ExtensionElement
    detail: str

    def __post_init__(self) -> None:
        remainder = self._validate_common()
        if remainder != self.basis.extension.field.zero:
            raise ValueError("an exact result needs a zero Hermite remainder")


@dataclass(frozen=True, slots=True)
class AlgebraicAllPlacesUnsupported(_AlgebraicAllPlaceResidueResult):
    """A Hermite decomposition whose global place collection is unsupported."""

    derivation: Derivation
    integrand: ExtensionElement
    basis: IntegralBasis
    exact_part: ExtensionElement
    remainder: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    detail: str

    def __post_init__(self) -> None:
        self._validate_common()
        if self.constant_field_analysis.preserved:
            raise ValueError("a preserved-constant result must collect every place")


@dataclass(frozen=True, slots=True)
class AlgebraicAllPlacesCollected(_AlgebraicAllPlaceResidueResult):
    """Complete local residue data at every finite and infinite pole."""

    derivation: Derivation
    integrand: ExtensionElement
    basis: IntegralBasis
    exact_part: ExtensionElement
    remainder: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    residues: tuple[AlgebraicLocalDifferentialResidue, ...]
    model: AlgebraicFunctionField
    differential_divisor: AlgebraicDivisor
    detail: str
    _verified_differential: InitVar[AlgebraicDifferential | None] = None

    def __post_init__(
        self, verified_differential: AlgebraicDifferential | None
    ) -> None:
        from ..algebraic_geometry.function_field import (
            AlgebraicDifferential,
            AlgebraicDivisor,
            AlgebraicFunctionField,
        )

        remainder = self._validate_common()
        analysis = self.constant_field_analysis
        if not analysis.preserved:
            raise ValueError("all-place collection requires proved preserved constants")
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a collected result needs an algebraic function field")
        if self.model.extension is not self.basis.extension:
            raise ValueError("the global model belongs to another extension")
        if remainder == self.basis.extension.field.zero:
            raise ValueError("a zero Hermite remainder needs exact status")
        if not isinstance(self.differential_divisor, AlgebraicDivisor):
            raise TypeError("a nonzero collected remainder needs its divisor")
        if verified_differential is not None:
            if not isinstance(verified_differential, AlgebraicDifferential):
                raise TypeError("a verified collection needs its differential")
            if verified_differential.model is not self.model:
                raise ValueError("the verified differential belongs to another model")
            if verified_differential.coefficient != remainder:
                raise ValueError("the verified differential has another coefficient")
            differential = verified_differential
        else:
            differential = self.model.differential(remainder)
        if self.differential_divisor != differential.divisor():
            raise ValueError("the stored differential divisor is inconsistent")
        expected_poles = tuple(
            (place, -order)
            for place, order in self.differential_divisor.terms
            if order < 0
        )
        if len(self.residues) != len(expected_poles):
            raise ValueError("the residue records do not cover every pole place")
        for record, (place, pole_order) in zip(
            self.residues,
            expected_poles,
            strict=True,
        ):
            if record.place is not place or record.pole_order != pole_order:
                raise ValueError("the residue records do not match the pole divisor")
            if record.value != differential.residue(place):
                raise ValueError("a recorded local residue is inconsistent")


AlgebraicAllPlaceResidueResult: TypeAlias = (
    AlgebraicHermiteExact | AlgebraicAllPlacesCollected | AlgebraicAllPlacesUnsupported
)


def _collect_algebraic_all_place_residues(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    lazy: bool,
    model: object | None,
) -> AlgebraicAllPlaceResidueResult:
    """Collect residues while reusing an optional validated global model."""
    if not isinstance(derivation, Derivation):
        raise TypeError("all-place residue collection requires a Derivation")
    if not isinstance(basis, IntegralBasis):
        raise TypeError("all-place residue collection requires an integral basis")
    if not isinstance(lazy, bool):
        raise TypeError("the lazy flag must be a bool")

    extension = basis.extension
    if model is not None:
        from ..algebraic_geometry.function_field import AlgebraicFunctionField

        if not isinstance(model, AlgebraicFunctionField):
            raise TypeError("all-place residue collection needs a function-field model")
        if model.extension is not extension:
            raise ValueError("the function-field model belongs to another extension")
    integrand_element = _extension_element(extension, integrand)
    if lazy:
        exact_part, remainder, active_basis = lazy_hermite_reduce_algebraic(
            integrand_element,
            derivation,
            basis,
        )
    else:
        exact_part, remainder = hermite_reduce_algebraic(
            integrand_element,
            derivation,
            basis,
        )
        active_basis = basis

    if remainder == extension.field.zero:
        return AlgebraicHermiteExact(
            derivation,
            integrand_element,
            active_basis,
            exact_part,
            remainder,
            "Hermite reduction proved exactness; no constant-field or global "
            "place computation was needed",
        )

    analysis = analyze_algebraic_constant_field(extension, derivation)
    if not analysis.preserved:
        return AlgebraicAllPlacesUnsupported(
            derivation,
            integrand_element,
            active_basis,
            exact_part,
            remainder,
            analysis,
            analysis.detail,
        )

    from ..algebraic_geometry.function_field import AlgebraicFunctionField

    if model is None:
        model = AlgebraicFunctionField(extension)
    differential = model.differential(remainder)
    differential_divisor = differential.divisor()
    residues = tuple(
        AlgebraicLocalDifferentialResidue(
            place,
            -order,
            place.uniformizer,
            differential.residue(place),
        )
        for place, order in differential_divisor.terms
        if order < 0
    )
    return AlgebraicAllPlacesCollected(
        derivation,
        integrand_element,
        active_basis,
        exact_part,
        remainder,
        analysis,
        residues,
        model,
        differential_divisor,
        "every finite and infinite pole was collected from the global "
        "differential divisor",
        _verified_differential=differential,
    )


def collect_algebraic_all_place_residues(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    lazy: bool = False,
) -> AlgebraicAllPlaceResidueResult:
    r"""Hermite-reduce and collect every pole place of a differential.

    A zero Hermite remainder returns :class:`AlgebraicHermiteExact` without
    global geometry. Otherwise, preserved constants are proved before a
    function-field model is constructed. The divisor of ``remainder*dx`` then
    supplies every finite, branch, and infinite pole with its true local
    uniformizer. An unsupported result retains the exact Hermite decomposition
    but never a partial residue list.
    """
    return _collect_algebraic_all_place_residues(
        integrand,
        derivation,
        basis,
        lazy=lazy,
        model=None,
    )
