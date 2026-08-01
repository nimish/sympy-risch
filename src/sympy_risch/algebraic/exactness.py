r"""Certified exactness decisions for residue-free algebraic differentials.

After algebraic Hermite reduction and logarithmic reconstruction, a remaining
differential has zero residues.  It is exact precisely when it is the
derivative of a function in the algebraic function field.  If

``E = sum_{ord_P(omega) < -1} (-ord_P(omega) - 1) P``,

then every primitive of ``omega`` lies in ``L(E)``.  This module uses only a
complete, certified Riemann--Roch space for that divisor: a derivative solve
inside a bounded partial space is never used to prove non-exactness.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
from typing import ClassVar, TypeAlias, cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polyerrors import HeuristicGCDFailed

from .._typing import FieldElement
from ..algebraic._field import _base_polynomial_ring, _extension_element
from ..algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from ..algebraic_geometry.function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
)
from ..algebraic_geometry.riemann_roch import (
    RiemannRochSpace,
    RiemannRochSpaceStatus,
    compute_riemann_roch_space,
    constant_coordinate_matrix,
)
from ..tower import Derivation


class AlgebraicExactnessStatus(Enum):
    """Outcome of an in-field exactness decision."""

    EXACT = "exact"
    NO_PRIMITIVE = "no_primitive"
    UNSUPPORTED = "unsupported"


class AlgebraicExactnessCertificate(Enum):
    """The complete fact behind an exactness or no-primitive conclusion."""

    ZERO_DIFFERENTIAL = "zero_differential"
    NONZERO_RESIDUE = "nonzero_residue"
    COMPLETE_PRIMITIVE_SPACE = "complete_primitive_space"


class _TrustedNoPrimitiveSolve:
    """Module-private marker for a no-solution result just computed below."""


_TRUSTED_NO_PRIMITIVE_SOLVE = _TrustedNoPrimitiveSolve()


def _primitive_pole_divisor(
    differential: AlgebraicDifferential,
) -> AlgebraicDivisor:
    r"""Return the pole divisor that contains every possible primitive.

    At a pole where ``ord_P(omega) = -m`` with ``m > 1``, a primitive can
    have a pole of order at most ``m - 1``.  A derivative has zero residue, so
    a simple pole contributes no possible primitive pole.
    """
    divisor = differential.divisor()
    return AlgebraicDivisor(
        differential.model,
        tuple((place, -order - 1) for place, order in divisor.terms if order < -1),
    )


def _has_nonzero_residue(differential: AlgebraicDifferential) -> bool:
    """Whether a pole residue proves that ``differential`` is not exact."""
    for place, order in differential.divisor().terms:
        if order < 0 and differential.residue(place) != place.residue_field.zero:
            return True
    return False


def _constant_linear_combination(
    model: AlgebraicFunctionField,
    basis: tuple[ExtensionElement, ...],
    coordinates: tuple[FieldElement, ...],
) -> ExtensionElement:
    """Rebuild a constant-linear combination in the native function field."""
    if len(basis) != len(coordinates):
        raise ValueError("constant coordinates have the wrong basis dimension")
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    result = model.extension.field.zero
    for coordinate, element in zip(coordinates, basis, strict=True):
        coefficient = constants.convert(coordinate)
        if coefficient != constants.zero:
            result += (
                _extension_element(
                    model.extension,
                    constants.to_sympy(coefficient),
                )
                * element
            )
    return cast(ExtensionElement, result)


def _solve_constant_derivative_equation(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    space: RiemannRochSpace,
    coefficient: ExtensionElement,
) -> ExtensionElement | None:
    r"""Solve ``D(g) = coefficient`` in a complete constant basis of ``space``."""
    basis = space.basis
    derivatives = tuple(
        _extension_element(model.extension, derivation(element)) for element in basis
    )
    augmented = constant_coordinate_matrix(model, (*derivatives, coefficient))
    reduced, pivots = augmented.rref()
    right_column = len(basis)
    if right_column in pivots:
        return None

    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    coordinates: list[FieldElement] = [constants.zero for _ in basis]
    reduced_rows = reduced.to_list()
    for row, pivot in enumerate(pivots):
        if pivot < right_column:
            coordinates[pivot] = constants.convert(reduced_rows[row][right_column])
    primitive = _constant_linear_combination(model, basis, tuple(coordinates))
    if derivation(primitive) - coefficient != model.extension.field.zero:
        raise ValueError("the exact constant derivative solve did not reconstruct")
    return primitive


class _AlgebraicExactnessResult:
    r"""A constructive or complete in-field exactness decision.

    At this low-level boundary, ``NO_PRIMITIVE`` says only that no primitive
    belongs to the current algebraic function field.  After the all-place
    integration pipeline has removed every logarithmic residue, a
    ``COMPLETE_PRIMITIVE_SPACE`` negative can be adapted to a Liouville
    non-elementarity obstruction; a nonzero-residue decision cannot.
    """

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    detail: str
    status: ClassVar[AlgebraicExactnessStatus]

    def _validate_common(self) -> ExtensionElement:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("an exactness result needs an algebraic function field")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an exactness result needs a derivation")
        if self.derivation.field != self.model.extension.field:
            raise ValueError("the exactness derivation belongs to another field")
        if self.derivation.level < 0 or (
            self.derivation.tower.levels[self.derivation.level]
            is not self.model.extension
        ):
            raise ValueError("the exactness derivation is not bound to the model")
        if not isinstance(
            self.constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError("an exactness result needs constant-field analysis")
        if self.constant_field_analysis.extension is not self.model.extension:
            raise ValueError("the constant analysis belongs to another field")
        if not self.constant_field_analysis.derivation.has_same_position(
            self.derivation,
        ):
            raise ValueError("the constant analysis belongs to another derivation")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an exactness result needs a nonempty detail")

        coefficient = _extension_element(self.model.extension, self.coefficient)
        object.__setattr__(self, "coefficient", coefficient)
        return coefficient

    def _validate_pole_divisor(
        self, coefficient: ExtensionElement, pole_divisor: AlgebraicDivisor
    ) -> AlgebraicDifferential:
        if not isinstance(pole_divisor, AlgebraicDivisor):
            raise TypeError("an exactness pole divisor has the wrong type")
        if pole_divisor.model is not self.model:
            raise ValueError("the primitive pole divisor belongs to another model")
        differential = self.model.differential(coefficient)
        if pole_divisor != _primitive_pole_divisor(differential):
            raise ValueError("the primitive pole divisor is inconsistent")
        return differential

    @property
    def is_exact(self) -> bool:
        """Whether this result carries a verified primitive."""
        return isinstance(self, AlgebraicExactPrimitive)

    @property
    def complete_negative(self) -> bool:
        """Whether a residue or complete space proves non-exactness."""
        return isinstance(self, AlgebraicNoPrimitive)

    @property
    def reconstructs(self) -> bool:
        """Whether the recorded primitive differentiates to the coefficient."""
        primitive = getattr(self, "primitive", None)
        return primitive is not None and (
            self.derivation(primitive) - self.coefficient
            == self.model.extension.field.zero
        )


@dataclass(frozen=True, slots=True)
class AlgebraicExactPrimitive(_AlgebraicExactnessResult):
    """A verified primitive in the current algebraic function field."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    primitive_pole_divisor: AlgebraicDivisor
    certificate: AlgebraicExactnessCertificate
    primitive: ExtensionElement
    space: RiemannRochSpace | None
    detail: str
    status: ClassVar[AlgebraicExactnessStatus] = AlgebraicExactnessStatus.EXACT

    def __post_init__(self) -> None:
        coefficient = self._validate_common()
        primitive = _extension_element(self.model.extension, self.primitive)
        object.__setattr__(self, "primitive", primitive)
        pole_divisor = self.primitive_pole_divisor
        zero = self.model.extension.field.zero
        if not isinstance(pole_divisor, AlgebraicDivisor):
            raise TypeError("an exactness pole divisor has the wrong type")
        if pole_divisor.model is not self.model:
            raise ValueError("the primitive pole divisor belongs to another model")

        if coefficient == zero:
            if (
                self.certificate is not AlgebraicExactnessCertificate.ZERO_DIFFERENTIAL
                or primitive != zero
                or not pole_divisor.is_zero
                or self.space is not None
            ):
                raise ValueError(
                    "the zero differential has inconsistent exactness data"
                )
            return
        differential = self._validate_pole_divisor(coefficient, pole_divisor)
        has_nonzero_residue = _has_nonzero_residue(differential)
        space = self.space
        if space is None or (
            not isinstance(space, RiemannRochSpace)
            or space.model is not self.model
            or space.divisor != pole_divisor
            or space.status is not RiemannRochSpaceStatus.CERTIFIED
        ):
            raise ValueError("the primitive space is inconsistent")
        if (
            self.certificate
            is not AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE
            or has_nonzero_residue
            or not space.contains(primitive)
            or self.derivation(primitive) - coefficient != zero
        ):
            raise ValueError("an exactness witness is inconsistent")


@dataclass(frozen=True, slots=True)
class AlgebraicNoPrimitive(_AlgebraicExactnessResult):
    """A complete certificate that no in-field primitive exists."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    primitive_pole_divisor: AlgebraicDivisor
    certificate: AlgebraicExactnessCertificate
    space: RiemannRochSpace | None
    detail: str
    _trusted_no_solution: InitVar[object | None] = None
    status: ClassVar[AlgebraicExactnessStatus] = AlgebraicExactnessStatus.NO_PRIMITIVE
    primitive: ClassVar[None] = None

    def __post_init__(self, trusted_no_solution: object | None) -> None:
        coefficient = self._validate_common()
        if coefficient == self.model.extension.field.zero:
            raise ValueError("the zero differential always has a primitive")
        differential = self._validate_pole_divisor(
            coefficient, self.primitive_pole_divisor
        )
        has_nonzero_residue = _has_nonzero_residue(differential)
        space = self.space
        if space is not None and (
            not isinstance(space, RiemannRochSpace)
            or space.model is not self.model
            or space.divisor != self.primitive_pole_divisor
            or space.status is not RiemannRochSpaceStatus.CERTIFIED
        ):
            raise ValueError("the primitive space is inconsistent")
        if self.certificate is AlgebraicExactnessCertificate.NONZERO_RESIDUE:
            if has_nonzero_residue and space is None:
                return
            raise ValueError("a residue obstruction has no nonzero residue")
        if self.certificate is AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE:
            if (
                self.constant_field_analysis.status
                is not AlgebraicConstantFieldStatus.PRESERVED
            ):
                raise ValueError(
                    "a complete primitive-space negative needs preserved constants"
                )
            if (
                trusted_no_solution is not None
                and trusted_no_solution is not _TRUSTED_NO_PRIMITIVE_SOLVE
            ):
                raise ValueError("the trusted exactness construction marker is invalid")
            if (
                has_nonzero_residue
                or space is None
                or (
                    trusted_no_solution is None
                    and _solve_constant_derivative_equation(
                        self.model, self.derivation, space, coefficient
                    )
                    is not None
                )
            ):
                raise ValueError("the complete primitive space has a solution")
            return
        raise ValueError("a no-primitive result needs a complete certificate")


@dataclass(frozen=True, slots=True)
class AlgebraicExactnessUnsupported(_AlgebraicExactnessResult):
    """An exact boundary where no complete exactness decision was possible."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    primitive_pole_divisor: AlgebraicDivisor | None
    detail: str
    status: ClassVar[AlgebraicExactnessStatus] = AlgebraicExactnessStatus.UNSUPPORTED
    certificate: ClassVar[None] = None
    primitive: ClassVar[None] = None
    space: ClassVar[None] = None

    def __post_init__(self) -> None:
        self._validate_common()
        pole_divisor = self.primitive_pole_divisor
        if pole_divisor is not None:
            if not isinstance(pole_divisor, AlgebraicDivisor):
                raise TypeError("an exactness pole divisor has the wrong type")
            if pole_divisor.model is not self.model:
                raise ValueError("the primitive pole divisor belongs to another model")


AlgebraicExactnessResult: TypeAlias = (
    AlgebraicExactPrimitive | AlgebraicNoPrimitive | AlgebraicExactnessUnsupported
)


def _unsupported_exactness(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    coefficient: ExtensionElement,
    analysis: AlgebraicConstantFieldAnalysis,
    pole_divisor: AlgebraicDivisor | None,
    detail: str,
) -> AlgebraicExactnessResult:
    """Construct an intentionally information-free unsupported result."""
    return AlgebraicExactnessUnsupported(
        model,
        derivation,
        coefficient,
        analysis,
        pole_divisor,
        detail,
    )


def solve_algebraic_exactness(
    coefficient: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
) -> AlgebraicExactnessResult:
    r"""Decide exactness of one algebraic differential in the current field.

    This is a complete exactness decision only when a complete
    Riemann--Roch space for the primitive pole divisor is certified. A
    nonzero residue is an immediate complete no-primitive certificate.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("algebraic exactness needs a derivation")
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("algebraic exactness needs a function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("algebraic exactness needs constant-field analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant analysis belongs to another function field")
    if not constant_field_analysis.derivation.has_same_position(derivation):
        raise ValueError("the constant analysis belongs to another derivation")
    if derivation.field != model.extension.field:
        raise ValueError("the exactness derivation belongs to another field")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not model.extension
    ):
        raise ValueError("the exactness derivation is not bound to the model")
    value = _extension_element(model.extension, coefficient)
    zero = model.extension.field.zero
    zero_divisor = AlgebraicDivisor(model)
    if value == zero:
        return AlgebraicExactPrimitive(
            model,
            derivation,
            value,
            constant_field_analysis,
            zero_divisor,
            AlgebraicExactnessCertificate.ZERO_DIFFERENTIAL,
            zero,
            None,
            "the zero differential has the zero primitive",
        )

    try:
        differential = model.differential(value)
        pole_divisor = _primitive_pole_divisor(differential)
        has_nonzero_residue = _has_nonzero_residue(differential)
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_exactness(
            model,
            derivation,
            value,
            constant_field_analysis,
            None,
            "the algebraic differential data needed for exactness is unavailable: "
            + str(exc),
        )

    if has_nonzero_residue:
        return AlgebraicNoPrimitive(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            AlgebraicExactnessCertificate.NONZERO_RESIDUE,
            None,
            "a nonzero residue proves that the differential has no algebraic primitive",
        )

    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_exactness(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            "algebraic exactness needs a certified preserved constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    space = compute_riemann_roch_space(model, pole_divisor)
    if space.status is not RiemannRochSpaceStatus.CERTIFIED:
        return _unsupported_exactness(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            "the complete primitive space is unavailable: " + space.detail,
        )

    try:
        primitive = _solve_constant_derivative_equation(
            model,
            derivation,
            space,
            value,
        )
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_exactness(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            "the exact constant derivative solve is unavailable: " + str(exc),
        )
    if primitive is not None:
        return AlgebraicExactPrimitive(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE,
            primitive,
            space,
            "a complete primitive space contains a verified algebraic primitive",
        )
    return AlgebraicNoPrimitive(
        model,
        derivation,
        value,
        constant_field_analysis,
        pole_divisor,
        AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE,
        space,
        "the complete primitive space contains no algebraic primitive",
        _trusted_no_solution=_TRUSTED_NO_PRIMITIVE_SOLVE,
    )
