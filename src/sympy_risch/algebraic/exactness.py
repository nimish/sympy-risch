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

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polyerrors import HeuristicGCDFailed

from .._typing import FieldElement
from ..algebraic._field import _base_polynomial_ring, _extension_element
from ..algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from ..algebraic_geometry.constraints import (
    AlgebraicNonspecialLatticeSearch,
    AlgebraicNonspecialSearchStatus,
    search_nonspecial_lattice_slice,
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
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


class AlgebraicExactnessCertificate(Enum):
    """The complete fact behind an exactness or no-primitive conclusion."""

    ZERO_DIFFERENTIAL = "zero_differential"
    NONZERO_RESIDUE = "nonzero_residue"
    COMPLETE_PRIMITIVE_SPACE = "complete_primitive_space"


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


@dataclass(frozen=True, slots=True)
class AlgebraicExactnessResult:
    r"""A constructive or complete in-field exactness decision.

    ``NO_PRIMITIVE`` says only that no primitive belongs to the current
    algebraic function field.  It is not an elementary-integrability
    obstruction: logarithmic terms are handled by the separate residue step.
    """

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    primitive_pole_divisor: AlgebraicDivisor | None
    status: AlgebraicExactnessStatus
    certificate: AlgebraicExactnessCertificate | None
    primitive: ExtensionElement | None
    space: RiemannRochSpace | None
    nonspecial_search: AlgebraicNonspecialLatticeSearch | None
    max_x_degree: int
    detail: str

    def __post_init__(self) -> None:
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
        pole_divisor = self.primitive_pole_divisor
        if pole_divisor is not None and not isinstance(pole_divisor, AlgebraicDivisor):
            raise TypeError("an exactness pole divisor has the wrong type")
        if pole_divisor is not None and pole_divisor.model is not self.model:
            raise ValueError("the primitive pole divisor belongs to another model")
        if not isinstance(self.status, AlgebraicExactnessStatus):
            raise TypeError("the exactness status has the wrong type")
        if self.certificate is not None and not isinstance(
            self.certificate,
            AlgebraicExactnessCertificate,
        ):
            raise TypeError("the exactness certificate has the wrong type")
        if self.space is not None and not isinstance(self.space, RiemannRochSpace):
            raise TypeError("an exactness space needs a Riemann--Roch result")
        if self.nonspecial_search is not None and not isinstance(
            self.nonspecial_search,
            AlgebraicNonspecialLatticeSearch,
        ):
            raise TypeError("an exactness search has the wrong type")
        if not isinstance(self.max_x_degree, int) or self.max_x_degree < 0:
            raise ValueError("the exactness lattice degree bound must be nonnegative")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an exactness result needs a nonempty detail")

        coefficient = _extension_element(self.model.extension, self.coefficient)
        primitive = (
            None
            if self.primitive is None
            else _extension_element(self.model.extension, self.primitive)
        )
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "primitive", primitive)

        zero = self.model.extension.field.zero
        if coefficient == zero:
            if (
                self.status is not AlgebraicExactnessStatus.EXACT
                or self.certificate
                is not AlgebraicExactnessCertificate.ZERO_DIFFERENTIAL
                or primitive != zero
                or pole_divisor is None
                or not pole_divisor.is_zero
                or self.space is not None
                or self.nonspecial_search is not None
            ):
                raise ValueError(
                    "the zero differential has inconsistent exactness data"
                )
            return

        if self.status is AlgebraicExactnessStatus.UNSUPPORTED:
            if (
                self.certificate is not None
                or primitive is not None
                or self.space is not None
                or self.nonspecial_search is not None
            ):
                raise ValueError("an unsupported exactness result has partial data")
            return
        if pole_divisor is None:
            raise ValueError("a supported exactness result needs a pole divisor")

        differential = self.model.differential(coefficient)
        expected_bound = _primitive_pole_divisor(differential)
        if pole_divisor != expected_bound:
            raise ValueError("the primitive pole divisor is inconsistent")
        has_nonzero_residue = _has_nonzero_residue(differential)

        search = self.nonspecial_search
        if search is not None and (
            search.model is not self.model
            or search.divisor != pole_divisor
            or search.constant_field_analysis is not self.constant_field_analysis
        ):
            raise ValueError(
                "the nonspecial search belongs to another exactness problem"
            )
        space = self.space
        if space is not None:
            if (
                space.model is not self.model
                or space.divisor != pole_divisor
                or space.constant_field_analysis is not self.constant_field_analysis
                or space.status is not RiemannRochSpaceStatus.CERTIFIED
            ):
                raise ValueError("the primitive space is inconsistent")
            if search is not None and (
                search.status is not AlgebraicNonspecialSearchStatus.CERTIFIED
                or search.space is not space
            ):
                raise ValueError("the exactness search does not certify its space")

        if self.status is AlgebraicExactnessStatus.EXACT:
            if (
                self.certificate
                is not AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE
                or primitive is None
                or space is None
                or has_nonzero_residue
                or not space.contains(primitive)
                or self.derivation(primitive) - coefficient != zero
            ):
                raise ValueError("an exactness witness is inconsistent")
            return

        if self.status is AlgebraicExactnessStatus.NO_PRIMITIVE:
            if primitive is not None:
                raise ValueError("a no-primitive result cannot carry a primitive")
            if self.certificate is AlgebraicExactnessCertificate.NONZERO_RESIDUE:
                if has_nonzero_residue and space is None and search is None:
                    return
                raise ValueError("a residue obstruction has no nonzero residue")
            if (
                self.certificate
                is AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE
            ):
                if (
                    has_nonzero_residue
                    or space is None
                    or _solve_constant_derivative_equation(
                        self.model,
                        self.derivation,
                        space,
                        coefficient,
                    )
                    is not None
                ):
                    raise ValueError("the complete primitive space has a solution")
                return
            raise ValueError("a no-primitive result needs a complete certificate")

        if self.status is AlgebraicExactnessStatus.INCOMPLETE:
            if (
                self.certificate is not None
                or primitive is not None
                or space is not None
                or has_nonzero_residue
                or search is None
                or search.status is not AlgebraicNonspecialSearchStatus.INCOMPLETE
            ):
                raise ValueError("an incomplete exactness result has inconsistent data")
            return

        raise ValueError("the exactness result has an unknown status")

    @property
    def is_exact(self) -> bool:
        """Whether this result carries a verified primitive."""
        return self.status is AlgebraicExactnessStatus.EXACT

    @property
    def complete_negative(self) -> bool:
        """Whether a residue or a complete primitive space proves non-exactness."""
        return self.status is AlgebraicExactnessStatus.NO_PRIMITIVE

    @property
    def reconstructs(self) -> bool:
        """Whether the recorded primitive differentiates to the coefficient."""
        if self.primitive is None:
            return False
        return (
            self.derivation(self.primitive) - self.coefficient
            == self.model.extension.field.zero
        )


def _unsupported_exactness(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    coefficient: ExtensionElement,
    analysis: AlgebraicConstantFieldAnalysis,
    pole_divisor: AlgebraicDivisor | None,
    max_x_degree: int,
    detail: str,
) -> AlgebraicExactnessResult:
    """Construct an intentionally information-free unsupported result."""
    return AlgebraicExactnessResult(
        model,
        derivation,
        coefficient,
        analysis,
        pole_divisor,
        AlgebraicExactnessStatus.UNSUPPORTED,
        None,
        None,
        None,
        None,
        max_x_degree,
        detail,
    )


def solve_algebraic_exactness(
    coefficient: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    *,
    max_x_degree: int = 1,
) -> AlgebraicExactnessResult:
    r"""Decide exactness of one algebraic differential in the current field.

    This is a complete exactness decision only when a complete
    Riemann--Roch space for the primitive pole divisor is certified.  A
    nonzero residue is an immediate complete no-primitive certificate.  A
    bounded nonspecial lattice miss is returned as ``INCOMPLETE``; it never
    becomes a claim about elementary integrability.
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
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the exactness lattice degree bound must be nonnegative")

    value = _extension_element(model.extension, coefficient)
    zero = model.extension.field.zero
    zero_divisor = AlgebraicDivisor(model)
    if value == zero:
        return AlgebraicExactnessResult(
            model,
            derivation,
            value,
            constant_field_analysis,
            zero_divisor,
            AlgebraicExactnessStatus.EXACT,
            AlgebraicExactnessCertificate.ZERO_DIFFERENTIAL,
            zero,
            None,
            None,
            max_x_degree,
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
            max_x_degree,
            "the algebraic differential data needed for exactness is unavailable: "
            + str(exc),
        )

    if has_nonzero_residue:
        return AlgebraicExactnessResult(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            AlgebraicExactnessStatus.NO_PRIMITIVE,
            AlgebraicExactnessCertificate.NONZERO_RESIDUE,
            None,
            None,
            None,
            max_x_degree,
            "a nonzero residue proves that the differential has no algebraic primitive",
        )

    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_exactness(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            max_x_degree,
            "algebraic exactness needs a certified preserved constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    search: AlgebraicNonspecialLatticeSearch | None = None
    if pole_divisor.is_zero:
        space = compute_riemann_roch_space(
            model,
            pole_divisor,
            constant_field_analysis,
        )
        if space.status is not RiemannRochSpaceStatus.CERTIFIED:
            return _unsupported_exactness(
                model,
                derivation,
                value,
                constant_field_analysis,
                pole_divisor,
                max_x_degree,
                "the constant primitive space is unavailable: " + space.detail,
            )
    else:
        try:
            search = search_nonspecial_lattice_slice(
                model,
                pole_divisor,
                constant_field_analysis,
                max_x_degree=max_x_degree,
            )
        except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
            return _unsupported_exactness(
                model,
                derivation,
                value,
                constant_field_analysis,
                pole_divisor,
                max_x_degree,
                "the primitive-space search is unavailable: " + str(exc),
            )
        if search.status is AlgebraicNonspecialSearchStatus.INCOMPLETE:
            return AlgebraicExactnessResult(
                model,
                derivation,
                value,
                constant_field_analysis,
                pole_divisor,
                AlgebraicExactnessStatus.INCOMPLETE,
                None,
                None,
                None,
                search,
                max_x_degree,
                search.detail,
            )
        if search.status is AlgebraicNonspecialSearchStatus.UNSUPPORTED:
            return _unsupported_exactness(
                model,
                derivation,
                value,
                constant_field_analysis,
                pole_divisor,
                max_x_degree,
                search.detail,
            )
        if search.space is None:
            raise ValueError("a certified nonspecial search lacks its space")
        space = search.space

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
            max_x_degree,
            "the exact constant derivative solve is unavailable: " + str(exc),
        )
    if primitive is not None:
        return AlgebraicExactnessResult(
            model,
            derivation,
            value,
            constant_field_analysis,
            pole_divisor,
            AlgebraicExactnessStatus.EXACT,
            AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE,
            primitive,
            space,
            search,
            max_x_degree,
            "a complete primitive space contains a verified algebraic primitive",
        )
    return AlgebraicExactnessResult(
        model,
        derivation,
        value,
        constant_field_analysis,
        pole_divisor,
        AlgebraicExactnessStatus.NO_PRIMITIVE,
        AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE,
        None,
        space,
        search,
        max_x_degree,
        "the complete primitive space contains no algebraic primitive",
    )
