r"""Recognize algebraic integrating factors from logarithmic derivatives.

For an algebraic function field ``L/C(x)``, an element ``a`` has an
integrating factor in ``L`` exactly when ``a = D(h)/h`` for some nonzero
``h`` in ``L``.  The differential ``a*dx`` must then have only simple poles
with integral residues, and its residue divisor must be ``div(h)``.  A
verified principal-divisor witness is checked against the original
coefficient, so this module never mistakes a residue-compatible differential
for a logarithmic derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed, HeuristicGCDFailed

from ..algebraic_geometry.function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)
from ..algebraic_geometry.principal import (
    AlgebraicPrincipalDivisorSearch,
    AlgebraicPrincipalDivisorStatus,
    search_algebraic_principal_divisor,
)
from ..tower import Derivation
from ._field import (
    _base_polynomial_ring,
    _extension_element,
    _quotient_extension_elements,
)
from .constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)


class AlgebraicIntegratingFactorStatus(Enum):
    """Outcome of recognizing ``a = D(h)/h`` in one algebraic field."""

    FOUND = "found"
    NO_FACTOR = "no_factor"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


class AlgebraicIntegratingFactorCertificate(Enum):
    """Mathematical fact behind an integrating-factor outcome."""

    ZERO_COEFFICIENT = "zero_coefficient"
    PRINCIPAL_RESIDUE_DIVISOR = "principal_residue_divisor"
    HIGHER_ORDER_POLE = "higher_order_pole"
    NONINTEGRAL_RESIDUE = "nonintegral_residue"
    NONPRINCIPAL_RESIDUE_DIVISOR = "nonprincipal_residue_divisor"
    RESIDUAL_DIFFERENTIAL = "residual_differential"


@dataclass(frozen=True, slots=True)
class AlgebraicIntegratingFactorResult:
    r"""A verified integrating factor or a precise recognition boundary.

    ``NO_FACTOR`` is complete only for the logarithmic-derivative question
    in the current field.  In particular, it does not decide a general Risch
    differential equation with coefficient :attr:`coefficient`.
    """

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    status: AlgebraicIntegratingFactorStatus
    certificate: AlgebraicIntegratingFactorCertificate | None
    factor: ExtensionElement | None
    residue_divisor: AlgebraicDivisor | None
    principal_search: AlgebraicPrincipalDivisorSearch | None
    max_x_degree: int
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("an integrating-factor result needs a function field")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an integrating-factor result needs a derivation")
        if self.derivation.field != self.model.extension.field:
            raise ValueError(
                "the integrating-factor derivation belongs to another field"
            )
        if self.derivation.level < 0 or (
            self.derivation.tower.levels[self.derivation.level]
            is not self.model.extension
        ):
            raise ValueError(
                "the integrating-factor derivation is not bound to the model"
            )
        if not isinstance(
            self.constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError("an integrating-factor result needs constant analysis")
        if self.constant_field_analysis.extension is not self.model.extension:
            raise ValueError("the constant analysis belongs to another function field")
        if not self.constant_field_analysis.derivation.has_same_position(
            self.derivation,
        ):
            raise ValueError("the constant analysis belongs to another derivation")
        if not isinstance(self.status, AlgebraicIntegratingFactorStatus):
            raise TypeError("the integrating-factor status has the wrong type")
        if self.certificate is not None and not isinstance(
            self.certificate,
            AlgebraicIntegratingFactorCertificate,
        ):
            raise TypeError("the integrating-factor certificate has the wrong type")
        if not isinstance(self.max_x_degree, int) or self.max_x_degree < 0:
            raise ValueError(
                "the integrating-factor lattice degree must be nonnegative"
            )
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an integrating-factor result needs a detail")

        coefficient = _extension_element(self.model.extension, self.coefficient)
        factor = (
            None
            if self.factor is None
            else _extension_element(self.model.extension, self.factor)
        )
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "factor", factor)

        if self.residue_divisor is not None and (
            not isinstance(self.residue_divisor, AlgebraicDivisor)
            or self.residue_divisor.model is not self.model
        ):
            raise ValueError("the residue divisor belongs to another model")
        if self.principal_search is not None and (
            not isinstance(self.principal_search, AlgebraicPrincipalDivisorSearch)
            or self.principal_search.model is not self.model
            or self.principal_search.divisor != self.residue_divisor
        ):
            raise ValueError("the principal search belongs to another residue divisor")

        if self.status is AlgebraicIntegratingFactorStatus.FOUND:
            if factor is None or factor == self.model.extension.field.zero:
                raise ValueError("a found integrating factor needs a nonzero witness")
            if (
                _quotient_extension_elements(
                    self.model.extension,
                    self.derivation(factor),
                    factor,
                )
                != coefficient
            ):
                raise ValueError("the integrating-factor witness does not reconstruct")
            return
        if factor is not None:
            raise ValueError("only a found integrating factor carries a witness")

    @property
    def found(self) -> bool:
        """Whether this result carries a verified nonzero integrating factor."""
        return self.status is AlgebraicIntegratingFactorStatus.FOUND

    @property
    def complete_negative(self) -> bool:
        """Whether no logarithmic-derivative factor exists in the current field."""
        return self.status is AlgebraicIntegratingFactorStatus.NO_FACTOR


def _integer_residue(
    place: AlgebraicPlace,
    value: ExtensionElement,
    constants: Domain,
) -> int | None:
    """Return a place residue when it is a displayed rational integer."""
    try:
        coordinates = place.constant_coordinates(value)
        zero = constants.zero
        if not coordinates or any(coordinate != zero for coordinate in coordinates[1:]):
            return None
        expression = constants.to_sympy(constants.convert(coordinates[0]))
    except (AttributeError, CoercionFailed, TypeError, ValueError):
        return None
    if expression.is_Integer is not True:
        return None
    return int(expression)


def _result(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    coefficient: ExtensionElement,
    analysis: AlgebraicConstantFieldAnalysis,
    status: AlgebraicIntegratingFactorStatus,
    certificate: AlgebraicIntegratingFactorCertificate | None,
    factor: ExtensionElement | None,
    residue_divisor: AlgebraicDivisor | None,
    principal_search: AlgebraicPrincipalDivisorSearch | None,
    max_x_degree: int,
    detail: str,
) -> AlgebraicIntegratingFactorResult:
    """Build one compact result while keeping the public constructor uniform."""
    return AlgebraicIntegratingFactorResult(
        model,
        derivation,
        coefficient,
        analysis,
        status,
        certificate,
        factor,
        residue_divisor,
        principal_search,
        max_x_degree,
        detail,
    )


def find_algebraic_integrating_factor(
    coefficient: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    *,
    max_x_degree: int = 1,
) -> AlgebraicIntegratingFactorResult:
    r"""Recognize a verified ``h`` satisfying ``D(h)/h = coefficient``.

    This is a bounded principal-divisor semialgorithm only at the global
    divisor step.  A failed finite principal search is ``INCOMPLETE``; all
    other negative outcomes are complete for the integrating-factor question.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("an algebraic integrating factor needs a derivation")
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("an algebraic integrating factor needs a function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("an algebraic integrating factor needs constant analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant analysis belongs to another function field")
    if not constant_field_analysis.derivation.has_same_position(derivation):
        raise ValueError("the constant analysis belongs to another derivation")
    if derivation.field != model.extension.field:
        raise ValueError("the integrating-factor derivation belongs to another field")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not model.extension
    ):
        raise ValueError("the integrating-factor derivation is not bound to the model")
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the integrating-factor lattice degree must be nonnegative")

    value = _extension_element(model.extension, coefficient)
    zero = model.extension.field.zero
    if value == zero:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.FOUND,
            AlgebraicIntegratingFactorCertificate.ZERO_COEFFICIENT,
            model.extension.field.one,
            AlgebraicDivisor(model),
            None,
            max_x_degree,
            "the zero coefficient has constant integrating factor one",
        )

    try:
        differential = model.differential(value)
        divisor = differential.divisor()
        _parent, constants, _variable = _base_polynomial_ring(model.extension)
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.UNSUPPORTED,
            None,
            None,
            None,
            None,
            max_x_degree,
            "the differential data needed for an integrating factor is unavailable: "
            + str(exc),
        )

    residue_terms: list[tuple[AlgebraicPlace, int]] = []
    try:
        for place, order in divisor.terms:
            if order < -1:
                return _result(
                    model,
                    derivation,
                    value,
                    constant_field_analysis,
                    AlgebraicIntegratingFactorStatus.NO_FACTOR,
                    AlgebraicIntegratingFactorCertificate.HIGHER_ORDER_POLE,
                    None,
                    None,
                    None,
                    max_x_degree,
                    "a logarithmic derivative cannot have a pole of order greater than one",
                )
            if order == -1:
                residue = _integer_residue(
                    place,
                    differential.residue(place),
                    constants,
                )
                if residue is None:
                    return _result(
                        model,
                        derivation,
                        value,
                        constant_field_analysis,
                        AlgebraicIntegratingFactorStatus.NO_FACTOR,
                        AlgebraicIntegratingFactorCertificate.NONINTEGRAL_RESIDUE,
                        None,
                        None,
                        None,
                        max_x_degree,
                        "a logarithmic-derivative residue is not a displayed integer",
                    )
                if residue:
                    residue_terms.append((place, residue))
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.UNSUPPORTED,
            None,
            None,
            None,
            None,
            max_x_degree,
            "the local residue data needed for an integrating factor is unavailable: "
            + str(exc),
        )

    residue_divisor = AlgebraicDivisor(model, tuple(residue_terms))
    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.UNSUPPORTED,
            None,
            None,
            residue_divisor,
            None,
            max_x_degree,
            "integrating-factor reconstruction requires a certified preserved "
            f"constant field; the current presentation is "
            f"{constant_field_analysis.status.value}",
        )

    search = search_algebraic_principal_divisor(
        model,
        residue_divisor,
        constant_field_analysis,
        max_x_degree=max_x_degree,
    )
    if search.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.INCOMPLETE,
            None,
            None,
            residue_divisor,
            search,
            max_x_degree,
            search.detail,
        )
    if search.status is AlgebraicPrincipalDivisorStatus.UNSUPPORTED:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.UNSUPPORTED,
            None,
            None,
            residue_divisor,
            search,
            max_x_degree,
            search.detail,
        )
    if search.status is AlgebraicPrincipalDivisorStatus.NONPRINCIPAL:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.NO_FACTOR,
            AlgebraicIntegratingFactorCertificate.NONPRINCIPAL_RESIDUE_DIVISOR,
            None,
            residue_divisor,
            search,
            max_x_degree,
            "the integral residue divisor cannot be the divisor of an integrating factor",
        )
    if search.witness is None:
        raise ValueError("a principal integrating-factor search lacks its witness")

    try:
        logarithmic_derivative = _quotient_extension_elements(
            model.extension,
            derivation(search.witness),
            search.witness,
        )
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.UNSUPPORTED,
            None,
            None,
            residue_divisor,
            search,
            max_x_degree,
            "the principal witness could not be differentiated exactly: " + str(exc),
        )
    if logarithmic_derivative != value:
        return _result(
            model,
            derivation,
            value,
            constant_field_analysis,
            AlgebraicIntegratingFactorStatus.NO_FACTOR,
            AlgebraicIntegratingFactorCertificate.RESIDUAL_DIFFERENTIAL,
            None,
            residue_divisor,
            search,
            max_x_degree,
            "the residue-compatible principal divisor leaves a nonzero differential",
        )
    return _result(
        model,
        derivation,
        value,
        constant_field_analysis,
        AlgebraicIntegratingFactorStatus.FOUND,
        AlgebraicIntegratingFactorCertificate.PRINCIPAL_RESIDUE_DIVISOR,
        search.witness,
        residue_divisor,
        search,
        max_x_degree,
        "a verified principal residue divisor reconstructs the logarithmic derivative",
    )
