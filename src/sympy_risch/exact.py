r"""Exact-derivative dispatch across implemented differential tower levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly

from ._typing import FieldElement, is_field_zero
from .monomial import (
    HyperexponentialMonomialReduction,
    HypertangentMonomialReduction,
    PrimitiveExactDerivativeResult,
    _normal_fraction_part,
    reduce_hyperexponential_monomial,
    reduce_hypertangent_monomial,
    solve_primitive_exact_derivative,
)
from .rational import (
    RationalExactDerivativeResult,
    _solve_rational_exact_derivative_over_domain,
)
from .tower import Derivation, MonomialExtension, MonomialFamily, _coerce

if TYPE_CHECKING:
    from .rde import RischDifferentialEquationResult


@dataclass(frozen=True, slots=True)
class TranscendentalExactDerivativeCertificate:
    r"""Family reductions and the recursively solved lower remainder.

    ``residual_reduction`` is present when a nonconstant normal residue proves
    non-exactness but a disjoint special or lower-field part can still be
    reduced.  Keeping that second reduction makes the returned exact part
    maximal without discarding the finite-residue obstruction.
    """

    reduction: HyperexponentialMonomialReduction | HypertangentMonomialReduction
    residual_reduction: (
        HyperexponentialMonomialReduction | HypertangentMonomialReduction | None
    )
    lower_result: ExactDerivativeResult | None


@dataclass(frozen=True, slots=True)
class HypertangentGaussianExactDerivativeCertificate:
    """Exactness decision obtained from the Gaussian scalar RDE fallback."""

    rde_result: RischDifferentialEquationResult


ExactDerivativeCertificate = (
    RationalExactDerivativeResult
    | PrimitiveExactDerivativeResult
    | TranscendentalExactDerivativeCertificate
    | HypertangentGaussianExactDerivativeCertificate
)


@dataclass(frozen=True, slots=True)
class ExactDerivativeResult:
    r"""Level-independent exact-derivative result using native domain elements."""

    exact_part: FieldElement
    obstruction: FieldElement
    certificate: ExactDerivativeCertificate | None

    @property
    def has_solution(self) -> bool:
        """Whether the input is a derivative in the current field."""
        return is_field_zero(self.obstruction)

    @property
    def antiderivative(self) -> FieldElement | None:
        """Return the in-field antiderivative, or ``None`` if none exists."""
        return self.exact_part if self.has_solution else None


def _standard_rational_constant_field(
    extension: MonomialExtension,
) -> Domain | None:
    """Return ``C`` for a supported standard rational base ``C(x)``."""
    constant_field = extension.parent_field
    if constant_field not in {QQ, QQ_I}:
        return None
    if extension.generator_derivative != Poly(
        1,
        extension.generator,
        domain=constant_field,
    ):
        return None
    return constant_field


def _transcendental_lower_remainder(
    reduction: (HyperexponentialMonomialReduction | HypertangentMonomialReduction),
) -> FieldElement | None:
    """Return the separated lower-field mode, even beside a top obstruction."""
    if isinstance(reduction, HyperexponentialMonomialReduction):
        if reduction.laurent_reduction is None:
            return None
        return reduction.laurent_reduction.lower_remainder
    if reduction.special_reduction is None:
        return None
    return reduction.special_reduction.lower_remainder


def _continue_past_finite_residue(
    reduction: (HyperexponentialMonomialReduction | HypertangentMonomialReduction),
    derivation: Derivation,
) -> HyperexponentialMonomialReduction | HypertangentMonomialReduction | None:
    r"""Reduce the part disjoint from a proved nonconstant normal residue."""
    if isinstance(reduction, HyperexponentialMonomialReduction):
        has_finite_obstruction = reduction.has_finite_residue_obstruction
    else:
        has_finite_obstruction = (
            reduction.common_reduction.has_finite_residue_obstruction
        )
    if not has_finite_obstruction:
        return None

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("transcendental exact reduction requires a monomial level")
    normal_numerator, normal_denominator = _normal_fraction_part(
        reduction.remainder,
        extension,
        derivation,
    )
    normal_obstruction = _coerce(
        extension.field,
        normal_numerator.as_expr() / normal_denominator.as_expr(),
    )
    residual = reduction.remainder - normal_obstruction
    if residual == extension.field.zero:
        return None

    if isinstance(reduction, HyperexponentialMonomialReduction):
        residual_reduction = reduce_hyperexponential_monomial(
            residual,
            derivation,
        )
        if residual_reduction.has_finite_residue_obstruction:
            raise ValueError(
                "normal-fraction separation left a hyperexponential residue"
            )
    else:
        residual_reduction = reduce_hypertangent_monomial(
            residual,
            derivation,
        )
        if residual_reduction.common_reduction.has_finite_residue_obstruction:
            raise ValueError("normal-fraction separation left a hypertangent residue")
    return residual_reduction


def _finish_transcendental_exact_derivative(
    element: FieldElement,
    derivation: Derivation,
    reduction: (HyperexponentialMonomialReduction | HypertangentMonomialReduction),
) -> ExactDerivativeResult:
    r"""Combine one family reduction with exact solving in its lower field."""
    field = derivation.field
    exact_part = reduction.exact_part
    obstruction = reduction.logarithmic_derivative + reduction.remainder
    residual_reduction = _continue_past_finite_residue(
        reduction,
        derivation,
    )
    active_reduction = reduction
    if residual_reduction is not None:
        active_reduction = residual_reduction
        exact_part += residual_reduction.exact_part
        obstruction -= derivation(residual_reduction.exact_part)

    lower_result = None
    lower_remainder = _transcendental_lower_remainder(active_reduction)
    if lower_remainder is not None:
        lower_derivation = derivation.lower()
        lower_result = solve_exact_derivative(
            lower_remainder,
            lower_derivation,
        )
        exact_part += _coerce(field, lower_result.exact_part)
        obstruction -= _coerce(
            field,
            lower_derivation(lower_result.exact_part),
        )

    if derivation(exact_part) + obstruction != element:
        raise ValueError("transcendental exact-derivative reconstruction failed")
    return ExactDerivativeResult(
        exact_part,
        obstruction,
        TranscendentalExactDerivativeCertificate(
            reduction,
            residual_reduction,
            lower_result,
        ),
    )


def _solve_hypertangent_exact_derivative_via_rde(
    element: FieldElement,
    derivation: Derivation,
) -> ExactDerivativeResult:
    """Use the complete scalar RDE fallback for a mixed tangent lower field."""
    from .rde import solve_risch_differential_equation

    result = solve_risch_differential_equation(
        derivation.field.zero,
        element,
        derivation,
    )
    certificate = HypertangentGaussianExactDerivativeCertificate(result)
    if result.solution is None:
        return ExactDerivativeResult(
            derivation.field.zero,
            element,
            certificate,
        )
    return ExactDerivativeResult(
        result.solution,
        derivation.field.zero,
        certificate,
    )


def solve_exact_derivative(
    integrand: object,
    derivation: Derivation,
) -> ExactDerivativeResult:
    r"""Dispatch the exact-derivative problem at an implemented tower level.

    Results are normalized into ``derivation.field`` even though the rational
    base solver has an expression API and the primitive solver works directly
    with native fraction-field elements.
    """
    field = derivation.field
    element = _coerce(field, integrand)

    if derivation.level == -1:
        return ExactDerivativeResult(field.zero, element, None)

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "exact-derivative dispatch does not yet support algebraic levels"
        )

    constant_field = _standard_rational_constant_field(extension)
    if constant_field is not None:
        certificate = _solve_rational_exact_derivative_over_domain(
            field.to_sympy(element),
            extension.generator,
            constant_field,
        )
    elif derivation.level > 0 and extension.family is MonomialFamily.PRIMITIVE:
        certificate = solve_primitive_exact_derivative(
            element,
            derivation,
        )
    elif extension.family is MonomialFamily.HYPEREXPONENTIAL:
        return _finish_transcendental_exact_derivative(
            element,
            derivation,
            reduce_hyperexponential_monomial(element, derivation),
        )
    elif extension.family is MonomialFamily.HYPERTANGENT:
        try:
            reduction = reduce_hypertangent_monomial(element, derivation)
        except NotImplementedError:
            return _solve_hypertangent_exact_derivative_via_rde(
                element,
                derivation,
            )
        return _finish_transcendental_exact_derivative(
            element,
            derivation,
            reduction,
        )
    else:
        raise NotImplementedError(
            "exact-derivative dispatch currently supports rational, "
            "primitive, hyperexponential, and hypertangent levels"
        )

    return ExactDerivativeResult(
        _coerce(field, certificate.exact_part),
        _coerce(field, certificate.obstruction),
        certificate,
    )
