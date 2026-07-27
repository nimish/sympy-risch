r"""Recursive primitive parametric logarithmic-derivative decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy.core.symbol import Dummy
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..tower.core import Derivation, MonomialExtension, MonomialFamily, _coerce
from ._common import (
    _constant_field_linear_ratio,
    _parametric_logarithmic_derivative_eta,
)
from .models import (
    ParametricLogarithmicDerivativeOutcome,
    ParametricLogarithmicDerivativeResult,
    PrimitiveParametricLogarithmicDerivativeCertificate,
    PrimitiveParametricLogarithmicDerivativeFactor,
)

if TYPE_CHECKING:
    from ..monomial import PrimitiveLogarithmicDerivativeSignature


def _primitive_parametric_signature_constraints(
    f_signature: PrimitiveLogarithmicDerivativeSignature,
    eta_signature: PrimitiveLogarithmicDerivativeSignature,
    derivation: Derivation,
) -> tuple[
    tuple[tuple[FieldElement, FieldElement], ...],
    tuple[PrimitiveParametricLogarithmicDerivativeFactor, ...],
    tuple[tuple[FieldElement, FieldElement], ...],
]:
    r"""Align two primitive signatures and project away every ``QQ*Dp``."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("primitive parametric constraints require a monomial level")
    parent = extension.parent_field
    f_polynomial = f_signature.polynomial_part
    eta_polynomial = eta_signature.polynomial_part
    maximum_degree = max(
        (polynomial.degree() if not polynomial.is_zero else 0)
        for polynomial in (f_polynomial, eta_polynomial)
    )
    polynomial_constraints = tuple(
        (
            parent.convert(f_polynomial.nth(degree)),
            parent.convert(eta_polynomial.nth(degree)),
        )
        for degree in range(1, int(maximum_degree) + 1)
    )

    f_by_expression = {
        factor.factor.as_expr(): factor for factor in f_signature.factors
    }
    eta_by_expression = {
        factor.factor.as_expr(): factor for factor in eta_signature.factors
    }
    factor_expressions = list(f_by_expression)
    factor_expressions.extend(
        expression
        for expression in eta_by_expression
        if expression not in f_by_expression
    )

    factor_certificates = []
    all_constraints = list(polynomial_constraints)
    lower_derivation = derivation.lower()
    for expression in factor_expressions:
        f_factor = f_by_expression.get(expression)
        eta_factor = eta_by_expression.get(expression)
        if f_factor is None and eta_factor is None:
            raise ValueError("a collected factor has no source signature")
        if f_factor is not None:
            factor = f_factor.factor
        else:
            assert eta_factor is not None
            factor = eta_factor.factor
        maximum_multiplicity = max(
            f_factor.multiplicity if f_factor is not None else 0,
            eta_factor.multiplicity if eta_factor is not None else 0,
        )

        if f_factor is None:
            f_principal_numerators = (factor.zero,) * maximum_multiplicity
        else:
            f_principal_numerators = f_factor.principal_numerators + (factor.zero,) * (
                maximum_multiplicity - len(f_factor.principal_numerators)
            )
        if eta_factor is None:
            eta_principal_numerators = (factor.zero,) * maximum_multiplicity
        else:
            eta_principal_numerators = eta_factor.principal_numerators + (
                factor.zero,
            ) * (maximum_multiplicity - len(eta_factor.principal_numerators))
        factor_derivative = derivation.polynomial(factor).rem(factor)
        if factor_derivative.is_zero:
            raise ValueError(
                "a primitive extension with preserved constants cannot "
                "have a nonconstant special factor"
            )

        factor_constraints = []
        for pole_order in range(2, maximum_multiplicity + 1):
            f_numerator = f_principal_numerators[pole_order - 1]
            eta_numerator = eta_principal_numerators[pole_order - 1]
            factor_constraints.extend(
                (
                    parent.convert(f_numerator.nth(degree)),
                    parent.convert(eta_numerator.nth(degree)),
                )
                for degree in range(factor.degree())
            )

        f_simple = f_principal_numerators[0]
        eta_simple = eta_principal_numerators[0]
        (pivot_degree,), pivot_value = factor_derivative.terms()[0]
        pivot_value = parent.convert(pivot_value)
        f_pivot = parent.convert(f_simple.nth(pivot_degree))
        eta_pivot = parent.convert(eta_simple.nth(pivot_degree))
        for degree in range(factor.degree()):
            if degree == pivot_degree:
                continue
            derivative_coefficient = parent.convert(factor_derivative.nth(degree))
            factor_constraints.append(
                (
                    parent.convert(f_simple.nth(degree)) * pivot_value
                    - f_pivot * derivative_coefficient,
                    parent.convert(eta_simple.nth(degree)) * pivot_value
                    - eta_pivot * derivative_coefficient,
                )
            )

        # The cross-products above make the residual numerator a K-multiple
        # of Dp. Its multiplier belongs to QQ exactly when its lower
        # derivative vanishes, under the same-constants hypothesis.
        factor_constraints.append(
            (
                lower_derivation(parent.exquo(f_pivot, pivot_value)),
                lower_derivation(parent.exquo(eta_pivot, pivot_value)),
            )
        )

        factor_constraints_tuple = tuple(factor_constraints)
        all_constraints.extend(factor_constraints_tuple)
        factor_certificates.append(
            PrimitiveParametricLogarithmicDerivativeFactor(
                factor,
                factor_derivative,
                f_principal_numerators,
                eta_principal_numerators,
                factor_constraints_tuple,
            )
        )

    return (
        polynomial_constraints,
        tuple(factor_certificates),
        tuple(all_constraints),
    )


def _solve_primitive_parametric_logarithmic_derivative(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Decide the parametric problem over a recursive primitive tower."""
    from .transcendental import solve_parametric_logarithmic_derivative

    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    field = lower_derivation.field
    element = _coerce(field, f)
    if lower_derivation.level < 1:
        raise ValueError(
            "primitive parametric solving requires a primitive lower field"
        )
    primitive_extension = lower_derivation.tower.levels[lower_derivation.level]
    if (
        not isinstance(primitive_extension, MonomialExtension)
        or primitive_extension.family is not MonomialFamily.PRIMITIVE
    ):
        raise ValueError(
            "primitive parametric solving requires a primitive lower field"
        )
    for level in range(1, lower_derivation.level + 1):
        recursive_extension = lower_derivation.tower.levels[level]
        if (
            not isinstance(recursive_extension, MonomialExtension)
            or recursive_extension.family is not MonomialFamily.PRIMITIVE
        ):
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
                detail=(
                    "recursive primitive parametric solving requires every "
                    "transcendental level above QQ(x) to be primitive"
                ),
            )

    from ..constant_proofs import first_nonpreserved_constant_level

    evidence = first_nonpreserved_constant_level(lower_derivation)
    if evidence is not None:
        suffix = f": {evidence.detail}" if evidence.detail else ""
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=(
                "primitive parametric solving requires Const(K(t)) = "
                "Const(K) at every primitive level"
                f"{suffix}"
            ),
        )

    # Local import avoids the module cycle: monomial family reducers use the
    # dependency analyses defined in this module.
    from ..monomial import primitive_logarithmic_derivative_signature

    try:
        f_signature = primitive_logarithmic_derivative_signature(
            element,
            lower_derivation,
        )
        eta_signature = primitive_logarithmic_derivative_signature(
            eta,
            lower_derivation,
        )
    except NotImplementedError as exc:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=str(exc),
        )

    (
        polynomial_constraints,
        factor_certificates,
        linear_constraints,
    ) = _primitive_parametric_signature_constraints(
        f_signature,
        eta_signature,
        lower_derivation,
    )
    consistent, candidate_ratio = _constant_field_linear_ratio(
        linear_constraints,
        primitive_extension.parent_field,
    )
    lower_parametric_result = None
    residual_signature = None

    def certificate(
        reason: str,
    ) -> PrimitiveParametricLogarithmicDerivativeCertificate:
        return PrimitiveParametricLogarithmicDerivativeCertificate(
            f_signature,
            eta_signature,
            polynomial_constraints,
            factor_certificates,
            linear_constraints,
            candidate_ratio,
            lower_parametric_result,
            residual_signature,
            reason,
        )

    if not consistent:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=certificate(
                "the primitive signature constraints on m/n are inconsistent"
            ),
        )

    if candidate_ratio is None:
        f_lower = primitive_extension.parent_field.convert(
            f_signature.polynomial_part.nth(0)
            if not f_signature.polynomial_part.is_zero
            else primitive_extension.parent_field.zero
        )
        eta_lower = primitive_extension.parent_field.convert(
            eta_signature.polynomial_part.nth(0)
            if not eta_signature.polynomial_part.is_zero
            else primitive_extension.parent_field.zero
        )
        if eta_lower == primitive_extension.parent_field.zero:
            candidate_ratio = QQ.zero
        else:
            lower_theta = Dummy("theta_lower")
            lower_extension = MonomialExtension(
                Poly(
                    primitive_extension.parent_field.to_sympy(eta_lower) * lower_theta,
                    lower_theta,
                    domain=primitive_extension.parent_field,
                )
            )
            lower_parametric_result = solve_parametric_logarithmic_derivative(
                f_lower,
                lower_extension,
                lower_derivation.lower(),
            )
            if lower_parametric_result.proves_no_solution:
                return ParametricLogarithmicDerivativeResult(
                    extension,
                    lower_derivation,
                    element,
                    ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
                    certificate=certificate(
                        "the top signature leaves m/n free, but the complete "
                        "recursive lower problem is inconsistent"
                    ),
                )
            if not lower_parametric_result.has_solution:
                return ParametricLogarithmicDerivativeResult(
                    extension,
                    lower_derivation,
                    element,
                    ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
                    certificate=certificate(
                        "the top signature leaves m/n free, but the lower "
                        "field family is unsupported"
                    ),
                    detail=lower_parametric_result.detail,
                )
            candidate_ratio = QQ(
                lower_parametric_result.m,
                lower_parametric_result.n,
            )

    ratio_numerator = int(QQ.numer(candidate_ratio))
    ratio_denominator = int(QQ.denom(candidate_ratio))
    integer_residual = ratio_denominator * element - ratio_numerator * eta
    residual_signature = primitive_logarithmic_derivative_signature(
        integer_residual,
        lower_derivation,
    )
    if not residual_signature.has_solution:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=certificate(
                "the selected primitive residual is not the logarithmic "
                "derivative of a radical"
            ),
        )

    logarithmic_power = residual_signature.minimal_power
    if logarithmic_power is None:
        raise ValueError("a solved radical signature lacks its minimal power")
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        element,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=logarithmic_power * ratio_denominator,
        m=logarithmic_power * ratio_numerator,
        argument=residual_signature.relation_argument,
        certificate=certificate(
            "the primitive signature constraints produced a verified witness"
        ),
    )
