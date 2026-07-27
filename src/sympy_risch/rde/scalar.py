r"""Admission and recursive dispatch for scalar Risch differential equations."""

from __future__ import annotations

from sympy.polys.domains import QQ_I
from sympy.polys.polytools import Poly, cancel

from ..admission import assess_scalar_rde_admission
from ..tower import Derivation, MonomialExtension, MonomialFamily, _coerce
from ._tower_polynomials import (
    _is_standard_rational_derivation,
    _is_standard_rational_derivation_over,
)
from .constant_field import _solve_rde_over_constant_field
from .hyperexponential import (
    _hyperexponential_laurent_coefficients,
    solve_hyperexponential_laurent_risch_differential_equation,
    solve_hyperexponential_risch_differential_equation,
)
from .models import (
    HypertangentGaussianRischDifferentialEquationResult,
    RationalRischDifferentialEquationResult,
    RischDifferentialEquationResult,
)
from .primitive import solve_primitive_risch_differential_equation
from .rational import solve_rational_risch_differential_equation


def solve_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> RischDifferentialEquationResult:
    r"""Dispatch ``Dy + a*y = b`` over an implemented differential level."""
    admission = assess_scalar_rde_admission(derivation)
    if not admission.accepted:
        if admission.reason is None:
            raise RuntimeError("unsupported scalar RDE admission omitted its reason")
        raise NotImplementedError(
            "scalar RDE dispatch is unsupported for this tower prefix: "
            f"{admission.reason.value}: {admission.detail}"
        )
    return _solve_admitted_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )


def _solve_admitted_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> RischDifferentialEquationResult:
    """Solve after admission, including exact certified coordinate changes."""
    field = derivation.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )

    if _is_standard_rational_derivation(derivation):
        extension = derivation.tower.levels[0]
        certificate = solve_rational_risch_differential_equation(
            field.to_sympy(coefficient_element),
            field.to_sympy(rhs_element),
            extension.generator,
        )
        solution = (
            _coerce(field, certificate.solution) if certificate.has_solution else None
        )
    elif _is_standard_rational_derivation_over(derivation, QQ_I):
        extension = derivation.tower.levels[0]
        domain_result = _solve_rde_over_constant_field(
            field.to_sympy(coefficient_element),
            field.to_sympy(rhs_element),
            extension.generator,
            QQ_I,
        )
        normalized_rhs = cancel(
            domain_result.normalization_factor.as_expr() * field.to_sympy(rhs_element)
        )
        certificate = RationalRischDifferentialEquationResult(
            field.to_sympy(coefficient_element),
            field.to_sympy(rhs_element),
            domain_result.solution,
            domain_result.obstruction,
            domain_result.normalization_factor,
            domain_result.normalized_coefficient,
            normalized_rhs,
            domain_result.denominator_bound,
            domain_result.degree_bound,
            domain_result.polynomial_result,
            obstruction_polynomial=(domain_result.obstruction_polynomial),
            obstruction_remainder=(domain_result.obstruction_remainder),
        )
        solution = (
            _coerce(field, domain_result.solution)
            if domain_result.solution is not None
            else None
        )
    else:
        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "RDE dispatch does not yet support algebraic tower levels"
            )
        if extension.family is MonomialFamily.PRIMITIVE:
            certificate = solve_primitive_risch_differential_equation(
                coefficient_element,
                rhs_element,
                derivation,
            )
            solution = certificate.solution
        elif extension.family is MonomialFamily.HYPEREXPONENTIAL:
            try:
                _hyperexponential_laurent_coefficients(
                    coefficient_element,
                    extension,
                    "coefficient",
                )
                _hyperexponential_laurent_coefficients(
                    rhs_element,
                    extension,
                    "right-hand side",
                )
            except NotImplementedError:
                certificate = solve_hyperexponential_risch_differential_equation(
                    coefficient_element,
                    rhs_element,
                    derivation,
                )
            else:
                certificate = (
                    solve_hyperexponential_laurent_risch_differential_equation(
                        coefficient_element,
                        rhs_element,
                        derivation,
                    )
                )
            solution = certificate.solution
        elif extension.family is MonomialFamily.HYPERTANGENT:
            from ..gaussian import build_hypertangent_gaussian_normal_form

            normal_form = build_hypertangent_gaussian_normal_form(
                derivation,
            )
            target_result = _solve_admitted_risch_differential_equation(
                normal_form.forward(coefficient_element),
                normal_form.forward(rhs_element),
                normal_form.target_derivation,
            )
            certificate = HypertangentGaussianRischDifferentialEquationResult(
                normal_form,
                target_result,
            )
            if target_result.solution is None:
                solution = None
            else:
                real_solution, _ = normal_form.split_inverse(
                    target_result.solution,
                )
                solution = real_solution
        else:
            raise NotImplementedError(
                f"RDE dispatch does not yet support "
                f"{extension.family.value} monomial levels"
            )

    result = RischDifferentialEquationResult(
        derivation,
        coefficient_element,
        rhs_element,
        solution,
        certificate.obstruction,
        certificate,
    )
    if result.has_solution and not result.reconstructs:
        raise ValueError("recursive RDE dispatch reconstruction failed")
    return result
