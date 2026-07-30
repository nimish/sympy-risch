r"""Direct real hypertangent relations and Cayley reduction oracles."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.numbers import I
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..constant_fields import (
    finite_constant_field_profile,
    gaussian_constant_field,
)
from ..constant_proofs import require_preserved_constant_field
from ..dependency import (
    analyze_hypertangent_extension,
)
from ..gaussian import (
    HypertangentGaussianNormalForm,
    build_hypertangent_gaussian_normal_form,
)
from ..rde import (
    CoupledRationalRDEResult,
    CoupledRischDifferentialEquationResult,
    solve_coupled_rational_rde,
)
from ..rde.coupled import (
    _prepare_gaussian_coupled_transport,
    _solve_general_coupled_risch_differential_equation,
)
from ..relations import (
    LogarithmicRelationSystemResult,
    solve_logarithmic_relation_system,
)
from ..relations.linear import (
    minimal_source_logarithmic_relation,
)
from ..tower.core import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
)
from .logarithmic import RadicalLogarithmicDerivativeResult
from .nonlinear import NonlinearMonomialReduction, reduce_nonlinear_monomial
from .rational_base import _standard_rational_base_variable


@dataclass(frozen=True, slots=True)
class HypertangentCayleyStep:
    r"""One positive Cayley mode and its real coupled RDE certificate."""

    power: int
    real_coefficient: FieldElement
    imaginary_coefficient: FieldElement
    exact_real_coefficient: FieldElement | None
    exact_imaginary_coefficient: FieldElement | None
    rde_result: CoupledRationalRDEResult | CoupledRischDifferentialEquationResult

    @property
    def has_solution(self) -> bool:
        """Whether this conjugate pair is an exact derivative."""
        return self.rde_result.has_solution


@dataclass(frozen=True, slots=True)
class HypertangentSpecialReduction:
    r"""Real paired-mode reduction at the special place ``1 + t**2``."""

    exact_part: FieldElement
    remainder: FieldElement
    lower_remainder: FieldElement
    steps: tuple[HypertangentCayleyStep, ...]

    @property
    def obstructed_steps(self) -> tuple[HypertangentCayleyStep, ...]:
        """Return all Cayley pairs with unsolvable coupled equations."""
        return tuple(step for step in self.steps if not step.has_solution)

    @property
    def has_obstruction(self) -> bool:
        """Whether some nonzero Cayley pair remains reduced."""
        return bool(self.obstructed_steps)

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only the Cayley power-zero coefficient remains."""
        return not self.has_obstruction


@dataclass(frozen=True, slots=True)
class HypertangentMonomialReduction:
    """Complete real reduction at one supported hypertangent level."""

    common_reduction: NonlinearMonomialReduction
    special_reduction: HypertangentSpecialReduction | None
    exact_part: FieldElement
    logarithmic_derivative: FieldElement
    remainder: FieldElement
    lower_remainder: FieldElement | None

    @property
    def has_common_obstruction(self) -> bool:
        """Whether a normal residue or infinity residue is nonconstant."""
        return self.common_reduction.has_obstruction

    @property
    def has_special_obstruction(self) -> bool:
        """Whether a Cayley pair has no rational coupled solution."""
        return (
            self.special_reduction is not None
            and self.special_reduction.has_obstruction
        )

    @property
    def is_non_elementary(self) -> bool:
        """Whether this level supplies a Liouville obstruction."""
        return self.has_common_obstruction or self.has_special_obstruction

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only elementary rational integration remains."""
        return not self.is_non_elementary and self.lower_remainder is not None


@dataclass(frozen=True, slots=True)
class HypertangentLogarithmicDerivativeSignature:
    r"""Direct real radical-logarithmic decision at a tangent level."""

    derivation: Derivation
    integrand: FieldElement
    relation_system: LogarithmicRelationSystemResult
    minimal_power: int | None
    relation_argument: FieldElement | None

    @property
    def has_solution(self) -> bool:
        """Whether a positive multiple is a real logarithmic derivative."""
        return self.minimal_power is not None

    @property
    def reconstructs_relation(self) -> bool:
        """Whether the recovered real witness satisfies the input identity."""
        if not self.has_solution or self.relation_argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.relation_argument)
        return (
            self.minimal_power * _coerce(field, self.integrand)
            == self.derivation(argument) / argument
        )


@dataclass(frozen=True, slots=True)
class HypertangentGaussianLogarithmicDerivativeOracle:
    """Independent Cayley/Gaussian result for a real tangent query."""

    derivation: Derivation
    integrand: FieldElement
    normal_form: HypertangentGaussianNormalForm
    gaussian_result: RadicalLogarithmicDerivativeResult
    minimal_power: int | None
    relation_argument: FieldElement | None

    @property
    def has_solution(self) -> bool:
        """Whether a positive multiple is a real logarithmic derivative."""
        return self.minimal_power is not None

    @property
    def reconstructs_relation(self) -> bool:
        """Whether the recovered real witness satisfies the input identity."""
        if not self.has_solution or self.relation_argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.relation_argument)
        return (
            self.minimal_power * _coerce(field, self.integrand)
            == self.derivation(argument) / argument
        )


def hypertangent_logarithmic_derivative_signature(
    integrand: object,
    derivation: Derivation,
) -> HypertangentLogarithmicDerivativeSignature:
    r"""Decide a real radical logarithmic derivative by Chapter-9 relations."""
    if not isinstance(derivation, Derivation):
        raise TypeError("hypertangent logarithmic signatures require a Derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "hypertangent logarithmic signatures require a level above the "
            "rational base"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("hypertangent logarithmic signatures require a monomial level")
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError(
            "hypertangent logarithmic signatures require Dt = eta*(1 + t**2)"
        )
    field = extension.field
    element = _coerce(field, integrand)
    relation_system = solve_logarithmic_relation_system(
        (element,),
        derivation,
    )
    relation = minimal_source_logarithmic_relation(relation_system)
    signature = HypertangentLogarithmicDerivativeSignature(
        derivation,
        element,
        relation_system,
        relation.source_coefficients[0] if relation is not None else None,
        relation.argument if relation is not None else None,
    )
    if signature.has_solution and not signature.reconstructs_relation:
        raise ValueError(
            "hypertangent logarithmic signature witness did not reconstruct"
        )
    return signature


def hypertangent_logarithmic_derivative_signature_gaussian(
    integrand: object,
    derivation: Derivation,
) -> HypertangentGaussianLogarithmicDerivativeOracle:
    r"""Independently decide the same real query through Cayley normalization."""
    if not isinstance(derivation, Derivation):
        raise TypeError("hypertangent logarithmic signatures require a Derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "hypertangent logarithmic signatures require a level above the "
            "rational base"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("hypertangent logarithmic signatures require a monomial level")
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError(
            "hypertangent logarithmic signatures require Dt = eta*(1 + t**2)"
        )
    field = extension.field
    element = _coerce(field, integrand)

    from .logarithmic import solve_radical_logarithmic_derivative

    normal_form = build_hypertangent_gaussian_normal_form(derivation)
    gaussian_result = solve_radical_logarithmic_derivative(
        normal_form.forward(element),
        normal_form.target_derivation,
    )
    minimal_power = gaussian_result.minimal_power
    relation_argument = None
    if minimal_power is not None:
        if gaussian_result.argument is None:
            raise ValueError("a solved Gaussian logarithmic result lacks an argument")
        real_part, imaginary_part = normal_form.split_inverse(
            gaussian_result.argument,
        )
        relation_argument = real_part if real_part != field.zero else imaginary_part
        if relation_argument == field.zero:
            raise ValueError(
                "a nonzero Gaussian logarithmic witness has no real coordinate"
            )

    signature = HypertangentGaussianLogarithmicDerivativeOracle(
        derivation,
        element,
        normal_form,
        gaussian_result,
        minimal_power,
        relation_argument,
    )
    if signature.has_solution and not signature.reconstructs_relation:
        raise ValueError(
            "hypertangent logarithmic signature witness did not reconstruct"
        )
    return signature


def _validate_hypertangent_reduction(
    derivation: Derivation,
) -> tuple[MonomialExtension, FieldElement]:
    """Validate ``Dt = eta*(1+t**2)`` over a supported lower tower."""
    if derivation.level < 1:
        raise TypeError("hypertangent reduction requires a nonconstant lower field")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("hypertangent reduction requires a monomial level")
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError("hypertangent reduction requires Dt = eta*(1 + t**2)")
    require_preserved_constant_field(derivation, "hypertangent reduction")

    lower_derivation = derivation.lower()
    _standard_rational_base_variable(Derivation(derivation.tower, 0))
    profile = finite_constant_field_profile(derivation.tower.constants)
    if profile is None or not profile.is_real:
        raise NotImplementedError(
            "real hypertangent reduction requires a provably real finite "
            "algebraic constant field"
        )
    for level in range(1, lower_derivation.level + 1):
        lower_extension = derivation.tower.levels[level]
        if not isinstance(
            lower_extension, MonomialExtension
        ) or lower_extension.family not in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            raise NotImplementedError(
                "hypertangent reduction currently supports primitive, "
                "hyperexponential, and hypertangent levels above C(x)"
            )
        require_preserved_constant_field(
            Derivation(derivation.tower, level),
            "hypertangent reduction",
        )
    dependency = analyze_hypertangent_extension(
        extension,
        lower_derivation,
    )
    if dependency.is_dependent:
        raise ValueError(
            "hypertangent reduction requires a transcendental generator; "
            "the Cayley generator is algebraically dependent"
        )
    eta = extension.parent_field.convert(extension.generator_derivative.nth(2))
    if eta == extension.parent_field.zero:
        raise ValueError("the hypertangent coefficient eta must be nonzero")
    return extension, eta


def _gaussian_conjugate(
    coefficient: FieldElement,
    gaussian_field: Domain,
) -> FieldElement:
    """Conjugate a member of ``K(i)`` while fixing all tower generators."""
    expression = gaussian_field.to_sympy(coefficient)
    return gaussian_field.from_sympy(expression.xreplace({I: -I}))


def _hypertangent_cayley_coefficients(
    element: FieldElement,
    extension: MonomialExtension,
    constants: Domain,
) -> tuple[Domain, dict[int, FieldElement]]:
    r"""Return Laurent coefficients after ``u=(1+i*t)/(1-i*t)``.

    The special irreducible ``1+t**2`` becomes a monomial in ``u``.  Thus an
    element whose only non-lower-field poles are special becomes a Laurent
    polynomial over ``K(i)``. This is the Cayley-coordinate form of the
    hypertangent reduction in Bronstein, *Symbolic Integration I*, Chapter 5.
    """
    generator = extension.generator
    cayley_generator = Dummy("u")
    parent = extension.parent_field
    parent_symbols = tuple(getattr(parent, "symbols", ()) or ())
    if getattr(parent, "domain", None) != constants or not parent_symbols:
        raise NotImplementedError(
            "hypertangent Cayley reduction requires a rational-function "
            "lower field over its declared finite constant field"
        )
    gaussian_field = gaussian_constant_field(constants).frac_field(*parent_symbols)
    inverse_cayley = I * (1 - cayley_generator) / (1 + cayley_generator)
    transformed = cancel(
        extension.field.to_sympy(element).subs(generator, inverse_cayley)
    )
    numerator_expression, denominator_expression = transformed.as_numer_denom()
    numerator = Poly(
        numerator_expression,
        cayley_generator,
        domain=gaussian_field,
    )
    denominator = Poly(
        denominator_expression,
        cayley_generator,
        domain=gaussian_field,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(coefficient)
    denominator_terms = denominator.terms()
    if len(denominator_terms) != 1:
        raise ValueError(
            "hypertangent special reduction requires a Laurent polynomial "
            "in the Cayley generator"
        )

    (denominator_power,), denominator_coefficient = denominator_terms[0]
    coefficient_scale = gaussian_field.one / gaussian_field.convert(
        denominator_coefficient
    )
    coefficients = {
        numerator_power - denominator_power: (
            gaussian_field.convert(term_coefficient) * coefficient_scale
        )
        for (numerator_power,), term_coefficient in numerator.terms()
    }
    return gaussian_field, coefficients


def _reduce_hypertangent_special_validated(
    element: FieldElement,
    derivation: Derivation,
    extension: MonomialExtension,
    eta: FieldElement,
) -> HypertangentSpecialReduction:
    """Reduce validated special modes without repeating tower checks."""
    field = extension.field
    parent = extension.parent_field
    generator = extension.generator
    gaussian_field, coefficients = _hypertangent_cayley_coefficients(
        element,
        extension,
        derivation.tower.constants,
    )
    zero = gaussian_field.zero

    # A real rational function of t has b[-n] = conjugate(b[n]) in Cayley
    # coordinates because real conjugation sends u to u**-1.
    powers = set(coefficients)
    for power in (candidate for candidate in powers if candidate != 0):
        if (
            coefficients.get(-power, zero)
            - _gaussian_conjugate(
                coefficients[power],
                gaussian_field,
            )
            != zero
        ):
            raise ValueError("Cayley coefficients do not form real conjugate pairs")

    lower_coefficient = coefficients.get(0, zero)
    if (
        lower_coefficient - _gaussian_conjugate(lower_coefficient, gaussian_field)
        != zero
    ):
        raise ValueError("the Cayley power-zero coefficient is not real")
    lower_remainder = _coerce(
        parent,
        gaussian_field.to_sympy(lower_coefficient),
    )

    cayley_expression = (1 + I * generator) / (1 - I * generator)
    exact_part = field.zero
    steps: list[HypertangentCayleyStep] = []
    positive_powers = sorted(candidate for candidate in powers if candidate > 0)
    lower_derivation = derivation.lower()
    use_specialized_rational_solver = (
        lower_derivation.level == 0 and derivation.tower.constants == QQ
    )
    gaussian_transport = (
        None
        if use_specialized_rational_solver or not positive_powers
        else _prepare_gaussian_coupled_transport(lower_derivation)
    )
    for power in positive_powers:
        coefficient = coefficients[power]
        conjugate_coefficient = _gaussian_conjugate(
            coefficient,
            gaussian_field,
        )
        real_coefficient = _coerce(
            parent,
            cancel(
                (
                    gaussian_field.to_sympy(coefficient)
                    + gaussian_field.to_sympy(conjugate_coefficient)
                )
                / 2
            ),
        )
        imaginary_coefficient = _coerce(
            parent,
            cancel(
                (
                    gaussian_field.to_sympy(coefficient)
                    - gaussian_field.to_sympy(conjugate_coefficient)
                )
                / (2 * I)
            ),
        )
        if use_specialized_rational_solver:
            variable = _standard_rational_base_variable(lower_derivation)
            rde_result = solve_coupled_rational_rde(
                2 * power * parent.to_sympy(eta),
                parent.to_sympy(real_coefficient),
                parent.to_sympy(imaginary_coefficient),
                variable,
            )
        else:
            rde_result = _solve_general_coupled_risch_differential_equation(
                parent.zero,
                parent.convert(2 * power) * eta,
                real_coefficient,
                imaginary_coefficient,
                lower_derivation,
                gaussian_transport=gaussian_transport,
            )
        exact_real_coefficient = None
        exact_imaginary_coefficient = None
        if rde_result.has_solution:
            exact_real_coefficient = _coerce(
                parent,
                rde_result.real_solution,
            )
            exact_imaginary_coefficient = _coerce(
                parent,
                rde_result.imaginary_solution,
            )
            real_expression = parent.to_sympy(exact_real_coefficient)
            imaginary_expression = parent.to_sympy(exact_imaginary_coefficient)
            paired_exact_expression = cancel(
                (real_expression + I * imaginary_expression) * cayley_expression**power
                + (real_expression - I * imaginary_expression)
                * cayley_expression ** (-power)
            )
            exact_part += _coerce(field, paired_exact_expression)

        steps.append(
            HypertangentCayleyStep(
                power,
                real_coefficient,
                imaginary_coefficient,
                exact_real_coefficient,
                exact_imaginary_coefficient,
                rde_result,
            )
        )

    remainder = element - derivation(exact_part)
    if derivation(exact_part) + remainder != element:
        raise ValueError("hypertangent special reduction reconstruction failed")
    if not any(not step.has_solution for step in steps) and remainder != _coerce(
        field, lower_remainder
    ):
        raise ValueError("solved hypertangent modes did not reduce to the lower field")
    return HypertangentSpecialReduction(
        exact_part,
        remainder,
        lower_remainder,
        tuple(steps),
    )


def reduce_hypertangent_special(
    integrand: object,
    derivation: Derivation,
) -> HypertangentSpecialReduction:
    r"""Reduce the special part of a rational-coefficient tangent extension.

    For ``Dt = eta*(1+t**2)``, introduce

    ``u = (1 + i*t)/(1 - i*t)``, so ``Du/u = 2*i*eta``.

    Real inputs have paired Laurent coefficients.  A positive mode
    ``(a + i*b)*u**n`` and its conjugate are exact precisely when the real
    coupled lower-field system

    ``Dp - 2*n*eta*q = a`` and ``Dq + 2*n*eta*p = b``

    has a solution. Pairing before reconstruction keeps the public result in
    the real field ``K(t)``.
    """
    extension, eta = _validate_hypertangent_reduction(derivation)
    element = _coerce(extension.field, integrand)
    return _reduce_hypertangent_special_validated(
        element,
        derivation,
        extension,
        eta,
    )


def reduce_hypertangent_monomial(
    integrand: object,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> HypertangentMonomialReduction:
    r"""Apply the complete reduction for ``Dt=eta*(1+t**2)``.

    Common nonlinear Hermite, normal-residue, polynomial, and infinity
    reductions run first.  If they leave no Liouville obstruction, the
    remaining powers of the special polynomial are reduced through conjugate
    Cayley modes and real coupled Risch equations over the supported lower
    tower.
    """
    extension, eta = _validate_hypertangent_reduction(derivation)
    field = extension.field
    element = _coerce(field, integrand)
    common_reduction = reduce_nonlinear_monomial(
        element,
        derivation,
        residue=residue,
    )

    special_reduction = None
    exact_part = common_reduction.exact_part
    remainder = common_reduction.remainder
    lower_remainder = None
    if (
        not common_reduction.has_obstruction
        and not common_reduction.needs_special_polynomial
    ):
        special_reduction = _reduce_hypertangent_special_validated(
            common_reduction.remainder,
            derivation,
            extension,
            eta,
        )
        exact_part += special_reduction.exact_part
        remainder = special_reduction.remainder
        if not special_reduction.has_obstruction:
            lower_remainder = special_reduction.lower_remainder

    if (
        derivation(exact_part) + common_reduction.logarithmic_derivative + remainder
        != element
    ):
        raise ValueError("hypertangent monomial reduction reconstruction failed")
    return HypertangentMonomialReduction(
        common_reduction,
        special_reduction,
        exact_part,
        common_reduction.logarithmic_derivative,
        remainder,
        lower_remainder,
    )
