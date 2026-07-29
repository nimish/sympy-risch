r"""Hyperexponential logarithmic signatures and Laurent reductions."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..constant_proofs import require_preserved_constant_field
from ..dependency import (
    ParametricLogarithmicDerivativeResult,
    analyze_hyperexponential_extension,
)
from ..rde import (
    RischDifferentialEquationResult,
    solve_risch_differential_equation,
)
from ..reduction.fractions import (
    _polynomial_and_proper_fraction,
    _reduced_polynomial_fraction,
    split_normal_special,
)
from ..reduction.polynomials import exact_field_equal
from ..tower.core import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
)
from .hermite import hermite_reduce_monomial
from .principal_parts import (
    _factor_principal_numerators,
    _qq_constant_polynomial_multiple,
)
from .residues import BronsteinResidueReduction, residue_reduce_monomial


@dataclass(frozen=True, slots=True)
class HyperexponentialLogarithmicDerivativeFactor:
    r"""Principal-part data for a normal factor in ``K(t)``, ``Dt=eta*t``.

    A radical logarithmic derivative permits only a simple pole at a normal
    factor, with a rational multiple of ``Dp`` as its numerator.  The special
    generator ``t`` is handled separately: its logarithmic derivative is the
    lower-field element ``eta`` rather than a pole in ``t``.
    """

    factor: Poly
    multiplicity: int
    principal_numerators: tuple[Poly, ...]
    factor_derivative: Poly
    logarithmic_multiplicity: FieldElement | None

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether a nonzero pole of order above one remains."""
        return any(not numerator.is_zero for numerator in self.principal_numerators[1:])

    @property
    def has_logarithmic_residue(self) -> bool:
        """Whether the simple numerator is a rational multiple of ``Dp``."""
        return self.logarithmic_multiplicity is not None


@dataclass(frozen=True, slots=True)
class HyperexponentialLogarithmicDerivativeSignature:
    r"""Radical-logarithmic data for one hyperexponential level.

    For a genuine hyperexponential monomial ``Dt=eta*t``, a witness has the
    form ``v0*t**e*prod(p**e_p)``.  Normal factors ``p`` contribute ordinary
    logarithmic principal parts.  Their full logarithmic derivatives can also
    have a lower-field polynomial quotient, so ``lower_component`` is the
    constant coordinate *after* those normal logarithmic contributions have
    been removed.  It descends to the lower parametric problem, which
    determines ``v0`` and the exponent ``e`` of the special generator.
    """

    derivation: Derivation
    integrand: FieldElement
    polynomial_part: Poly
    proper_numerator: Poly
    proper_denominator: Poly
    normal_denominator: Poly
    special_denominator: Poly
    lower_component: FieldElement
    lower_parametric_result: ParametricLogarithmicDerivativeResult | None
    factors: tuple[HyperexponentialLogarithmicDerivativeFactor, ...]
    minimal_power: int | None
    relation_argument: FieldElement | None

    @property
    def has_positive_degree_polynomial_obstruction(self) -> bool:
        """Whether the polynomial part has a nonconstant ``t`` component."""
        return not self.polynomial_part.is_zero and self.polynomial_part.degree() > 0

    @property
    def has_special_pole_obstruction(self) -> bool:
        """Whether a pole at the special generator remains."""
        return self.special_denominator.degree() > 0

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether any normal factor has a forbidden repeated pole."""
        return any(factor.has_repeated_pole_obstruction for factor in self.factors)

    @property
    def has_nonlogarithmic_residue_obstruction(self) -> bool:
        """Whether a normal simple residue is not a rational logarithm."""
        return any(not factor.has_logarithmic_residue for factor in self.factors)

    @property
    def has_lower_field_obstruction(self) -> bool:
        """Whether the lower parametric decision did not produce a witness."""
        return (
            self.lower_parametric_result is None
            or not self.lower_parametric_result.has_solution
        )

    @property
    def has_solution(self) -> bool:
        """Whether a positive multiple of the input is logarithmic."""
        return self.minimal_power is not None

    @property
    def reconstructs_decomposition(self) -> bool:
        """Whether the stored reduced fraction equals ``integrand``."""
        extension = self.derivation.tower.levels[self.derivation.level]
        if not isinstance(extension, MonomialExtension):
            return False
        expression = self.polynomial_part.as_expr()
        if not self.proper_numerator.is_zero:
            expression += (
                self.proper_numerator.as_expr() / self.proper_denominator.as_expr()
            )
        return exact_field_equal(
            _coerce(extension.field, expression),
            _coerce(extension.field, self.integrand),
            extension.field,
        )

    @property
    def reconstructs_relation(self) -> bool:
        """Whether the successful radical-logarithmic identity is exact."""
        if not self.has_solution or self.relation_argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.relation_argument)
        difference = (
            self.minimal_power * _coerce(field, self.integrand)
            - self.derivation(argument) / argument
        )
        return difference == field.zero or cancel(field.to_sympy(difference)) == 0


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentStep:
    r"""One nonzero Laurent mode reduced by a lower-field Risch equation.

    For ``Dt/t = eta`` and the input coefficient ``b`` at power ``n``, the
    nested result decides

    ``Dy + n*eta*y = b``.

    A solution contributes ``y*t**n`` to the exact part.  Otherwise the
    original ``b*t**n`` remains as a certified special-mode obstruction.
    """

    power: int
    coefficient: FieldElement
    exact_coefficient: FieldElement | None
    rde_result: RischDifferentialEquationResult

    @property
    def has_solution(self) -> bool:
        """Whether this Laurent mode is an exact derivative."""
        return self.rde_result.has_solution


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentReduction:
    r"""Mode-by-mode reduction of a Laurent polynomial in ``t``.

    The result satisfies

    ``integrand = D(exact_part) + remainder``.

    The power-zero coefficient belongs to the lower field and is returned as
    ``lower_remainder``.  Every nonzero mode is represented by one
    :class:`HyperexponentialLaurentStep`.
    """

    exact_part: FieldElement
    remainder: FieldElement
    lower_remainder: FieldElement
    steps: tuple[HyperexponentialLaurentStep, ...]

    @property
    def obstructed_steps(self) -> tuple[HyperexponentialLaurentStep, ...]:
        """Return all nonzero modes whose rational RDE has no solution."""
        return tuple(step for step in self.steps if not step.has_solution)

    @property
    def has_obstruction(self) -> bool:
        """Whether a nonzero Laurent mode cannot be integrated in the field."""
        return bool(self.obstructed_steps)

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only the power-zero lower-field coefficient remains."""
        return not self.has_obstruction


@dataclass(frozen=True, slots=True)
class HyperexponentialMonomialReduction:
    r"""Complete elementary reduction at one hyperexponential level.

    The flattened fields satisfy

    ``f = D(exact_part) + logarithmic_derivative + remainder``.

    Normal finite poles are handled by generalized residue reduction.  Once
    those residues are constant, the remaining special Laurent modes are
    handled by :class:`HyperexponentialLaurentReduction`.
    """

    hermite_exact_part: FieldElement
    residue_reduction: BronsteinResidueReduction
    laurent_reduction: HyperexponentialLaurentReduction | None
    exact_part: FieldElement
    logarithmic_derivative: FieldElement
    remainder: FieldElement
    lower_remainder: FieldElement | None

    @property
    def has_finite_residue_obstruction(self) -> bool:
        """Whether a nonconstant normal residue proves non-elementarity."""
        return not self.residue_reduction.all_normal_residues_removed

    @property
    def has_laurent_obstruction(self) -> bool:
        """Whether a special Laurent mode has an unsolvable rational RDE."""
        return (
            self.laurent_reduction is not None
            and self.laurent_reduction.has_obstruction
        )

    @property
    def is_non_elementary(self) -> bool:
        """Whether this level supplies a Liouville obstruction."""
        return self.has_finite_residue_obstruction or self.has_laurent_obstruction

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only elementary integration in ``QQ(x)`` remains."""
        return not self.is_non_elementary and self.lower_remainder is not None


def hyperexponential_logarithmic_derivative_signature(
    integrand: object,
    derivation: Derivation,
) -> HyperexponentialLogarithmicDerivativeSignature:
    r"""Compute a radical-logarithmic signature at ``Dt=eta*t``.

    Bronstein's hyperexponential descent separates normal logarithmic factors
    from the special generator.  Once normal factors have been checked, the
    remaining constant-in-``t`` coordinate is solved by the lower
    parametric problem with direction ``eta=Dt/t``.  This constructs the
    special ``t`` exponent in the witness rather than treating a ``t`` pole
    as an ordinary logarithmic factor.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("hyperexponential logarithmic signatures require a Derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "hyperexponential logarithmic signatures require a level above "
            "the rational base"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "hyperexponential logarithmic signatures require a monomial level"
        )
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError("hyperexponential logarithmic signatures require Dt = eta*t")
    require_preserved_constant_field(
        derivation,
        "hyperexponential logarithmic signatures",
    )

    field = extension.field
    parent = extension.parent_field
    element = _coerce(field, integrand)
    polynomial_part, proper_numerator, proper_denominator = (
        _polynomial_and_proper_fraction(element, extension)
    )
    normal_denominator, special_denominator = split_normal_special(
        proper_denominator,
        derivation,
    )
    raw_lower_component = (
        parent.zero
        if polynomial_part.is_zero
        else parent.convert(polynomial_part.nth(0))
    )
    lower_component = raw_lower_component

    factor_signatures: tuple[HyperexponentialLogarithmicDerivativeFactor, ...]
    if special_denominator.degree() > 0:
        factor_signatures = ()
    else:
        factors = []
        for raw_factor, multiplicity in proper_denominator.factor_list()[1]:
            factor = raw_factor.monic()
            principal_numerators = _factor_principal_numerators(
                proper_numerator,
                proper_denominator,
                factor,
                multiplicity,
            )
            factor_derivative = derivation.polynomial(factor)
            factor_remainder = factor_derivative.rem(factor)
            if factor_remainder.is_zero:
                raise ValueError("a normal hyperexponential factor has zero derivative")
            logarithmic_multiplicity = _qq_constant_polynomial_multiple(
                principal_numerators[0],
                factor_remainder,
                derivation.lower(),
            )
            factors.append(
                HyperexponentialLogarithmicDerivativeFactor(
                    factor,
                    multiplicity,
                    principal_numerators,
                    factor_remainder,
                    logarithmic_multiplicity,
                )
            )
        factor_signatures = tuple(factors)

    lower_result = None
    has_top_obstruction = (
        (not polynomial_part.is_zero and polynomial_part.degree() > 0)
        or special_denominator.degree() > 0
        or any(factor.has_repeated_pole_obstruction for factor in factor_signatures)
        or any(not factor.has_logarithmic_residue for factor in factor_signatures)
    )
    minimal_power = None
    relation_argument = None
    if not has_top_obstruction:
        from ..dependency import solve_parametric_logarithmic_derivative

        # Unlike a primitive monomial, ``Dp`` can have the same degree as
        # ``p`` here.  Its quotient in ``Dp/p`` lies in the lower field and
        # must be removed along with the normal simple principal part.  For
        # example, with ``Dt=x*t`` and ``p=t+x``,
        # ``Dp/p = x + (1-x**2)/(t+x)``.  Passing the raw polynomial
        # coordinate ``x`` down would add that contribution a second time.
        normal_logarithmic_part = field.zero
        for factor in factor_signatures:
            multiplicity = factor.logarithmic_multiplicity
            if multiplicity is None:
                raise ValueError("a logarithmic factor lacks its multiplicity")
            full_derivative = derivation.polynomial(factor.factor)
            normal_logarithmic_part += _coerce(
                field,
                multiplicity,
            ) * _coerce(
                field,
                full_derivative.as_expr() / factor.factor.as_expr(),
            )
        normal_residual = element - normal_logarithmic_part
        residual_polynomial, residual_numerator, _ = _polynomial_and_proper_fraction(
            normal_residual, extension
        )
        if not residual_numerator.is_zero or (
            not residual_polynomial.is_zero and residual_polynomial.degree() > 0
        ):
            raise ValueError(
                "normal logarithmic factors did not leave a lower-field residual"
            )
        lower_component = (
            parent.zero
            if residual_polynomial.is_zero
            else parent.convert(residual_polynomial.nth(0))
        )
        lower_result = solve_parametric_logarithmic_derivative(
            lower_component,
            extension,
            derivation.lower(),
        )
        if not (lower_result.has_solution or lower_result.proves_no_solution):
            detail = lower_result.detail
            raise NotImplementedError(
                "hyperexponential radical logarithmic descent requires a "
                "complete lower parametric logarithmic-derivative decision"
                + (f": {detail}" if detail else "")
            )
        if lower_result.has_solution:
            if (
                lower_result.n is None
                or lower_result.m is None
                or lower_result.argument is None
            ):
                raise ValueError("a solved lower parametric problem lacks a witness")
            minimal_power = lower_result.n
            for factor in factor_signatures:
                multiplicity = factor.logarithmic_multiplicity
                if multiplicity is None:
                    raise ValueError("a logarithmic factor lacks its multiplicity")
                minimal_power = lcm(
                    minimal_power,
                    int(QQ.denom(QQ.convert(multiplicity))),
                )
            lower_scale = minimal_power // lower_result.n
            relation_argument = _coerce(field, lower_result.argument) ** lower_scale
            generator = _coerce(field, extension.generator)
            relation_argument *= generator ** (lower_result.m * lower_scale)
            for factor in factor_signatures:
                multiplicity = factor.logarithmic_multiplicity
                if multiplicity is None:
                    raise ValueError("a logarithmic factor lacks its multiplicity")
                exponent = QQ.convert(minimal_power) * QQ.convert(multiplicity)
                if QQ.denom(exponent) != 1:
                    raise ValueError(
                        "clearing hyperexponential residues left a "
                        "nonintegral valuation"
                    )
                relation_argument *= _coerce(
                    field,
                    factor.factor.as_expr(),
                ) ** int(QQ.numer(exponent))

    signature = HyperexponentialLogarithmicDerivativeSignature(
        derivation,
        element,
        polynomial_part,
        proper_numerator,
        proper_denominator,
        normal_denominator,
        special_denominator,
        lower_component,
        lower_result,
        factor_signatures,
        minimal_power,
        relation_argument,
    )
    if not signature.reconstructs_decomposition:
        raise ValueError(
            "hyperexponential logarithmic signature did not reconstruct its input"
        )
    if signature.has_solution and not signature.reconstructs_relation:
        raise ValueError(
            "hyperexponential logarithmic signature witness did not reconstruct"
        )
    return signature


def _validate_hyperexponential_reduction(
    derivation: Derivation,
) -> tuple[MonomialExtension, Derivation, FieldElement]:
    """Validate a supported transcendental hyperexponential extension."""
    if derivation.level < 1:
        raise TypeError("hyperexponential reduction requires a nonconstant lower field")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("hyperexponential reduction requires a monomial level")
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError("hyperexponential reduction requires Dt = eta*t")
    require_preserved_constant_field(derivation, "hyperexponential reduction")

    lower_derivation = derivation.lower()
    eta = extension.parent_field.convert(extension.generator_derivative.nth(1))
    dependency = analyze_hyperexponential_extension(
        extension,
        lower_derivation,
    )
    if dependency.is_dependent:
        raise ValueError(
            "hyperexponential reduction requires a transcendental "
            "generator; the proposed generator is algebraically dependent"
        )

    return extension, lower_derivation, eta


def _reduce_hyperexponential_laurent_validated(
    element: FieldElement,
    derivation: Derivation,
    extension: MonomialExtension,
    lower_derivation: Derivation,
    eta: FieldElement,
) -> HyperexponentialLaurentReduction:
    """Reduce a validated Laurent element without repeating dependency tests."""
    field = extension.field
    parent = extension.parent_field
    generator = extension.generator
    numerator, denominator = _reduced_polynomial_fraction(
        element,
        extension,
    )
    denominator_terms = denominator.terms()
    if len(denominator_terms) != 1:
        raise ValueError(
            "hyperexponential Laurent reduction requires a denominator "
            "that is a power of the generator"
        )
    (denominator_power,), denominator_coefficient = denominator_terms[0]
    denominator_coefficient = parent.convert(denominator_coefficient)
    coefficient_scale = parent.one / denominator_coefficient

    coefficients = {
        numerator_power - denominator_power: (
            parent.convert(coefficient) * coefficient_scale
        )
        for (numerator_power,), coefficient in numerator.terms()
    }
    exact_part = field.zero
    remainder = field.zero
    lower_remainder = parent.zero
    steps: list[HyperexponentialLaurentStep] = []

    for power in sorted(coefficients):
        coefficient = coefficients[power]
        coefficient_expression = parent.to_sympy(coefficient)
        term = _coerce(
            field,
            coefficient_expression * generator**power,
        )
        if power == 0:
            lower_remainder = coefficient
            remainder += term
            continue

        rde_result = solve_risch_differential_equation(
            parent.convert(power) * eta,
            coefficient,
            lower_derivation,
        )
        exact_coefficient = None
        if rde_result.has_solution:
            exact_coefficient = rde_result.solution
            exact_part += _coerce(
                field,
                parent.to_sympy(exact_coefficient) * generator**power,
            )
        else:
            remainder += term
        steps.append(
            HyperexponentialLaurentStep(
                power,
                coefficient,
                exact_coefficient,
                rde_result,
            )
        )

    if derivation(exact_part) + remainder != element:
        raise ValueError("hyperexponential Laurent reduction reconstruction failed")
    return HyperexponentialLaurentReduction(
        exact_part,
        remainder,
        lower_remainder,
        tuple(steps),
    )


def reduce_hyperexponential_laurent(
    integrand: object,
    derivation: Derivation,
) -> HyperexponentialLaurentReduction:
    r"""Reduce a Laurent polynomial in a hyperexponential generator.

    For ``Dt/t = eta`` and

    ``f = sum(b_n*t**n)``,

    every nonzero mode is an exact derivative precisely when the lower-field
    Risch differential equation

    ``Dy_n + n*eta*y_n = b_n``

    has a solution in the supported lower field. The zero mode is returned for
    recursive lower-field integration.
    """
    extension, lower_derivation, eta = _validate_hyperexponential_reduction(derivation)
    element = _coerce(extension.field, integrand)
    return _reduce_hyperexponential_laurent_validated(
        element,
        derivation,
        extension,
        lower_derivation,
        eta,
    )


def reduce_hyperexponential_monomial(
    integrand: object,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> HyperexponentialMonomialReduction:
    r"""Apply the complete reduction for a supported hyperexponential level.

    Repeated normal poles are Hermite-reduced, constant normal residues are
    extracted as logarithmic derivatives, and remaining special Laurent modes
    are reduced through lower-field Risch differential equations.
    """
    extension, lower_derivation, eta = _validate_hyperexponential_reduction(derivation)
    field = extension.field
    element = _coerce(field, integrand)
    hermite_exact_part, hermite_remainder = hermite_reduce_monomial(
        element,
        derivation,
    )
    residue_reduction = residue_reduce_monomial(
        hermite_remainder,
        derivation,
        residue,
    )

    laurent_reduction = None
    exact_part = hermite_exact_part
    remainder = residue_reduction.remainder
    lower_remainder = None
    if residue_reduction.all_normal_residues_removed:
        laurent_reduction = _reduce_hyperexponential_laurent_validated(
            residue_reduction.remainder,
            derivation,
            extension,
            lower_derivation,
            eta,
        )
        exact_part += laurent_reduction.exact_part
        remainder = laurent_reduction.remainder
        if not laurent_reduction.has_obstruction:
            lower_remainder = laurent_reduction.lower_remainder

    if (
        derivation(exact_part) + residue_reduction.logarithmic_derivative + remainder
        != element
    ):
        raise ValueError("hyperexponential monomial reduction reconstruction failed")
    return HyperexponentialMonomialReduction(
        hermite_exact_part,
        residue_reduction,
        laurent_reduction,
        exact_part,
        residue_reduction.logarithmic_derivative,
        remainder,
        lower_remainder,
    )
