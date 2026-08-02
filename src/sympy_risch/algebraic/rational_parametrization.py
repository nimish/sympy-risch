r"""Rational parametrization of one-factor simple-radical fields.

If ``z**n = c*L(x)**m`` with ``L`` linear and ``gcd(m, n) = 1``, choose
integers ``r, s`` with ``m*r + n*s = 1`` and put

``u = z**r * L(x)**s``.

Then ``L(x) = c**(-r)*u**n`` and ``z = c**s*u**m``.  The algebraic function
field is therefore a rational function field in ``u``; integrating there
avoids constructing maximal orders and places for a singular plane model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import ZZ
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polytools import Poly

from .._typing import SupportsAsExpr
from ..rational import (
    hermite_reduce_rational_over_domain,
    integrate_rational_over_domain,
)
from ..tower import AlgebraicExtension, Derivation, MonomialExtension
from ._field import (
    _base_polynomial_ring,
    _embed_parent,
    _evaluate_constant_polynomial,
    _extension_element,
    _quotient_extension_elements,
)
from .basis import SimpleRadicalData, recognize_simple_radical


def _as_constant_polynomial(
    value: SupportsAsExpr,
    variable: Symbol,
    constants: Domain,
) -> Poly:
    """Convert one native univariate polynomial without changing its domain."""
    return Poly(value.as_expr(), variable, domain=constants)


def _evaluate_in_fraction_field(
    polynomial: Poly,
    value: FracElement,
    field: FractionField,
) -> FracElement:
    """Evaluate a constant polynomial by Horner's rule in ``field``."""
    if polynomial.is_zero:
        return cast(FracElement, field.zero)
    result = field.zero
    for power in range(polynomial.degree(), -1, -1):
        result = result * value + field.convert(polynomial.nth(power))
    return cast(FracElement, result)


def _extension_integer_power(
    extension: AlgebraicExtension,
    value: ExtensionElement,
    exponent: int,
) -> ExtensionElement:
    """Raise a source element without relying on its broken native inverse."""
    if exponent >= 0:
        return cast(ExtensionElement, value**exponent)
    return _quotient_extension_elements(
        extension,
        extension.field.one,
        value ** (-exponent),
    )


def _same_fraction(
    left: FracElement,
    right: FracElement,
    field: FractionField,
) -> bool:
    """Compare exact fractions without assuming canonical stored quotients."""
    return left - right == field.zero


def _differentiate_fraction(value: FracElement) -> FracElement:
    """Differentiate a native fraction even when its unit is not normalized."""
    (generator,) = value.numer.ring.gens
    numerator = value.numer.diff(
        generator
    ) * value.denom - value.numer * value.denom.diff(generator)
    return cast(FracElement, value.new(numerator, value.denom**2))


@dataclass(frozen=True, slots=True)
class RationalRadicalParametrization:
    """Checked isomorphism from a one-factor radical field to ``C(u)``."""

    radical: SimpleRadicalData
    constants: Domain
    variable: Symbol
    linear_factor: Poly
    factor_multiplicity: int
    bezout_radical: int
    bezout_degree: int
    parameter: Symbol
    target_field: FractionField
    source_parameter: ExtensionElement
    target_variable: FracElement
    target_generator: FracElement

    @property
    def extension(self) -> AlgebraicExtension:
        """The source algebraic extension."""
        return self.radical.extension

    @property
    def target_parameter(self) -> FracElement:
        """The native target-field element represented by ``parameter``."""
        return cast(FracElement, self.target_field.from_sympy(self.parameter))

    def _forward_parent(self, value: object) -> FracElement:
        parent, constants, variable = _base_polynomial_ring(
            self.extension,
            constants=self.constants,
            variable=self.variable,
        )
        element = parent.convert(value)
        numerator = _as_constant_polynomial(element.numer, variable, constants)
        denominator = _as_constant_polynomial(element.denom, variable, constants)
        return cast(
            FracElement,
            _evaluate_in_fraction_field(
                numerator,
                self.target_variable,
                self.target_field,
            )
            / _evaluate_in_fraction_field(
                denominator,
                self.target_variable,
                self.target_field,
            ),
        )

    def forward(self, value: object) -> FracElement:
        """Map a source algebraic-function element into ``C(u)``."""
        element = _extension_element(self.extension, value)
        result = self.target_field.zero
        generator_power = self.target_field.one
        for coordinate in self.extension.power_basis_coordinates(element):
            result += self._forward_parent(coordinate) * generator_power
            generator_power *= self.target_generator
        return cast(FracElement, result)

    def inverse(self, value: object) -> ExtensionElement:
        """Map a rational function of ``u`` back into the source field."""
        element = self.target_field.convert(value)
        if element == self.target_field.zero:
            return self.extension.field.zero
        _parent, constants, _variable = _base_polynomial_ring(
            self.extension,
            constants=self.constants,
            variable=self.variable,
        )
        numerator = _as_constant_polynomial(
            element.numer,
            self.parameter,
            constants,
        )
        denominator = _as_constant_polynomial(
            element.denom,
            self.parameter,
            constants,
        )
        return _quotient_extension_elements(
            self.extension,
            _evaluate_constant_polynomial(
                self.extension,
                numerator,
                self.source_parameter,
            ),
            _evaluate_constant_polynomial(
                self.extension,
                denominator,
                self.source_parameter,
            ),
        )

    def __post_init__(self) -> None:
        extension = self.extension
        _base_polynomial_ring(
            extension,
            constants=self.constants,
            variable=self.variable,
        )
        if self.linear_factor.domain != self.radical.normalized_radicand.domain:
            raise ValueError("the linear factor has another constant domain")
        if self.linear_factor.degree() != 1:
            raise ValueError("a rational radical parametrization needs a linear factor")
        if self.factor_multiplicity <= 0:
            raise ValueError("a rational radical parametrization needs a multiplicity")
        if (
            self.factor_multiplicity * self.bezout_radical
            + self.radical.degree * self.bezout_degree
            != 1
        ):
            raise ValueError("the radical exponents do not satisfy Bezout's identity")
        if not isinstance(self.parameter, Symbol):
            raise TypeError("a rational radical parametrization needs a parameter")
        if self.target_field.domain != self.constants or self.target_field.symbols != (
            self.parameter,
        ):
            raise ValueError("the target field uses another rational parameter")

        source_variable = _extension_element(extension, self.variable)
        if not _same_fraction(
            self.forward(source_variable),
            self.target_variable,
            self.target_field,
        ):
            raise ValueError("the parametrization has the wrong source variable image")
        if not _same_fraction(
            self.forward(extension.field.generator),
            self.target_generator,
            self.target_field,
        ):
            raise ValueError("the parametrization has the wrong generator image")
        if not _same_fraction(
            self.forward(self.source_parameter),
            self.target_parameter,
            self.target_field,
        ):
            raise ValueError(
                "the source parameter does not map to the target generator"
            )
        if self.inverse(self.target_variable) - source_variable != extension.field.zero:
            raise ValueError("the inverse map does not recover the source variable")
        if (
            self.inverse(self.target_generator) - extension.field.generator
            != extension.field.zero
        ):
            raise ValueError("the inverse map does not recover the algebraic generator")


def rational_simple_radical_parametrization(
    extension: AlgebraicExtension,
    *,
    constants: Domain | None = None,
    variable: Symbol | None = None,
) -> RationalRadicalParametrization | None:
    r"""Parametrize ``z**n = c*L(x)**m`` when ``L`` is linear.

    Other simple radicals return ``None`` and continue through the general
    algebraic-function-field algorithms.
    """
    try:
        radical = recognize_simple_radical(
            extension,
            constants=constants,
            variable=variable,
        )
    except NotImplementedError:
        return None
    if radical is None:
        return None
    unit, factors = radical.normalized_radicand.sqf_list()
    if len(factors) != 1:
        return None
    linear_factor, multiplicity = factors[0]
    if linear_factor.degree() != 1:
        return None

    bezout_radical, bezout_degree, gcd = ZZ.gcdex(multiplicity, radical.degree)
    if gcd != ZZ.one:
        return None
    r = int(bezout_radical)
    s = int(bezout_degree)

    _parent, constants, variable = _base_polynomial_ring(
        extension,
        constants=constants,
        variable=variable,
    )
    parameter = Dummy("risch_radical_parameter")
    target_field = constants.frac_field(parameter)
    u = cast(FracElement, target_field.gens[0])
    c = target_field.convert(unit)
    leading = target_field.convert(linear_factor.nth(1))
    constant = target_field.convert(linear_factor.nth(0))
    target_linear_factor = c ** (-r) * u**radical.degree
    target_variable = cast(
        FracElement,
        (target_linear_factor - constant) / leading,
    )
    target_normalized_generator = c**s * u**multiplicity
    target_extracted_factor = _evaluate_in_fraction_field(
        radical.extracted_factor,
        target_variable,
        target_field,
    )
    target_denominator = _evaluate_in_fraction_field(
        radical.denominator,
        target_variable,
        target_field,
    )
    target_generator = cast(
        FracElement,
        target_normalized_generator * target_extracted_factor / target_denominator,
    )

    source_linear_factor = _embed_parent(extension, linear_factor)
    source_parameter = cast(
        ExtensionElement,
        _extension_integer_power(extension, radical.normalized_generator, r)
        * _extension_integer_power(extension, source_linear_factor, s),
    )
    return RationalRadicalParametrization(
        radical,
        constants,
        variable,
        linear_factor,
        multiplicity,
        r,
        s,
        parameter,
        target_field,
        source_parameter,
        target_variable,
        target_generator,
    )


@dataclass(frozen=True, slots=True)
class RationalRadicalIntegrationReduction:
    """Rational integration transported through a checked parametrization."""

    parametrization: RationalRadicalParametrization
    derivation: Derivation
    integrand: ExtensionElement
    target_integrand: FracElement
    target_exact_part: FracElement
    target_logarithmic_derivative: FracElement
    exact_part: ExtensionElement
    logarithmic_derivative: ExtensionElement
    logarithmic_antiderivative: Expr

    def __post_init__(self) -> None:
        extension = self.parametrization.extension
        if self.derivation.field is not extension.field:
            raise ValueError("the parametrization reduction has another derivation")
        if self.derivation.tower.constants != self.parametrization.constants:
            raise ValueError("the parametrization reduction has other constants")
        if (
            _extension_element(extension, self.integrand) - self.integrand
            != extension.field.zero
        ):
            raise ValueError("the parametrization reduction has another integrand")
        if not _same_fraction(
            _differentiate_fraction(self.target_exact_part)
            + self.target_logarithmic_derivative,
            self.target_integrand,
            self.parametrization.target_field,
        ):
            raise ValueError("the rational reduction does not reconstruct")
        if (
            self.derivation(self.exact_part)
            + self.logarithmic_derivative
            - self.integrand
            != extension.field.zero
        ):
            raise ValueError("the pulled-back radical reduction does not reconstruct")

    @property
    def reconstructs(self) -> bool:
        """Whether the transported native decomposition is exact."""
        return (
            self.derivation(self.exact_part)
            + self.logarithmic_derivative
            - self.integrand
            == self.parametrization.extension.field.zero
        )


def reduce_rational_simple_radical_integral(
    integrand: object,
    derivation: Derivation,
) -> RationalRadicalIntegrationReduction | None:
    """Transport one rational simple-radical integral to ``C(u)``."""
    if not isinstance(derivation, Derivation):
        raise TypeError("rational radical integration needs a derivation")
    if derivation.level < 1:
        return None
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, AlgebraicExtension):
        return None
    if derivation.level != 1:
        return None
    base = derivation.tower.levels[0]
    if not isinstance(base, MonomialExtension) or base.generator_derivative != Poly(
        1,
        base.generator,
        domain=derivation.tower.constants,
    ):
        return None
    parametrization = rational_simple_radical_parametrization(
        extension,
        constants=derivation.tower.constants,
        variable=base.generator,
    )
    if parametrization is None:
        return None

    element = _extension_element(extension, integrand)
    dx_du = _differentiate_fraction(parametrization.target_variable)
    target_integrand = cast(FracElement, parametrization.forward(element) * dx_du)
    target_expression = parametrization.target_field.to_sympy(target_integrand)
    _parent, constants, _variable = _base_polynomial_ring(
        extension,
        constants=parametrization.constants,
        variable=parametrization.variable,
    )
    exact_expression, logarithmic_derivative_expression = (
        hermite_reduce_rational_over_domain(
            target_expression,
            parametrization.parameter,
            constants,
        )
    )
    target_exact_part = cast(
        FracElement,
        parametrization.target_field.from_sympy(exact_expression),
    )
    target_logarithmic_derivative = cast(
        FracElement,
        parametrization.target_field.from_sympy(logarithmic_derivative_expression),
    )
    exact_part = parametrization.inverse(target_exact_part)
    logarithmic_derivative = cast(
        ExtensionElement,
        parametrization.inverse(target_logarithmic_derivative)
        * derivation(parametrization.source_parameter),
    )
    target_logarithmic_antiderivative = (
        integrate_rational_over_domain(
            logarithmic_derivative_expression,
            parametrization.parameter,
            constants,
        )
        if logarithmic_derivative_expression != 0
        else logarithmic_derivative_expression
    )
    logarithmic_antiderivative = target_logarithmic_antiderivative.xreplace(
        {
            parametrization.parameter: extension.field.to_sympy(
                parametrization.source_parameter,
            )
        },
    )
    return RationalRadicalIntegrationReduction(
        parametrization,
        derivation,
        element,
        target_integrand,
        target_exact_part,
        target_logarithmic_derivative,
        exact_part,
        logarithmic_derivative,
        logarithmic_antiderivative,
    )


__all__ = [
    "RationalRadicalIntegrationReduction",
    "RationalRadicalParametrization",
    "rational_simple_radical_parametrization",
    "reduce_rational_simple_radical_integral",
]
