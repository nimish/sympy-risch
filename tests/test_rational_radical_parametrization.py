"""Rational parametrization of genus-zero simple radicals."""

from sympy import Poly, log, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
    integrate_elementary,
)
from sympy_risch.algebraic.rational_parametrization import (
    RationalRadicalIntegrationReduction,
    rational_simple_radical_parametrization,
)

x, y = symbols("x y")


def _radical_tower(modulus, constants=QQ):
    base = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=base.field()))
    return base.extend(extension), extension


def test_scaled_shifted_radical_has_verified_inverse_maps() -> None:
    _tower, extension = _radical_tower(y**3 - 2 * (x + 1) ** 2)

    parametrization = rational_simple_radical_parametrization(extension)

    assert parametrization is not None
    u = parametrization.target_parameter
    assert parametrization.target_variable == 2 * u**3 - 1
    assert parametrization.target_generator == 2 * u**2
    assert parametrization.source_parameter == extension.field.from_sympy(
        y**2 / (2 * x + 2),
    )
    assert parametrization.inverse(parametrization.forward(y + 1 / (x + 1))) == (
        extension.field.from_sympy(y + 1 / (x + 1))
    )


def test_rational_radical_integration_transports_logarithms() -> None:
    tower, extension = _radical_tower(y**3 - 2 * (x + 1) ** 2)
    derivation = tower.derivation()
    parametrization = rational_simple_radical_parametrization(extension)
    assert parametrization is not None
    argument = parametrization.source_parameter + extension.field.one
    integrand = derivation(argument) / argument

    result = integrate_elementary(integrand, derivation)

    assert result.reconstructs
    assert isinstance(result.certificate, RationalRadicalIntegrationReduction)
    assert result.antiderivative == log(y**2 / (2 * x + 2) + 1)


def test_reciprocal_radical_is_parametrized_without_global_geometry() -> None:
    _tower, extension = _radical_tower(y**2 - 1 / x)

    parametrization = rational_simple_radical_parametrization(extension)

    assert parametrization is not None
    u = parametrization.target_parameter
    assert parametrization.target_variable == u**2
    assert parametrization.target_generator == 1 / u
    assert parametrization.source_parameter == extension.field.from_sympy(x * y)


def test_parametrization_supports_algebraic_constant_fields() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    _tower, extension = _radical_tower(
        y**3 - sqrt(2) * (x + 1) ** 2,
        constants,
    )

    parametrization = rational_simple_radical_parametrization(extension)

    assert parametrization is not None
    assert (
        parametrization.forward(parametrization.source_parameter)
        - parametrization.target_parameter
        == parametrization.target_field.zero
    )


def test_positive_genus_radical_does_not_claim_a_rational_parameter() -> None:
    _tower, extension = _radical_tower(y**2 - x**3 + x)

    assert rational_simple_radical_parametrization(extension) is None
