"""Curated integration examples from public symbolic-integration suites.

The primitive examples below are adapted to this project's explicit tower API
from SymPy's ``sympy/integrals/tests/test_risch.py``, whose module notes that
most cases come from Bronstein's book:

https://github.com/sympy/sympy/blob/master/sympy/integrals/tests/test_risch.py

The logarithmic-power families mirror the logarithm category in Rubi's public
integration test suite:

https://rulebasedintegration.org/testProblems.html

We retain only cases whose required tower families are currently implemented.
Each test checks a differential identity or an inspectable obstruction rather
than requiring presentation-identical antiderivatives.
"""

from math import factorial

import pytest
from sympy import Poly, Rational, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    reduce_primitive_monomial,
)

x, t, z = symbols("x t z")


def logarithmic_tower() -> DifferentialTower:
    """Return ``QQ(x)(t)`` with ``Dx = 1`` and ``Dt = 1/x``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_bronstein_primitive_residue_example_with_algebraic_constants() -> None:
    # SymPy test_residue_reduce: two constant residues are encoded by
    # z**2 - 1/4 without adjoining either root, while the third residue x
    # remains as a non-elementarity certificate.
    tower = logarithmic_tower()
    parent = tower.field(0)
    field = tower.field()
    numerator = 2*t**2 - t - x**2
    denominator = t**3 - x**2*t
    integrand = field.from_sympy(numerator/denominator)

    result = reduce_primitive_monomial(
        integrand,
        tower.derivation(),
        residue=z,
    )

    assert result.has_finite_residue_obstruction
    assert result.is_non_elementary
    assert result.logarithmic_derivative == field.from_sympy(
        (1 - t)/(x**2 - t**2)
    )
    assert result.remainder == field.from_sympy(1/t)
    factor = result.residue_reduction.certificate.factors[0]
    assert factor.constant_factor == Poly(
        z**2 - Rational(1, 4),
        z,
        domain=parent,
    )
    assert factor.nonconstant_factor == Poly(z - x, z, domain=parent)
    assert result.residue_reduction.logarithmic_parts
    assert result.logarithmic_derivative + result.remainder == integrand


def test_bronstein_primitive_residue_example_with_rational_residues() -> None:
    # SymPy test_residue_reduce expects -log(t - 1) + log(t + 1).
    tower = logarithmic_tower()
    parent = tower.field(0)
    field = tower.field()
    integrand = field.from_sympy(-2/(x*(t**2 - 1)))

    result = reduce_primitive_monomial(
        integrand,
        tower.derivation(),
        residue=z,
    )

    assert result.reduced_to_lower_field
    assert result.logarithmic_derivative == integrand
    assert result.remainder == field.zero
    assert (
        result.residue_reduction.certificate.residue_polynomial.monic()
        == Poly(z**2 - 1, z, domain=parent)
    )


def test_bronstein_primitive_example_separates_exact_and_lower_parts() -> None:
    # SymPy integrate_primitive returns x*log(x) plus the lower remainder -1.
    # Our polynomial reducer immediately integrates that rational remainder.
    tower = logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(t)

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.exact_part == field.from_sympy(x*t - x)
    assert result.remainder == field.zero
    assert result.reduced_to_lower_field


def test_bronstein_primitive_example_certifies_x_over_log_x() -> None:
    # SymPy integrate_primitive records this as a NonElementaryIntegral.
    tower = logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(x/t)

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.has_finite_residue_obstruction
    assert result.is_non_elementary
    assert result.remainder == integrand
    assert not result.residue_reduction.certificate.all_residues_constant


def test_bronstein_primitive_example_combines_exact_and_nonelementary_parts() -> None:
    # This is the rational function in t=log(x) used by SymPy's
    # test_integrate_primitive.  Its decomposition is
    #
    #   D(-1/(t + 1)) + 1/(t + 1),
    #
    # and the remaining simple pole has the nonconstant residue x.
    tower = logarithmic_tower()
    field = tower.field()
    numerator = (
        x**2*t**3
        + (3*x**2 + x)*t**2
        + (3*x**2 + 2*x)*t
        + x**2
        + x
    )
    denominator = x**2*(t + 1)**4
    integrand = field.from_sympy(numerator/denominator)

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.exact_part == field.from_sympy(-1/(t + 1))
    assert result.remainder == field.from_sympy(1/(t + 1))
    assert result.has_finite_residue_obstruction
    assert result.is_non_elementary
    assert (
        tower.derivation()(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )


def logarithmic_power_antiderivative(power_x: int, power_log: int):
    r"""Return the standard antiderivative of ``x**m*log(x)**n``.

    This formula applies for ``m != -1`` and uses ``t`` for ``log(x)``.
    """
    scale = power_x + 1
    return x**scale*sum(
        Rational(
            (-1)**index*factorial(power_log),
            factorial(power_log - index)*scale**(index + 1),
        )
        * t**(power_log - index)
        for index in range(power_log + 1)
    )


@pytest.mark.parametrize(
    ("power_x", "power_log"),
    [
        (0, 3),
        (1, 1),
        (2, 3),
        (4, 5),
        (-2, 2),
        (-3, 4),
    ],
)
def test_rubi_logarithmic_power_family(
    power_x: int,
    power_log: int,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(x**power_x*t**power_log)
    expected = field.from_sympy(
        logarithmic_power_antiderivative(power_x, power_log)
    )

    result = reduce_primitive_monomial(integrand, derivation)

    assert result.exact_part == expected
    assert result.remainder == field.zero
    assert derivation(result.exact_part) == integrand
    assert result.reduced_to_lower_field


@pytest.mark.parametrize("power_log", range(9))
def test_rubi_logarithmic_power_over_x_family(power_log: int) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(t**power_log/x)
    expected = field.from_sympy(t**(power_log + 1)/(power_log + 1))

    result = reduce_primitive_monomial(integrand, derivation)

    assert result.exact_part == expected
    assert result.remainder == field.zero
    assert derivation(result.exact_part) == integrand


@pytest.mark.parametrize(
    "shift",
    [-3, -1, 1, 2, 5],
)
def test_shifted_logarithmic_poles_are_nonelementary_reference_cases(
    shift: int,
) -> None:
    # Generalizes the standard 1/log(x) negative example.  The residue x at
    # t=shift is nonconstant for every rational shift.
    tower = logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(1/(t - shift))

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.has_finite_residue_obstruction
    assert result.is_non_elementary
    assert result.remainder == integrand
    assert cancel(
        result.residue_reduction.remainder.as_expr() - integrand.as_expr()
    ) == 0
