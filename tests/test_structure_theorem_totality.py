"""Representative totality checks for the classical P/E/T structure cell."""

from itertools import permutations

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import ConstantFieldStatus, DifferentialTower, MonomialExtension
from sympy_risch.dependency import (
    MonomialRelationSystemParametricCertificate,
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
    certify_primitive_extension,
    solve_parametric_logarithmic_derivative,
)
from sympy_risch.monomial import (
    HypertangentLogarithmicDerivativeSignature,
    solve_radical_logarithmic_derivative,
)

x, p, e, t, theta = symbols("x p e t theta")


def _checked_base() -> DifferentialTower:
    analysis = certify_primitive_extension(
        Poly(1, x, domain=QQ),
        DifferentialTower(QQ).derivation(),
    )
    return DifferentialTower(QQ).extend(analysis.extension)


def _checked_mixed_tower(order: str) -> DifferentialTower:
    tower = _checked_base()
    for family, generator in zip(order, (p, e, t), strict=True):
        field = tower.field()
        derivation = tower.derivation()
        if family == "P":
            analysis = certify_primitive_extension(
                Poly(1 / x, generator, domain=field),
                derivation,
            )
        elif family == "E":
            analysis = certify_hyperexponential_extension(
                Poly(x * generator, generator, domain=field),
                derivation,
            )
        else:
            analysis = certify_hypertangent_extension(
                Poly(1 + generator**2, generator, domain=field),
                derivation,
            )
        assert analysis.status is ConstantFieldStatus.PRESERVED
        tower = tower.extend(analysis.extension)
    return tower


@pytest.mark.parametrize("order", tuple(map("".join, permutations("PET"))))
def test_structure_decisions_are_total_for_every_mixed_family_order(
    order: str,
) -> None:
    tower = _checked_mixed_tower(order)
    field = tower.field()
    derivation = tower.derivation()
    argument = field.from_sympy((p + 1) * (e + 1) * (t + 1))
    logarithmic_derivative = derivation(argument) / argument
    eta = field.from_sympy(x + 1 + 1 / x)
    proposal = MonomialExtension(
        Poly(field.to_sympy(eta) * theta, theta, domain=field),
    )

    radical = solve_radical_logarithmic_derivative(
        logarithmic_derivative,
        derivation,
    )
    parametric = solve_parametric_logarithmic_derivative(
        (3 * eta + logarithmic_derivative) / 2,
        proposal,
        derivation,
    )
    negative = solve_parametric_logarithmic_derivative(
        field.from_sympy(1 / (x**2 + 1)),
        MonomialExtension(Poly(theta, theta, domain=field)),
        derivation,
    )

    assert radical.has_solution
    assert radical.reconstructs
    assert parametric.has_solution
    assert (parametric.n, parametric.m) == (2, 3)
    assert parametric.reconstructs
    assert isinstance(
        parametric.certificate,
        MonomialRelationSystemParametricCertificate,
    )
    assert negative.proves_no_solution
    assert isinstance(
        negative.certificate,
        MonomialRelationSystemParametricCertificate,
    )


def test_repeated_tangent_levels_use_the_direct_real_relation_system() -> None:
    tower = _checked_base()
    generators = (p, e, t)
    for power, generator in enumerate(generators):
        analysis = certify_hypertangent_extension(
            Poly(
                x**power * (1 + generator**2),
                generator,
                domain=tower.field(),
            ),
            tower.derivation(),
        )
        assert analysis.status is ConstantFieldStatus.PRESERVED
        tower = tower.extend(analysis.extension)

    field = tower.field()
    argument = field.from_sympy((p + 1) * (e + 1) * (t + 1))
    result = solve_radical_logarithmic_derivative(
        tower.derivation()(argument) / argument,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.minimal_power == 1
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentLogarithmicDerivativeSignature,
    )
