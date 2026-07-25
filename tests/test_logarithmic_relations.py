"""Independent-facing checks for the finite logarithmic relation compiler."""

from sympy import I, Poly, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
)
from sympy_risch.relations import solve_logarithmic_relation_system

x, t = symbols("x t")


def rational_tower(constants=QQ) -> DifferentialTower:
    """Return the standard rational differential field over ``constants``."""
    return DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_rational_relation_system_removes_repeated_poles() -> None:
    tower = rational_tower()

    system = solve_logarithmic_relation_system(
        (1 / x, 1 / (x + 1), 1 / x**2),
        tower.derivation(),
    )

    assert system.reconstructs
    assert len(system.relations) == 2
    assert all(relation.source_coefficients[2] == 0 for relation in system.relations)
    assert {relation.source_coefficients[:2] for relation in system.relations} == {
        (1, 0),
        (0, 1),
    }


def test_gaussian_relation_system_keeps_valuations_rational() -> None:
    tower = rational_tower(QQ_I)

    system = solve_logarithmic_relation_system(
        (I / x, 1 / x),
        tower.derivation(),
    )

    assert system.reconstructs
    assert len(system.relations) == 1
    relation = system.relations[0]
    assert relation.source_coefficients == (0, 1)
    assert relation.argument == tower.field().from_sympy(x)


def test_hyperexponential_relation_system_records_normal_and_special_witnesses() -> (
    None
):
    base = rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(x * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    factor = field.from_sympy(t + x)

    system = solve_logarithmic_relation_system(
        (
            x,
            1 / x,
            tower.derivation()(factor) / factor,
        ),
        tower.derivation(),
    )

    assert system.reconstructs
    assert len(system.relations) == 3
    assert {relation.source_coefficients for relation in system.relations} == {
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    }
