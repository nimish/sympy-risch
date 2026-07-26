"""Adversarial construction tests for public RDE result certificates."""

from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    CoupledRDEObstruction,
    DifferentialTower,
    MonomialExtension,
    RationalRDEObstruction,
    solve_bounded_coupled_rde_matrix,
    solve_bounded_rational_rde_matrix,
    solve_coupled_rational_rde,
    solve_coupled_risch_differential_equation,
    solve_hyperexponential_laurent_risch_differential_equation,
    solve_primitive_risch_differential_equation,
    solve_rational_risch_differential_equation,
    solve_risch_differential_equation,
)

x, t = symbols("x t")


def rational_tower() -> DifferentialTower:
    """Return the standard rational differential field."""
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_rational_certificate_rejects_contradictions_and_false_solution() -> None:
    solved = solve_rational_risch_differential_equation(1, x, x)
    with pytest.raises(ValueError, match="both a solution and an obstruction"):
        replace(solved, obstruction=RationalRDEObstruction.DENOMINATOR)
    with pytest.raises(ValueError, match="does not reconstruct"):
        replace(solved, solution=0)

    obstructed = solve_rational_risch_differential_equation(x, 1, x)
    with pytest.raises(ValueError, match="obstructed polynomial result"):
        replace(obstructed, polynomial_result=None)


def test_recursive_certificate_rejects_dispatch_mismatch_and_false_solution() -> None:
    tower = rational_tower()
    field = tower.field()
    result = solve_risch_differential_equation(
        field.one,
        field.from_sympy(x),
        tower.derivation(),
    )

    with pytest.raises(ValueError, match="does not reconstruct"):
        replace(result, solution=field.zero)
    with pytest.raises(ValueError, match="disagrees with its level certificate"):
        replace(result, solution=None, obstruction=RationalRDEObstruction.DENOMINATOR)


def test_primitive_obstruction_requires_its_polynomial_subcertificate() -> None:
    base = rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    result = solve_primitive_risch_differential_equation(
        field.from_sympy(t),
        field.from_sympy(t + 1),
        tower.derivation(),
    )

    with pytest.raises(ValueError, match="obstructed polynomial result"):
        replace(result, polynomial_result=None)


def test_laurent_lower_field_obstruction_requires_failed_mode() -> None:
    base = rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    result = solve_hyperexponential_laurent_risch_differential_equation(
        field.zero,
        field.from_sympy((1 + 2 * x) * t**2 + t / x + 2 * x),
        tower.derivation(),
    )

    with pytest.raises(ValueError, match="obstructed mode"):
        replace(result, steps=())


def test_bounded_matrix_certificates_reject_rank_and_pair_mismatches() -> None:
    scalar = solve_bounded_rational_rde_matrix(1, x, x, 1, 1)
    with pytest.raises(ValueError, match="solved system must have equal ranks"):
        replace(scalar, augmented_rank=scalar.coefficient_rank + 1)

    coupled = solve_bounded_coupled_rde_matrix(
        2,
        1 - 2 / x,
        2 * x + 2 - 1 / x**2,
        x,
        x,
        2,
    )
    with pytest.raises(ValueError, match="both solution coordinates"):
        replace(coupled, imaginary_solution=None)


def test_coupled_rational_certificate_rejects_pair_and_gaussian_mismatch() -> None:
    solved = solve_coupled_rational_rde(
        2,
        1 - 2 / x,
        2 * x + 2 - 1 / x**2,
        x,
    )
    with pytest.raises(ValueError, match="both solution coordinates"):
        replace(solved, imaginary_solution=None)
    with pytest.raises(ValueError, match="does not reconstruct"):
        replace(solved, real_solution=0)

    gaussian_failure = solve_coupled_rational_rde(
        1 / (x**2 + 1),
        1 / x,
        0,
        x,
    )
    with pytest.raises(ValueError, match="disagrees with its Gaussian certificate"):
        replace(gaussian_failure, obstruction=CoupledRDEObstruction.DEGREE)


def test_recursive_coupled_certificate_requires_one_subordinate_result() -> None:
    tower = rational_tower()
    field = tower.field()
    result = solve_coupled_risch_differential_equation(
        field.from_sympy(2),
        field.from_sympy(1),
        field.from_sympy(2 * x),
        tower.derivation(),
    )

    with pytest.raises(ValueError, match="exactly one scalar or direct result"):
        replace(result, scalar_result=None)
