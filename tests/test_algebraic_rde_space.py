"""Finite-dimensional algebraic RDE decisions in certified global spaces."""

from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicConstantFieldStatus,
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
    analyze_algebraic_constant_field,
)
from sympy_risch.algebraic.rde import (
    AlgebraicIntegratingFactorRDEStatus,
    solve_algebraic_integrating_factor_rde,
)
from sympy_risch.algebraic.rde_space import (
    AlgebraicRDEInSpaceSolution,
    AlgebraicRDEInSpaceUnsupported,
    AlgebraicRDENoSolutionInSpace,
    AlgebraicRDEZeroSolution,
    solve_algebraic_rde_in_space,
)
from sympy_risch.algebraic_geometry.function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
)

x, y = symbols("x y")


def algebraic_model(modulus):
    """Return a direct algebraic model and its constant-field analysis."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    derivation = tower.derivation()
    return (
        extension,
        derivation,
        AlgebraicFunctionField(extension),
        analyze_algebraic_constant_field(extension, derivation),
    )


def test_general_coefficient_is_solved_in_constant_function_space() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)
    generator = extension.field.generator

    result = solve_algebraic_rde_in_space(
        generator,
        2 * generator,
        derivation,
        model,
        analysis,
        AlgebraicDivisor(model),
    )

    assert isinstance(result, AlgebraicRDEInSpaceSolution)
    assert result.solution == extension.field.convert(2)
    assert result.reconstructs
    # This coefficient is outside the older integrating-factor endpoint.
    old_result = solve_algebraic_integrating_factor_rde(
        generator,
        2 * generator,
        derivation,
        model,
        analysis,
    )
    assert old_result.status is AlgebraicIntegratingFactorRDEStatus.UNSUPPORTED


def test_inconsistent_system_is_negative_only_inside_requested_space() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)
    outside_solution = extension.field.generator
    coefficient = outside_solution
    right_hand_side = derivation(outside_solution) + coefficient * outside_solution

    result = solve_algebraic_rde_in_space(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
        AlgebraicDivisor(model),
    )

    assert isinstance(result, AlgebraicRDENoSolutionInSpace)
    assert result.complete_within_space
    assert result.space.basis == (extension.field.one,)
    assert result.operator_rank == 1
    assert result.augmented_rank == 2
    assert not hasattr(result, "complete_negative")
    # The same unrestricted RDE does have a solution, but it lies outside L(0).
    assert not result.space.contains(outside_solution)
    assert derivation(outside_solution) + coefficient * outside_solution == (
        right_hand_side
    )
    with pytest.raises(TypeError, match="created by the solver"):
        AlgebraicRDENoSolutionInSpace()


def test_nonconstant_solution_is_found_in_its_complete_pole_space() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)
    solution = extension.field.generator
    pole_divisor = model.principal_divisor(solution).negative_part
    coefficient = solution
    right_hand_side = derivation(solution) + coefficient * solution

    result = solve_algebraic_rde_in_space(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
        pole_divisor,
    )

    assert isinstance(result, AlgebraicRDEInSpaceSolution)
    assert result.solution == solution
    assert result.reconstructs
    assert len(result.space.basis) == 2


def test_positive_genus_space_uses_the_same_exact_solver() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)
    solution = extension.field.from_sympy(x)
    pole_divisor = model.principal_divisor(solution).negative_part
    coefficient = extension.field.generator
    right_hand_side = derivation(solution) + coefficient * solution

    result = solve_algebraic_rde_in_space(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
        pole_divisor,
    )

    assert isinstance(result, AlgebraicRDEInSpaceSolution)
    assert result.reconstructs


def test_unproved_constants_and_foreign_divisors_are_not_admitted() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)
    unsupported_analysis = replace(
        analysis,
        status=AlgebraicConstantFieldStatus.UNSUPPORTED,
        embedding=None,
        detail="forced unsupported constant-field boundary",
    )

    unsupported = solve_algebraic_rde_in_space(
        extension.field.one,
        extension.field.one,
        derivation,
        model,
        unsupported_analysis,
        AlgebraicDivisor(model),
    )
    assert isinstance(unsupported, AlgebraicRDEInSpaceUnsupported)

    foreign_model = AlgebraicFunctionField(extension)
    with pytest.raises(ValueError, match="another model"):
        solve_algebraic_rde_in_space(
            extension.field.one,
            extension.field.one,
            derivation,
            model,
            analysis,
            AlgebraicDivisor(foreign_model),
        )


def test_homogeneous_rde_keeps_zero_without_constant_field_proof() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)
    unsupported_analysis = replace(
        analysis,
        status=AlgebraicConstantFieldStatus.UNSUPPORTED,
        embedding=None,
        detail="forced unsupported constant-field boundary",
    )

    result = solve_algebraic_rde_in_space(
        extension.field.generator,
        extension.field.zero,
        derivation,
        model,
        unsupported_analysis,
        AlgebraicDivisor(model),
    )

    assert isinstance(result, AlgebraicRDEZeroSolution)
    assert result.solution == extension.field.zero
    assert result.reconstructs
    assert not hasattr(result, "space")


def test_solver_builds_one_constant_linear_system(monkeypatch) -> None:
    """Increasing the space dimension does not rebuild the global matrix."""
    import sympy_risch.algebraic.linear as linear_module
    import sympy_risch.algebraic.rde_space as rde_space_module

    extension, derivation, model, analysis = algebraic_model(y**2 - x)
    solution = extension.field.generator
    pole_divisor = model.principal_divisor(solution).negative_part.scale(3)
    original_matrix = linear_module.constant_coordinate_matrix
    original_combination = rde_space_module.constant_linear_combination
    matrix_calls = 0
    combination_calls = 0

    def counted_matrix(*args, **kwargs):
        nonlocal matrix_calls
        matrix_calls += 1
        return original_matrix(*args, **kwargs)

    def counted_combination(*args, **kwargs):
        nonlocal combination_calls
        combination_calls += 1
        return original_combination(*args, **kwargs)

    monkeypatch.setattr(
        linear_module,
        "constant_coordinate_matrix",
        counted_matrix,
    )
    monkeypatch.setattr(
        rde_space_module,
        "constant_linear_combination",
        counted_combination,
    )
    result = solve_algebraic_rde_in_space(
        solution,
        derivation(solution) + solution**2,
        derivation,
        model,
        analysis,
        pole_divisor,
    )

    assert isinstance(result, AlgebraicRDEInSpaceSolution)
    assert len(result.space.basis) == 4
    assert matrix_calls == 1
    assert combination_calls == 1
