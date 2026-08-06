"""Complete parametric RDEs on direct algebraic curves."""

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
from sympy_risch.algebraic.exactness import (
    AlgebraicNoPrimitive,
    solve_algebraic_exactness,
)
from sympy_risch.algebraic.rde_parametric import (
    AlgebraicParametricRDEPoleBoundCertificate,
    AlgebraicParametricRDESolutionSpace,
    AlgebraicParametricRDEUnsupported,
    certify_algebraic_parametric_rde_pole_bound,
    solve_algebraic_parametric_rde,
)
from sympy_risch.algebraic_geometry.function_field import AlgebraicFunctionField

x, y = symbols("x y")


def algebraic_model(modulus):
    """Return a direct algebraic model and its full-constant analysis."""
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


@pytest.fixture(scope="module")
def sqrt_x_model():
    return algebraic_model(y**2 - x)


def test_two_parameter_family_is_reconstructed_completely(sqrt_x_model) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    generator = extension.field.generator
    right_hand_sides = (1, derivation(generator) + generator)

    result = solve_algebraic_parametric_rde(
        1,
        right_hand_sides,
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicParametricRDESolutionSpace)
    assert result.complete
    assert len(result.basis) == 2
    assert {vector.parameters for vector in result.basis} == {
        (QQ.one, QQ.zero),
        (QQ.zero, QQ.one),
    }
    solutions = {vector.parameters: vector.solution for vector in result.basis}
    assert solutions[(QQ.one, QQ.zero)] == extension.field.one
    assert solutions[(QQ.zero, QQ.one)] == generator


def test_nonexact_parameter_is_excluded_by_independent_exactness_oracle(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    right_hand_side = extension.field.from_sympy(1 / x)

    oracle = solve_algebraic_exactness(
        right_hand_side,
        derivation,
        model,
        analysis,
    )
    result = solve_algebraic_parametric_rde(
        extension.field.zero,
        (right_hand_side,),
        derivation,
        model,
        analysis,
    )

    assert isinstance(oracle, AlgebraicNoPrimitive)
    assert isinstance(result, AlgebraicParametricRDESolutionSpace)
    assert len(result.basis) == 1
    assert result.basis[0].parameters == (QQ.zero,)
    assert result.basis[0].solution == extension.field.one


def test_shared_bound_retains_simple_pole_resonance(sqrt_x_model) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    coefficient = extension.field.from_sympy(1 + 1 / x)
    right_hand_sides = (extension.field.from_sympy(1 / x), extension.field.one)

    result = solve_algebraic_parametric_rde(
        coefficient,
        right_hand_sides,
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicParametricRDESolutionSpace)
    resonant = next(
        bound
        for bound in result.pole_bound.local_bounds
        if bound.resonant_order is not None
    )
    assert resonant.resonant_order == -2
    assert any(
        vector.parameters[0] != QQ.zero
        and model.principal_divisor(vector.solution).negative_part.coefficient(
            resonant.place
        )
        == 2
        for vector in result.basis
    )


def test_positive_genus_uses_the_same_parametric_solver() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)
    generator = extension.field.generator
    solution = extension.field.from_sympy(x)
    right_hand_sides = (
        generator,
        derivation(solution) + generator * solution,
    )

    result = solve_algebraic_parametric_rde(
        generator,
        right_hand_sides,
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicParametricRDESolutionSpace)
    assert result.complete
    assert len(result.basis) == 2


def test_unproved_constants_and_forged_certificates_are_rejected(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    unsupported_analysis = replace(
        analysis,
        status=AlgebraicConstantFieldStatus.UNSUPPORTED,
        embedding=None,
        detail="forced unsupported constant-field boundary",
    )
    unsupported = solve_algebraic_parametric_rde(
        extension.field.one,
        (extension.field.one,),
        derivation,
        model,
        unsupported_analysis,
    )
    certificate = certify_algebraic_parametric_rde_pole_bound(
        extension.field.one,
        (extension.field.one,),
        derivation,
        model,
        analysis,
    )

    assert isinstance(unsupported, AlgebraicParametricRDEUnsupported)
    assert unsupported.pole_bound is None
    with pytest.raises(TypeError, match="created by the factory"):
        AlgebraicParametricRDEPoleBoundCertificate()
    with pytest.raises(TypeError, match="created by the factory"):
        replace(certificate, pole_divisor=certificate.pole_divisor)


def test_parametric_solver_builds_one_space_and_one_kernel(
    sqrt_x_model,
    monkeypatch,
) -> None:
    import sympy_risch.algebraic.linear as linear_module
    import sympy_risch.algebraic.rde_parametric as parametric_module

    extension, derivation, model, analysis = sqrt_x_model
    original_space = parametric_module.compute_riemann_roch_space
    original_matrix = linear_module.constant_coordinate_matrix
    space_calls = 0
    matrix_calls = 0

    def counted_space(*args, **kwargs):
        nonlocal space_calls
        space_calls += 1
        return original_space(*args, **kwargs)

    def counted_matrix(*args, **kwargs):
        nonlocal matrix_calls
        matrix_calls += 1
        return original_matrix(*args, **kwargs)

    monkeypatch.setattr(parametric_module, "compute_riemann_roch_space", counted_space)
    monkeypatch.setattr(linear_module, "constant_coordinate_matrix", counted_matrix)
    result = solve_algebraic_parametric_rde(
        extension.field.one,
        tuple(extension.field.convert(value) for value in (1, 2, 3, 4)),
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicParametricRDESolutionSpace)
    assert space_calls == 1
    assert matrix_calls == 1
