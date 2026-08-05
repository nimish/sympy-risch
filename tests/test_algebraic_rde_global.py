"""Automatic all-place bounds and global scalar algebraic RDE decisions."""

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
from sympy_risch.algebraic.rde_bounds import (
    AlgebraicRDELocalCase,
    AlgebraicRDEPoleBoundCertificate,
    certify_algebraic_rde_pole_bound,
)
from sympy_risch.algebraic.rde_global import (
    AlgebraicScalarRDENoSolution,
    AlgebraicScalarRDESolution,
    AlgebraicScalarRDEUnsupported,
    AlgebraicScalarRDEZeroSolution,
    solve_algebraic_rde,
)
from sympy_risch.algebraic_geometry.function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
)

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
    """Reuse the certified curve ``y**2 = x`` across local-bound examples."""
    return algebraic_model(y**2 - x)


def finite_place_above_x(model: AlgebraicFunctionField):
    """Return the unique ramified place above ``x = 0`` on ``y**2 = x``."""
    return model.finite_places(Poly(x, x, domain=QQ))[0]


def test_local_bounds_cover_derivative_and_multiplication_dominance(
    sqrt_x_model,
) -> None:
    extension, derivation, model, _analysis = sqrt_x_model
    place = finite_place_above_x(model)
    solution = extension.field.from_sympy(1 / x)

    derivative_bound = certify_algebraic_rde_pole_bound(
        extension.field.one,
        derivation(solution) + solution,
        derivation,
        model,
    )
    derivative_local = next(
        bound for bound in derivative_bound.local_bounds if bound.place is place
    )
    assert derivative_local.case is AlgebraicRDELocalCase.DERIVATIVE_DOMINANT
    assert derivative_local.coefficient_order == 1
    assert derivative_local.right_hand_side_order == -3
    assert derivative_local.lower_bound == -2

    coefficient = extension.field.from_sympy(1 / x**2)
    multiplication_bound = certify_algebraic_rde_pole_bound(
        coefficient,
        derivation(solution) + coefficient * solution,
        derivation,
        model,
    )
    multiplication_local = next(
        bound for bound in multiplication_bound.local_bounds if bound.place is place
    )
    assert multiplication_local.case is AlgebraicRDELocalCase.MULTIPLICATION_DOMINANT
    assert multiplication_local.coefficient_order == -3
    assert multiplication_local.right_hand_side_order == -5
    assert multiplication_local.lower_bound == -2


def test_positive_integer_residue_retains_the_resonant_pole(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    place = finite_place_above_x(model)
    coefficient = extension.field.from_sympy(1 + 1 / x)
    right_hand_side = extension.field.from_sympy(1 / x)

    pole_bound = certify_algebraic_rde_pole_bound(
        coefficient,
        right_hand_side,
        derivation,
        model,
    )
    local = next(bound for bound in pole_bound.local_bounds if bound.place is place)
    assert local.case is AlgebraicRDELocalCase.SIMPLE_POLE_RESONANCE
    assert local.coefficient_order == -1
    assert local.right_hand_side_order == -1
    assert local.resonant_order == -2
    assert local.lower_bound == -2
    assert pole_bound.pole_divisor.coefficient(place) == 2

    result = solve_algebraic_rde(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
    )
    assert isinstance(result, AlgebraicScalarRDESolution)
    assert result.reconstructs
    # This solution cannot be replaced by a regular solution at x = 0.
    witness = extension.field.from_sympy(1 / x)
    assert result.solution == witness
    assert derivation(witness) + coefficient * witness == right_hand_side
    assert result.pole_bound.pole_divisor.coefficient(place) == 2


def test_zero_coefficient_uses_the_exact_derivative_bound(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    solution = extension.field.generator

    result = solve_algebraic_rde(
        extension.field.zero,
        derivation(solution),
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicScalarRDESolution)
    assert result.reconstructs
    assert all(
        bound.case is AlgebraicRDELocalCase.ZERO_COEFFICIENT
        for bound in result.pole_bound.local_bounds
    )


def test_higher_degree_closed_place_is_bounded_and_solved(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    solution = extension.field.from_sympy(1 / (x**2 + 1))
    coefficient = extension.field.one
    right_hand_side = derivation(solution) + coefficient * solution

    result = solve_algebraic_rde(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicScalarRDESolution)
    assert result.solution == solution
    assert result.reconstructs
    assert any(
        bound.place.degree > 1 and bound.pole_order > 0
        for bound in result.pole_bound.local_bounds
    )


def test_positive_genus_infinity_uses_the_same_global_solver() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)
    solution = extension.field.from_sympy(x)
    coefficient = extension.field.generator
    right_hand_side = derivation(solution) + coefficient * solution

    result = solve_algebraic_rde(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
    )

    assert isinstance(result, AlgebraicScalarRDESolution)
    assert result.reconstructs
    assert any(
        bound.place in model.infinite_places()
        for bound in result.pole_bound.local_bounds
    )


def test_global_negative_agrees_with_integrating_factor_oracle(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    coefficient = extension.field.from_sympy(1 / (2 * x))
    right_hand_side = extension.field.from_sympy(1 / (x - 1)) / (
        extension.field.generator
    )

    oracle = solve_algebraic_integrating_factor_rde(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
    )
    result = solve_algebraic_rde(
        coefficient,
        right_hand_side,
        derivation,
        model,
        analysis,
    )

    assert oracle.status is AlgebraicIntegratingFactorRDEStatus.NO_SOLUTION
    assert isinstance(result, AlgebraicScalarRDENoSolution)
    assert result.complete_negative
    assert result.bounded_obstruction.complete_within_space


def test_zero_solution_precedes_full_constant_field_requirement(
    sqrt_x_model,
) -> None:
    extension, derivation, model, analysis = sqrt_x_model
    unsupported_analysis = replace(
        analysis,
        status=AlgebraicConstantFieldStatus.UNSUPPORTED,
        embedding=None,
        detail="forced unsupported constant-field boundary",
    )

    zero = solve_algebraic_rde(
        extension.field.generator,
        extension.field.zero,
        derivation,
        model,
        unsupported_analysis,
    )
    unsupported = solve_algebraic_rde(
        extension.field.generator,
        extension.field.one,
        derivation,
        model,
        unsupported_analysis,
    )

    assert isinstance(zero, AlgebraicScalarRDEZeroSolution)
    assert zero.reconstructs
    assert isinstance(unsupported, AlgebraicScalarRDEUnsupported)
    assert unsupported.pole_bound is None


def test_global_negative_and_pole_certificate_are_factory_owned(
    sqrt_x_model,
) -> None:
    extension, derivation, model, _analysis = sqrt_x_model
    certificate = certify_algebraic_rde_pole_bound(
        extension.field.one,
        extension.field.one,
        derivation,
        model,
    )

    with pytest.raises(TypeError, match="created by the factory"):
        AlgebraicRDEPoleBoundCertificate()
    with pytest.raises(TypeError, match="created by the factory"):
        replace(certificate, pole_divisor=AlgebraicDivisor(model))
    with pytest.raises(TypeError, match="created by the solver"):
        AlgebraicScalarRDENoSolution()


def test_pole_certificate_replays_the_actual_resonance(sqrt_x_model) -> None:
    extension, derivation, model, _analysis = sqrt_x_model
    coefficient = extension.field.from_sympy(1 + 1 / x)
    right_hand_side = extension.field.from_sympy(1 / x)
    certificate = certify_algebraic_rde_pole_bound(
        coefficient,
        right_hand_side,
        derivation,
        model,
    )
    place = finite_place_above_x(model)
    local = next(bound for bound in certificate.local_bounds if bound.place is place)
    forged_local = replace(
        local,
        residue=place.residue_field.zero,
        resonant_order=None,
        lower_bound=0,
    )
    forged_bounds = tuple(
        forged_local if bound.place is place else bound
        for bound in certificate.local_bounds
    )
    forged_divisor = AlgebraicDivisor(
        model,
        tuple(
            (bound.place, bound.pole_order)
            for bound in forged_bounds
            if bound.pole_order
        ),
    )

    with pytest.raises(ValueError, match="local bound is inconsistent"):
        AlgebraicRDEPoleBoundCertificate._create(
            model,
            derivation,
            coefficient,
            right_hand_side,
            certificate.coefficient_differential,
            certificate.right_hand_side_differential,
            forged_bounds,
            forged_divisor,
            "forged resonance",
        )


def test_pole_bound_reuses_global_divisors_and_canonical_data(monkeypatch) -> None:
    extension, derivation, model, _analysis = algebraic_model(y**2 - x)
    original_principal_divisor = model.principal_divisor
    principal_calls = 0
    order_calls = 0

    def counted_principal_divisor(element):
        nonlocal principal_calls
        principal_calls += 1
        return original_principal_divisor(element)

    original_order_at = AlgebraicDifferential.order_at

    def counted_order_at(differential, place):
        nonlocal order_calls
        order_calls += 1
        return original_order_at(differential, place)

    monkeypatch.setattr(model, "principal_divisor", counted_principal_divisor)
    monkeypatch.setattr(AlgebraicDifferential, "order_at", counted_order_at)
    small = certify_algebraic_rde_pole_bound(
        extension.field.from_sympy(1 / x),
        extension.field.from_sympy(1 / x),
        derivation,
        model,
    )
    assert principal_calls == 2
    assert order_calls == 0

    wider_value = extension.field.from_sympy(1 / (x * (x - 1) * (x + 1) * (x**2 + 1)))
    wider = certify_algebraic_rde_pole_bound(
        wider_value,
        wider_value,
        derivation,
        model,
    )

    assert len(wider.local_bounds) > len(small.local_bounds)
    assert principal_calls == 4
    assert order_calls == 0
    assert model.canonical_divisor() is model.canonical_divisor()
