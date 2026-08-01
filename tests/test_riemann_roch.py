"""Generic Riemann--Roch spaces, including Hess ideal reduction."""

from dataclasses import replace

import pytest
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    MonomialExtension,
)
from sympy_risch.algebraic_geometry.riemann_roch import (
    CertifiedRiemannRochSpace,
    RiemannRochCertificate,
    RiemannRochUnsupported,
    compute_riemann_roch_space,
)

x, y = symbols("x y")


def algebraic_model(modulus, constants=QQ):
    """Return a certified direct algebraic ``QQ(x)`` function field."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return AlgebraicFunctionField(extension), extension, tower.derivation()


def test_degree_one_parameter_constructs_a_complete_rational_space() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    parameter = extension.field.generator
    pole_divisor = model.principal_divisor(parameter).negative_part

    space = compute_riemann_roch_space(
        model,
        pole_divisor.scale(3),
        parameter=parameter,
    )

    assert isinstance(space, CertifiedRiemannRochSpace)
    assert not hasattr(space, "status")
    assert space.certificate is RiemannRochCertificate.RATIONAL_PARAMETER
    assert space.dimension == 4
    assert space.basis == tuple(parameter**power for power in range(4))
    assert pole_divisor.degree == 1
    assert all(space.contains(element) for element in space.basis)


def test_constant_linear_independence_uses_an_exact_algebraic_constant_field() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    model, extension, _derivation = algebraic_model(y**2 - x, constants)
    parameter = extension.field.generator
    divisor = model.principal_divisor(parameter).negative_part

    space = compute_riemann_roch_space(
        model,
        divisor,
        parameter=parameter,
    )

    assert isinstance(space, CertifiedRiemannRochSpace)
    assert space.basis == (extension.field.one, parameter)

    with pytest.raises(ValueError, match="constant-linearly dependent"):
        compute_riemann_roch_space(
            model,
            divisor,
            candidate_basis=(extension.field.one, extension.field.from_sympy(sqrt(2))),
        )


def test_nonspecial_certificate_uses_all_place_membership_on_genus_one_curve() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x**3 + x)
    field = extension.field
    x_element = field.from_sympy(x)
    divisor = model.principal_divisor(x_element).negative_part

    space = compute_riemann_roch_space(
        model,
        divisor,
        candidate_basis=(field.one, x_element),
    )

    assert model.genus == 1
    assert divisor.degree == 2
    assert isinstance(space, CertifiedRiemannRochSpace)
    assert space.certificate is RiemannRochCertificate.NONSPECIAL
    assert space.dimension == 2
    assert all(space.contains(element) for element in space.basis)


def test_degree_zero_principal_witness_is_exactly_one_dimensional() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    parameter = extension.field.generator
    divisor = -model.principal_divisor(parameter)

    space = compute_riemann_roch_space(
        model,
        divisor,
        candidate_basis=(parameter,),
    )

    assert isinstance(space, CertifiedRiemannRochSpace)
    assert space.certificate is RiemannRochCertificate.DEGREE_ZERO_WITNESS
    assert space.dimension == 1
    assert (model.principal_divisor(space.basis[0]) + divisor).is_zero


def test_negative_degree_and_zero_divisor_have_canonical_complete_spaces() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    field = extension.field
    (finite_place,) = model.finite_places(Poly(x, x, domain=QQ))
    negative_divisor = AlgebraicDivisor(model, ((finite_place, -1),))

    zero_space = compute_riemann_roch_space(
        model,
        AlgebraicDivisor(model),
    )
    negative_space = compute_riemann_roch_space(
        model,
        negative_divisor,
    )

    assert isinstance(zero_space, CertifiedRiemannRochSpace)
    assert zero_space.certificate is RiemannRochCertificate.ZERO_DIVISOR
    assert zero_space.basis == (field.one,)
    assert isinstance(negative_space, CertifiedRiemannRochSpace)
    assert negative_space.certificate is RiemannRochCertificate.NEGATIVE_DEGREE
    assert negative_space.basis == ()
    assert negative_space.dimension == 0


def test_positive_genus_special_range_uses_hess_ideal_intersection() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x**5 + x)
    field = extension.field
    divisor = model.principal_divisor(field.from_sympy(x)).negative_part

    space = compute_riemann_roch_space(model, divisor)

    assert model.genus == 2
    assert divisor.degree == 2
    assert isinstance(space, CertifiedRiemannRochSpace)
    assert space.certificate is RiemannRochCertificate.HESS_IDEAL_INTERSECTION
    assert space.dimension == 2
    assert all(space.contains(element) for element in space.basis)


def test_candidate_cannot_forge_nonspecial_completeness_or_skip_valuations() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x**3 + x)
    field = extension.field
    divisor = model.principal_divisor(field.from_sympy(x)).negative_part

    with pytest.raises(ValueError, match=r"outside L\(D\)"):
        compute_riemann_roch_space(
            model,
            divisor,
            candidate_basis=(field.one, field.generator),
        )


def test_candidate_constant_linear_dependence_and_dataclass_tampering_are_rejected() -> (
    None
):
    model, extension, _derivation = algebraic_model(y**2 - x)
    field = extension.field
    divisor = model.principal_divisor(field.generator).negative_part

    with pytest.raises(ValueError, match="constant-linearly dependent"):
        compute_riemann_roch_space(
            model,
            divisor,
            candidate_basis=(field.one, field.from_sympy(2)),
        )

    space = compute_riemann_roch_space(
        model,
        divisor,
        parameter=field.generator,
    )
    with pytest.raises(ValueError, match="consecutive parameter powers"):
        replace(space, basis=(field.one, field.from_sympy(2)))


def test_parameter_of_higher_pole_degree_is_explicitly_unsupported() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    field = extension.field
    x_element = field.from_sympy(x)
    divisor = model.principal_divisor(x_element).negative_part

    space = compute_riemann_roch_space(
        model,
        divisor,
        parameter=x_element,
    )

    assert isinstance(space, RiemannRochUnsupported)
    assert not hasattr(space, "basis")
    assert "degree one" in space.detail


def test_parameter_and_candidate_basis_are_deliberately_mutually_exclusive() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    field = extension.field
    divisor = model.principal_divisor(field.generator).negative_part

    with pytest.raises(ValueError, match="alternative completeness inputs"):
        compute_riemann_roch_space(
            model,
            divisor,
            parameter=field.generator,
            candidate_basis=(field.one, field.generator),
        )


def test_zero_divisor_rejects_an_ignored_rational_parameter() -> None:
    model, _extension, _derivation = algebraic_model(y**2 - x)

    with pytest.raises(ValueError, match="unnecessary for the zero divisor"):
        compute_riemann_roch_space(
            model,
            AlgebraicDivisor(model),
            parameter=object(),
        )


def test_public_constructor_rejects_a_forged_complete_certificate() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    field = extension.field
    divisor = model.principal_divisor(field.generator).negative_part

    with pytest.raises(ValueError, match="wrong dimension"):
        CertifiedRiemannRochSpace(
            model,
            divisor,
            (field.one,),
            RiemannRochCertificate.NONSPECIAL,
            "forged incomplete nonspecial certificate",
        )


def test_result_rejects_partial_unsupported_data_and_wrong_divisor_ownership() -> None:
    model, extension, _derivation = algebraic_model(y**2 - x)
    field = extension.field
    other_model = AlgebraicFunctionField(extension)

    with pytest.raises(TypeError):
        RiemannRochUnsupported(
            model,
            AlgebraicDivisor(model),
            "forged unsupported partial result",
            basis=(field.one,),
        )
    unsupported = RiemannRochUnsupported(
        model,
        AlgebraicDivisor(model),
        "unsupported boundary",
    )
    with pytest.raises(TypeError):
        replace(unsupported, status="certified")
    with pytest.raises(ValueError, match="belongs to another function field"):
        compute_riemann_roch_space(
            model,
            AlgebraicDivisor(other_model),
        )
