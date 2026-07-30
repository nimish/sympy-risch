"""Hess finite/infinite ideal representations and Riemann--Roch bases."""

from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    MonomialExtension,
    analyze_algebraic_constant_field,
    compute_riemann_roch_space,
    hess_divisor_ideals,
)
from sympy_risch.algebraic_geometry import hess_parametric_basis
from sympy_risch.algebraic_geometry.constraints import (
    AlgebraicValuationKernelStatus,
    finite_lattice_slice,
    solve_algebraic_valuation_constraints,
)
from sympy_risch.algebraic_geometry.riemann_roch import (
    RiemannRochCertificate,
    constant_coordinate_matrix,
)

x, y = symbols("x y")


def elliptic_model() -> tuple[AlgebraicFunctionField, AlgebraicExtension]:
    """Return the smooth elliptic field ``QQ(x, y)``, ``y**2 = x**3 - x``."""
    rational = DifferentialTower(QQ).extend(MonomialExtension(Poly(1, x, domain=QQ)))
    extension = AlgebraicExtension(
        Poly(y**2 - x**3 + x, y, domain=rational.field()),
    )
    return AlgebraicFunctionField(extension), extension


def split_infinity_model() -> tuple[AlgebraicFunctionField, AlgebraicExtension]:
    """Return a rational quadratic curve with two places above infinity."""
    rational = DifferentialTower(QQ).extend(MonomialExtension(Poly(1, x, domain=QQ)))
    extension = AlgebraicExtension(
        Poly(y**2 - x**2 + 1, y, domain=rational.field()),
    )
    return AlgebraicFunctionField(extension), extension


def test_hess_representation_separates_finite_and_infinite_conditions() -> None:
    model, extension = elliptic_model()
    (finite_place,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinite_place, 1)))
    representation = hess_divisor_ideals(model, divisor)

    assert representation.finite_ideal.order is model.order
    assert representation.infinite_ideal.order is model.reciprocal_chart.closure.order
    for element in (extension.field.one, extension.field.from_sympy(x)):
        assert (
            representation.contains(element)
            == (model.principal_divisor(element) + divisor).is_effective
        )


def test_hess_computes_a_mixed_support_space_without_local_expansions() -> None:
    model, _extension = elliptic_model()
    (finite_place,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinite_place, 2)))

    space = compute_riemann_roch_space(model, divisor)

    assert space.certificate is RiemannRochCertificate.HESS_IDEAL_INTERSECTION
    assert space.dimension == divisor.degree
    assert all(space.contains(element) for element in space.basis)


def test_parametric_hess_basis_reuses_one_reduction_across_infinity_shifts() -> None:
    """Theorem 7 expansions agree with fresh Algorithm 13 spaces."""
    model, extension = elliptic_model()
    (finite_place,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinite_place, 2)))

    parametric = hess_parametric_basis(model, divisor)

    assert parametric.parameter == extension.field.from_sympy(x)
    assert (
        parametric.parameter_pole_divisor
        == model.principal_divisor(parametric.parameter).negative_part
    )
    assert len(parametric.generators) == extension.field.rank
    assert len(parametric.invariant_degrees) == extension.field.rank

    for shift in (-1, 0, 1, 2):
        shifted_divisor = parametric.shifted_divisor(shift)
        expanded = parametric.expand(shift)
        fresh = compute_riemann_roch_space(model, shifted_divisor)

        assert fresh.complete
        assert len(expanded) == parametric.dimension(shift) == fresh.dimension
        assert all(fresh.contains(element) for element in expanded)
        assert (
            constant_coordinate_matrix(
                model,
                fresh.basis + expanded,
            ).rank()
            == fresh.dimension
        )


def test_parametric_hess_dimension_does_not_expand_large_shifted_spaces() -> None:
    """Dimension is the sum of shifted invariant-degree bounds."""
    model, extension = elliptic_model()
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((infinite_place, 2),))
    parametric = hess_parametric_basis(model, divisor)
    shift = 25

    assert parametric.dimension(shift) == sum(
        max(degree + shift + 1, 0) for degree in parametric.invariant_degrees
    )
    assert (
        parametric.dimension(shift) - parametric.dimension(shift - 1)
        == extension.field.rank
    )
    assert len(parametric.expand(shift)) == parametric.dimension(shift)


def test_parametric_hess_shift_uses_every_place_above_infinity() -> None:
    """The parameter pole divisor is not assumed to be a single place."""
    model, _extension = split_infinity_model()
    infinite_places = model.infinite_places()
    assert len(infinite_places) == 2
    divisor = AlgebraicDivisor(
        model,
        tuple((place, place.ramification_index) for place in infinite_places),
    )
    parametric = hess_parametric_basis(model, divisor)

    assert parametric.parameter_pole_divisor == divisor
    for shift in (0, 1):
        expanded = parametric.expand(shift)
        fresh = compute_riemann_roch_space(
            model,
            parametric.shifted_divisor(shift),
        )

        assert len(expanded) == parametric.dimension(shift) == fresh.dimension
        assert all(fresh.contains(element) for element in expanded)
        assert (
            constant_coordinate_matrix(model, fresh.basis + expanded).rank()
            == fresh.dimension
        )


def test_parametric_hess_basis_agrees_with_local_valuation_constraints() -> None:
    """A separate Laurent-coordinate kernel finds the same small space."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(
        Poly(y**2 - x**3 + x, y, domain=rational.field()),
    )
    tower = rational.extend(extension)
    model = AlgebraicFunctionField(extension)
    constant_field = analyze_algebraic_constant_field(
        extension,
        tower.derivation(),
    )
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((infinite_place, 4),))
    parametric = hess_parametric_basis(model, divisor)

    local = solve_algebraic_valuation_constraints(
        model,
        divisor,
        constant_field,
        finite_lattice_slice(model, divisor, max_x_degree=4),
    )
    expanded = parametric.expand()

    assert local.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert local.dimension == parametric.dimension() == len(expanded)
    assert (
        constant_coordinate_matrix(model, local.basis + expanded).rank()
        == local.dimension
    )
