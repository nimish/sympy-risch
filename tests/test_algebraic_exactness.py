"""Certified in-field exactness decisions for algebraic differentials."""

from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
    analyze_algebraic_constant_field,
)
from sympy_risch.algebraic.exactness import (
    AlgebraicExactnessCertificate,
    AlgebraicExactnessStatus,
    solve_algebraic_exactness,
)
from sympy_risch.algebraic_geometry.function_field import AlgebraicFunctionField

x, y = symbols("x y")


def algebraic_model(modulus, constants=QQ):
    """Return the standard direct algebraic model and its derivation proof."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return (
        extension,
        tower.derivation(),
        AlgebraicFunctionField(extension),
        analyze_algebraic_constant_field(extension, tower.derivation()),
    )


def test_exactness_finds_a_genus_zero_primitive_from_a_complete_space() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)

    result = solve_algebraic_exactness(
        extension.field.one / extension.field.generator,
        derivation,
        model,
        analysis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicExactnessStatus.EXACT
    assert result.is_exact
    assert result.reconstructs
    assert result.primitive == 2 * extension.field.generator
    assert result.space is not None and result.space.complete


def test_exactness_solves_over_an_algebraic_constant_field() -> None:
    extension, derivation, model, analysis = algebraic_model(
        y**2 - x,
        QQ.algebraic_field(sqrt(2)),
    )

    result = solve_algebraic_exactness(
        extension.field.from_sympy(sqrt(2)) / extension.field.generator,
        derivation,
        model,
        analysis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicExactnessStatus.EXACT
    assert result.reconstructs
    assert result.primitive == extension.field.from_sympy(2 * sqrt(2) * y)


def test_holomorphic_elliptic_differential_has_no_algebraic_primitive() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)

    result = solve_algebraic_exactness(
        extension.field.one / extension.field.generator,
        derivation,
        model,
        analysis,
    )

    assert result.status is AlgebraicExactnessStatus.NO_PRIMITIVE
    assert result.complete_negative
    assert result.primitive is None
    assert result.primitive_pole_divisor is not None
    assert result.primitive_pole_divisor.is_zero
    assert result.space is not None and result.space.basis == (extension.field.one,)


def test_nonzero_residue_is_an_immediate_no_primitive_certificate() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x)

    result = solve_algebraic_exactness(
        extension.field.from_sympy(1 / (x - 1)),
        derivation,
        model,
        analysis,
    )

    assert result.status is AlgebraicExactnessStatus.NO_PRIMITIVE
    assert result.certificate is AlgebraicExactnessCertificate.NONZERO_RESIDUE
    assert result.complete_negative
    assert result.space is None


def test_second_kind_elliptic_differential_uses_a_complete_nonspecial_space() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)

    result = solve_algebraic_exactness(
        extension.field.from_sympy(x) / extension.field.generator,
        derivation,
        model,
        analysis,
        max_x_degree=1,
    )

    assert result.status is AlgebraicExactnessStatus.NO_PRIMITIVE
    assert result.complete_negative
    assert result.primitive is None
    assert result.nonspecial_search is not None
    assert result.nonspecial_search.space is not None


def test_bounded_primitive_space_miss_remains_incomplete() -> None:
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)

    incomplete = solve_algebraic_exactness(
        extension.field.one,
        derivation,
        model,
        analysis,
        max_x_degree=0,
    )
    exact = solve_algebraic_exactness(
        extension.field.one,
        derivation,
        model,
        analysis,
        max_x_degree=1,
    )

    assert incomplete.status is AlgebraicExactnessStatus.INCOMPLETE
    assert not incomplete.complete_negative
    assert exact.status is AlgebraicExactnessStatus.EXACT
    assert exact.primitive == extension.field.from_sympy(x)


def test_backend_local_arithmetic_gap_is_an_honest_unsupported_boundary() -> None:
    """A valid second-kind differential must not leak a SymPy GCD failure."""
    extension, derivation, model, analysis = algebraic_model(y**2 - x**3 + x)
    primitive = extension.field.from_sympy(x) / extension.field.generator

    result = solve_algebraic_exactness(
        derivation(primitive),
        derivation,
        model,
        analysis,
        max_x_degree=5,
    )

    assert result.status is AlgebraicExactnessStatus.UNSUPPORTED
    assert result.primitive is None
    assert not result.complete_negative
