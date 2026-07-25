"""Trager's finite-residue polynomial for algebraic differentials."""

from dataclasses import FrozenInstanceError

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, expand, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    TragerAlgebraicResidueResult,
    hermite_reduce_algebraic,
    rothstein_trager_residue_polynomial,
    trager_algebraic_residue_polynomial,
)

x, y, z = symbols("x y z")


def algebraic_tower(modulus, constants=QQ):
    """Return ``C(x)(y)``, its algebraic extension, and its power basis."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(
        Poly(modulus, y, domain=rational.field()),
    )
    tower = rational.extend(extension)
    return tower, extension, IntegralBasis.power_basis(extension)


def test_quadratic_residue_polynomial_records_complete_certificate() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)

    result = trager_algebraic_residue_polynomial(
        y/(x - 1),
        tower.derivation(),
        basis,
        z,
    )

    assert isinstance(result, TragerAlgebraicResidueResult)
    assert result.defining_polynomial == Poly(
        y**2 - x,
        y,
        x,
        domain=QQ,
    )
    assert result.numerator == Poly(y, y, x, domain=QQ)
    assert result.denominator == Poly(x - 1, x, domain=QQ)
    assert result.discriminant == Poly(4*x, x, domain=QQ)
    assert result.branch_gcd == Poly(1, x, domain=QQ)
    assert result.inner_resultant == Poly(
        -x + z**2,
        x,
        z,
        domain=QQ,
    )
    assert result.residue_polynomial == Poly(z**2 - 1, z, domain=QQ)
    assert result.square_free_residue_polynomial == result.residue_polynomial
    assert result.normalizes_resultant
    assert result.has_finite_poles
    assert result.reconstructs_input


def test_repeated_equal_residues_retain_place_multiplicity() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)

    result = trager_algebraic_residue_polynomial(
        1/(x - 1),
        tower.derivation(),
        basis,
        z,
    )

    assert result.residue_polynomial == Poly((z - 1)**2, z, domain=QQ)
    assert result.square_free_residue_polynomial == Poly(
        z - 1,
        z,
        domain=QQ,
    )


def test_y_independent_case_matches_rational_rothstein_trager() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    integrand = 1/(x - 1) + 2/(x + 1)

    algebraic_result = trager_algebraic_residue_polynomial(
        integrand,
        tower.derivation(),
        basis,
        z,
    )
    rational_result = rothstein_trager_residue_polynomial(
        integrand,
        x,
        z,
    ).monic()

    # Every rational place has one conjugate lift for each algebraic sheet.
    assert algebraic_result.residue_polynomial == (
        rational_result**extension.field.rank
    ).monic()
    assert algebraic_result.square_free_residue_polynomial == rational_result


def test_dense_nonradical_cubic_residues() -> None:
    tower, _, basis = algebraic_tower(y**3 + y - x)

    result = trager_algebraic_residue_polynomial(
        y/x,
        tower.derivation(),
        basis,
        z,
    )

    # At x = 0 the three residues are the roots of y**3 + y.
    assert result.residue_polynomial == Poly(z**3 + z, z, domain=QQ)
    assert result.discriminant == Poly(-27*x**2 - 4, x, domain=QQ)
    assert result.reconstructs_input


def test_nonpower_integral_basis_preserves_the_residue_polynomial() -> None:
    tower, extension, _ = algebraic_tower(y**3 + y - x)
    field = extension.field
    basis = IntegralBasis(
        extension,
        (
            field.one,
            field.generator,
            field.generator + field.generator**2,
        ),
    )

    result = trager_algebraic_residue_polynomial(
        y/x,
        tower.derivation(),
        basis,
        z,
    )

    assert result.residue_polynomial == Poly(z**3 + z, z, domain=QQ)
    assert result.reconstructs_input


def test_irreducible_base_poles_are_eliminated_without_root_objects() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)

    result = trager_algebraic_residue_polynomial(
        y/(x**2 + 1),
        tower.derivation(),
        basis,
        z,
    )

    # If x**2=-1, then residue r=y/(2*x) satisfies
    # r**2=1/(4*x). Eliminating the two conjugate x-values gives
    # (z**2 + I/4)*(z**2 - I/4) = z**4 + 1/16.
    assert result.residue_polynomial == Poly(
        z**4 + QQ(1, 16),
        z,
        domain=QQ,
    )
    assert result.reconstructs_input


def test_residue_polynomial_composes_with_algebraic_hermite_reduction() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    field = extension.field
    derivation = tower.derivation()
    planted_exact = field.from_sympy((1 + y)/(x + 2))
    planted_remainder = field.from_sympy(y/(x - 1))
    integrand = derivation(planted_exact) + planted_remainder

    exact, remainder = hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
    )
    result = trager_algebraic_residue_polynomial(
        remainder,
        derivation,
        basis,
        z,
    )

    assert derivation(exact) + remainder == integrand
    assert result.residue_polynomial == Poly(z**2 - 1, z, domain=QQ)


def test_no_finite_poles_have_unit_residue_polynomial() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)

    result = trager_algebraic_residue_polynomial(
        x + y,
        tower.derivation(),
        basis,
        z,
    )

    assert not result.has_finite_poles
    assert result.inner_resultant is None
    assert result.residue_polynomial == Poly(1, z, domain=QQ)
    assert result.square_free_residue_polynomial == Poly(
        1,
        z,
        domain=QQ,
    )
    assert result.reconstructs_input


def test_residues_preserve_an_algebraic_constant_field() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower, _, basis = algebraic_tower(y**2 - x, constants)

    result = trager_algebraic_residue_polynomial(
        (sqrt(2) + y)/(x - 1),
        tower.derivation(),
        basis,
        z,
    )

    assert result.residue_polynomial == Poly(
        z**2 - 2*sqrt(2)*z + 1,
        z,
        domain=constants,
    )
    assert result.reconstructs_input


def test_high_degree_multiple_poles_match_explicit_local_residues() -> None:
    tower, _, basis = algebraic_tower(y**5 - (x + 2))

    result = trager_algebraic_residue_polynomial(
        y/((x - 1)*(x + 1)),
        tower.derivation(),
        basis,
        z,
    )

    # At x=1, D'=2 and y**5=3. At x=-1, D'=-2 and y**5=1.
    expected = Poly(
        expand((z**5 - QQ(3, 32))*(z**5 + QQ(1, 32))),
        z,
        domain=QQ,
    )
    assert result.residue_polynomial == expected
    assert result.raw_residue_polynomial.monic() == expected
    assert result.normalizes_resultant
    assert result.residue_polynomial.degree() == 10


@settings(max_examples=50, deadline=None)
@given(
    pole=st.integers(min_value=-4, max_value=4),
    shift=st.integers(min_value=-4, max_value=4),
    constant=st.integers(min_value=-4, max_value=4),
    generator_coefficient=st.integers(min_value=-4, max_value=4),
)
def test_generated_quadratic_residues_match_specialization_oracle(
    pole,
    shift,
    constant,
    generator_coefficient,
) -> None:
    assume(pole + shift != 0)
    assume(constant != 0 or generator_coefficient != 0)
    tower, _, basis = algebraic_tower(y**2 - (x + shift))

    result = trager_algebraic_residue_polynomial(
        (constant + generator_coefficient*y)/(x - pole),
        tower.derivation(),
        basis,
        z,
    )

    # Specializing x=pole gives the conjugate residues
    # constant +/- generator_coefficient*sqrt(pole + shift).
    expected = Poly(
        (
            (z - constant)**2
            - generator_coefficient**2*(pole + shift)
        ),
        z,
        domain=QQ,
    )
    assert result.residue_polynomial == expected
    assert result.reconstructs_input


def test_branch_place_is_rejected_with_its_common_factor() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)

    with pytest.raises(
        NotImplementedError,
        match="branch locus x",
    ):
        trager_algebraic_residue_polynomial(
            y/x,
            tower.derivation(),
            basis,
            z,
        )


def test_repeated_finite_pole_requires_hermite_reduction() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)

    with pytest.raises(ValueError, match="square-free denominator"):
        trager_algebraic_residue_polynomial(
            y/(x - 1)**2,
            tower.derivation(),
            basis,
            z,
        )


def test_nonpolynomial_plane_model_is_explicitly_unsupported() -> None:
    tower, _, basis = algebraic_tower(y**2 - x/(x + 1))

    with pytest.raises(
        NotImplementedError,
        match=r"defining polynomial in C\[x, y\]",
    ):
        trager_algebraic_residue_polynomial(
            y/(x - 1),
            tower.derivation(),
            basis,
            z,
        )


def test_nonpolynomial_power_numerator_is_explicitly_unsupported() -> None:
    tower, extension, _ = algebraic_tower(y**2 - x**2*(x + 1))
    field = extension.field
    integral_generator = field.from_sympy(y/x)
    basis = IntegralBasis(
        extension,
        (field.one, integral_generator),
    )

    with pytest.raises(
        NotImplementedError,
        match="integral numerator",
    ):
        trager_algebraic_residue_polynomial(
            integral_generator/(x - 1),
            tower.derivation(),
            basis,
            z,
        )


def test_residue_input_contract_rejects_mismatched_or_reused_data() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)
    other_tower, _, _ = algebraic_tower(y**2 - (x + 1))

    with pytest.raises(TypeError, match="Derivation"):
        trager_algebraic_residue_polynomial(  # type: ignore[arg-type]
            y/(x - 1),
            object(),
            basis,
            z,
        )
    with pytest.raises(TypeError, match="IntegralBasis"):
        trager_algebraic_residue_polynomial(  # type: ignore[arg-type]
            y/(x - 1),
            tower.derivation(),
            object(),
            z,
        )
    with pytest.raises(TypeError, match="must be a Symbol"):
        trager_algebraic_residue_polynomial(  # type: ignore[arg-type]
            y/(x - 1),
            tower.derivation(),
            basis,
            1,
        )
    with pytest.raises(ValueError, match="same algebraic field"):
        trager_algebraic_residue_polynomial(
            y/(x - 1),
            other_tower.derivation(),
            basis,
            z,
        )
    for reused_symbol in (x, y):
        with pytest.raises(ValueError, match="must be new"):
            trager_algebraic_residue_polynomial(
                y/(x - 1),
                tower.derivation(),
                basis,
                reused_symbol,
            )


def test_nonstandard_rational_derivation_is_explicitly_unsupported() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(2, x, domain=QQ)),
    )
    extension = AlgebraicExtension(
        Poly(y**2 - x, y, domain=rational.field()),
    )
    tower = rational.extend(extension)
    basis = IntegralBasis.power_basis(extension)

    with pytest.raises(NotImplementedError, match="Dx = 1"):
        trager_algebraic_residue_polynomial(
            y/(x - 1),
            tower.derivation(),
            basis,
            z,
        )


def test_residue_result_is_immutable() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)
    result = trager_algebraic_residue_polynomial(
        y/(x - 1),
        tower.derivation(),
        basis,
        z,
    )

    with pytest.raises(FrozenInstanceError):
        result.denominator = Poly(1, x, domain=QQ)  # type: ignore[misc]
