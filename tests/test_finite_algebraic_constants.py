"""Finite algebraic constant fields in the rational P/E Risch kernel."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import I, Poly, sqrt, symbols
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    FundamentalSignatureOutcome,
    FundamentalSignatureReason,
    MonomialExtension,
    build_fundamental_signature,
)
from sympy_risch.constant_fields import (
    is_supported_finite_constant_field,
    rational_constant_coordinates,
)
from sympy_risch.dependency import (
    certify_hyperexponential_extension,
    certify_primitive_extension,
)
from sympy_risch.exact import solve_exact_derivative
from sympy_risch.relations import solve_logarithmic_relation_system

x, t, e, y = symbols("x t e y")


def _certified_rational_base(constants: Domain) -> DifferentialTower:
    """Construct ``C(x)`` through the checked primitive-extension factory."""
    analysis = certify_primitive_extension(
        Poly(1, x, domain=constants),
        DifferentialTower(constants).derivation(),
    )
    assert analysis.status is ConstantFieldStatus.PRESERVED
    return DifferentialTower(constants).extend(analysis.extension)


def _algebraic_rational_base() -> tuple[DifferentialTower, Domain]:
    """Return the standard native ``QQ(sqrt(2))(x)`` differential field."""
    constants = QQ.algebraic_field(sqrt(2))
    return _certified_rational_base(constants), constants


def test_native_power_basis_coordinates_are_exact_and_round_trip() -> None:
    constants = QQ.algebraic_field(sqrt(2))

    assert is_supported_finite_constant_field(constants)
    assert rational_constant_coordinates(constants, constants.zero) == (
        QQ.zero,
        QQ.zero,
    )
    assert rational_constant_coordinates(
        constants, constants.convert(3 + 5 * sqrt(2))
    ) == (
        QQ.convert(3),
        QQ.convert(5),
    )

    coordinates = rational_constant_coordinates(
        constants,
        constants.convert(3 + 5 * sqrt(2)),
    )
    assert constants.new(list(reversed(coordinates))) == constants.convert(
        3 + 5 * sqrt(2),
    )


def test_native_power_basis_coordinates_handle_a_degree_four_number_field() -> None:
    constants = QQ.algebraic_field(sqrt(2), sqrt(3))
    value = constants.convert(1 + sqrt(2) + 2 * sqrt(3))

    coordinates = rational_constant_coordinates(constants, value)

    assert len(coordinates) == 4
    assert constants.new(list(reversed(coordinates))) == value


@given(
    constant=st.integers(min_value=-8, max_value=8),
    theta_coefficient=st.integers(min_value=-8, max_value=8),
)
@settings(max_examples=16, deadline=None)
def test_generated_degree_two_power_basis_coordinates_round_trip(
    constant: int,
    theta_coefficient: int,
) -> None:
    constants = QQ.algebraic_field(sqrt(2))
    value = constants.convert(constant + theta_coefficient * sqrt(2))

    coordinates = rational_constant_coordinates(constants, value)

    assert constants.new(list(reversed(coordinates))) == value


def test_relation_kernel_uses_every_degree_four_power_coordinate() -> None:
    constants = QQ.algebraic_field(sqrt(2), sqrt(3))
    tower = _certified_rational_base(constants)
    field = tower.field()
    theta = constants.ext

    system = solve_logarithmic_relation_system(
        (
            field.from_sympy(theta / x),
            field.from_sympy(theta**2 / x),
            field.from_sympy(theta**3 / x),
            field.from_sympy((1 - theta - theta**2 - theta**3) / x),
        ),
        tower.derivation(),
    )

    assert system.reconstructs
    assert tuple(relation.source_coefficients for relation in system.relations) == (
        (1, 1, 1, 1),
    )


def test_native_algebraic_i_uses_the_generic_power_basis_path() -> None:
    constants = QQ.algebraic_field(I)
    tower = _certified_rational_base(constants)
    field = tower.field()

    system = solve_logarithmic_relation_system(
        (
            field.from_sympy(I / x),
            field.from_sympy((1 - I) / x),
        ),
        tower.derivation(),
    )

    assert system.reconstructs
    assert tuple(relation.source_coefficients for relation in system.relations) == (
        (1, 1),
    )


def test_finite_algebraic_constants_have_a_rational_relation_kernel() -> None:
    tower, _constants = _algebraic_rational_base()
    field = tower.field()

    system = solve_logarithmic_relation_system(
        (
            field.from_sympy(sqrt(2) / x),
            field.from_sympy((1 - sqrt(2)) / x),
        ),
        tower.derivation(),
    )

    assert system.fundamental_signature.supported
    assert system.reconstructs
    assert tuple(relation.source_coefficients for relation in system.relations) == (
        (1, 1),
    )
    assert system.relations[0].argument == field.from_sympy(x)


@given(offset=st.integers(min_value=-4, max_value=4))
@settings(max_examples=12, deadline=None)
def test_relation_coordinates_do_not_admit_algebraic_scalar_valuations(
    offset: int,
) -> None:
    tower, _constants = _algebraic_rational_base()
    field = tower.field()

    system = solve_logarithmic_relation_system(
        (
            field.from_sympy((sqrt(2) + offset) / x),
            field.from_sympy((1 - sqrt(2) - offset) / x),
        ),
        tower.derivation(),
    )

    assert system.reconstructs
    assert tuple(relation.source_coefficients for relation in system.relations) == (
        (1, 1),
    )


def test_finite_constant_nonbase_signatures_wait_for_proof_carrying_evidence() -> None:
    base, _constants = _algebraic_rational_base()
    primitive_analysis = certify_primitive_extension(
        Poly(1 / x, t, domain=base.field()),
        base.derivation(),
    )
    assert primitive_analysis.status is ConstantFieldStatus.PRESERVED
    tower = base.extend(primitive_analysis.extension)
    field = tower.field()

    signature = build_fundamental_signature(tower.derivation())

    assert signature.outcome is FundamentalSignatureOutcome.UNSUPPORTED
    assert signature.reason is FundamentalSignatureReason.CONSTANT_FIELD_PROVENANCE
    with pytest.raises(NotImplementedError, match="proof-carrying"):
        solve_logarithmic_relation_system(
            (field.from_sympy(sqrt(2) / x),),
            tower.derivation(),
        )

    recursive_hyperexponential = certify_hyperexponential_extension(
        Poly(e / x, e, domain=field),
        tower.derivation(),
    )
    assert recursive_hyperexponential.status is ConstantFieldStatus.UNKNOWN
    assert recursive_hyperexponential.dependency is None


def test_direct_exact_and_extension_factories_support_finite_constants() -> None:
    base, _constants = _algebraic_rational_base()
    field = base.field()

    exact = solve_exact_derivative(field.from_sympy(sqrt(2) * x), base.derivation())
    dependent_primitive = certify_primitive_extension(
        Poly(1, t, domain=field),
        base.derivation(),
    )
    independent_primitive = certify_primitive_extension(
        Poly(1 / x, t, domain=field),
        base.derivation(),
    )
    dependent_hyperexponential = certify_hyperexponential_extension(
        Poly(e / x, e, domain=field),
        base.derivation(),
    )
    independent_hyperexponential = certify_hyperexponential_extension(
        Poly(e, e, domain=field),
        base.derivation(),
    )

    assert exact.has_solution
    assert exact.antiderivative == field.from_sympy(sqrt(2) * x**2 / 2)
    assert dependent_primitive.status is ConstantFieldStatus.ENLARGED
    assert independent_primitive.status is ConstantFieldStatus.PRESERVED
    assert dependent_hyperexponential.status is ConstantFieldStatus.ENLARGED
    assert dependent_hyperexponential.dependency is not None
    assert dependent_hyperexponential.dependency.minimal_power == 1
    assert independent_hyperexponential.status is ConstantFieldStatus.PRESERVED


def test_unsupported_constant_fields_and_hypertangent_remain_explicit() -> None:
    nonconstant_base = DifferentialTower(QQ.frac_field(y)).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ.frac_field(y)),
            ConstantFieldStatus.PRESERVED,
        )
    )
    nonconstant_signature = build_fundamental_signature(
        nonconstant_base.derivation(),
    )

    assert not is_supported_finite_constant_field(QQ.frac_field(y))
    assert nonconstant_signature.outcome is FundamentalSignatureOutcome.UNSUPPORTED
    assert nonconstant_signature.reason is FundamentalSignatureReason.BASE_FIELD

    malformed_constants = QQ.algebraic_field(sqrt(2))
    malformed_base = DifferentialTower(malformed_constants).extend(
        MonomialExtension(
            Poly(2, x, domain=malformed_constants),
            ConstantFieldStatus.PRESERVED,
        )
    )
    malformed_signature = build_fundamental_signature(malformed_base.derivation())
    assert malformed_signature.outcome is FundamentalSignatureOutcome.UNSUPPORTED
    assert malformed_signature.reason is FundamentalSignatureReason.BASE_DERIVATIVE

    base, _constants = _algebraic_rational_base()
    hypertangent = base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    signature = build_fundamental_signature(hypertangent.derivation())

    assert signature.outcome is FundamentalSignatureOutcome.UNSUPPORTED
    assert signature.reason is FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY
    assert "hypertangent" in signature.detail
