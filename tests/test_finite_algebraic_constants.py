"""Finite algebraic constant fields in the rational P/E Risch kernel."""

from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import I, Poly, sqrt, symbols
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    FundamentalSignatureOutcome,
    FundamentalSignatureReason,
    MonomialExtension,
    build_fundamental_signature,
)
from sympy_risch.constant_fields import (
    finite_constant_field_profile,
    gaussian_constant_field,
    is_supported_finite_constant_field,
    is_supported_rational_constant_field,
    rational_constant_coordinates,
    rational_constant_relation_rows,
)
from sympy_risch.dependency import (
    MonomialRelationSystemParametricCertificate,
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
    certify_primitive_extension,
    solve_parametric_logarithmic_derivative,
)
from sympy_risch.exact import solve_exact_derivative
from sympy_risch.gaussian import build_hypertangent_gaussian_normal_form
from sympy_risch.relations import solve_logarithmic_relation_system

x, t, e, u, y, a, b = symbols("x t e u y a b")


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
    profile = finite_constant_field_profile(constants)

    assert profile is not None
    assert profile.degree_over_qq == 2
    assert profile.is_real
    assert not profile.contains_i
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


def test_rational_constant_relation_rows_find_a_planted_relation() -> None:
    constants = QQ.frac_field(a, b)
    values = tuple(
        constants.from_sympy(value)
        for value in (
            1 / (a + 1),
            1 / (b + 1),
            1 / (a + 1) + 2 / (b + 1),
        )
    )

    rows = rational_constant_relation_rows(constants, values)
    matrix = DomainMatrix([list(row) for row in rows], (len(rows), 3), QQ)

    assert matrix.rank() == 2
    assert all(-row[0] - 2 * row[1] + row[2] == QQ.zero for row in rows)


def test_rational_constant_relation_rows_scale_by_candidate_dimension() -> None:
    parameters = symbols("a0:12")
    constants = QQ.frac_field(*parameters)
    values = tuple(
        constants.from_sympy(1 / (parameter + 1)) for parameter in parameters
    )

    rows = rational_constant_relation_rows(constants, values)

    assert len(rows) == len(values)


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


def test_recursive_signatures_and_relations_support_finite_constants() -> None:
    base, _constants = _algebraic_rational_base()
    primitive_analysis = certify_primitive_extension(
        Poly(1 / x, t, domain=base.field()),
        base.derivation(),
    )
    assert primitive_analysis.status is ConstantFieldStatus.PRESERVED
    tower = base.extend(primitive_analysis.extension)
    field = tower.field()

    signature = build_fundamental_signature(tower.derivation())

    assert signature.outcome is FundamentalSignatureOutcome.SUPPORTED
    relation_value = 1 / (x * (t + 1))
    system = solve_logarithmic_relation_system(
        (
            field.from_sympy(sqrt(2) * relation_value),
            field.from_sympy((1 - sqrt(2)) * relation_value),
        ),
        tower.derivation(),
    )
    assert system.reconstructs
    assert tuple(relation.source_coefficients for relation in system.relations) == (
        (1, 1),
    )
    assert system.relations[0].argument == field.from_sympy(t + 1)

    recursive_hyperexponential = certify_hyperexponential_extension(
        Poly((sqrt(2) + 1 / x) * e, e, domain=field),
        tower.derivation(),
    )
    assert recursive_hyperexponential.status is ConstantFieldStatus.PRESERVED

    hyperexponential_tower = tower.extend(recursive_hyperexponential.extension)
    hyperexponential_field = hyperexponential_tower.field()
    hyperexponential_system = solve_logarithmic_relation_system(
        (hyperexponential_field.from_sympy(sqrt(2) + 1 / x),),
        hyperexponential_tower.derivation(),
    )
    assert hyperexponential_system.reconstructs
    assert hyperexponential_system.relations[0].argument == (
        hyperexponential_field.from_sympy(e)
    )


def test_parametric_logarithmic_derivative_keeps_parameters_rational() -> None:
    base, _constants = _algebraic_rational_base()
    field = base.field()
    extension = MonomialExtension(
        Poly(x * e, e, domain=field),
        ConstantFieldStatus.UNKNOWN,
    )

    solution = solve_parametric_logarithmic_derivative(
        field.from_sympy(x / 2 + 1 / (x + 1)),
        extension,
        base.derivation(),
    )
    irrational_parameter = solve_parametric_logarithmic_derivative(
        field.from_sympy(sqrt(2) * x),
        extension,
        base.derivation(),
    )

    assert solution.has_solution
    assert (solution.n, solution.m) == (2, 1)
    assert solution.reconstructs
    assert irrational_parameter.proves_no_solution


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


def test_hypertangent_uses_the_exact_gaussian_compositum() -> None:
    base, constants = _algebraic_rational_base()
    primitive = certify_primitive_extension(
        Poly(1 / x, t, domain=base.field()),
        base.derivation(),
    )
    assert primitive.status is ConstantFieldStatus.PRESERVED
    lower_tower = base.extend(primitive.extension)
    analysis = certify_hypertangent_extension(
        Poly((1 + u**2) / x, u, domain=lower_tower.field()),
        lower_tower.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.PRESERVED
    tower = lower_tower.extend(analysis.extension)
    signature = build_fundamental_signature(tower.derivation())
    normal_form = build_hypertangent_gaussian_normal_form(tower.derivation())
    target_constants = gaussian_constant_field(constants)
    field = tower.field()
    outer = MonomialExtension(
        Poly(x**2 * e, e, domain=field),
        ConstantFieldStatus.UNKNOWN,
    )
    parametric = solve_parametric_logarithmic_derivative(
        field.from_sympy(x**2 + 2 * u / x),
        outer,
        tower.derivation(),
    )

    assert signature.outcome is FundamentalSignatureOutcome.SUPPORTED
    assert normal_form.target_derivation.tower.constants == target_constants
    assert target_constants.from_sympy(sqrt(2)) != target_constants.zero
    assert target_constants.from_sympy(I) != target_constants.zero
    assert normal_form.verifies_derivation()
    assert parametric.has_solution
    assert (parametric.n, parametric.m) == (1, 1)
    assert parametric.reconstructs
    assert isinstance(
        parametric.certificate,
        MonomialRelationSystemParametricCertificate,
    )


def test_rational_constant_fields_and_malformed_bases_remain_explicit() -> None:
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
    assert is_supported_rational_constant_field(QQ.frac_field(y))
    assert nonconstant_signature.outcome is FundamentalSignatureOutcome.SUPPORTED

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

    nonreal_constants = QQ.algebraic_field(sqrt(2) + I)
    nonreal_base = _certified_rational_base(nonreal_constants)
    nonreal_analysis = certify_hypertangent_extension(
        Poly(1 + t**2, t, domain=nonreal_base.field()),
        nonreal_base.derivation(),
    )
    hypertangent = nonreal_base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=nonreal_base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    signature = build_fundamental_signature(hypertangent.derivation())

    assert nonreal_analysis.status is ConstantFieldStatus.UNKNOWN
    assert nonreal_analysis.detail is not None
    assert "provably real" in nonreal_analysis.detail
    assert signature.outcome is FundamentalSignatureOutcome.UNSUPPORTED
    assert signature.reason is FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY
    assert "provably real constant field" in signature.detail
