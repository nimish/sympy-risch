"""Parametric logarithmic derivatives over recursive primitive towers."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Dummy, EmptySet, Poly, cancel, linsolve, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    ParametricLogarithmicDerivativeOutcome,
    PrimitiveParametricLogarithmicDerivativeCertificate,
    solve_parametric_logarithmic_derivative,
)


x, t, u, theta = symbols("x t u theta")


def logarithmic_tower(
    *,
    eta=1/x,
    constant_field_status=ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    """Return ``QQ(x)(t)``, with ``Dx = 1`` and ``Dt = eta``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(eta, t, domain=rational.field()),
            constant_field_status,
        ),
    )


def nested_logarithmic_tower(
    *,
    intermediate_status=ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    """Return ``QQ(x)(t)(u)`` with iterated-logarithm derivatives."""
    tower = logarithmic_tower(
        constant_field_status=intermediate_status,
    )
    return tower.extend(
        MonomialExtension(
            Poly(1/(x*t), u, domain=tower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def hyperexponential_extension(
    lower: DifferentialTower,
    eta,
) -> MonomialExtension:
    """Return the proposed extension ``Dtheta = eta*theta``."""
    return MonomialExtension(
        Poly(eta*theta, theta, domain=lower.field()),
    )


def _fraction_polynomials(expression, extension):
    numerator_expression, denominator_expression = cancel(
        expression,
    ).as_numer_denom()
    numerator = Poly(
        numerator_expression,
        t,
        domain=extension.parent_field,
    )
    denominator = Poly(
        denominator_expression,
        t,
        domain=extension.parent_field,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(coefficient)
    leading_inverse = denominator.domain.one/denominator.domain.convert(
        denominator.LC()
    )
    return (
        numerator.mul_ground(leading_inverse),
        denominator.monic(),
    )


def _independent_parametric_solution_set(f, eta, derivation):
    r"""Solve one global identity without production signature coordinates."""
    extension = derivation.tower.levels[derivation.level]
    field = extension.field
    f_expression = field.to_sympy(field.from_sympy(f))
    eta_expression = field.to_sympy(field.from_sympy(eta))
    f_numerator, f_denominator = _fraction_polynomials(
        f_expression,
        extension,
    )
    eta_numerator, eta_denominator = _fraction_polynomials(
        eta_expression,
        extension,
    )
    f_polynomial, _ = f_numerator.div(f_denominator)
    eta_polynomial, _ = eta_numerator.div(eta_denominator)

    top_factors = []
    seen_top_factors = set()
    for denominator in (f_denominator, eta_denominator):
        for factor, _multiplicity in denominator.factor_list()[1]:
            factor = factor.monic()
            expression = factor.as_expr()
            if expression not in seen_top_factors:
                seen_top_factors.add(expression)
                top_factors.append(factor)

    lower_factors = []
    seen_lower_factors = set()
    for polynomial in (f_polynomial, eta_polynomial):
        constant = (
            extension.parent_field.zero
            if polynomial.is_zero
            else extension.parent_field.convert(polynomial.nth(0))
        )
        denominator_expression = cancel(
            extension.parent_field.to_sympy(constant),
        ).as_numer_denom()[1]
        denominator = Poly(denominator_expression, x, domain=QQ)
        for factor, _multiplicity in denominator.factor_list()[1]:
            factor = factor.monic()
            expression = factor.as_expr()
            if expression not in seen_lower_factors:
                seen_lower_factors.add(expression)
                lower_factors.append(factor)

    ratio = Dummy("ratio")
    top_residues = tuple(
        Dummy(f"top_residue_{index}")
        for index in range(len(top_factors))
    )
    lower_residues = tuple(
        Dummy(f"lower_residue_{index}")
        for index in range(len(lower_factors))
    )
    candidate = ratio*eta_expression
    for residue, factor in zip(top_residues, top_factors, strict=True):
        candidate += (
            residue
            * derivation.polynomial(factor).as_expr()
            / factor.as_expr()
        )
    for residue, factor in zip(
        lower_residues,
        lower_factors,
        strict=True,
    ):
        candidate += residue*factor.diff().as_expr()/factor.as_expr()

    identity_numerator = cancel(
        f_expression - candidate,
    ).as_numer_denom()[0]
    equations = Poly(identity_numerator, x, t).coeffs()
    return linsolve(
        equations,
        (ratio, *top_residues, *lower_residues),
    )


def _independent_nested_parametric_solution_set(f, eta, derivation):
    r"""Solve one fixed global ansatz directly over ``QQ(x,t,u)``."""
    field = derivation.field
    f_expression = field.to_sympy(field.from_sympy(f))
    eta_expression = field.to_sympy(field.from_sympy(eta))
    factor_expressions = (x, x**2 + 1, t, u, u**2 + t)
    ratio = Dummy("nested_ratio")
    residues = tuple(
        Dummy(f"nested_residue_{index}")
        for index in range(len(factor_expressions))
    )
    candidate = ratio*eta_expression
    for residue, factor_expression in zip(
        residues,
        factor_expressions,
        strict=True,
    ):
        factor = field.from_sympy(factor_expression)
        candidate += residue*(derivation(factor)/factor).as_expr()
    identity_numerator = cancel(
        f_expression - candidate,
    ).as_numer_denom()[0]
    equations = Poly(identity_numerator, x, t, u).coeffs()
    return linsolve(equations, (ratio, *residues))


def test_polynomial_coordinate_determines_ratio_and_witness() -> None:
    lower = logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    eta = t + 1/(x + 1)
    extension = hyperexponential_extension(lower, eta)
    argument = field.from_sympy(x*t)
    f = (derivation(argument)/argument - 2*field.from_sympy(eta))/3

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert (result.n, result.m) == (3, -2)
    assert result.argument == argument
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        PrimitiveParametricLogarithmicDerivativeCertificate,
    )
    assert result.certificate.candidate_ratio == QQ(-2, 3)
    assert result.certificate.polynomial_constraints
    assert result.certificate.residual_signature.reconstructs_relation


def test_repeated_poles_can_cancel_before_logarithmic_reconstruction() -> None:
    lower = logarithmic_tower()
    eta = 1/t + 1/t**2
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        2*eta + 1/(x*t),
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 2)
    assert result.argument == lower.field().from_sympy(t)
    factor_certificate = result.certificate.factors[0]
    assert factor_certificate.factor == Poly(
        t,
        t,
        domain=lower.levels[1].parent_field,
    )
    assert len(factor_certificate.f_principal_numerators) == 2
    assert factor_certificate.linear_constraints


def test_nonlogarithmic_simple_residues_can_cancel() -> None:
    lower = logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    factor = field.from_sympy(t**2 + x)
    logarithmic_derivative = derivation(factor)/factor
    eta = 1/(t**2 + x)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        2*field.from_sympy(eta) + logarithmic_derivative/3,
        extension,
        derivation,
    )

    assert result.has_solution
    assert (result.n, result.m) == (3, 6)
    assert result.argument == factor
    assert result.reconstructs
    assert result.certificate.candidate_ratio == 2


def test_top_constraints_can_leave_ratio_to_complete_lower_solver() -> None:
    lower = logarithmic_tower()
    eta = 1/(x**2 + 1) + 1/(x*t)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        2/(x**2 + 1) + 1/(3*x*t),
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (3, 6)
    assert result.argument == lower.field().from_sympy(t**-5)
    assert result.certificate.candidate_ratio == 2
    assert result.certificate.lower_parametric_result.has_solution
    assert (
        result.certificate.lower_parametric_result.certificate
        is not None
    )


def test_conflicting_polynomial_and_repeated_pole_ratios_prove_no_solution() -> None:
    lower = logarithmic_tower()
    extension = hyperexponential_extension(
        lower,
        t + 2/t**2,
    )

    result = solve_parametric_logarithmic_derivative(
        t + 1/t**2,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert isinstance(
        result.certificate,
        PrimitiveParametricLogarithmicDerivativeCertificate,
    )
    assert result.certificate.candidate_ratio is None
    assert "inconsistent" in result.certificate.reason


def test_top_ratio_can_fail_complete_lower_logarithmic_test() -> None:
    lower = logarithmic_tower()
    eta = t + 1/(x**2 + 1)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        2*t + 1/(x**2 + 1),
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert result.certificate.candidate_ratio == 2
    assert result.certificate.residual_signature is not None
    assert not result.certificate.residual_signature.has_solution


def test_free_ratio_uses_zero_when_f_is_already_radical_logarithmic() -> None:
    lower = logarithmic_tower()
    eta = 1/x + 1/(x*t)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        1/(2*x) + 1/(3*x*t),
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (6, 0)
    assert result.argument == lower.field().from_sympy(x**3*t**2)
    assert result.reconstructs


@pytest.mark.parametrize(
    "status",
    [
        ConstantFieldStatus.UNKNOWN,
        ConstantFieldStatus.ENLARGED,
    ],
)
def test_unproved_primitive_constant_field_is_unsupported(status) -> None:
    lower = logarithmic_tower(constant_field_status=status)
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative(
        1,
        extension,
        lower.derivation(),
    )

    assert result.is_unsupported
    assert not result.proves_no_solution
    assert "Const" in result.detail


def test_nested_primitive_lower_field_is_decided_recursively() -> None:
    nested = nested_logarithmic_tower()
    extension = MonomialExtension(
        Poly(theta, theta, domain=nested.field()),
    )

    result = solve_parametric_logarithmic_derivative(
        1,
        extension,
        nested.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 1)
    assert result.argument == nested.field().one
    assert result.reconstructs
    assert result.certificate.lower_parametric_result.has_solution


def test_nested_top_coordinate_determines_ratio() -> None:
    lower = nested_logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    eta = u + 1/u**2
    extension = hyperexponential_extension(lower, eta)
    argument = field.from_sympy(u)
    f = 2*field.from_sympy(eta) + derivation(argument)/(3*argument)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert (result.n, result.m) == (3, 6)
    assert result.argument == argument
    assert result.reconstructs
    assert result.certificate.candidate_ratio == 2


def test_nested_free_ratio_descends_to_rational_base() -> None:
    lower = nested_logarithmic_tower()
    eta = 1/(x**2 + 1) + 1/(x*t) + 1/(x*t*u)
    extension = hyperexponential_extension(lower, eta)
    f = 2/(x**2 + 1) + 1/(3*x*t) + 1/(5*x*t*u)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (15, 30)
    assert result.argument == lower.field().from_sympy(t**-25*u**-27)
    assert result.reconstructs
    assert result.certificate.candidate_ratio == 2
    level_one = result.certificate.lower_parametric_result
    assert level_one.has_solution
    assert level_one.certificate.lower_parametric_result.has_solution


def test_nested_top_ratio_can_expose_lower_obstruction() -> None:
    lower = nested_logarithmic_tower()
    eta = u + 1/(x**2 + 1)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        2*u + 1/(x**2 + 1),
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert result.certificate.candidate_ratio == 2
    assert result.certificate.residual_signature is not None
    assert not result.certificate.residual_signature.has_solution


def test_unproved_intermediate_constant_field_is_unsupported() -> None:
    lower = nested_logarithmic_tower(
        intermediate_status=ConstantFieldStatus.UNKNOWN,
    )
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative(
        1,
        extension,
        lower.derivation(),
    )

    assert result.is_unsupported
    assert "every primitive level" in result.detail


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=-6, max_value=6),
    x_power=st.integers(min_value=-3, max_value=3),
    t_power=st.integers(min_value=-3, max_value=3),
    u_power=st.integers(min_value=-3, max_value=3),
    mixed_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_nested_primitive_parametric_witnesses(
    n,
    m,
    x_power,
    t_power,
    u_power,
    mixed_power,
) -> None:
    lower = nested_logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    eta = u + 1/u**2 + 1/(u**2 + t)
    extension = hyperexponential_extension(lower, eta)
    argument = field.from_sympy(
        x**x_power
        * t**t_power
        * u**u_power
        * (u**2 + t)**mixed_power
    )
    f = (
        derivation(argument)/argument
        + m*field.from_sympy(eta)
    )/n

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert QQ(result.m, result.n) == QQ(m, n)
    assert result.reconstructs


@settings(max_examples=25, deadline=None)
@given(
    f_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=9,
        max_size=9,
    ).map(tuple),
    eta_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=9,
        max_size=9,
    ).filter(any).map(tuple),
)
def test_nested_solver_matches_independent_global_parametric_oracle(
    f_coefficients,
    eta_coefficients,
) -> None:
    lower = nested_logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    mixed = field.from_sympy(u**2 + t)
    basis = (
        1/x,
        2*x/(x**2 + 1),
        1/(x*t),
        1/(x*t*u),
        (derivation(mixed)/mixed).as_expr(),
        u,
        1/u**2,
        1/u,
        1/(u**2 + t),
    )
    f = sum(
        coefficient*term
        for coefficient, term in zip(f_coefficients, basis, strict=True)
    )
    eta = sum(
        coefficient*term
        for coefficient, term in zip(eta_coefficients, basis, strict=True)
    )
    assume(cancel(eta) != 0)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )
    oracle_solutions = _independent_nested_parametric_solution_set(
        f,
        eta,
        derivation,
    )

    assert result.outcome in {
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
    }
    assert result.has_solution is (oracle_solutions is not EmptySet)
    if result.has_solution:
        assert result.reconstructs


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=-6, max_value=6),
    x_power=st.integers(min_value=-3, max_value=3),
    t_power=st.integers(min_value=-3, max_value=3),
    quadratic_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_planted_primitive_parametric_witnesses(
    n,
    m,
    x_power,
    t_power,
    quadratic_power,
) -> None:
    lower = logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    eta = t + 1/t**2 + 1/(t**2 + x)
    extension = hyperexponential_extension(lower, eta)
    argument = field.from_sympy(
        x**x_power*t**t_power*(t**2 + x)**quadratic_power
    )
    f = (
        derivation(argument)/argument
        + m*field.from_sympy(eta)
    )/n

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert QQ(result.m, result.n) == QQ(m, n)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        PrimitiveParametricLogarithmicDerivativeCertificate,
    )


@settings(max_examples=80, deadline=None)
@given(
    f_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=13,
        max_size=13,
    ).map(tuple),
    eta_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=13,
        max_size=13,
    ).filter(any).map(tuple),
)
def test_complete_solver_matches_independent_global_primitive_oracle(
    f_coefficients,
    eta_coefficients,
) -> None:
    lower = logarithmic_tower()
    field = lower.field()
    derivation = lower.derivation()
    quadratic = field.from_sympy(t**2 + x)
    quadratic_logarithmic_derivative = (
        derivation(quadratic)/quadratic
    ).as_expr()
    cubic = field.from_sympy(t**3 + t + x)
    cubic_logarithmic_derivative = (
        derivation(cubic)/cubic
    ).as_expr()
    basis = (
        1/x,
        1/x**2,
        1/(x**2 + 1),
        t,
        t**2,
        1/(x*t),
        1/t,
        quadratic_logarithmic_derivative,
        1/(t**2 + x),
        cubic_logarithmic_derivative,
        1/(t**3 + t + x),
        1/t**2,
        1/(t - x)**2,
    )
    f = sum(
        coefficient*term
        for coefficient, term in zip(f_coefficients, basis, strict=True)
    )
    eta = sum(
        coefficient*term
        for coefficient, term in zip(eta_coefficients, basis, strict=True)
    )
    assume(cancel(eta) != 0)
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )
    oracle_solutions = _independent_parametric_solution_set(
        f,
        eta,
        derivation,
    )

    assert result.outcome in {
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
    }
    assert result.has_solution is (oracle_solutions is not EmptySet)
    if result.has_solution:
        assert result.reconstructs
