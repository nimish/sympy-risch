import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, Rational, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    NonlinearNoSpecialStatus,
    SpecialPolynomialStatus,
    bronstein_residue_criterion,
    hermite_reduce_monomial,
    hermite_reduce_rational_recursive,
    integrate_rational,
    polynomial_reduce_monomial,
    polynomial_reduce_primitive,
    reduce_monomial_reduced_function,
    reduce_nonlinear_monomial,
    reduce_nonlinear_no_special,
    reduce_primitive_monomial,
    residue_reduce_monomial,
    split_normal_special,
)


x, t = symbols("x t")


def monomial_tower(
    generator_derivative,
    *,
    special_polynomial_status=SpecialPolynomialStatus.UNKNOWN,
) -> DifferentialTower:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
            special_polynomial_status,
        ),
    )


def test_exponential_normal_special_decomposition() -> None:
    tower = monomial_tower(t)
    parent = tower.field(0)
    polynomial = Poly(6*t**2*(t - 1)**3*(t + 1), t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(6*(t - 1)**3*(t + 1), t, domain=parent)
    assert special == Poly(t**2, t, domain=parent)
    assert normal*special == polynomial


def test_tangent_normal_special_decomposition() -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(t**2*(1 + t**2)**3, t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(t**2, t, domain=parent)
    assert special == Poly((1 + t**2)**3, t, domain=parent)


def test_coefficient_derivatives_expose_a_special_new_constant() -> None:
    # With Dx = Dt = 1, D(t - x) = 0.  This checks that decomposition uses
    # the total tower derivative rather than only differentiation in t.
    tower = monomial_tower(1)
    parent = tower.field(0)
    polynomial = Poly((t - x)**2*(t + 1)**3, t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly((t + 1)**3, t, domain=parent)
    assert special == Poly((t - x)**2, t, domain=parent)


def test_general_polynomial_monomials_use_the_same_decomposition() -> None:
    tower = monomial_tower(t**3)
    parent = tower.field(0)
    polynomial = Poly(t**4*(t - 1)**2, t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly((t - 1)**2, t, domain=parent)
    assert special == Poly(t**4, t, domain=parent)


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
)
def test_generated_exponential_decompositions_reconstruct_the_input(
    coefficient,
    special_multiplicity,
    normal_multiplicity,
) -> None:
    tower = monomial_tower(t)
    parent = tower.field(0)
    polynomial = Poly(
        coefficient*t**special_multiplicity*(t - 1)**normal_multiplicity,
        t,
        domain=parent,
    )

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(
        coefficient*(t - 1)**normal_multiplicity,
        t,
        domain=parent,
    )
    assert special == Poly(t**special_multiplicity, t, domain=parent)
    assert normal*special == polynomial


@given(
    coefficient=st.integers(min_value=1, max_value=6),
    special_multiplicity=st.integers(min_value=0, max_value=5),
    normal_multiplicity=st.integers(min_value=0, max_value=5),
)
def test_generated_tangent_decompositions_reconstruct_the_input(
    coefficient,
    special_multiplicity,
    normal_multiplicity,
) -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(
        coefficient
        * (1 + t**2)**special_multiplicity
        * t**normal_multiplicity,
        t,
        domain=parent,
    )

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(
        coefficient*t**normal_multiplicity,
        t,
        domain=parent,
    )
    assert special == Poly(
        (1 + t**2)**special_multiplicity,
        t,
        domain=parent,
    )
    assert normal*special == polynomial


def test_general_nonlinear_decomposition_can_have_multiple_special_factors() -> None:
    tower = monomial_tower(t*(1 + t**2))
    parent = tower.field(0)
    polynomial = Poly(
        3*t**2*(1 + t**2)**3*(t - 1)**4,
        t,
        domain=parent,
    )

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(3*(t - 1)**4, t, domain=parent)
    assert special == Poly(t**2*(1 + t**2)**3, t, domain=parent)
    assert normal*special == polynomial


def test_normal_special_decomposition_validates_its_polynomial() -> None:
    tower = monomial_tower(1/x)

    with pytest.raises(TypeError, match="SymPy Poly"):
        split_normal_special(t, tower.derivation())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero polynomial"):
        split_normal_special(
            Poly(0, t, domain=tower.field(0)),
            tower.derivation(),
        )
    with pytest.raises(ValueError, match="generator does not match"):
        split_normal_special(Poly(x + 1, x, domain=QQ), tower.derivation())


def test_tangent_polynomial_reduction() -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(t**4, t, domain=parent)

    exact, remainder = polynomial_reduce_monomial(
        polynomial,
        tower.derivation(),
    )

    assert exact == Poly(t**3/3 - t, t, domain=parent)
    assert remainder == Poly(1, t, domain=parent)
    assert tower.derivation().polynomial(exact) + remainder == polynomial


def test_polynomial_reduction_differentiates_lower_field_coefficients() -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(x*t**3, t, domain=parent)

    exact, remainder = polynomial_reduce_monomial(
        polynomial,
        tower.derivation(),
    )

    assert exact == Poly(x*t**2/2 - t/2, t, domain=parent)
    assert remainder == Poly(-x*t + 1/2, t, domain=parent)
    assert tower.derivation().polynomial(exact) + remainder == polynomial


@pytest.mark.parametrize(
    "generator_derivative",
    [1 + t**2, t**3 + x, x*t**2 + t + 1],
)
def test_nonlinear_polynomial_reduction_reaches_the_degree_bound(
    generator_derivative,
) -> None:
    tower = monomial_tower(generator_derivative)
    parent = tower.field(0)
    polynomial = Poly((x + 1)*t**8 + x*t**5 + t + 1, t, domain=parent)
    delta = tower.levels[-1].generator_derivative.degree()

    exact, remainder = polynomial_reduce_monomial(
        polynomial,
        tower.derivation(),
    )

    assert tower.derivation().polynomial(exact) + remainder == polynomial
    assert remainder.is_zero or remainder.degree() < delta

    second_exact, second_remainder = polynomial_reduce_monomial(
        remainder,
        tower.derivation(),
    )
    assert second_exact.is_zero
    assert second_remainder == remainder


def test_nonlinear_polynomial_reduction_handles_zero_and_low_degree() -> None:
    tower = monomial_tower(t**3 + x)
    parent = tower.field(0)

    for polynomial in (
        Poly(0, t, domain=parent),
        Poly(x*t**2 + 1, t, domain=parent),
    ):
        exact, remainder = polynomial_reduce_monomial(
            polynomial,
            tower.derivation(),
        )

        assert exact.is_zero
        assert remainder == polynomial


@pytest.mark.parametrize("generator_derivative", [1/x, t, t + 1])
def test_nonlinear_polynomial_reduction_rejects_linear_families(
    generator_derivative,
) -> None:
    tower = monomial_tower(generator_derivative)

    with pytest.raises(ValueError, match=r"degree\(Dt\) >= 2"):
        polynomial_reduce_monomial(
            Poly(t**3, t, domain=tower.field(0)),
            tower.derivation(),
        )


def test_nonlinear_polynomial_reduction_validates_the_polynomial() -> None:
    tower = monomial_tower(1 + t**2)

    with pytest.raises(TypeError, match="SymPy Poly"):
        polynomial_reduce_monomial(t**3, tower.derivation())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="generator does not match"):
        polynomial_reduce_monomial(
            Poly(x**3, x, domain=QQ),
            tower.derivation(),
        )


@given(
    generator_derivative=st.sampled_from(
        (1 + t**2, t**3 + x, x*t**2 + t + 1),
    ),
    coefficient=st.integers(min_value=-4, max_value=4),
    constant=st.integers(min_value=-4, max_value=4),
    exponent=st.integers(min_value=0, max_value=7),
    remainder_linear=st.integers(min_value=-4, max_value=4),
    remainder_constant=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_nonlinear_polynomial_reductions_reconstruct_input(
    generator_derivative,
    coefficient,
    constant,
    exponent,
    remainder_linear,
    remainder_constant,
) -> None:
    tower = monomial_tower(generator_derivative)
    parent = tower.field(0)
    derivation = tower.derivation()
    planted_exact = Poly(
        (coefficient*x + constant)*t**exponent,
        t,
        domain=parent,
    )
    planted_remainder = Poly(
        remainder_linear*t + remainder_constant,
        t,
        domain=parent,
    )
    polynomial = derivation.polynomial(planted_exact) + planted_remainder

    exact, remainder = polynomial_reduce_monomial(polynomial, derivation)

    delta = tower.levels[-1].generator_derivative.degree()
    assert derivation.polynomial(exact) + remainder == polynomial
    assert remainder.is_zero or remainder.degree() < delta


@pytest.mark.parametrize(
    ("polynomial_expression", "expected_exact_expression"),
    [
        (t, x*t - x),
        (t**2, x*(t**2 - 2*t + 2)),
        (t**3/x, t**4/4),
    ],
)
def test_primitive_polynomial_reduction_basic_logarithmic_examples(
    polynomial_expression,
    expected_exact_expression,
) -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    polynomial = Poly(polynomial_expression, t, domain=parent)

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.exact_part == Poly(
        expected_exact_expression,
        t,
        domain=parent,
    )
    assert result.remainder.is_zero
    assert not result.has_obstruction
    assert tower.derivation().polynomial(result.exact_part) == polynomial


def test_primitive_polynomial_reduction_records_the_eta_direction() -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    polynomial = Poly(t**5/x, t, domain=parent)

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.exact_part == Poly(t**6/6, t, domain=parent)
    assert result.remainder.is_zero
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.degree == 5
    assert step.exact_coefficient == parent.zero
    assert step.eta_multiple == QQ.one
    assert step.obstruction == parent.zero


@pytest.mark.parametrize(
    "eta",
    [
        1/x - 1/(x - 1)**2,
        1/x + 2/(x - 1),
        1/(x**2 + 1),
    ],
)
def test_primitive_polynomial_reduction_handles_general_rational_eta(
    eta,
) -> None:
    tower = monomial_tower(eta)
    parent = tower.field(0)
    polynomial = Poly(eta*t**4, t, domain=parent)

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.exact_part == Poly(t**5/5, t, domain=parent)
    assert result.remainder.is_zero
    assert result.steps[0].eta_multiple == QQ.one


def test_primitive_polynomial_reduction_uses_one_residue_coordinate_as_pivot() -> None:
    # In QQ[x]/(x**2 + 1), the residue numerators 1 and x are independent
    # over QQ.  Quotienting by C*eta removes the constant coordinate while
    # retaining the x coordinate as a canonical obstruction.
    eta = 1/(x**2 + 1)
    tower = monomial_tower(eta)
    parent = tower.field(0)
    independent_obstruction = parent.from_sympy(x/(x**2 + 1))
    polynomial = Poly((x + 3)*t**2/(x**2 + 1), t, domain=parent)

    result = polynomial_reduce_primitive(polynomial, tower.derivation())
    leading_step = result.steps[0]

    assert leading_step.eta_multiple == QQ(3)
    assert leading_step.obstruction == independent_obstruction
    assert parent.convert(result.remainder.nth(2)) == independent_obstruction
    assert (
        tower.derivation().polynomial(result.exact_part)
        + result.remainder
        == polynomial
    )

    second = polynomial_reduce_primitive(
        result.remainder,
        tower.derivation(),
    )
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


def test_primitive_polynomial_reduction_exposes_a_nonproportional_class() -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    polynomial = Poly(t**3/(x - 1), t, domain=parent)

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.exact_part.is_zero
    assert result.remainder == polynomial
    assert result.has_obstruction
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.exact_coefficient == parent.zero
    assert step.eta_multiple == QQ.zero
    assert step.obstruction == parent.from_sympy(1/(x - 1))

    # The obstruction has a pole at x=1 while eta=1/x does not, so no
    # rational constant multiple of eta can cancel its residue class.
    assert cancel(
        step.obstruction.as_expr()/(parent.from_sympy(1/x)).as_expr()
    ) == x/(x - 1)

    second = polynomial_reduce_primitive(
        result.remainder,
        tower.derivation(),
    )
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


def test_primitive_polynomial_reduction_separates_all_three_components() -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    derivation = tower.derivation()
    exact_coefficient = parent.from_sympy(1/(x - 2))
    eta = parent.from_sympy(1/x)
    obstruction = parent.from_sympy(1/(x - 1))
    coefficient = (
        derivation.lower()(exact_coefficient)
        + 2*eta
        + obstruction
    )
    polynomial = Poly.from_dict({(2,): coefficient}, t, domain=parent)

    result = polynomial_reduce_primitive(polynomial, derivation)
    leading_step = result.steps[0]

    assert leading_step.degree == 2
    assert leading_step.coefficient == coefficient
    assert leading_step.exact_coefficient == exact_coefficient
    assert leading_step.eta_multiple == QQ(2)
    assert leading_step.obstruction == obstruction
    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial
    assert parent.convert(result.remainder.nth(2)) == obstruction


def test_primitive_polynomial_steps_are_independently_reconstructible() -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    derivation = tower.derivation()
    eta = parent.from_sympy(1/x)
    polynomial = Poly(
        (1/(x - 1) - 1/(x - 2)**2 + 3/x)*t**3
        + (x + 1)*t,
        t,
        domain=parent,
    )

    result = polynomial_reduce_primitive(polynomial, derivation)

    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial
    for step in result.steps:
        assert step.coefficient == (
            derivation.lower()(step.exact_coefficient)
            + parent.convert(step.eta_multiple)*eta
            + step.obstruction
        )


def test_primitive_polynomial_reduction_handles_zero() -> None:
    tower = monomial_tower(1/x)
    zero = Poly(0, t, domain=tower.field(0))

    result = polynomial_reduce_primitive(zero, tower.derivation())

    assert result.exact_part.is_zero
    assert result.remainder.is_zero
    assert result.steps == ()


@pytest.mark.parametrize("eta", [1, 1/x**2, 2*x + 1])
def test_primitive_polynomial_reduction_rejects_an_exact_Dt(eta) -> None:
    # If eta = Du in QQ(x), then D(t-u)=0, contradicting the asserted
    # no-new-constants hypothesis.
    tower = monomial_tower(eta)

    with pytest.raises(
        ValueError,
        match="Dt is exact in the lower field",
    ):
        polynomial_reduce_primitive(
            Poly(t, t, domain=tower.field(0)),
            tower.derivation(),
        )


def test_primitive_polynomial_reduction_requires_constant_field_evidence() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    unknown = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
        ),
    )

    with pytest.raises(ValueError, match="status is unknown"):
        polynomial_reduce_primitive(
            Poly(t, t, domain=rational.field()),
            unknown.derivation(),
        )


def test_primitive_polynomial_reduction_validates_family_and_base() -> None:
    hyperexponential = monomial_tower(t)
    with pytest.raises(ValueError, match="requires Dt in K"):
        polynomial_reduce_primitive(
            Poly(t, t, domain=hyperexponential.field(0)),
            hyperexponential.derivation(),
        )

    scaled_base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(2, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    primitive = scaled_base.extend(
        MonomialExtension(
            Poly(1/x, t, domain=scaled_base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    with pytest.raises(NotImplementedError, match=r"Dx = 1"):
        polynomial_reduce_primitive(
            Poly(t, t, domain=scaled_base.field()),
            primitive.derivation(),
        )

    with pytest.raises(TypeError, match="SymPy Poly"):
        polynomial_reduce_primitive(
            t,  # type: ignore[arg-type]
            monomial_tower(1/x).derivation(),
        )


@given(
    eta=st.sampled_from(
        (
            1/x,
            1/x - 1/(x - 1)**2,
            1/x + 2/(x - 1),
            1/(x**2 + 1),
        )
    ),
    numerator_x=st.integers(min_value=-4, max_value=4),
    numerator_constant=st.integers(min_value=-4, max_value=4),
    pole_root=st.integers(min_value=1, max_value=5),
    pole_power=st.integers(min_value=1, max_value=4),
    degree=st.integers(min_value=0, max_value=6),
    raised_coefficient=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=45, deadline=None)
def test_generated_exact_primitive_polynomials_reduce_to_zero(
    eta,
    numerator_x,
    numerator_constant,
    pole_root,
    pole_power,
    degree,
    raised_coefficient,
) -> None:
    tower = monomial_tower(eta)
    parent = tower.field(0)
    derivation = tower.derivation()
    coefficient = (
        parent.from_sympy(
            (numerator_x*x + numerator_constant)/(x - pole_root)**pole_power
        )
    )
    planted_exact = Poly.from_dict(
        {
            (degree,): coefficient,
            (degree + 1,): parent.convert(raised_coefficient),
        },
        t,
        domain=parent,
    )
    polynomial = derivation.polynomial(planted_exact)

    result = polynomial_reduce_primitive(polynomial, derivation)

    assert derivation.polynomial(result.exact_part) == polynomial
    assert result.remainder.is_zero
    assert not result.has_obstruction


@given(
    exact_numerator=st.integers(min_value=-3, max_value=3),
    exact_root=st.integers(min_value=-5, max_value=-1),
    exact_power=st.integers(min_value=1, max_value=4),
    obstruction_residue=st.integers(min_value=-4, max_value=4).filter(bool),
    obstruction_root=st.integers(min_value=1, max_value=5),
    obstruction_degree=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=45, deadline=None)
def test_generated_nonproportional_primitive_obstructions_are_canonical(
    exact_numerator,
    exact_root,
    exact_power,
    obstruction_residue,
    obstruction_root,
    obstruction_degree,
) -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    derivation = tower.derivation()
    planted_exact = Poly(
        exact_numerator*t**(obstruction_degree + 1)
        /(x - exact_root)**exact_power,
        t,
        domain=parent,
    )
    planted_obstruction = Poly(
        obstruction_residue*t**obstruction_degree/(x - obstruction_root),
        t,
        domain=parent,
    )
    polynomial = (
        derivation.polynomial(planted_exact)
        + planted_obstruction
    )

    result = polynomial_reduce_primitive(polynomial, derivation)

    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial
    assert result.remainder == planted_obstruction
    assert result.has_obstruction

    second = polynomial_reduce_primitive(result.remainder, derivation)
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


@pytest.mark.parametrize(
    ("integrand_expression", "expected_exact_expression"),
    [
        (t, x*t - x),
        (t**2, x*(t**2 - 2*t + 2)),
        (t**4/x, t**5/5),
    ],
)
def test_primitive_pipeline_integrates_polynomial_examples(
    integrand_expression,
    expected_exact_expression,
) -> None:
    tower = monomial_tower(1/x)
    field = tower.field()

    result = reduce_primitive_monomial(
        field.from_sympy(integrand_expression),
        tower.derivation(),
    )

    assert result.exact_part == field.from_sympy(expected_exact_expression)
    assert result.logarithmic_derivative == field.zero
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert result.reduced_to_lower_field
    assert not result.is_non_elementary


def test_primitive_pipeline_extracts_log_log_x() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(1/(x*t))

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.exact_part == field.zero
    assert result.logarithmic_derivative == integrand
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert result.residue_reduction.logarithmic_parts
    assert result.reduced_to_lower_field


def test_primitive_pipeline_certifies_one_over_log_x_non_elementary() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(1/t)

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.has_finite_residue_obstruction
    assert not result.has_polynomial_obstruction
    assert result.is_non_elementary
    assert not result.reduced_to_lower_field
    assert result.lower_remainder is None
    assert result.remainder == integrand
    assert not result.residue_reduction.certificate.all_residues_constant

    second = reduce_primitive_monomial(result.remainder, tower.derivation())
    assert second.exact_part == field.zero
    assert second.logarithmic_derivative == field.zero
    assert second.remainder == result.remainder


def test_primitive_pipeline_certifies_a_positive_degree_obstruction() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(t/(x - 1))

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert not result.has_finite_residue_obstruction
    assert result.has_polynomial_obstruction
    assert result.is_non_elementary
    assert result.lower_remainder is None
    assert result.remainder == integrand

    second = reduce_primitive_monomial(result.remainder, tower.derivation())
    assert second.exact_part == field.zero
    assert second.logarithmic_derivative == field.zero
    assert second.remainder == result.remainder


def test_primitive_pipeline_returns_a_constant_polynomial_to_the_lower_field() -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    field = tower.field()
    integrand = field.from_sympy(1/(x - 1))

    result = reduce_primitive_monomial(integrand, tower.derivation())

    assert result.reduced_to_lower_field
    assert not result.is_non_elementary
    assert result.lower_remainder == parent.from_sympy(1/(x - 1))
    lower_integral = integrate_rational(
        result.lower_remainder.as_expr(),
        x,
    )
    assert cancel(lower_integral.diff(x) - integrand.as_expr()) == 0


def test_primitive_pipeline_combines_every_reduction_stage() -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    field = tower.field()
    derivation = tower.derivation()
    repeated_pole_exact = field.from_sympy((x + 1)/(t - 2)**2)
    polynomial_exact = field.from_sympy(t**4)
    normal_factor = field.from_sympy(t - 1)
    logarithmic_derivative = 3*derivation(normal_factor)/normal_factor
    lower_remainder = parent.from_sympy(1/(x - 3))
    integrand = (
        derivation(repeated_pole_exact + polynomial_exact)
        + logarithmic_derivative
        + field.convert(lower_remainder)
    )

    result = reduce_primitive_monomial(integrand, derivation)

    assert result.hermite_exact_part == repeated_pole_exact
    assert result.exact_part == repeated_pole_exact + polynomial_exact
    assert result.logarithmic_derivative == logarithmic_derivative
    assert result.lower_remainder == lower_remainder
    assert result.reduced_to_lower_field
    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )


def test_primitive_pipeline_preserves_simultaneous_obstructions() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(t - 1)
    finite_obstruction = (
        field.from_sympy(x)*derivation(normal_factor)/normal_factor
    )
    polynomial_obstruction = field.from_sympy(t/(x - 2))
    integrand = finite_obstruction + polynomial_obstruction

    result = reduce_primitive_monomial(integrand, derivation)

    assert result.has_finite_residue_obstruction
    assert result.has_polynomial_obstruction
    assert result.is_non_elementary
    assert result.lower_remainder is None
    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )


def test_primitive_pipeline_enforces_public_preconditions() -> None:
    hyperexponential = monomial_tower(t)
    with pytest.raises(ValueError, match="requires Dt in K"):
        reduce_primitive_monomial(
            hyperexponential.field().from_sympy(t),
            hyperexponential.derivation(),
        )

    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    unknown = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
        ),
    )
    with pytest.raises(ValueError, match="status is unknown"):
        reduce_primitive_monomial(
            unknown.field().from_sympy(t),
            unknown.derivation(),
        )

    exact_eta = monomial_tower(1)
    with pytest.raises(ValueError, match="Dt is exact in the lower field"):
        reduce_primitive_monomial(
            exact_eta.field().from_sympy(1),
            exact_eta.derivation(),
        )

    residue_collision = monomial_tower(1/x)
    with pytest.raises(ValueError, match="residue symbol must be new"):
        reduce_primitive_monomial(
            residue_collision.field().from_sympy(1/t),
            residue_collision.derivation(),
            residue=x,
        )


@given(
    repeated_coefficient=st.integers(min_value=-3, max_value=3),
    repeated_power=st.integers(min_value=2, max_value=4),
    polynomial_coefficient=st.integers(min_value=-3, max_value=3),
    polynomial_power=st.integers(min_value=1, max_value=6),
    logarithmic_residue=st.integers(min_value=-3, max_value=3),
    lower_residue=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=35, deadline=None)
def test_generated_primitive_pipelines_reconstruct_successful_stages(
    repeated_coefficient,
    repeated_power,
    polynomial_coefficient,
    polynomial_power,
    logarithmic_residue,
    lower_residue,
) -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        repeated_coefficient/(t - 2)**repeated_power
        + polynomial_coefficient*t**polynomial_power
    )
    normal_factor = field.from_sympy(t - 1)
    planted_logarithmic_derivative = (
        logarithmic_residue*derivation(normal_factor)/normal_factor
    )
    lower_remainder = parent.from_sympy(lower_residue/(x - 3))
    integrand = (
        derivation(planted_exact)
        + planted_logarithmic_derivative
        + field.convert(lower_remainder)
    )

    result = reduce_primitive_monomial(integrand, derivation)

    assert result.reduced_to_lower_field
    assert result.lower_remainder == lower_remainder
    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )


@given(
    residue_x_coefficient=st.integers(min_value=-4, max_value=4).filter(bool),
    residue_constant=st.integers(min_value=-4, max_value=4),
    factor_shift=st.integers(min_value=1, max_value=5),
    exact_coefficient=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_generated_primitive_finite_residue_obstructions_are_preserved(
    residue_x_coefficient,
    residue_constant,
    factor_shift,
    exact_coefficient,
) -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t - factor_shift)
    residue_value = field.from_sympy(
        residue_x_coefficient*x + residue_constant
    )
    finite_obstruction = residue_value*derivation(factor)/factor
    planted_exact = field.from_sympy(exact_coefficient/(t + 1)**2)
    integrand = derivation(planted_exact) + finite_obstruction

    result = reduce_primitive_monomial(integrand, derivation)

    assert result.has_finite_residue_obstruction
    assert result.is_non_elementary
    assert result.lower_remainder is None
    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )


@given(
    obstruction_residue=st.integers(min_value=-4, max_value=4).filter(bool),
    obstruction_root=st.integers(min_value=1, max_value=5),
    obstruction_degree=st.integers(min_value=1, max_value=5),
    exact_coefficient=st.integers(min_value=-3, max_value=3),
    exact_power=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=30, deadline=None)
def test_generated_primitive_polynomial_obstructions_are_preserved(
    obstruction_residue,
    obstruction_root,
    obstruction_degree,
    exact_coefficient,
    exact_power,
) -> None:
    tower = monomial_tower(1/x)
    parent = tower.field(0)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(exact_coefficient*t**exact_power)
    planted_obstruction = Poly(
        obstruction_residue*t**obstruction_degree/(x - obstruction_root),
        t,
        domain=parent,
    )
    integrand = (
        derivation(planted_exact)
        + field.from_sympy(planted_obstruction.as_expr())
    )

    result = reduce_primitive_monomial(integrand, derivation)

    assert not result.has_finite_residue_obstruction
    assert result.has_polynomial_obstruction
    assert result.is_non_elementary
    assert result.polynomial_reduction.remainder == planted_obstruction
    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )


def test_tangent_reduced_function_extracts_the_logarithm_at_infinity() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_coefficient == 1
    assert reduction.infinity_residue == 1
    assert reduction.infinity_residue_derivative == 0
    assert reduction.logarithmic_coefficient == tower.field(0).convert(QQ(1, 2))
    assert reduction.logarithmic_argument == Poly(
        1 + t**2,
        t,
        domain=tower.field(0),
    )
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert not reduction.has_infinity_obstruction
    assert not reduction.needs_special_polynomial


def test_reduced_function_handles_a_nonunit_derivative_leading_coefficient() -> None:
    tower = monomial_tower(x*(1 + t**2))
    field = tower.field()
    integrand = field.from_sympy(x*t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_coefficient == tower.field(0).from_sympy(x)
    assert reduction.infinity_residue == 1
    assert reduction.infinity_residue_derivative == 0
    assert reduction.logarithmic_coefficient == tower.field(0).convert(QQ(1, 2))
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero


@given(
    scale=st.integers(min_value=-5, max_value=5).filter(bool),
    exponent=st.integers(min_value=1, max_value=4),
)
def test_structure_theorems_reject_new_constant_counterexamples(
    scale,
    exponent,
) -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = rational.extend(
        MonomialExtension(
            Poly(scale*t**(exponent + 1), t, domain=rational.field()),
            ConstantFieldStatus.ENLARGED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    generator = field.from_sympy(t)
    new_constant = field.from_sympy(
        x + 1/(exponent*scale*t**exponent),
    )
    integrand = field.from_sympy(scale*x*t**exponent)
    normal_factor = field.from_sympy(t - 1)
    logarithmic_integrand = (
        new_constant*derivation(normal_factor)/normal_factor
    )

    assert derivation(new_constant) == field.zero
    # D(new_constant*log(t) - x/exponent) equals the polynomial integrand.
    assert (
        new_constant*derivation(generator)/generator
        - field.from_sympy(Rational(1, exponent))
        == integrand
    )
    with pytest.raises(ValueError, match=r"Const\(K\(t\)\) = Const\(K\)"):
        reduce_monomial_reduced_function(integrand, derivation)
    with pytest.raises(ValueError, match=r"Const\(K\(t\)\) = Const\(K\)"):
        bronstein_residue_criterion(logarithmic_integrand, derivation)


def test_reduced_function_certifies_a_nonconstant_infinity_residue() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(x*t + 3)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_coefficient == tower.field(0).from_sympy(x)
    assert reduction.infinity_residue == tower.field(0).from_sympy(x)
    assert reduction.infinity_residue_derivative == 1
    assert reduction.logarithmic_argument is None
    assert reduction.logarithmic_derivative == field.zero
    assert reduction.remainder == integrand
    assert reduction.has_infinity_obstruction
    assert not reduction.needs_special_polynomial


def test_reduced_function_uses_its_special_denominator() -> None:
    tower = monomial_tower(t**3)
    field = tower.field()
    integrand = field.from_sympy(2*t**2 + 1/t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.special_numerator == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert reduction.special_denominator == Poly(
        t,
        t,
        domain=tower.field(0),
    )
    assert reduction.logarithmic_coefficient == 2
    assert reduction.logarithmic_argument == reduction.special_denominator
    assert reduction.logarithmic_derivative == field.from_sympy(2*t**2)
    assert reduction.remainder == field.from_sympy(1/t)


def test_reduced_function_can_use_a_supplied_special_polynomial() -> None:
    # q = t - x is special because Dq = t*q, but it is not a factor of Dt.
    tower = monomial_tower(t**2 - x*t + 1)
    parent = tower.field(0)
    field = tower.field()
    special = Poly(t - x, t, domain=parent)

    reduction = reduce_monomial_reduced_function(
        field.from_sympy(t),
        tower.derivation(),
        special,
    )

    assert reduction.logarithmic_coefficient == 1
    assert reduction.logarithmic_argument == special
    assert reduction.logarithmic_derivative == field.from_sympy(t)
    assert reduction.remainder == field.zero


def test_reduced_function_defers_when_no_special_polynomial_is_known() -> None:
    tower = monomial_tower(t**2 + x)
    field = tower.field()
    integrand = field.from_sympy(t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_residue == 1
    assert reduction.infinity_residue_derivative == 0
    assert reduction.logarithmic_argument is None
    assert reduction.remainder == integrand
    assert not reduction.has_infinity_obstruction
    assert reduction.needs_special_polynomial


def test_reduced_function_leaves_lower_degree_and_special_parts() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(4*t + 3 + 1/(1 + t**2))

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.logarithmic_coefficient == 2
    assert reduction.logarithmic_derivative == field.from_sympy(4*t)
    assert reduction.remainder == field.from_sympy(3 + 1/(1 + t**2))
    assert reduction.logarithmic_derivative + reduction.remainder == integrand


def test_reduced_function_requires_the_previous_reductions() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()

    with pytest.raises(ValueError, match="normal residues"):
        reduce_monomial_reduced_function(
            field.from_sympy(t + 1/(t - 1)),
            derivation,
        )
    with pytest.raises(ValueError, match="polynomial reduction"):
        reduce_monomial_reduced_function(field.from_sympy(t**2), derivation)
    with pytest.raises(ValueError, match="must divide"):
        reduce_monomial_reduced_function(
            field.from_sympy(t),
            derivation,
            Poly(t - 1, t, domain=tower.field(0)),
        )


@given(
    example=st.sampled_from(
        (
            (1 + t**2, 1 + t**2),
            (t**3, t),
            (t**3, t**2),
        ),
    ),
    logarithmic_coefficient=st.integers(min_value=-4, max_value=4),
    background=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_generated_reduced_function_logarithms_reconstruct_input(
    example,
    logarithmic_coefficient,
    background,
) -> None:
    generator_derivative, special_expression = example
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    special = field.from_sympy(special_expression)
    planted_logarithmic_derivative = (
        logarithmic_coefficient*derivation(special)/special
    )
    integrand = planted_logarithmic_derivative + background

    reduction = reduce_monomial_reduced_function(integrand, derivation)

    assert reduction.logarithmic_derivative + reduction.remainder == integrand
    assert reduction.remainder == field.from_sympy(background)
    assert not reduction.has_infinity_obstruction
    assert not reduction.needs_special_polynomial


def test_nonlinear_pipeline_combines_every_common_reduction_stage() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    repeated_pole_exact = field.from_sympy((x + 1)/(t - 1)**2)
    polynomial_exact = field.from_sympy(t**5)
    normal_factor = field.from_sympy(t - 2)
    normal_logarithmic_derivative = (
        3*derivation(normal_factor)/normal_factor
    )
    infinity_logarithmic_derivative = field.from_sympy(4*t)
    final_remainder = field.from_sympy(3 + 1/(1 + t**2))
    integrand = (
        derivation(repeated_pole_exact + polynomial_exact)
        + normal_logarithmic_derivative
        + infinity_logarithmic_derivative
        + final_remainder
    )

    reduction = reduce_nonlinear_monomial(integrand, derivation)

    assert reduction.hermite_exact_part == repeated_pole_exact
    assert reduction.polynomial_exact_part == Poly(
        t**5,
        t,
        domain=tower.field(0),
    )
    assert reduction.polynomial_remainder == Poly(
        4*t + 3,
        t,
        domain=tower.field(0),
    )
    assert reduction.exact_part == repeated_pole_exact + polynomial_exact
    assert reduction.residue_reduction.logarithmic_derivative == (
        normal_logarithmic_derivative
    )
    assert reduction.reduced_function_reduction is not None
    assert reduction.reduced_function_reduction.logarithmic_derivative == (
        infinity_logarithmic_derivative
    )
    assert reduction.logarithmic_derivative == (
        normal_logarithmic_derivative
        + infinity_logarithmic_derivative
    )
    assert reduction.remainder == final_remainder
    assert (
        derivation(reduction.exact_part)
        + reduction.logarithmic_derivative
        + reduction.remainder
        == integrand
    )
    assert not reduction.has_obstruction
    assert not reduction.needs_special_polynomial

    second = reduce_nonlinear_monomial(reduction.remainder, derivation)
    assert second.exact_part == field.zero
    assert second.logarithmic_derivative == field.zero
    assert second.remainder == reduction.remainder


def test_nonlinear_pipeline_stops_after_a_finite_residue_obstruction() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(t - 1)
    finite_obstruction = (
        field.from_sympy(x)*derivation(normal_factor)/normal_factor
    )
    polynomial_exact = field.from_sympy(t**4)
    integrand = derivation(polynomial_exact) + finite_obstruction

    reduction = reduce_nonlinear_monomial(integrand, derivation)

    assert reduction.has_finite_residue_obstruction
    assert reduction.has_obstruction
    assert not reduction.has_infinity_obstruction
    assert reduction.reduced_function_reduction is None
    assert reduction.polynomial_exact_part == Poly(
        t**4,
        t,
        domain=tower.field(0),
    )
    assert (
        reduction.polynomial_remainder.is_zero
        or reduction.polynomial_remainder.degree()
        < tower.levels[-1].generator_derivative.degree()
    )
    assert (
        derivation(reduction.exact_part)
        + reduction.logarithmic_derivative
        + reduction.remainder
        == integrand
    )


def test_nonlinear_pipeline_reports_the_infinity_obstruction() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(x*t + 3)

    reduction = reduce_nonlinear_monomial(
        integrand,
        tower.derivation(),
    )

    assert not reduction.has_finite_residue_obstruction
    assert reduction.has_infinity_obstruction
    assert reduction.has_obstruction
    assert reduction.remainder == integrand


def test_nonlinear_pipeline_records_an_unknown_special_polynomial() -> None:
    tower = monomial_tower(t**2 + x)
    field = tower.field()
    integrand = field.from_sympy(t)

    reduction = reduce_nonlinear_monomial(
        integrand,
        tower.derivation(),
    )

    assert not reduction.has_obstruction
    assert reduction.needs_special_polynomial
    assert reduction.remainder == integrand


def test_nonlinear_pipeline_accepts_a_supplied_special_polynomial() -> None:
    tower = monomial_tower(t**2 - x*t + 1)
    parent = tower.field(0)
    field = tower.field()
    special = Poly(t - x, t, domain=parent)
    integrand = field.from_sympy(t)

    reduction = reduce_nonlinear_monomial(
        integrand,
        tower.derivation(),
        special_polynomial=special,
    )

    assert reduction.reduced_function_reduction is not None
    assert reduction.reduced_function_reduction.logarithmic_argument == special
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert not reduction.needs_special_polynomial


def test_nonlinear_pipeline_enforces_its_family_and_constant_hypotheses() -> None:
    primitive_tower = monomial_tower(1/x)
    with pytest.raises(ValueError, match=r"degree\(Dt\) >= 2"):
        reduce_nonlinear_monomial(
            primitive_tower.field().from_sympy(t),
            primitive_tower.derivation(),
        )

    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    unknown = rational.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=rational.field()),
        ),
    )
    with pytest.raises(ValueError, match="status is unknown"):
        reduce_nonlinear_monomial(
            unknown.field().from_sympy(t),
            unknown.derivation(),
        )


def test_nonlinear_no_special_reduces_recursively_to_the_lower_field() -> None:
    # The no-special status is an explicit structure-theorem assumption here;
    # automatically proving it from Dt = t**2 + x belongs to the tower builder.
    tower = monomial_tower(
        t**2 + x,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )
    field = tower.field()
    derivation = tower.derivation()
    repeated_pole_exact = field.from_sympy((x + 1)/(t - 2)**2)
    polynomial_exact = field.from_sympy(t**5)
    normal_factor = field.from_sympy(t - 1)
    logarithmic_derivative = 3*derivation(normal_factor)/normal_factor
    lower_remainder = tower.field(0).from_sympy((x**2 + 1)/(x + 1))
    integrand = (
        derivation(repeated_pole_exact + polynomial_exact)
        + logarithmic_derivative
        + field.convert(lower_remainder)
    )

    result = reduce_nonlinear_no_special(integrand, derivation)
    common = result.common_reduction

    assert result.status is NonlinearNoSpecialStatus.REDUCED_TO_LOWER_FIELD
    assert result.reduced_to_lower_field
    assert not result.is_non_elementary
    assert result.lower_remainder == lower_remainder
    assert common.remainder == field.convert(lower_remainder)
    assert (
        derivation(common.exact_part)
        + common.logarithmic_derivative
        + common.remainder
        == integrand
    )


def test_nonlinear_no_special_proves_a_reduced_t_term_non_elementary() -> None:
    tower = monomial_tower(
        t**2 + x,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )
    field = tower.field()

    result = reduce_nonlinear_no_special(
        field.from_sympy(t),
        tower.derivation(),
    )

    assert (
        result.status
        is NonlinearNoSpecialStatus.NONCONSTANT_REDUCED_POLYNOMIAL
    )
    assert result.is_non_elementary
    assert not result.reduced_to_lower_field
    assert result.lower_remainder is None
    assert result.common_reduction.remainder == field.from_sympy(t)


def test_nonlinear_no_special_preserves_the_infinity_certificate() -> None:
    tower = monomial_tower(
        t**2 + x,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )
    field = tower.field()

    result = reduce_nonlinear_no_special(
        field.from_sympy(x*t),
        tower.derivation(),
    )

    assert result.status is NonlinearNoSpecialStatus.INFINITY_RESIDUE_OBSTRUCTION
    assert result.is_non_elementary
    reduced = result.common_reduction.reduced_function_reduction
    assert reduced is not None
    assert reduced.infinity_residue == tower.field(0).from_sympy(x)
    assert reduced.infinity_residue_derivative == tower.field(0).one


def test_nonlinear_no_special_preserves_the_finite_residue_certificate() -> None:
    tower = monomial_tower(
        t**2 + x,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(t - 1)
    integrand = field.from_sympy(x)*derivation(normal_factor)/normal_factor

    result = reduce_nonlinear_no_special(integrand, derivation)

    assert result.status is NonlinearNoSpecialStatus.FINITE_RESIDUE_OBSTRUCTION
    assert result.is_non_elementary
    assert result.common_reduction.has_finite_residue_obstruction
    assert not result.common_reduction.residue_reduction.certificate.all_residues_constant


@pytest.mark.parametrize(
    "status",
    (SpecialPolynomialStatus.UNKNOWN, SpecialPolynomialStatus.PRESENT),
)
def test_nonlinear_no_special_requires_structure_theorem_evidence(
    status: SpecialPolynomialStatus,
) -> None:
    tower = monomial_tower(t**2 + x, special_polynomial_status=status)

    with pytest.raises(
        ValueError,
        match=f"special-polynomial status is {status.value}",
    ):
        reduce_nonlinear_no_special(
            tower.field().from_sympy(t),
            tower.derivation(),
        )


def test_nonlinear_no_special_rejects_an_obvious_metadata_contradiction() -> None:
    tower = monomial_tower(
        t*(1 + t**2),
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )

    with pytest.raises(ValueError, match="Dt has a nonconstant special factor"):
        reduce_nonlinear_no_special(
            tower.field().from_sympy(1),
            tower.derivation(),
        )


def test_nonlinear_no_special_rejects_a_hidden_special_denominator() -> None:
    # For Dt = t**2 - x*t + 1, D(t - x) = t*(t - x), although
    # t - x is not a factor of Dt.  This exercises the second consistency
    # check rather than the inexpensive factor check above.
    tower = monomial_tower(
        t**2 - x*t + 1,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )

    with pytest.raises(
        ValueError,
        match="reduced remainder has a nonconstant special denominator",
    ):
        reduce_nonlinear_no_special(
            tower.field().from_sympy(1/(t - x)),
            tower.derivation(),
        )


@given(
    fraction_coefficient=st.integers(min_value=-3, max_value=3),
    fraction_power=st.integers(min_value=2, max_value=4),
    polynomial_coefficient=st.integers(min_value=-3, max_value=3),
    polynomial_power=st.integers(min_value=1, max_value=6),
    residue_value=st.integers(min_value=-3, max_value=3),
    lower_numerator_x=st.integers(min_value=-3, max_value=3),
    lower_numerator_constant=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=25, deadline=None)
def test_generated_nonlinear_no_special_inputs_reduce_to_the_lower_field(
    fraction_coefficient,
    fraction_power,
    polynomial_coefficient,
    polynomial_power,
    residue_value,
    lower_numerator_x,
    lower_numerator_constant,
) -> None:
    tower = monomial_tower(
        t**2 + x,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )
    field = tower.field()
    derivation = tower.derivation()
    exact = field.from_sympy(
        fraction_coefficient/(t - 2)**fraction_power
        + polynomial_coefficient*t**polynomial_power
    )
    normal_factor = field.from_sympy(t - 1)
    lower_remainder = tower.field(0).from_sympy(
        (lower_numerator_x*x + lower_numerator_constant)/(x + 1)
    )
    integrand = (
        derivation(exact)
        + residue_value*derivation(normal_factor)/normal_factor
        + field.convert(lower_remainder)
    )

    result = reduce_nonlinear_no_special(integrand, derivation)
    common = result.common_reduction

    assert result.status is NonlinearNoSpecialStatus.REDUCED_TO_LOWER_FIELD
    assert result.lower_remainder == lower_remainder
    assert (
        derivation(common.exact_part)
        + common.logarithmic_derivative
        + common.remainder
        == integrand
    )


@given(
    top_coefficient=st.integers(min_value=-5, max_value=5).filter(bool),
    lower_constant=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=20, deadline=None)
def test_generated_nonconstant_reduced_polynomials_are_certified(
    top_coefficient,
    lower_constant,
) -> None:
    tower = monomial_tower(
        t**2 + x,
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )

    result = reduce_nonlinear_no_special(
        tower.field().from_sympy(top_coefficient*t + lower_constant),
        tower.derivation(),
    )

    assert (
        result.status
        is NonlinearNoSpecialStatus.NONCONSTANT_REDUCED_POLYNOMIAL
    )
    assert result.is_non_elementary
    assert result.common_reduction.remainder == tower.field().from_sympy(
        top_coefficient*t + lower_constant
    )
    assert result.common_reduction.needs_special_polynomial


@given(
    example=st.sampled_from(
        (
            (1 + t**2, t - 1),
            (t + t**3, t - 1),
            (t**2 + x, t - 1),
        ),
    ),
    fraction_coefficient=st.integers(min_value=-3, max_value=3),
    fraction_power=st.integers(min_value=2, max_value=4),
    polynomial_coefficient=st.integers(min_value=-3, max_value=3),
    polynomial_power=st.integers(min_value=1, max_value=5),
    residue_value=st.integers(min_value=-3, max_value=3),
    final_constant=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=35, deadline=None)
def test_generated_nonlinear_pipelines_reconstruct_planted_stages(
    example,
    fraction_coefficient,
    fraction_power,
    polynomial_coefficient,
    polynomial_power,
    residue_value,
    final_constant,
) -> None:
    generator_derivative, normal_factor_expression = example
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    fraction_exact = field.from_sympy(
        fraction_coefficient/(t - 2)**fraction_power,
    )
    polynomial_exact = field.from_sympy(
        polynomial_coefficient*t**polynomial_power,
    )
    normal_factor = field.from_sympy(normal_factor_expression)
    planted_logarithmic_derivative = (
        residue_value*derivation(normal_factor)/normal_factor
    )
    final_remainder = field.from_sympy(final_constant)
    integrand = (
        derivation(fraction_exact + polynomial_exact)
        + planted_logarithmic_derivative
        + final_remainder
    )

    reduction = reduce_nonlinear_monomial(integrand, derivation)

    assert (
        derivation(reduction.exact_part)
        + reduction.logarithmic_derivative
        + reduction.remainder
        == integrand
    )
    assert reduction.remainder == final_remainder
    assert not reduction.has_obstruction
    assert not reduction.needs_special_polynomial


def denominator_parts(element, tower: DifferentialTower) -> tuple[Poly, Poly]:
    extension = tower.levels[tower.top]
    assert isinstance(extension, MonomialExtension)
    _, denominator_expression = element.as_expr().as_numer_denom()
    denominator = Poly(
        denominator_expression,
        extension.generator,
        domain=extension.parent_field,
    )
    return split_normal_special(denominator, tower.derivation())


def assert_normal_part_is_square_free(
    element,
    tower: DifferentialTower,
) -> None:
    normal, _ = denominator_parts(element, tower)
    assert all(
        multiplicity == 1
        for _, multiplicity in normal.sqf_list()[1]
    )


def test_monomial_hermite_reduces_a_logarithmic_repeated_pole() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(x*t**2))

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == field.from_sympy(-1/t)
    assert remainder == field.zero
    assert derivation(exact) + remainder == integrand


def test_monomial_hermite_preserves_an_exponential_special_part() -> None:
    tower = monomial_tower(t)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(t**2*(t - 1)**2))

    exact, remainder = hermite_reduce_monomial(integrand, derivation)
    normal, special = denominator_parts(remainder, tower)

    assert derivation(exact) + remainder == integrand
    assert normal == Poly(t - 1, t, domain=tower.field(0))
    assert special == Poly(t**2, t, domain=tower.field(0))


def test_monomial_hermite_preserves_a_tangent_special_part() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(t**2*(1 + t**2)**2))

    exact, remainder = hermite_reduce_monomial(integrand, derivation)
    normal, special = denominator_parts(remainder, tower)

    assert derivation(exact) + remainder == integrand
    assert normal == Poly(1, t, domain=tower.field(0))
    assert special == Poly((1 + t**2)**2, t, domain=tower.field(0))


@pytest.mark.parametrize("expression", [t**3 + x*t, 1/t**3, 1/(t - 1)])
def test_monomial_hermite_leaves_polynomial_and_special_terms_for_later(
    expression,
) -> None:
    tower = monomial_tower(t)
    field = tower.field()
    integrand = field.from_sympy(expression)

    exact, remainder = hermite_reduce_monomial(
        integrand,
        tower.derivation(),
    )

    assert exact == field.zero
    assert remainder == integrand


@given(st.integers(min_value=1, max_value=6))
def test_generated_logarithmic_exact_derivatives_are_recovered(power) -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(t**(-power))
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == field.zero


def test_monomial_hermite_requires_a_monomial_level() -> None:
    constants = DifferentialTower(QQ)

    with pytest.raises(TypeError, match="monomial level"):
        hermite_reduce_monomial(QQ.one, constants.derivation())


@given(
    generator_derivative=st.sampled_from((1/x, t, 1 + t**2, t**3 + x)),
    left_power=st.integers(min_value=1, max_value=5),
    right_power=st.integers(min_value=1, max_value=5),
    coefficient=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_planted_derivatives_across_monomial_families(
    generator_derivative,
    left_power,
    right_power,
    coefficient,
) -> None:
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        (coefficient*x + 1)/(t - 1)**left_power
        + (x - coefficient)/(t + 1)**right_power,
    )
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == field.zero
    assert derivation(exact) + remainder == integrand


@given(
    family=st.sampled_from(
        (
            (t, t),
            (1 + t**2, 1 + t**2),
            (t**3, t),
        ),
    ),
    special_power=st.integers(min_value=1, max_value=4),
    normal_power=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_generated_mixed_special_and_normal_denominators(
    family,
    special_power,
    normal_power,
) -> None:
    generator_derivative, special_factor = family
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        1/(special_factor**special_power*(t - 1)**normal_power),
    )
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)
    normal, special = denominator_parts(remainder, tower)
    expected_special = Poly(
        special_factor**special_power,
        t,
        domain=tower.field(0),
    )

    assert derivation(exact) + remainder == integrand
    assert all(
        multiplicity == 1
        for _, multiplicity in normal.sqf_list()[1]
    )
    assert expected_special.rem(special).is_zero

    second_exact, second_remainder = hermite_reduce_monomial(
        remainder,
        derivation,
    )
    assert second_exact == field.zero
    assert second_remainder == remainder


@given(
    left_power=st.integers(min_value=2, max_value=5),
    right_power=st.integers(min_value=2, max_value=5),
    linear=st.integers(min_value=-5, max_value=5),
    constant=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_monomial_reducer_agrees_with_classical_rational_hermite_reduction(
    left_power,
    right_power,
    linear,
    constant,
) -> None:
    tower = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    integrand = (linear*x + constant) / (
        (x - 1)**left_power*(x + 1)**right_power
    )

    exact, remainder = hermite_reduce_monomial(
        integrand,
        tower.derivation(),
    )
    rational_exact, rational_remainder = hermite_reduce_rational_recursive(
        integrand,
        x,
    )

    assert cancel(exact.as_expr() - rational_exact) == 0
    assert cancel(remainder.as_expr() - rational_remainder) == 0
    assert_normal_part_is_square_free(remainder, tower)


def test_monomial_hermite_handles_multiple_multiplicities_and_lower_terms() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        (x + 1)/(t - 2)**4 + (x - 1)/(t + 2)**2,
    )
    planted_remainder = field.from_sympy(
        t**2 + 1/x**2 + 1/(t - 1),
    )
    integrand = derivation(planted_exact) + planted_remainder

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == planted_remainder
    assert derivation(exact) + remainder == integrand
    assert_normal_part_is_square_free(remainder, tower)


@pytest.mark.parametrize(
    ("generator_derivative", "normal_factor"),
    [
        (1/x, t**2 + x),
        (t, t**2 + t + 1),
        (1 + t**2, t**2 + t + 2),
        (t**3 + x, t**2 + x*t + 1),
    ],
)
def test_monomial_hermite_reduces_nonlinear_normal_factors(
    generator_derivative,
    normal_factor,
) -> None:
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy((x + 1)/normal_factor**3)
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == field.zero
    assert derivation(exact) + remainder == integrand


def test_residue_algorithms_require_preserved_constants() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
        ),
    )
    integrand = tower.field().from_sympy(1/t)

    with pytest.raises(ValueError, match="status is unknown"):
        bronstein_residue_criterion(integrand, tower.derivation())
    with pytest.raises(ValueError, match="status is unknown"):
        residue_reduce_monomial(integrand, tower.derivation())


def test_bronstein_criterion_requires_a_symbolic_residue_indeterminate() -> None:
    tower = monomial_tower(1/x)

    with pytest.raises(TypeError, match="must be a Symbol"):
        bronstein_residue_criterion(
            tower.field().from_sympy(1/t),
            tower.derivation(),
            0,  # type: ignore[arg-type]
        )


def test_bronstein_criterion_rejects_the_logarithmic_integral_residue() -> None:
    # With t = log(x), 1/t has residue x because D(t) = 1/x.  Since Dx != 0,
    # Liouville's theorem rules out an elementary logarithmic term.
    tower = monomial_tower(1/x)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(1/t),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator == Poly(1, t, domain=tower.field(0))
    assert certificate.normal_denominator == Poly(t, t, domain=tower.field(0))
    assert certificate.residue_polynomial == Poly(z - x, z, domain=tower.field(0))
    assert certificate.kappa_residue_polynomial == Poly(
        -1,
        z,
        domain=tower.field(0),
    )
    assert not certificate.all_residues_constant
    assert certificate.factors[0].constant_factor == Poly(
        1,
        z,
        domain=tower.field(0),
    )
    assert certificate.factors[0].nonconstant_factor == Poly(
        z - x,
        z,
        domain=tower.field(0),
    )


def test_bronstein_criterion_accepts_log_log_residue() -> None:
    # 1/(x*t) has the constant residue 1 and integrates to log(t).
    tower = monomial_tower(1/x)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(1/(x*t)),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator == Poly(1/x, t, domain=tower.field(0))
    assert certificate.normal_denominator == Poly(t, t, domain=tower.field(0))
    assert certificate.residue_polynomial == Poly(z - 1, z, domain=tower.field(0))
    assert certificate.kappa_residue_polynomial.is_zero
    assert certificate.all_residues_constant
    assert certificate.factors[0].constant_factor == Poly(
        z - 1,
        z,
        domain=tower.field(0),
    )
    assert certificate.factors[0].nonconstant_factor == Poly(
        1,
        z,
        domain=tower.field(0),
    )


def test_bronstein_criterion_splits_constant_and_nonconstant_roots() -> None:
    tower = monomial_tower(1/x)
    z = symbols("z")
    # The residues at t=0 and t=1 are respectively 1 and x.
    remainder = tower.field().from_sympy(
        1/(x*t) + 1/(t - 1),
    )

    certificate = bronstein_residue_criterion(
        remainder,
        tower.derivation(),
        z,
    )

    factor = certificate.factors[0]
    assert certificate.residue_polynomial == Poly(
        (z - 1)*(z - x),
        z,
        domain=tower.field(0),
    )
    assert factor.constant_factor == Poly(z - 1, z, domain=tower.field(0))
    assert factor.nonconstant_factor == Poly(z - x, z, domain=tower.field(0))
    assert not certificate.all_residues_constant


def test_bronstein_criterion_isolates_normal_from_special_fraction() -> None:
    # For Dt=t, t is special and t-1 is normal:
    # 1/(t*(t-1)) = 1/(t-1) - 1/t.
    tower = monomial_tower(t)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(1/(t*(t - 1))),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator == Poly(1, t, domain=tower.field(0))
    assert certificate.normal_denominator == Poly(t - 1, t, domain=tower.field(0))
    assert certificate.residue_polynomial == Poly(z - 1, z, domain=tower.field(0))
    assert certificate.all_residues_constant


def test_bronstein_criterion_ignores_polynomial_and_special_only_parts() -> None:
    tower = monomial_tower(t)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(t**2 + 1/t),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator.is_zero
    assert certificate.normal_denominator.is_one
    assert certificate.residue_polynomial.is_one
    assert certificate.kappa_residue_polynomial.is_zero
    assert certificate.factors == ()
    assert certificate.all_residues_constant


def test_bronstein_criterion_requires_prior_hermite_reduction() -> None:
    tower = monomial_tower(1/x)

    with pytest.raises(ValueError, match="Hermite-reduced normal denominator"):
        bronstein_residue_criterion(
            tower.field().from_sympy(1/t**2),
            tower.derivation(),
        )
    with pytest.raises(ValueError, match="new over the tower field"):
        bronstein_residue_criterion(
            tower.field().from_sympy(1/t),
            tower.derivation(),
            x,
        )


@pytest.mark.parametrize(
    ("generator_derivative", "normal_factor"),
    [
        (1/x, t),
        (t, t - 1),
        (1 + t**2, t),
        (t**3 + x, t - 1),
    ],
)
def test_bronstein_criterion_across_monomial_families(
    generator_derivative,
    normal_factor,
) -> None:
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    logarithmic_derivative = tower.derivation()(field.from_sympy(normal_factor))
    logarithmic_derivative /= field.from_sympy(normal_factor)

    constant_certificate = bronstein_residue_criterion(
        2*logarithmic_derivative,
        tower.derivation(),
    )
    variable_certificate = bronstein_residue_criterion(
        field.from_sympy(x)*logarithmic_derivative,
        tower.derivation(),
    )

    assert constant_certificate.all_residues_constant
    assert not variable_certificate.all_residues_constant


@given(
    constant_residue=st.integers(min_value=-3, max_value=3),
    slope=st.integers(min_value=-3, max_value=3),
    intercept=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_generated_bronstein_criterion_matches_planted_residues(
    constant_residue,
    slope,
    intercept,
) -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    variable_residue = slope*x + intercept
    remainder = field.from_sympy(
        constant_residue/(x*t)
        + variable_residue/(x*(t - 1)),
    )

    certificate = bronstein_residue_criterion(
        remainder,
        tower.derivation(),
    )

    assert certificate.all_residues_constant is (slope == 0)


def test_residue_reduction_extracts_log_log_term() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    integrand = field.from_sympy(1/(x*t))

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand == Poly(t, t, domain=tower.field(0))
    assert residue_factor == Poly(z - 1, z, domain=tower.field(0))
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert reduction.all_normal_residues_removed


def test_residue_reduction_preserves_nonconstant_residue_obstruction() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(1/t)

    reduction = residue_reduce_monomial(integrand, tower.derivation())

    assert reduction.logarithmic_parts == ()
    assert reduction.logarithmic_derivative == field.zero
    assert reduction.remainder == integrand
    assert not reduction.all_normal_residues_removed


def test_residue_reduction_extracts_only_constant_residue_roots() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    integrand = field.from_sympy(1/(x*t) + 1/(t - 1))

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand.as_expr() == t
    assert residue_factor == Poly(z - 1, z, domain=tower.field(0))
    assert reduction.logarithmic_derivative == field.from_sympy(1/(x*t))
    assert reduction.remainder == field.from_sympy(1/(t - 1))
    assert not reduction.all_normal_residues_removed

    remaining_certificate = bronstein_residue_criterion(
        reduction.remainder,
        tower.derivation(),
        z,
    )
    assert remaining_certificate.factors[0].constant_factor.is_one
    assert remaining_certificate.factors[0].nonconstant_factor == Poly(
        z - x,
        z,
        domain=tower.field(0),
    )


def test_residue_reduction_traces_algebraic_constant_residues() -> None:
    # The residues are +/-i/2.  Their encoded logarithmic sum differentiates
    # to the input without constructing either algebraic constant explicitly.
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    integrand = field.from_sympy(1/(x*(t**2 + 1)))

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand.as_expr() == t + 2*z
    assert residue_factor == Poly(z**2 + 1/4, z, domain=tower.field(0))
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert reduction.all_normal_residues_removed


def test_residue_reduction_traces_degree_five_constant_residues() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(
        (t**2 + 1)/(x*(t**5 + t + 1)),
    )

    reduction = residue_reduce_monomial(integrand, tower.derivation())

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand.degree() == 1
    assert residue_factor.degree() == 5
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert reduction.all_normal_residues_removed


def test_residue_reduction_groups_repeated_equal_residues() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    normal_factor = field.from_sympy(t*(t - 1))
    integrand = tower.derivation()(normal_factor)/normal_factor

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand == Poly(t*(t - 1), t, domain=tower.field(0))
    assert residue_factor == Poly(z - 1, z, domain=tower.field(0))
    assert reduction.certificate.factors[0].multiplicity == 2
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero


def test_residue_reduction_leaves_only_exponential_special_terms() -> None:
    tower = monomial_tower(t)
    field = tower.field()
    integrand = field.from_sympy(1/(t*(t - 1)))

    reduction = residue_reduce_monomial(integrand, tower.derivation())

    assert reduction.logarithmic_derivative == field.from_sympy(t/(t - 1))
    assert reduction.remainder == field.from_sympy(-1 - 1/t)
    assert reduction.logarithmic_derivative + reduction.remainder == integrand
    remaining_certificate = bronstein_residue_criterion(
        reduction.remainder,
        tower.derivation(),
    )
    assert remaining_certificate.normal_denominator.is_one


@given(
    example=st.sampled_from(
        (
            (1/x, t**2 + 1),
            (t, t**2 - 1),
            (1 + t**2, t),
            (t**3 + x, t - 1),
        ),
    ),
    residue_value=st.integers(min_value=-4, max_value=4),
    background_constant=st.integers(min_value=-4, max_value=4),
    background_linear=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_residue_reductions_reconstruct_planted_logarithms(
    example,
    residue_value,
    background_constant,
    background_linear,
) -> None:
    generator_derivative, normal_factor_expression = example
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(normal_factor_expression)
    planted_logarithmic_derivative = (
        residue_value*derivation(normal_factor)/normal_factor
    )
    background = field.from_sympy(background_linear*t + background_constant)
    integrand = planted_logarithmic_derivative + background

    reduction = residue_reduce_monomial(integrand, derivation)

    assert reduction.logarithmic_derivative == planted_logarithmic_derivative
    assert reduction.remainder == background
    assert reduction.logarithmic_derivative + reduction.remainder == integrand
    assert reduction.all_normal_residues_removed
