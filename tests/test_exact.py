"""Exact-derivative decisions across the implemented tower levels."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import I, Poly, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    ExactDerivativeResult,
    HyperexponentialMonomialReduction,
    HypertangentMonomialReduction,
    MonomialExtension,
    PrimitiveExactDerivativeResult,
    RationalExactDerivativeResult,
    TranscendentalExactDerivativeCertificate,
    reduce_primitive_monomial,
    solve_exact_derivative,
    solve_primitive_exact_derivative,
)

x, t, u, y = symbols("x t u y")


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


def hyperexponential_tower(eta=1) -> DifferentialTower:
    """Return ``QQ(x)(t)``, with ``Dx = 1`` and ``Dt = eta*t``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(eta*t, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def hypertangent_tower(eta=1) -> DifferentialTower:
    """Return ``QQ(x)(t)``, with ``Dt = eta*(1 + t**2)``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(eta*(1 + t**2), t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


@pytest.mark.parametrize(
    "antiderivative_expression",
    [
        0,
        x*t - x,
        x*(t**2 - 2*t + 2),
        (x + 1)/(t - 2)**3 + t**5 + (x**2 + 1)/(x - 1),
    ],
)
def test_primitive_exact_solver_recovers_planted_derivatives(
    antiderivative_expression,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(antiderivative_expression)
    integrand = derivation(planted)

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert result.antiderivative == result.exact_part
    assert result.obstruction == field.zero
    assert not result.has_proper_fraction_obstruction
    assert not result.has_polynomial_obstruction
    assert derivation(result.exact_part) == integrand


def test_logarithmic_derivative_is_elementary_but_not_exact_in_the_field() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(x*t))

    exact_result = solve_primitive_exact_derivative(integrand, derivation)
    elementary_result = reduce_primitive_monomial(integrand, derivation)

    assert not exact_result.has_solution
    assert exact_result.antiderivative is None
    assert exact_result.has_proper_fraction_obstruction
    assert not exact_result.has_polynomial_obstruction
    assert exact_result.obstruction == integrand
    assert elementary_result.logarithmic_derivative == integrand
    assert elementary_result.remainder == field.zero

    denominator = exact_result.proper_denominator
    assert denominator.gcd(denominator.diff()).degree() == 0
    assert denominator.gcd(derivation.polynomial(denominator)).degree() == 0


@pytest.mark.parametrize(
    "integrand_expression",
    [
        1/t,
        1/(x*(t - 1)),
    ],
)
def test_simple_poles_are_primitive_exact_derivative_obstructions(
    integrand_expression,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(integrand_expression)

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.has_proper_fraction_obstruction
    assert not result.has_polynomial_obstruction
    assert result.obstruction == integrand
    assert derivation(result.exact_part) + result.obstruction == integrand


@pytest.mark.parametrize(
    "integrand_expression",
    [
        1/(x - 1),
        t/(x - 1),
        t**4/(x - 2),
    ],
)
def test_reduced_polynomials_are_primitive_exact_derivative_obstructions(
    integrand_expression,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(integrand_expression)

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert not result.has_proper_fraction_obstruction
    assert result.has_polynomial_obstruction
    assert result.obstruction == integrand
    assert derivation(result.exact_part) + result.obstruction == integrand


def test_primitive_exact_solver_combines_both_obstruction_classes() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy((x + 1)/(t - 2)**3 + x*t**3)
    planted_obstruction = field.from_sympy(
        3/(x*(t - 1)) + 2*t**2/(x - 1)
    )
    integrand = derivation(planted_exact) + planted_obstruction

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.has_proper_fraction_obstruction
    assert result.has_polynomial_obstruction
    assert result.obstruction == planted_obstruction
    assert derivation(result.exact_part) + result.obstruction == integrand

    second = solve_primitive_exact_derivative(
        result.obstruction,
        derivation,
    )
    assert second.exact_part == field.zero
    assert second.obstruction == result.obstruction


@given(
    rational_numerator=st.integers(min_value=-4, max_value=4),
    rational_shift=st.integers(min_value=-4, max_value=4),
    rational_power=st.integers(min_value=1, max_value=4),
    polynomial_coefficient=st.integers(min_value=-4, max_value=4),
    polynomial_degree=st.integers(min_value=0, max_value=6),
    lower_numerator=st.integers(min_value=-4, max_value=4),
    lower_root=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=40, deadline=None)
def test_generated_primitive_rational_derivatives_reduce_to_zero(
    rational_numerator,
    rational_shift,
    rational_power,
    polynomial_coefficient,
    polynomial_degree,
    lower_numerator,
    lower_root,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(
        rational_numerator/(t - rational_shift)**rational_power
        + polynomial_coefficient*t**polynomial_degree
        + lower_numerator/(x - lower_root)
    )
    integrand = derivation(planted)

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert result.obstruction == field.zero
    assert derivation(result.exact_part) == integrand


@given(
    exact_numerator=st.integers(min_value=-3, max_value=3),
    exact_shift=st.integers(min_value=-4, max_value=-1),
    exact_power=st.integers(min_value=1, max_value=4),
    residue=st.integers(min_value=-4, max_value=4).filter(bool),
    obstruction_shift=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_generated_logarithmic_derivatives_remain_canonical_obstructions(
    exact_numerator,
    exact_shift,
    exact_power,
    residue,
    obstruction_shift,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        exact_numerator/(t - exact_shift)**exact_power
    )
    planted_obstruction = field.from_sympy(
        residue/(x*(t - obstruction_shift))
    )
    integrand = derivation(planted_exact) + planted_obstruction

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.obstruction == planted_obstruction
    assert derivation(result.exact_part) + result.obstruction == integrand

    second = solve_primitive_exact_derivative(result.obstruction, derivation)
    assert second.exact_part == field.zero
    assert second.obstruction == result.obstruction


@given(
    exact_numerator=st.integers(min_value=-3, max_value=3),
    exact_root=st.integers(min_value=-5, max_value=-1),
    exact_power=st.integers(min_value=1, max_value=4),
    obstruction_coefficient=st.integers(min_value=-4, max_value=4).filter(bool),
    obstruction_root=st.integers(min_value=1, max_value=5),
    obstruction_degree=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_generated_polynomial_obstructions_are_canonical(
    exact_numerator,
    exact_root,
    exact_power,
    obstruction_coefficient,
    obstruction_root,
    obstruction_degree,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        exact_numerator/(x - exact_root)**exact_power
    )
    planted_obstruction = field.from_sympy(
        obstruction_coefficient*t**obstruction_degree/(x - obstruction_root)
    )
    integrand = derivation(planted_exact) + planted_obstruction

    result = solve_primitive_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.obstruction == planted_obstruction
    assert derivation(result.exact_part) + result.obstruction == integrand

    second = solve_primitive_exact_derivative(result.obstruction, derivation)
    assert second.exact_part == field.zero
    assert second.obstruction == result.obstruction


def test_primitive_exact_solver_validates_its_structure_theorem_hypotheses() -> None:
    hyperexponential = logarithmic_tower(eta=t)
    with pytest.raises(ValueError, match="requires Dt in K"):
        solve_primitive_exact_derivative(
            hyperexponential.field().one,
            hyperexponential.derivation(),
        )

    unknown = logarithmic_tower(
        constant_field_status=ConstantFieldStatus.UNKNOWN,
    )
    with pytest.raises(ValueError, match="status is unknown"):
        solve_primitive_exact_derivative(
            unknown.field().one,
            unknown.derivation(),
        )

    for exact_eta in (1, 1/x**2):
        invalid = logarithmic_tower(eta=exact_eta)
        with pytest.raises(ValueError, match="Dt is exact in the lower field"):
            solve_primitive_exact_derivative(
                invalid.field().one,
                invalid.derivation(),
            )


def test_generic_dispatches_to_the_rational_solver_with_native_outputs() -> None:
    tower = logarithmic_tower()
    derivation = tower.derivation(0)
    field = tower.field(0)

    exact = solve_exact_derivative(field.from_sympy(-1/x**2), derivation)
    obstructed = solve_exact_derivative(field.from_sympy(1/x), derivation)

    assert exact.has_solution
    assert exact.antiderivative == field.from_sympy(1/x)
    assert exact.obstruction == field.zero
    assert isinstance(exact.certificate, RationalExactDerivativeResult)
    assert derivation(exact.exact_part) == field.from_sympy(-1/x**2)

    assert not obstructed.has_solution
    assert obstructed.antiderivative is None
    assert obstructed.exact_part == field.zero
    assert obstructed.obstruction == field.from_sympy(1/x)


def test_generic_dispatch_supports_the_gaussian_rational_base() -> None:
    tower = DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(I*x**2/2 - 1/(x + I))

    result = solve_exact_derivative(derivation(planted), derivation)

    assert result.has_solution
    assert result.obstruction == field.zero
    assert derivation(result.exact_part) == derivation(planted)
    assert isinstance(result.certificate, RationalExactDerivativeResult)


def test_gaussian_rational_exact_dispatch_keeps_simple_poles_as_obstructions() -> None:
    tower = DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    integrand = field.from_sympy(1/(x + I))

    result = solve_exact_derivative(integrand, tower.derivation())

    assert not result.has_solution
    assert result.obstruction == integrand


def test_generic_dispatches_to_the_primitive_solver() -> None:
    tower = logarithmic_tower()
    derivation = tower.derivation()
    field = tower.field()
    planted = field.from_sympy((x + 1)/(t - 2)**2 + t**3)
    integrand = derivation(planted)

    exact = solve_exact_derivative(integrand, derivation)
    obstructed = solve_exact_derivative(
        field.from_sympy(1/(x*t)),
        derivation,
    )

    assert exact.has_solution
    assert isinstance(exact.certificate, PrimitiveExactDerivativeResult)
    assert derivation(exact.exact_part) == integrand
    assert not obstructed.has_solution
    assert obstructed.obstruction == field.from_sympy(1/(x*t))


@pytest.mark.parametrize(
    "antiderivative_expression",
    [
        t,
        x*t,
        1/t,
        1/(t - 1)**2,
        (x + 1)*t**2 + 1/x,
    ],
)
def test_generic_dispatch_recovers_hyperexponential_derivatives(
    antiderivative_expression,
) -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(antiderivative_expression)
    integrand = derivation(planted)

    result = solve_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert result.obstruction == field.zero
    assert derivation(result.exact_part) == integrand
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert isinstance(
        result.certificate.reduction,
        HyperexponentialMonomialReduction,
    )


@pytest.mark.parametrize(
    "obstruction_expression",
    [
        t/x,
        t/(t - 1),
        1/(x*(t - 1)),
        1/x,
    ],
)
def test_generic_dispatch_preserves_hyperexponential_obstructions(
    obstruction_expression,
) -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x + 1)*t**2 + 1/(t - 2)**2)
    obstruction = field.from_sympy(obstruction_expression)
    integrand = derivation(planted) + obstruction

    result = solve_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.antiderivative is None
    assert result.obstruction == obstruction
    assert derivation(result.exact_part) + result.obstruction == integrand
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert isinstance(
        result.certificate.reduction,
        HyperexponentialMonomialReduction,
    )
    second = solve_exact_derivative(result.obstruction, derivation)
    assert second.exact_part == field.zero
    assert second.obstruction == result.obstruction


def test_hyperexponential_exact_dispatch_descends_a_primitive_lower_tower() -> None:
    logarithmic = logarithmic_tower()
    tower = logarithmic.extend(
        MonomialExtension(
            Poly(u, u, domain=logarithmic.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x*t + 1)*u + t**2)
    integrand = derivation(planted)

    result = solve_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert derivation(result.exact_part) == integrand
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert result.certificate.lower_result is not None
    assert result.certificate.lower_result.has_solution


@pytest.mark.parametrize(
    "antiderivative_expression",
    [
        t,
        x*t,
        1/(t - 1)**2,
        t/(1 + t**2),
        (x + 1)*t/(1 + t**2) + 1/x,
    ],
)
def test_generic_dispatch_recovers_hypertangent_derivatives(
    antiderivative_expression,
) -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(antiderivative_expression)
    integrand = derivation(planted)

    result = solve_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert result.obstruction == field.zero
    assert derivation(result.exact_part) == integrand
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert isinstance(
        result.certificate.reduction,
        HypertangentMonomialReduction,
    )


@pytest.mark.parametrize(
    "obstruction_expression",
    [
        2*t,
        2*(1 - t**2)/(x*(1 + t**2)),
        1/(x*t),
        1/x,
    ],
)
def test_generic_dispatch_preserves_hypertangent_obstructions(
    obstruction_expression,
) -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x + 1)/(t - 2)**2)
    obstruction = field.from_sympy(obstruction_expression)
    integrand = derivation(planted) + obstruction

    result = solve_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.antiderivative is None
    assert result.obstruction == obstruction
    assert derivation(result.exact_part) + result.obstruction == integrand
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert isinstance(
        result.certificate.reduction,
        HypertangentMonomialReduction,
    )
    second = solve_exact_derivative(result.obstruction, derivation)
    assert second.exact_part == field.zero
    assert second.obstruction == result.obstruction


def test_transcendental_dispatch_separates_layered_obstructions() -> None:
    cases = (
        (
            hyperexponential_tower(),
            1/(x*(t - 1)) + t/(t - 2) + t/x + 1/x,
        ),
        (
            hypertangent_tower(),
            1/(x*t)
            + 2*t
            + 2*(1 - t**2)/(x*(1 + t**2))
            + 1/x,
        ),
    )
    for tower, obstruction_expression in cases:
        field = tower.field()
        derivation = tower.derivation()
        planted = field.from_sympy(
            (x + 1)*t**2 + 1/(t - 3)**2
        )
        obstruction = field.from_sympy(obstruction_expression)
        integrand = derivation(planted) + obstruction

        result = solve_exact_derivative(integrand, derivation)

        assert not result.has_solution
        assert derivation(result.exact_part) == derivation(planted)
        assert result.obstruction == obstruction
        assert isinstance(
            result.certificate,
            TranscendentalExactDerivativeCertificate,
        )
        assert result.certificate.residual_reduction is not None

        second = solve_exact_derivative(result.obstruction, derivation)
        assert second.exact_part == field.zero
        assert second.obstruction == result.obstruction


@given(
    coefficient=st.integers(min_value=-4, max_value=4),
    x_degree=st.integers(min_value=0, max_value=3),
    t_power=st.integers(min_value=-3, max_value=3),
    pole_shift=st.integers(min_value=1, max_value=4),
    pole_power=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=25, deadline=None)
def test_generated_transcendental_derivatives_dispatch_exactly(
    coefficient,
    x_degree,
    t_power,
    pole_shift,
    pole_power,
) -> None:
    for tower in (hyperexponential_tower(), hypertangent_tower()):
        field = tower.field()
        derivation = tower.derivation()
        planted = field.from_sympy(
            coefficient*x**x_degree*t**t_power
            + 1/(t - pole_shift)**pole_power
        )
        integrand = derivation(planted)

        result = solve_exact_derivative(integrand, derivation)

        assert result.has_solution
        assert result.obstruction == field.zero
        assert derivation(result.exact_part) == integrand


def test_generic_constant_level_has_only_the_zero_derivative() -> None:
    derivation = DifferentialTower(QQ).derivation()

    zero = solve_exact_derivative(QQ.zero, derivation)
    nonzero = solve_exact_derivative(QQ(3, 2), derivation)

    assert zero.has_solution
    assert zero.antiderivative == QQ.zero
    assert zero.certificate is None
    assert not nonzero.has_solution
    assert nonzero.antiderivative is None
    assert nonzero.obstruction == QQ(3, 2)


def test_exact_result_recognizes_zero_in_a_finite_algebraic_field() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    algebraic = base.extend(
        AlgebraicExtension(
            Poly(y**2 - x, y, domain=base.field()),
        ),
    )
    for zero in (QQ_I.zero, algebraic.field().zero):
        result = ExactDerivativeResult(zero, zero, None)

        assert result.has_solution
        assert result.antiderivative == zero


def test_generic_dispatch_rejects_unimplemented_tower_families() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    algebraic = base.extend(
        AlgebraicExtension(
            Poly(y**2 - x, y, domain=base.field()),
        ),
    )
    with pytest.raises(NotImplementedError, match="algebraic levels"):
        solve_exact_derivative(
            algebraic.field().one,
            algebraic.derivation(),
        )

    scaled_base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(2, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    with pytest.raises(NotImplementedError, match="supports rational"):
        solve_exact_derivative(
            scaled_base.field().one,
            scaled_base.derivation(),
        )


def test_generic_dispatch_rejects_invalid_transcendental_metadata() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    unknown = rational.extend(
        MonomialExtension(
            Poly(t, t, domain=rational.field()),
        )
    )
    with pytest.raises(ValueError, match="status is unknown"):
        solve_exact_derivative(
            unknown.field().one,
            unknown.derivation(),
        )

    dependent_exponential = rational.extend(
        MonomialExtension(
            Poly(t/x, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    with pytest.raises(ValueError, match="algebraically dependent"):
        solve_exact_derivative(
            dependent_exponential.field().one,
            dependent_exponential.derivation(),
        )

    dependent_tangent = rational.extend(
        MonomialExtension(
            Poly(
                (1 + t**2)/(x**2 + 1),
                t,
                domain=rational.field(),
            ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    with pytest.raises(ValueError, match="Cayley generator"):
        solve_exact_derivative(
            dependent_tangent.field().one,
            dependent_tangent.derivation(),
        )
