"""Contract tests for Bronstein's parametric logarithmic-derivative problem."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, Rational, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    ParametricLogarithmicDerivativeHeuristicBranch,
    ParametricLogarithmicDerivativeHeuristicCertificate,
    ParametricLogarithmicDerivativeOutcome,
    ParametricLogarithmicDerivativeResult,
    solve_parametric_logarithmic_derivative_heuristic,
    verify_parametric_logarithmic_derivative_witness,
)


x, t, theta = symbols("x t theta")


def rational_base() -> DifferentialTower:
    """Return the standard differential field ``QQ(x)``, with ``Dx = 1``."""
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def hyperexponential_extension(
    lower_tower: DifferentialTower,
    eta,
) -> MonomialExtension:
    """Return a proposed ``Dtheta = eta*theta`` extension."""
    return MonomialExtension(
        Poly(
            eta*theta,
            theta,
            domain=lower_tower.field(),
        ),
        ConstantFieldStatus.UNKNOWN,
    )


def test_verified_rational_witness_uses_native_field_elements() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    field = lower.field()
    extension = hyperexponential_extension(lower, 1/(x + 1))
    argument = field.from_sympy(x**2*(x - 1))
    n = 5
    m = -3
    f = (
        derivation(argument)/argument
        + m*field.from_sympy(1/(x + 1))
    )/n

    result = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        f,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=n,
        m=m,
        argument=argument,
    )

    assert result.has_solution
    assert not result.proves_no_solution
    assert not result.is_inconclusive
    assert not result.is_unsupported
    assert result.reconstructs
    assert result.f == f
    assert result.eta == field.from_sympy(1/(x + 1))
    assert result.argument == argument


def test_negative_n_is_normalized_by_inverting_the_argument() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    result = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        1/x,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=-2,
        m=-1,
        argument=1/x,
    )

    assert result.n == 2
    assert result.m == 1
    assert result.argument == lower.field().from_sympy(x)
    assert result.reconstructs


def test_zero_theta_coefficient_is_a_valid_solution_case() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/(x + 1))

    result = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        1/x,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=1,
        m=0,
        argument=x,
    )

    assert result.m == 0
    assert result.reconstructs


def test_contract_works_over_a_primitive_lower_field() -> None:
    rational = rational_base()
    lower = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    derivation = lower.derivation()
    field = lower.field()
    eta = field.from_sympy(t + 1/(x + 1))
    extension = hyperexponential_extension(lower, t + 1/(x + 1))
    argument = field.from_sympy(x*t)
    n = 3
    m = -2
    f = (derivation(argument)/argument + m*eta)/n

    result = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        f,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=n,
        m=m,
        argument=argument,
    )

    assert result.reconstructs
    assert result.argument == argument
    assert result.eta == eta


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(min_value=-6, max_value=6).filter(lambda value: value != 0),
    m=st.integers(min_value=-6, max_value=6),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_planted_witnesses_reconstruct_exactly(
    n,
    m,
    x_power,
    shifted_power,
) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    field = lower.field()
    extension = hyperexponential_extension(lower, 1/(x + 1))
    argument = field.from_sympy(
        x**x_power*(x - 1)**shifted_power
    )
    eta = field.from_sympy(1/(x + 1))
    f = (derivation(argument)/argument + m*eta)/n

    result = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        f,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=n,
        m=m,
        argument=argument,
    )

    assert result.n > 0
    assert result.reconstructs
    assert verify_parametric_logarithmic_derivative_witness(
        result.f,
        extension,
        derivation,
        result.n,
        result.m,
        result.argument,
    )


def test_incorrect_witness_is_rejected_exactly() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    assert not verify_parametric_logarithmic_derivative_witness(
        0,
        extension,
        derivation,
        1,
        0,
        x,
    )
    with pytest.raises(ValueError, match="defining equation"):
        ParametricLogarithmicDerivativeResult(
            extension,
            derivation,
            0,
            ParametricLogarithmicDerivativeOutcome.SOLUTION,
            n=1,
            m=0,
            argument=x,
        )


@pytest.mark.parametrize(
    ("n", "argument"),
    [
        (0, x),
        (1, 0),
    ],
)
def test_degenerate_witness_is_not_a_solution(n, argument) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    assert not verify_parametric_logarithmic_derivative_witness(
        0,
        extension,
        derivation,
        n,
        0,
        argument,
    )
    with pytest.raises(ValueError):
        ParametricLogarithmicDerivativeResult(
            extension,
            derivation,
            0,
            ParametricLogarithmicDerivativeOutcome.SOLUTION,
            n=n,
            m=0,
            argument=argument,
        )


@pytest.mark.parametrize("bad_integer", [True, Rational(1, 2)])
def test_witness_parameters_must_be_exact_integers(bad_integer) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    with pytest.raises(TypeError, match="n must be an integer"):
        verify_parametric_logarithmic_derivative_witness(
            1/x,
            extension,
            derivation,
            bad_integer,
            0,
            x,
        )


def test_logical_nonsolution_outcomes_are_distinct() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    proved = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        1,
        ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
        certificate={"obstruction": "degree"},
    )
    inconclusive = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        1,
        ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
        detail="Section 7.3 split-factor heuristic returned failed",
    )
    unsupported = ParametricLogarithmicDerivativeResult(
        extension,
        derivation,
        1,
        ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
        detail="constant field does not support exact linear algebra",
    )

    assert proved.proves_no_solution
    assert not proved.has_solution
    assert not proved.reconstructs
    assert inconclusive.is_inconclusive
    assert not inconclusive.proves_no_solution
    assert unsupported.is_unsupported
    assert not unsupported.proves_no_solution


def test_claimed_nonsolution_requires_evidence() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    with pytest.raises(ValueError, match="obstruction certificate"):
        ParametricLogarithmicDerivativeResult(
            extension,
            derivation,
            1,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
        )
    with pytest.raises(ValueError, match="requires a detail"):
        ParametricLogarithmicDerivativeResult(
            extension,
            derivation,
            1,
            ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
        )


def test_nonsolution_cannot_smuggle_in_a_witness() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, 1/x)

    with pytest.raises(ValueError, match="cannot carry solution"):
        ParametricLogarithmicDerivativeResult(
            extension,
            derivation,
            1,
            ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
            n=1,
            m=0,
            argument=x,
            detail="heuristic returned failed",
        )


def test_extension_must_be_hyperexponential_over_the_lower_field() -> None:
    lower = rational_base()
    derivation = lower.derivation()
    primitive = MonomialExtension(
        Poly(1, theta, domain=lower.field()),
        ConstantFieldStatus.PRESERVED,
    )
    other_lower = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, symbols("z"), domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    mismatched = hyperexponential_extension(other_lower, 1)

    with pytest.raises(ValueError, match="Dtheta = eta\\*theta"):
        verify_parametric_logarithmic_derivative_witness(
            0,
            primitive,
            derivation,
            1,
            0,
            1,
        )
    with pytest.raises(ValueError, match="parent field"):
        verify_parametric_logarithmic_derivative_witness(
            0,
            mismatched,
            derivation,
            1,
            0,
            1,
        )


def test_polynomial_comparison_finds_a_verified_witness() -> None:
    lower = rational_base()
    eta = x**2 + 1/x
    extension = hyperexponential_extension(lower, eta)
    f = (1/(x - 1) + 3*eta)/2

    result = solve_parametric_logarithmic_derivative_heuristic(
        f,
        extension,
        lower.derivation(),
    )

    assert result.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION
    assert result.n == 2
    assert result.m == 3
    assert result.argument == lower.field().from_sympy(x - 1)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        ParametricLogarithmicDerivativeHeuristicCertificate,
    )
    assert (
        result.certificate.branch
        is ParametricLogarithmicDerivativeHeuristicBranch.POLYNOMIAL_PART
    )
    assert result.certificate.candidate_ratio == QQ(3, 2)
    assert result.certificate.logarithmic_power == 1
    assert result.certificate.logarithmic_result.has_solution


def test_polynomial_comparison_clears_rational_logarithmic_residues() -> None:
    lower = rational_base()
    eta = x**2
    extension = hyperexponential_extension(lower, eta)

    # The high-degree comparison first yields m/n = 3/2.  Its residual is
    # 1/(2*x), so a second factor of two is needed before it is D(x)/x.
    result = solve_parametric_logarithmic_derivative_heuristic(
        3*x**2/2 + 1/(4*x),
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (4, 6)
    assert result.argument == lower.field().from_sympy(x)
    assert result.certificate.logarithmic_power == 2
    assert result.reconstructs


def test_polynomial_comparison_allows_zero_theta_coefficient() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, x**2)

    result = solve_parametric_logarithmic_derivative_heuristic(
        1/x,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 0)
    assert result.argument == lower.field().from_sympy(x)
    assert result.reconstructs


def test_nonproportional_polynomial_parts_prove_no_solution() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, x)

    result = solve_parametric_logarithmic_derivative_heuristic(
        x**2,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert (
        result.certificate.branch
        is ParametricLogarithmicDerivativeHeuristicBranch.POLYNOMIAL_PART
    )
    assert "not proportional" in result.certificate.reason


def test_polynomial_degree_without_matching_eta_degree_proves_no_solution() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, 1/x)

    result = solve_parametric_logarithmic_derivative_heuristic(
        x,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert "degree above the bound" in result.certificate.reason


def test_unique_ratio_with_nonlogarithmic_residual_proves_no_solution() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, x)

    result = solve_parametric_logarithmic_derivative_heuristic(
        x + 1/x**2,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert result.certificate.candidate_ratio == QQ.one
    assert (
        result.certificate.logarithmic_result.has_repeated_pole_obstruction
    )


def test_denominator_congruence_finds_a_verified_witness() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, 1/x**2)

    # Modulo z=x, x**2*f has residue 3/2 while x**2*eta has residue 1.
    result = solve_parametric_logarithmic_derivative_heuristic(
        (5/x + 3/x**2)/2,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (2, 3)
    assert result.argument == lower.field().from_sympy(x**5)
    assert result.reconstructs
    assert (
        result.certificate.branch
        is ParametricLogarithmicDerivativeHeuristicBranch.DENOMINATOR_CONGRUENCE
    )
    assert result.certificate.comparison_modulus == Poly(x, x, domain=QQ)
    assert result.certificate.candidate_ratio == QQ(3, 2)


def test_inconsistent_denominator_congruences_prove_no_solution() -> None:
    lower = rational_base()
    denominator = (x**2 + 1)**2
    extension = hyperexponential_extension(lower, 1/denominator)

    result = solve_parametric_logarithmic_derivative_heuristic(
        x/denominator,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert result.certificate.comparison_modulus == Poly(
        x**2 + 1,
        x,
        domain=QQ,
    )
    assert "congruences are not proportional" in result.certificate.reason


def test_bronstein_example_7_3_2_is_inconclusive_not_a_false_negative() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative_heuristic(
        11,
        extension,
        lower.derivation(),
    )

    # The actual witness is (n,m,v)=(1,11,1), but neither constant
    # polynomial parts nor constant denominators determine m/n.
    assert verify_parametric_logarithmic_derivative_witness(
        11,
        extension,
        lower.derivation(),
        1,
        11,
        1,
    )
    assert result.is_inconclusive
    assert not result.proves_no_solution
    assert result.certificate.comparison_modulus.degree() == 0
    assert "structure-theorem fallback" in result.detail


def test_nonrational_lower_field_is_reported_as_unsupported() -> None:
    rational = rational_base()
    lower = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative_heuristic(
        1,
        extension,
        lower.derivation(),
    )

    assert result.is_unsupported
    assert not result.proves_no_solution
    assert "requires QQ(x)" in result.detail


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=-6, max_value=6).filter(lambda value: value != 0),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_polynomial_branch_witnesses_reconstruct(
    n,
    m,
    x_power,
    shifted_power,
) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    field = lower.field()
    eta_expression = x**2 + 1/(x + 2)
    eta = field.from_sympy(eta_expression)
    extension = hyperexponential_extension(lower, eta_expression)
    argument = field.from_sympy(
        x**x_power*(x - 1)**shifted_power
    )
    f = (derivation(argument)/argument + m*eta)/n

    result = solve_parametric_logarithmic_derivative_heuristic(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert (
        result.certificate.branch
        is ParametricLogarithmicDerivativeHeuristicBranch.POLYNOMIAL_PART
    )
    assert QQ(result.m, result.n) == QQ(m, n)


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=-6, max_value=6).filter(lambda value: value != 0),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_denominator_branch_witnesses_reconstruct(
    n,
    m,
    x_power,
    shifted_power,
) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    field = lower.field()
    eta_expression = 1/x**2
    eta = field.from_sympy(eta_expression)
    extension = hyperexponential_extension(lower, eta_expression)
    argument = field.from_sympy(
        x**x_power*(x - 1)**shifted_power
    )
    f = (derivation(argument)/argument + m*eta)/n

    result = solve_parametric_logarithmic_derivative_heuristic(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert (
        result.certificate.branch
        is ParametricLogarithmicDerivativeHeuristicBranch.DENOMINATOR_CONGRUENCE
    )
    assert QQ(result.m, result.n) == QQ(m, n)
