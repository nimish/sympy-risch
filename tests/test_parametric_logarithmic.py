"""Contract tests for Bronstein's parametric logarithmic-derivative problem."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Dummy, EmptySet, I, Poly, Rational, cancel, linsolve, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    ParametricLogarithmicDerivativeHeuristicBranch,
    ParametricLogarithmicDerivativeHeuristicCertificate,
    ParametricLogarithmicDerivativeOutcome,
    ParametricLogarithmicDerivativeResult,
    RationalParametricLogarithmicDerivativeCertificate,
    solve_parametric_logarithmic_derivative,
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


def gaussian_rational_base() -> DifferentialTower:
    """Return the standard Gaussian rational field ``QQ_I(x)``."""
    return DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
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


def test_gaussian_rational_base_solves_a_parametric_witness() -> None:
    """The integer parameters remain rational after adjoining ``i``."""
    lower = gaussian_rational_base()
    derivation = lower.derivation()
    extension = hyperexponential_extension(lower, x)

    result = solve_parametric_logarithmic_derivative(
        x/2 + 1/(x + 1),
        extension,
        derivation,
    )

    assert result.has_solution
    assert result.n == 2
    assert result.m == 1
    assert result.reconstructs


def test_gaussian_rational_base_rejects_a_nonrational_parameter() -> None:
    """A Gaussian coefficient cannot silently become the integer ``m/n``."""
    lower = gaussian_rational_base()
    extension = hyperexponential_extension(lower, x)

    result = solve_parametric_logarithmic_derivative(
        I*x,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution


@pytest.mark.parametrize(
    ("f", "eta", "n", "m", "argument"),
    [
        (I/(x - I), I/(x - I), 1, 1, 1),
        ((1 + I)/(x - I), I/(x - I), 1, 1, x - I),
    ],
)
def test_gaussian_rational_fallback_constrains_residues_to_qq(
    f,
    eta,
    n,
    m,
    argument,
) -> None:
    """A Gaussian residue direction still has a rational valuation."""
    lower = gaussian_rational_base()
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (n, m)
    assert result.argument == lower.field().from_sympy(argument)
    assert result.reconstructs


def test_gaussian_rational_fallback_rejects_nonrational_residue_ratio() -> None:
    """The fallback never promotes a Gaussian ratio to ``m/n``."""
    lower = gaussian_rational_base()
    extension = hyperexponential_extension(lower, 1/(x - I))

    result = solve_parametric_logarithmic_derivative(
        I/(x - I),
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution


def _independent_rational_solution_set(f, eta):
    """Solve the rational problem without using the production certificate.

    Over ``QQ(x)``, ``f - c*eta`` is the logarithmic derivative of a rational
    radical exactly when it is a rational linear combination of ``p'/p`` for
    the irreducible factors ``p`` in its denominator.  After multiplication by
    a common denominator, coefficient comparison gives one global linear
    system.  In particular, polynomial parts and repeated-pole terms must
    cancel automatically; they are not copied from the implementation's
    locally constructed constraints.
    """

    def fraction_polynomials(expression):
        numerator, denominator = cancel(expression).as_numer_denom()
        return (
            Poly(numerator, x, domain=QQ),
            Poly(denominator, x, domain=QQ),
        )

    f_numerator, f_denominator = fraction_polynomials(f)
    eta_numerator, eta_denominator = fraction_polynomials(eta)
    common_denominator = f_denominator.lcm(eta_denominator)
    irreducible_factors = tuple(
        factor
        for factor, _multiplicity in common_denominator.factor_list()[1]
    )

    ratio = Dummy("ratio")
    residues = tuple(
        Dummy(f"residue_{index}")
        for index in range(len(irreducible_factors))
    )
    identity = (
        common_denominator.exquo(f_denominator)*f_numerator
        - ratio*common_denominator.exquo(eta_denominator)*eta_numerator
    ).as_expr()
    for residue, factor in zip(
        residues,
        irreducible_factors,
        strict=True,
    ):
        identity -= (
            residue
            * common_denominator.exquo(factor).as_expr()
            * factor.diff().as_expr()
        )

    equations = Poly(identity, x).all_coeffs()
    return linsolve(equations, (ratio, *residues))


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


def test_complete_solver_promotes_bronstein_example_7_3_2_to_solution() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative(
        11,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 11)
    assert result.argument == lower.field().one
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        RationalParametricLogarithmicDerivativeCertificate,
    )
    assert result.certificate.heuristic_result.is_inconclusive
    assert result.certificate.linear_constraints == ((QQ(11), QQ.one),)
    assert result.certificate.candidate_ratio == QQ(11)


def test_complete_solver_cancels_a_nonlogarithmic_residue() -> None:
    lower = rational_base()
    nonlogarithmic_part = 1/(x**2 + 1)
    extension = hyperexponential_extension(lower, nonlogarithmic_part)

    result = solve_parametric_logarithmic_derivative(
        nonlogarithmic_part + 1/x,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 1)
    assert result.argument == lower.field().from_sympy(x)
    assert result.reconstructs
    assert result.certificate.candidate_ratio == QQ.one

    quadratic_residue = next(
        residue
        for residue in result.certificate.residues
        if residue.factor.degree() == 2
    )
    assert quadratic_residue.factor == Poly(x**2 + 1, x, domain=QQ)
    assert quadratic_residue.f_obstruction == (QQ.one,)
    assert quadratic_residue.eta_obstruction == (QQ.one,)

    ratio = symbols("ratio")
    equations = [
        left - ratio*right
        for left, right in result.certificate.linear_constraints
    ]
    assert (QQ.one,) in linsolve(equations, ratio)


def test_fallback_projects_high_degree_residues_modulo_factor_derivative() -> None:
    lower = rational_base()
    factor = x**3 + x + 1
    eta = (x + 1)/factor
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        2*eta + (3*x**2 + 1)/factor,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 2)
    assert result.argument == lower.field().from_sympy(factor)
    assert result.reconstructs

    cubic_residue = next(
        residue
        for residue in result.certificate.residues
        if residue.factor.degree() == 3
    )
    assert cubic_residue.eta_obstruction == (QQ.one, QQ.one)
    assert cubic_residue.f_obstruction == (QQ(2), QQ(2))


def test_independent_residue_factors_can_impose_conflicting_ratios() -> None:
    lower = rational_base()
    first = x**2 + 1
    second = x**2 + 2
    extension = hyperexponential_extension(
        lower,
        1/first + 2/second,
    )

    result = solve_parametric_logarithmic_derivative(
        1/first + 1/second,
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert (QQ.one, QQ.one) in result.certificate.linear_constraints
    assert (QQ.one, QQ(2)) in result.certificate.linear_constraints


def test_complete_solver_chooses_m_zero_when_every_ratio_is_free() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, 1/(x + 1))

    result = solve_parametric_logarithmic_derivative(
        x/(x**2 + 1),
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (2, 0)
    assert result.argument == lower.field().from_sympy(x**2 + 1)
    assert result.certificate.candidate_ratio == QQ.zero
    assert result.certificate.logarithmic_power == 2
    assert result.reconstructs


def test_nonlogarithmic_residue_proves_no_solution_after_heuristic_failure() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative(
        1/(x**2 + 1),
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert isinstance(
        result.certificate,
        RationalParametricLogarithmicDerivativeCertificate,
    )
    assert result.certificate.heuristic_result.is_inconclusive
    assert result.certificate.candidate_ratio is None

    ratio = symbols("ratio")
    equations = [
        left - ratio*right
        for left, right in result.certificate.linear_constraints
    ]
    assert linsolve(equations, ratio) is EmptySet


def test_constant_and_residue_constraints_can_conflict() -> None:
    lower = rational_base()
    extension = hyperexponential_extension(
        lower,
        1 + 2/(x**2 + 1),
    )

    result = solve_parametric_logarithmic_derivative(
        1 + 1/(x**2 + 1),
        extension,
        lower.derivation(),
    )

    assert result.proves_no_solution
    assert (QQ.one, QQ.one) in result.certificate.linear_constraints
    assert (QQ.one, QQ(2)) in result.certificate.linear_constraints

    ratio = symbols("ratio")
    equations = [
        left - ratio*right
        for left, right in result.certificate.linear_constraints
    ]
    assert linsolve(equations, ratio) is EmptySet


def test_complete_solver_keeps_conclusive_fast_path_result() -> None:
    lower = rational_base()
    eta = x**2 + 1/x
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        (1/(x - 1) + 3*eta)/2,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (2, 3)
    assert isinstance(
        result.certificate,
        ParametricLogarithmicDerivativeHeuristicCertificate,
    )
    assert result.reconstructs


def test_complete_solver_dispatches_to_primitive_lower_field() -> None:
    rational = rational_base()
    lower = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    extension = hyperexponential_extension(lower, 1)

    result = solve_parametric_logarithmic_derivative(
        1,
        extension,
        lower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 1)
    assert result.argument == lower.field().one
    assert result.reconstructs


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=-6, max_value=6).filter(lambda value: value != 0),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_structure_fallback_residue_cancellations(
    n,
    m,
    x_power,
    shifted_power,
) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    field = lower.field()
    eta_expression = 1/(x**2 + 1)
    eta = field.from_sympy(eta_expression)
    extension = hyperexponential_extension(lower, eta_expression)
    argument = field.from_sympy(
        x**x_power*(x - 1)**shifted_power
    )
    f = (derivation(argument)/argument + m*eta)/n

    heuristic = solve_parametric_logarithmic_derivative_heuristic(
        f,
        extension,
        derivation,
    )
    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )

    assert heuristic.is_inconclusive
    assert result.has_solution
    assert result.reconstructs
    assert QQ(result.m, result.n) == QQ(m, n)
    assert isinstance(
        result.certificate,
        RationalParametricLogarithmicDerivativeCertificate,
    )


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=6),
    m=st.integers(min_value=-6, max_value=6),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_structure_fallback_constant_constraints(
    n,
    m,
    x_power,
    shifted_power,
) -> None:
    lower = rational_base()
    derivation = lower.derivation()
    field = lower.field()
    extension = hyperexponential_extension(lower, 1)
    argument = field.from_sympy(
        x**x_power*(x - 1)**shifted_power
    )
    f = (derivation(argument)/argument + m*field.one)/n

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert QQ(result.m, result.n) == QQ(m, n)
    assert isinstance(
        result.certificate,
        RationalParametricLogarithmicDerivativeCertificate,
    )


@settings(max_examples=60, deadline=None)
@given(
    f_quadratic=st.integers(min_value=-3, max_value=3),
    f_linear=st.integers(min_value=-3, max_value=3),
    f_constant=st.integers(min_value=-3, max_value=3),
    eta_quadratic=st.integers(min_value=1, max_value=3),
    eta_linear=st.integers(min_value=-3, max_value=3),
    eta_constant=st.integers(min_value=-3, max_value=3),
)
def test_complete_solver_is_total_on_generated_squarefree_rational_inputs(
    f_quadratic,
    f_linear,
    f_constant,
    eta_quadratic,
    eta_linear,
    eta_constant,
) -> None:
    lower = rational_base()
    f = (
        f_quadratic/(x**2 + 1)
        + f_linear/x
        + f_constant
    )
    eta = (
        eta_quadratic/(x**2 + 1)
        + eta_linear/x
        + eta_constant
    )
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        lower.derivation(),
    )

    assert result.outcome in {
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
    }
    assert isinstance(
        result.certificate,
        RationalParametricLogarithmicDerivativeCertificate,
    )

    oracle_solutions = _independent_rational_solution_set(f, eta)
    assert result.has_solution is (oracle_solutions is not EmptySet)
    if result.has_solution:
        assert result.reconstructs


@settings(max_examples=80, deadline=None)
@given(
    f_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=8,
        max_size=8,
    ).map(tuple),
    eta_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=8,
        max_size=8,
    ).filter(any).map(tuple),
)
def test_complete_solver_matches_independent_global_rational_oracle(
    f_coefficients,
    eta_coefficients,
) -> None:
    """Exercise polynomial, repeated-pole, and higher-degree obstructions."""

    basis = (
        x**2,
        x,
        1,
        1/x,
        1/x**3,
        1/(x - 1)**2,
        (x + 1)/(x**2 + 1),
        1/(x**3 + x + 1),
    )
    f = sum(
        coefficient*term
        for coefficient, term in zip(f_coefficients, basis, strict=True)
    )
    eta = sum(
        coefficient*term
        for coefficient, term in zip(eta_coefficients, basis, strict=True)
    )
    lower = rational_base()
    extension = hyperexponential_extension(lower, eta)

    result = solve_parametric_logarithmic_derivative(
        f,
        extension,
        lower.derivation(),
    )
    oracle_solutions = _independent_rational_solution_set(f, eta)

    assert result.outcome in {
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
    }
    assert result.has_solution is (oracle_solutions is not EmptySet)
    if result.has_solution:
        assert result.reconstructs
