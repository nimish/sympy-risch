"""Radical logarithmic-derivative signatures over one primitive level."""

from dataclasses import FrozenInstanceError
from math import lcm

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Dummy, EmptySet, Poly, cancel, linsolve, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    PrimitiveLogarithmicDerivativeSignature,
    primitive_logarithmic_derivative_signature,
)


x, t, u = symbols("x t u")


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


def _independent_radical_logarithmic_oracle(
    expression,
    derivation,
) -> bool:
    r"""Solve a global logarithmic ansatz without local principal parts.

    Unique factorization says a radical logarithmic derivative in
    ``QQ(x)(t)`` is a rational linear combination of ``Dp/p`` for monic
    irreducible top factors, plus ``q'/q`` for lower-field factors.  This
    oracle constructs that global identity directly and asks ``linsolve`` for
    its rational residue coefficients.  It does not inspect the production
    signature or any production obstruction flag.
    """
    extension = derivation.tower.levels[derivation.level]
    field = extension.field
    element_expression = field.to_sympy(field.from_sympy(expression))
    numerator_expression, denominator_expression = cancel(
        element_expression,
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
    numerator = numerator.mul_ground(leading_inverse)
    denominator = denominator.monic()
    polynomial_part, _proper_numerator = numerator.div(denominator)

    top_factors = tuple(
        factor.monic()
        for factor, _multiplicity in denominator.factor_list()[1]
    )
    lower_constant = (
        extension.parent_field.zero
        if polynomial_part.is_zero
        else extension.parent_field.convert(polynomial_part.nth(0))
    )
    lower_numerator_expression, lower_denominator_expression = cancel(
        extension.parent_field.to_sympy(lower_constant),
    ).as_numer_denom()
    lower_denominator = Poly(
        lower_denominator_expression,
        x,
        domain=QQ,
    )
    lower_factors = tuple(
        factor.monic()
        for factor, _multiplicity in lower_denominator.factor_list()[1]
    )

    top_residues = tuple(
        Dummy(f"top_residue_{index}")
        for index in range(len(top_factors))
    )
    lower_residues = tuple(
        Dummy(f"lower_residue_{index}")
        for index in range(len(lower_factors))
    )
    candidate = 0
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
        element_expression - candidate,
    ).as_numer_denom()[0]
    equations = Poly(identity_numerator, x, t).coeffs()
    unknowns = (*top_residues, *lower_residues)
    if not unknowns:
        return all(equation == 0 for equation in equations)
    return linsolve(equations, unknowns) is not EmptySet


def test_signature_combines_lower_and_top_radical_residues() -> None:
    tower = logarithmic_tower()
    derivation = tower.derivation()

    signature = primitive_logarithmic_derivative_signature(
        1/(2*x) + 1/(3*x*t),
        derivation,
    )

    assert isinstance(signature, PrimitiveLogarithmicDerivativeSignature)
    assert signature.has_solution
    assert signature.minimal_power == 6
    assert signature.relation_argument == tower.field().from_sympy(x**3*t**2)
    assert signature.reconstructs_decomposition
    assert signature.reconstructs_relation
    assert not signature.has_positive_degree_polynomial_obstruction
    assert not signature.has_repeated_pole_obstruction
    assert not signature.has_nonlogarithmic_residue_obstruction
    assert not signature.has_lower_field_obstruction


def test_signature_uses_total_derivative_of_coefficient_dependent_factor() -> None:
    tower = logarithmic_tower()
    derivation = tower.derivation()
    field = tower.field()
    factor_expression = t**2 + x
    argument = field.from_sympy(factor_expression)
    integrand = derivation(argument)/(2*argument)

    signature = primitive_logarithmic_derivative_signature(
        integrand,
        derivation,
    )

    assert signature.has_solution
    assert signature.minimal_power == 2
    assert signature.relation_argument == argument
    assert len(signature.factors) == 1
    factor = signature.factors[0]
    assert factor.factor == Poly(
        factor_expression,
        t,
        domain=tower.levels[1].parent_field,
    )
    assert factor.factor_derivative == Poly(
        2*t/x + 1,
        t,
        domain=tower.levels[1].parent_field,
    )
    assert factor.logarithmic_multiplicity == QQ(1, 2)


def test_signature_records_every_repeated_principal_part() -> None:
    tower = logarithmic_tower()
    derivation = tower.derivation()
    field = tower.field()
    factor = field.from_sympy(t - x)
    factor_derivative = derivation(factor)
    integrand = (
        3*factor_derivative/factor
        + 5/factor**2
        + 7/factor**3
    )

    signature = primitive_logarithmic_derivative_signature(
        integrand,
        derivation,
    )

    assert not signature.has_solution
    assert signature.minimal_power is None
    assert signature.relation_argument is None
    assert signature.has_repeated_pole_obstruction
    assert not signature.has_nonlogarithmic_residue_obstruction
    assert signature.reconstructs_decomposition
    assert len(signature.factors) == 1
    factor_signature = signature.factors[0]
    assert factor_signature.multiplicity == 3
    assert tuple(
        cancel(
            numerator.as_expr() - expected,
        )
        for numerator, expected in zip(
            factor_signature.principal_numerators,
            (3/x - 3, 5, 7),
            strict=True,
        )
    ) == (0, 0, 0)
    assert factor_signature.logarithmic_multiplicity == 3


@pytest.mark.parametrize(
    (
        "expression",
        "polynomial_obstruction",
        "repeated_obstruction",
        "residue_obstruction",
        "lower_obstruction",
    ),
    [
        (t + 1/(x*t), True, False, False, False),
        (1/t**2 + 1/(x*t), False, True, False, False),
        (1/t, False, False, True, False),
        (1/(x**2 + 1), False, False, False, True),
        (1/x**2, False, False, False, True),
        (
            t + 1/t**2 + 1/t + 1/(x**2 + 1),
            True,
            True,
            True,
            True,
        ),
    ],
)
def test_signature_distinguishes_independent_obstruction_families(
    expression,
    polynomial_obstruction,
    repeated_obstruction,
    residue_obstruction,
    lower_obstruction,
) -> None:
    signature = primitive_logarithmic_derivative_signature(
        expression,
        logarithmic_tower().derivation(),
    )

    assert not signature.has_solution
    assert (
        signature.has_positive_degree_polynomial_obstruction
        is polynomial_obstruction
    )
    assert signature.has_repeated_pole_obstruction is repeated_obstruction
    assert (
        signature.has_nonlogarithmic_residue_obstruction
        is residue_obstruction
    )
    assert signature.has_lower_field_obstruction is lower_obstruction
    assert signature.reconstructs_decomposition
    assert not signature.reconstructs_relation


def test_signature_is_immutable() -> None:
    signature = primitive_logarithmic_derivative_signature(
        1/(x*t),
        logarithmic_tower().derivation(),
    )

    with pytest.raises(FrozenInstanceError):
        signature.minimal_power = 2
    with pytest.raises(FrozenInstanceError):
        signature.factors[0].logarithmic_multiplicity = 2


@pytest.mark.parametrize(
    "status",
    [
        ConstantFieldStatus.UNKNOWN,
        ConstantFieldStatus.ENLARGED,
    ],
)
def test_signature_rejects_unproved_constant_field_hypothesis(status) -> None:
    derivation = logarithmic_tower(
        constant_field_status=status,
    ).derivation()

    with pytest.raises(ValueError, match="requires Const"):
        primitive_logarithmic_derivative_signature(1/t, derivation)


def test_signature_detects_inconsistent_special_factor_hypothesis() -> None:
    derivation = logarithmic_tower(eta=1).derivation()

    with pytest.raises(ValueError, match="special factor"):
        primitive_logarithmic_derivative_signature(
            1/(t - x),
            derivation,
        )


def test_signature_rejects_nonprimitive_top_level() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = rational.extend(
        MonomialExtension(
            Poly(t, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    with pytest.raises(ValueError, match="require Dt in the lower field"):
        primitive_logarithmic_derivative_signature(
            1,
            tower.derivation(),
        )


def test_signature_rejects_unsupported_tower_depths() -> None:
    rational_derivation = logarithmic_tower().derivation().lower()
    primitive = logarithmic_tower()
    nested = primitive.extend(
        MonomialExtension(
            Poly(1/(x*t), u, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    with pytest.raises(NotImplementedError, match="one primitive level"):
        primitive_logarithmic_derivative_signature(1/x, rational_derivation)
    with pytest.raises(NotImplementedError, match="one primitive level"):
        primitive_logarithmic_derivative_signature(
            1/u,
            nested.derivation(),
        )


@settings(max_examples=60, deadline=None)
@given(
    power=st.integers(min_value=1, max_value=6),
    x_power=st.integers(min_value=-4, max_value=4),
    shifted_x_power=st.integers(min_value=-4, max_value=4),
    t_power=st.integers(min_value=-4, max_value=4),
    quadratic_power=st.integers(min_value=-4, max_value=4),
)
def test_generated_planted_radical_logarithmic_derivatives(
    power,
    x_power,
    shifted_x_power,
    t_power,
    quadratic_power,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    argument = field.from_sympy(
        x**x_power
        * (x - 1)**shifted_x_power
        * t**t_power
        * (t**2 + x)**quadratic_power
    )
    integrand = derivation(argument)/(power*argument)

    signature = primitive_logarithmic_derivative_signature(
        integrand,
        derivation,
    )

    expected_power = 1
    for exponent in (
        x_power,
        shifted_x_power,
        t_power,
        quadratic_power,
    ):
        expected_power = lcm(expected_power, int(QQ.denom(QQ(exponent, power))))
    assert signature.has_solution
    assert signature.minimal_power == expected_power
    assert signature.reconstructs_decomposition
    assert signature.reconstructs_relation


@settings(max_examples=80, deadline=None)
@given(
    coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=13,
        max_size=13,
    ).map(tuple),
)
def test_signature_matches_independent_global_linear_oracle(
    coefficients,
) -> None:
    tower = logarithmic_tower()
    derivation = tower.derivation()
    field = tower.field()
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
        1/(x - 1),
        1/x**2,
        1/(x**2 + 1),
        t,
        1/(x*t),
        1/t,
        quadratic_logarithmic_derivative,
        1/(t**2 + x),
        cubic_logarithmic_derivative,
        1/(t**3 + t + x),
        1/t**2,
        1/(t - x)**2,
    )
    expression = sum(
        coefficient*term
        for coefficient, term in zip(coefficients, basis, strict=True)
    )

    signature = primitive_logarithmic_derivative_signature(
        expression,
        derivation,
    )
    oracle_has_solution = _independent_radical_logarithmic_oracle(
        expression,
        derivation,
    )

    assert signature.has_solution is oracle_has_solution
    assert signature.reconstructs_decomposition
    if signature.has_solution:
        assert signature.reconstructs_relation
