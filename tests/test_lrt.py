import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, Rational, cancel, prod, symbols

from sympy_risch import (
    integrate_rational,
    lazard_rioboo_trager,
    rothstein_trager_residue_polynomial,
)

x, z = symbols("x z")


def test_lrt_quadratic_with_algebraic_residues() -> None:
    parts = lazard_rioboo_trager(1, x**2 + x + 1, x, z)

    assert len(parts) == 1
    logand, residue_factor = parts[0]
    assert logand.as_expr() == x + 3 * z / 2 + Rational(1, 2)
    assert residue_factor == Poly(3 * z**2 + 1, z)


def test_lrt_groups_poles_with_the_same_residue() -> None:
    parts = lazard_rioboo_trager(2 * x - 1, x**2 - x, x, z)

    assert len(parts) == 1
    logand, residue_factor = parts[0]
    assert logand == Poly(x**2 - x, x, domain="QQ")
    assert residue_factor.monic() == Poly(z - 1, z, domain="QQ")


def test_lrt_two_distinct_residues() -> None:
    parts = lazard_rioboo_trager(-1, x * (x + 1), x, z)

    assert len(parts) == 1
    logand, residue_factor = parts[0]
    assert logand.as_expr() == x + z / 2 + Rational(1, 2)
    assert residue_factor.monic() == Poly(z**2 - 1, z, domain="QQ")


@pytest.mark.parametrize(
    ("numerator", "denominator", "message"),
    [
        (x, 1, "nonconstant denominator"),
        (x, x + 1, "proper rational function"),
        (x, x * (x + 1), "coprime"),
        (1, x**2, "square-free"),
    ],
)
def test_lrt_rejects_inputs_outside_its_preconditions(
    numerator,
    denominator,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        lazard_rioboo_trager(numerator, denominator, x, z)


def test_lrt_requires_a_residue_symbol_new_over_the_coefficients() -> None:
    with pytest.raises(ValueError, match="coefficient field"):
        lazard_rioboo_trager(z, x + 1, x, z)
    with pytest.raises(TypeError, match="must be a Symbol"):
        lazard_rioboo_trager(1, x + 1, x, 0)  # type: ignore[arg-type]


@st.composite
def rational_functions_with_rational_poles(draw):
    roots = draw(
        st.lists(
            st.integers(-5, 5),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    coefficients = draw(
        st.lists(
            st.integers(-4, 4),
            min_size=len(roots),
            max_size=len(roots),
        )
    )
    assume(any(coefficients))

    denominator = Poly(prod(x - root for root in roots), x, domain="QQ")
    numerator = Poly(
        sum(coefficient * x**degree for degree, coefficient in enumerate(coefficients)),
        x,
        domain="QQ",
    )
    assume(numerator.gcd(denominator).degree() == 0)

    return numerator, denominator, roots


@st.composite
def arbitrary_reduced_rational_parts(draw):
    denominator_degree = draw(st.integers(1, 5))
    denominator_leading = draw(st.sampled_from([-3, -2, -1, 1, 2, 3]))
    denominator_coefficients = draw(
        st.lists(
            st.integers(-3, 3),
            min_size=denominator_degree,
            max_size=denominator_degree,
        )
    )
    numerator_coefficients = draw(
        st.lists(
            st.integers(-3, 3),
            min_size=denominator_degree,
            max_size=denominator_degree,
        )
    )
    assume(any(numerator_coefficients))

    denominator = Poly(
        denominator_leading * x**denominator_degree
        + sum(
            coefficient * x**degree
            for degree, coefficient in enumerate(denominator_coefficients)
        ),
        x,
        domain="QQ",
    )
    numerator = Poly(
        sum(
            coefficient * x**degree
            for degree, coefficient in enumerate(numerator_coefficients)
        ),
        x,
        domain="QQ",
    )
    assume(denominator.gcd(denominator.diff()).degree() == 0)
    assume(numerator.gcd(denominator).degree() == 0)

    return numerator, denominator


@given(rational_functions_with_rational_poles())
@settings(max_examples=75, deadline=None)
def test_lrt_specializations_group_poles_by_residue(example) -> None:
    numerator, denominator, roots = example
    parts = lazard_rioboo_trager(numerator, denominator, x, z)
    residues = {
        root: numerator.eval(root) / denominator.diff().eval(root) for root in roots
    }

    for residue in set(residues.values()):
        matching_parts = [
            logand
            for logand, residue_factor in parts
            if residue_factor.eval(residue) == 0
        ]
        assert len(matching_parts) == 1

        specialized_logand = Poly(
            matching_parts[0].as_expr().subs(z, residue),
            x,
            domain="QQ",
        ).monic()
        expected_logand = Poly(
            prod(
                x - root
                for root, root_residue in residues.items()
                if root_residue == residue
            ),
            x,
            domain="QQ",
        ).monic()

        assert specialized_logand == expected_logand


@given(arbitrary_reduced_rational_parts())
@settings(max_examples=75, deadline=None)
def test_lrt_parts_reconstruct_residue_resultant(example) -> None:
    numerator, denominator = example
    parts = lazard_rioboo_trager(numerator, denominator, x, z)
    residue_resultant = rothstein_trager_residue_polynomial(
        numerator.as_expr() / denominator.as_expr(),
        x,
        z,
    )
    reconstructed_resultant = Poly(
        prod(
            residue_factor.as_expr() ** logand.degree()
            for logand, residue_factor in parts
        ),
        z,
        domain="QQ",
    )

    assert reconstructed_resultant.monic() == residue_resultant.monic()

    for _, residue_factor in parts:
        assert residue_factor.gcd(residue_factor.diff()).degree() == 0


@pytest.mark.parametrize(
    "integrand",
    (
        1 / (x**3 + x + 1),
        (x**2 + 1) / (x**4 + x + 1),
        (x**2 + 1) / (x**5 + x + 1),
    ),
)
def test_integrate_rational_with_higher_degree_algebraic_poles(integrand) -> None:
    antiderivative = integrate_rational(integrand, x)

    assert cancel(antiderivative.diff(x) - integrand) == 0


@given(rational_functions_with_rational_poles())
@settings(max_examples=50, deadline=None)
def test_integrate_rational_property_differentiates_to_input(example) -> None:
    numerator, denominator, _ = example
    integrand = numerator.as_expr() / denominator.as_expr()
    antiderivative = integrate_rational(integrand, x)

    assert cancel(antiderivative.diff(x) - integrand) == 0
