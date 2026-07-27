from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import QQ, Poly, Rational, cancel, prod, sqrt, symbols

from sympy_risch import (
    czichowski_groebner,
    lazard_rioboo_trager,
    rothstein_trager_residue_polynomial,
)

x, z = symbols("x z")


def test_czichowski_reproduces_raab_section_1_3_example() -> None:
    parts = czichowski_groebner(
        x**3 - x + 1,
        (x + 1) * (x + 2),
        x,
        z,
    )

    assert parts == [
        (
            Poly(
                x + z / 4 + Rational(3, 4),
                x,
                domain="QQ(z)",
            ),
            Poly((z - 1) * (z - 5), z, domain="QQ"),
        )
    ]

    logand, residue_factor = parts[0]
    assert Poly(logand.as_expr().subs(z, 1), x, domain="QQ").monic() == Poly(
        x + 1, x, domain="QQ"
    )
    assert Poly(logand.as_expr().subs(z, 5), x, domain="QQ").monic() == Poly(
        x + 2, x, domain="QQ"
    )
    assert residue_factor.monic() == Poly(z**2 - 6 * z + 5, z, domain="QQ")


def test_czichowski_extracts_every_level_of_the_basis_content_chain() -> None:
    numerator = 14 * x**5 + 13 * x**4 - 112 * x**3 - 99 * x**2 + 144 * x + 108
    denominator = x**6 + x**5 - 13 * x**4 - 13 * x**3 + 36 * x**2 + 36 * x

    parts = czichowski_groebner(numerator, denominator, x, z)

    assert [factor for _, factor in parts] == [
        Poly(z - 1, z, domain="QQ"),
        Poly(z - 2, z, domain="QQ"),
        Poly(z - 3, z, domain="QQ"),
    ]
    expected_logands = {
        1: Poly(x + 1, x, domain="QQ"),
        2: Poly(x**2 - 4, x, domain="QQ"),
        3: Poly(x**3 - 9 * x, x, domain="QQ"),
    }
    for logand, residue_factor in parts:
        root = -residue_factor.nth(0)
        assert (
            Poly(logand.as_expr().subs(z, root), x, domain="QQ").monic()
            == expected_logands[root]
        )


def test_czichowski_and_lrt_agree_on_distinct_algebraic_residues() -> None:
    numerator = x**2 + 1
    denominator = x**5 + x + 1

    groebner_parts = czichowski_groebner(numerator, denominator, x, z)
    lrt_parts = lazard_rioboo_trager(numerator, denominator, x, z)

    assert len(groebner_parts) == len(lrt_parts)
    for groebner_logand, residue_factor in groebner_parts:
        matching_lrt = [
            lrt_logand
            for lrt_logand, lrt_factor in lrt_parts
            if lrt_factor.monic() == residue_factor.monic()
        ]
        assert len(matching_lrt) == 1
        lrt_logand = matching_lrt[0]
        difference = Poly(
            groebner_logand.as_expr() - lrt_logand.as_expr(),
            x,
            domain="QQ(z)",
        )
        for coefficient in difference.all_coeffs():
            numerator_in_z, _ = coefficient.as_expr().as_numer_denom()
            assert Poly(numerator_in_z, z).rem(residue_factor).is_zero


def test_czichowski_supports_exact_algebraic_coefficient_fields() -> None:
    algebraic_field = QQ.algebraic_field(sqrt(2))
    numerator = Poly(1, x, domain=QQ)
    denominator = Poly(
        x**2 + sqrt(2) * x + 1,
        x,
        domain=algebraic_field,
    )

    parts = czichowski_groebner(numerator, denominator, x, z)

    assert len(parts) == 1
    logand, residue_factor = parts[0]
    assert logand.as_expr() == x + z + sqrt(2) / 2
    assert residue_factor.monic().as_expr() == z**2 + Rational(1, 2)


def test_czichowski_parts_reconstruct_the_rothstein_trager_resultant() -> None:
    numerator = 3 * x**4 - 2 * x**2 + x + 4
    denominator = x**5 + 2 * x**3 - x + 2
    assert cancel(numerator / denominator) == numerator / denominator

    parts = czichowski_groebner(numerator, denominator, x, z)
    reconstructed = Poly(
        prod(
            residue_factor.as_expr() ** logand.degree()
            for logand, residue_factor in parts
        ),
        z,
        domain="QQ",
    )
    resultant = rothstein_trager_residue_polynomial(
        numerator / denominator,
        x,
        z,
    )

    assert reconstructed.monic() == resultant.monic()


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


@given(rational_functions_with_rational_poles())
@settings(max_examples=50, deadline=None)
def test_czichowski_specializations_group_poles_by_residue(example) -> None:
    numerator, denominator, roots = example
    parts = czichowski_groebner(numerator, denominator, x, z)
    residues = {
        root: numerator.eval(root) / denominator.diff().eval(root) for root in roots
    }

    for residue in set(residues.values()):
        matching_logands = [
            logand
            for logand, residue_factor in parts
            if residue_factor.eval(residue) == 0
        ]
        assert len(matching_logands) == 1
        actual = Poly(
            matching_logands[0].as_expr().subs(z, residue),
            x,
            domain="QQ",
        ).monic()
        expected = Poly(
            prod(
                x - root
                for root, root_residue in residues.items()
                if root_residue == residue
            ),
            x,
            domain="QQ",
        ).monic()
        assert actual == expected
