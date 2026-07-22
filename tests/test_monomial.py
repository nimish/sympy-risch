import pytest
from hypothesis import given
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import DifferentialTower, MonomialExtension, split_normal_special


x, t = symbols("x t")


def monomial_tower(generator_derivative) -> DifferentialTower:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    return rational.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=rational.field()),
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
