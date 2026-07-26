"""Sound conservative certificates for special monomial polynomials."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    SpecialPolynomialStatus,
    certify_primitive_extension,
)
from sympy_risch.special import (
    analyze_special_polynomials,
    certify_primitive_no_special_polynomials,
    certify_special_polynomial,
)

x, t = symbols("x t")


def monomial_derivation(
    generator_derivative,
    *,
    constant_status=ConstantFieldStatus.PRESERVED,
):
    """Return ``QQ(x)(t)`` with the supplied derivative of ``t``."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=base.field()),
            constant_status,
        ),
    )
    return tower.derivation()


def certified_primitive_derivation():
    """Build QQ(x)(t), Dt = 1/x, through the checked primitive factory."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    proposal = certify_primitive_extension(
        Poly(1 / x, t, domain=base.field()),
        base.derivation(),
    )

    assert proposal.status is ConstantFieldStatus.PRESERVED
    return base.extend(proposal.extension).derivation()


def test_analyzer_discovers_the_hyperexponential_canonical_factor() -> None:
    derivation = monomial_derivation(x * t)

    analysis = analyze_special_polynomials(derivation)

    assert analysis.status is SpecialPolynomialStatus.PRESENT
    assert analysis.has_special_polynomials
    assert analysis.reconstructs
    assert [certificate.factor.as_expr() for certificate in analysis.certificates] == [
        t,
    ]
    assert analysis.certificates[0].cofactor == Poly(
        x,
        t,
        domain=derivation.tower.field(0),
    )


def test_analyzer_discovers_the_hypertangent_canonical_factor() -> None:
    derivation = monomial_derivation(1 + t**2)

    analysis = analyze_special_polynomials(derivation)

    assert analysis.status is SpecialPolynomialStatus.PRESENT
    assert analysis.reconstructs
    assert [certificate.factor.as_expr() for certificate in analysis.certificates] == [
        t**2 + 1,
    ]
    assert analysis.certificates[0].cofactor == Poly(
        2 * t,
        t,
        domain=derivation.tower.field(0),
    )


def test_analyzer_discovers_visible_general_polynomial_special_factors() -> None:
    derivation = monomial_derivation(t * (1 + t**2))

    analysis = analyze_special_polynomials(derivation)

    assert analysis.status is SpecialPolynomialStatus.PRESENT
    assert analysis.reconstructs
    assert {certificate.factor.as_expr() for certificate in analysis.certificates} == {
        t,
        t**2 + 1,
    }


def test_analyzer_discovers_the_hyperbolic_mobius_special_factors() -> None:
    derivation = monomial_derivation(1 - t**2)

    analysis = analyze_special_polynomials(derivation)

    assert analysis.status is SpecialPolynomialStatus.PRESENT
    assert analysis.reconstructs
    assert {certificate.factor.as_expr() for certificate in analysis.certificates} == {
        t - 1,
        t + 1,
    }


def test_analyzer_verifies_a_hidden_caller_supplied_special_factor() -> None:
    derivation = monomial_derivation(t**2 - x * t + 1)
    parent = derivation.tower.field(0)
    hidden = Poly(t - x, t, domain=parent)

    without_hint = analyze_special_polynomials(derivation)
    with_hint = analyze_special_polynomials(derivation, candidates=(hidden,))

    assert without_hint.status is SpecialPolynomialStatus.UNKNOWN
    assert without_hint.certificates == ()
    assert with_hint.status is SpecialPolynomialStatus.PRESENT
    assert with_hint.reconstructs
    certificate = with_hint.certificates[0]
    assert certificate.factor == hidden
    assert certificate.cofactor == Poly(t, t, domain=parent)


def test_analyzer_never_infers_absence_from_an_empty_search() -> None:
    derivation = monomial_derivation(t**2 + x)

    analysis = analyze_special_polynomials(derivation)

    assert analysis.status is SpecialPolynomialStatus.UNKNOWN
    assert not analysis.has_special_polynomials
    assert analysis.certificates == ()


def test_primitive_shape_does_not_masquerade_as_an_absence_certificate() -> None:
    # Here t-x is a new constant.  The primitive shape alone must not be
    # confused with the primitive structure theorem's no-new-constants
    # hypothesis, so an empty automatic search remains UNKNOWN.
    derivation = monomial_derivation(
        1,
        constant_status=ConstantFieldStatus.UNKNOWN,
    )
    parent = derivation.tower.field(0)
    hidden = Poly(t - x, t, domain=parent)

    without_hint = analyze_special_polynomials(derivation)
    with_hint = analyze_special_polynomials(derivation, candidates=(hidden,))

    assert without_hint.status is SpecialPolynomialStatus.UNKNOWN
    assert with_hint.status is SpecialPolynomialStatus.PRESENT
    assert with_hint.certificates[0].cofactor.is_zero


def test_analyzer_certifies_primitive_absence_from_checked_constants() -> None:
    derivation = certified_primitive_derivation()

    analysis = analyze_special_polynomials(derivation)
    certificate = certify_primitive_no_special_polynomials(derivation)

    assert analysis.status is SpecialPolynomialStatus.ABSENT
    assert not analysis.has_special_polynomials
    assert analysis.certificates == ()
    assert analysis.absence_certificate == certificate
    assert analysis.reconstructs


@given(
    degree=st.integers(min_value=1, max_value=5),
    lower_coefficients=st.lists(
        st.integers(min_value=-4, max_value=4),
        min_size=0,
        max_size=8,
    ),
)
@settings(max_examples=30, deadline=None)
def test_generated_primitive_candidates_are_rejected_by_the_absence_theorem(
    degree,
    lower_coefficients,
) -> None:
    derivation = certified_primitive_derivation()
    parent = derivation.tower.field(0)
    candidate = Poly(
        t**degree
        + sum(
            coefficient * x**index * t ** (index % degree)
            for index, coefficient in enumerate(lower_coefficients)
        ),
        t,
        domain=parent,
    )

    with pytest.raises(ValueError, match="not a special polynomial"):
        certify_special_polynomial(candidate, derivation)


@pytest.mark.parametrize(
    ("status", "generator_derivative"),
    [
        (ConstantFieldStatus.UNKNOWN, 1 / x),
        # Dt = 1 makes t - x a genuine new constant.
        (ConstantFieldStatus.ENLARGED, 1),
    ],
)
def test_primitive_absence_requires_certified_constants_and_primitive_shape(
    status,
    generator_derivative,
) -> None:
    unproved = monomial_derivation(
        generator_derivative,
        constant_status=status,
    )
    hyperexponential = monomial_derivation(t)

    assert (
        analyze_special_polynomials(unproved).status is SpecialPolynomialStatus.UNKNOWN
    )
    with pytest.raises(ValueError, match=f"status is {status.value}"):
        certify_primitive_no_special_polynomials(unproved)
    with pytest.raises(ValueError, match="requires Dt in the lower field"):
        certify_primitive_no_special_polynomials(hyperexponential)


def test_special_candidate_exposes_an_inconsistent_primitive_status() -> None:
    # This deliberately lies about preservation: Dt = 1 makes t - x a new
    # constant.  A supplied Darboux identity must expose the contradiction,
    # rather than allowing the analyzer to return an unsound ABSENT result.
    derivation = monomial_derivation(1)
    parent = derivation.tower.field(0)
    hidden = Poly(t - x, t, domain=parent)

    assert certify_special_polynomial(hidden, derivation).cofactor.is_zero
    with pytest.raises(ValueError, match="contradicts"):
        analyze_special_polynomials(derivation, candidates=(hidden,))


def test_hyperexponential_canonical_factor_is_not_an_exhaustive_claim() -> None:
    # Dt/t = 1/x is a logarithmic derivative, so t/x is a new constant.
    # Accordingly t-x is special as well as the canonical t factor.  This
    # counterexample makes it essential that the bounded analyzer report only
    # positive evidence, not a complete enumeration.
    derivation = monomial_derivation(
        t / x,
        constant_status=ConstantFieldStatus.UNKNOWN,
    )
    parent = derivation.tower.field(0)
    hidden = Poly(t - x, t, domain=parent)

    canonical = analyze_special_polynomials(derivation)
    with_hint = analyze_special_polynomials(derivation, candidates=(hidden,))

    assert [certificate.factor.as_expr() for certificate in canonical.certificates] == [
        t,
    ]
    assert {certificate.factor.as_expr() for certificate in with_hint.certificates} == {
        t,
        t - x,
    }
    assert with_hint.reconstructs


def test_certification_rejects_a_non_special_supplied_candidate() -> None:
    derivation = monomial_derivation(t**2 + x)
    parent = derivation.tower.field(0)

    with pytest.raises(ValueError, match="not a special polynomial"):
        certify_special_polynomial(Poly(t, t, domain=parent), derivation)
    with pytest.raises(ValueError, match="not a special polynomial"):
        analyze_special_polynomials(
            derivation,
            candidates=(Poly(t, t, domain=parent),),
        )
