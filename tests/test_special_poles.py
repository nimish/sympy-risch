"""Certified local reduction at caller-supplied linear special factors."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    RationalRDEObstruction,
    SpecialPolynomialCertificate,
    certify_special_polynomial,
)
from sympy_risch.monomial import (
    LinearSpecialPoleReduction,
    LinearSpecialPoleReductionStatus,
    LinearSpecialPoleStep,
    reduce_certified_linear_special_poles,
)

x, t, u, y = symbols("x t u y")


def rational_base() -> DifferentialTower:
    """Return the standard rational base ``QQ(x)``."""
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )


def monomial_tower(
    generator_derivative,
) -> DifferentialTower:
    """Return ``QQ(x)(t)`` with the selected polynomial derivative law."""
    base = rational_base()
    return base.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def linear_certificate(
    tower: DifferentialTower,
    factor,
) -> SpecialPolynomialCertificate:
    """Certify one active linear factor in the test tower."""
    return certify_special_polynomial(
        Poly(factor, t, domain=tower.field(0)),
        tower.derivation(),
    )


def test_reduces_a_moving_riccati_special_pole_with_the_residue_cofactor() -> None:
    # D(t-x)=(t+1/x)(t-x), so the relevant lower-field coefficient is
    # (t+1/x) mod (t-x)=x+1/x, not the constant coefficient 1/x.
    tower = monomial_tower(t**2 + (1 / x - x) * t)
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t - x)
    planted_exact = field.from_sympy(x / (t - x) ** 2)

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact),
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.is_fully_reduced
    assert result.reconstructs
    assert result.exact_part == planted_exact
    assert result.remainder == field.zero
    assert result.remaining_pole_order == 0
    assert certificate.cofactor == Poly(t + 1 / x, t, domain=tower.field(0))
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.pole_order == 2
    assert cancel(tower.field(0).to_sympy(step.cofactor_residue) - (x + 1 / x)) == 0
    assert step.rde_result is not None
    assert step.rde_result.reconstructs
    assert (
        cancel(tower.field(0).to_sympy(step.rde_result.coefficient) + 2 * (x + 1 / x))
        == 0
    )
    assert tower.field(0).to_sympy(step.principal_coefficient) == -2 * x**2 - 1
    assert tower.field(0).to_sympy(step.rde_result.solution) == x


def test_retains_completed_local_work_before_an_unresolved_lower_rde() -> None:
    tower = monomial_tower(t**2 + (1 / x - x) * t)
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t - x)
    planted_exact = field.from_sympy(x / (t - x) ** 2)
    unresolved_part = field.from_sympy(1 / (t - x))

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact) + unresolved_part,
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.UNRESOLVED_RDE
    assert result.reconstructs
    assert [step.pole_order for step in result.steps] == [2, 1]
    assert result.steps[0].correction is not None
    assert cancel(field.to_sympy(result.steps[0].correction - planted_exact)) == 0
    assert cancel(field.to_sympy(result.exact_part - planted_exact)) == 0
    assert cancel(field.to_sympy(result.remainder - unresolved_part)) == 0


def test_recomputes_lower_orders_after_each_full_special_derivative() -> None:
    tower = monomial_tower(t * (t - 1))
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)
    planted_exact = field.from_sympy(x / t**2 + (x + 1) / t)

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact),
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.exact_part == planted_exact
    assert result.remainder == field.zero
    assert [step.pole_order for step in result.steps] == [2, 1]
    assert [
        tower.field(0).to_sympy(step.principal_coefficient) for step in result.steps
    ] == [2 * x + 1, x + 2]
    assert [
        tower.field(0).to_sympy(step.rde_result.solution)
        for step in result.steps
        if step.rde_result is not None and step.rde_result.solution is not None
    ] == [x, x + 1]


def test_reduces_a_zero_residue_cofactor() -> None:
    # Here Dt=t**2 and p=t, so h=t and h mod p is zero.  The leading
    # equation is an ordinary lower-field exact-derivative problem.
    tower = monomial_tower(t**2)
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)

    result = reduce_certified_linear_special_poles(
        field.from_sympy(1 / t**2),
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.remaining_pole_order == 0
    assert [step.pole_order for step in result.steps] == [2, 1]
    assert all(step.cofactor_residue == tower.field(0).zero for step in result.steps)


def test_reduces_a_special_factor_with_a_quadratic_cofactor() -> None:
    # D(t)=(t**2+1)*t.  Although the cofactor has degree two, its residue at
    # p=t is the lower-field scalar 1 and the local RDE remains scalar.
    tower = monomial_tower(t * (t**2 + 1))
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)
    planted_exact = field.from_sympy(x / t)

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact),
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.exact_part == planted_exact
    assert result.remainder == field.zero
    assert len(result.steps) == 1
    assert result.steps[0].cofactor_residue == tower.field(0).one


def test_reduces_over_a_gaussian_rational_lower_field() -> None:
    base = DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
            ConstantFieldStatus.UNKNOWN,
        )
    )
    tower = base.extend(
        MonomialExtension(
            Poly(t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    certificate = certify_special_polynomial(
        Poly(t, t, domain=base.field()),
        derivation,
    )
    planted_exact = field.from_sympy(x / t)

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact),
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.exact_part == planted_exact
    assert result.remainder == field.zero


def test_reduction_certificate_rejects_a_forged_lifted_correction() -> None:
    tower = monomial_tower(t * (t - 1))
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)
    planted_exact = field.from_sympy(x / t**2)
    result = reduce_certified_linear_special_poles(
        derivation(planted_exact),
        derivation,
        certificate,
    )
    step = result.steps[0]
    forged_step = LinearSpecialPoleStep(
        step.pole_order,
        step.principal_coefficient,
        step.cofactor_residue,
        step.rde_result,
        field.one,
    )

    with pytest.raises(ValueError, match="wrong lifted correction"):
        LinearSpecialPoleReduction(
            result.derivation,
            result.certificate,
            result.integrand,
            result.exact_part,
            result.remainder,
            (forged_step,),
            result.status,
            result.remaining_pole_order,
        )


def test_reduces_only_the_certified_factor_and_preserves_other_poles() -> None:
    tower = monomial_tower(t * (t - 1))
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)
    planted_exact = field.from_sympy(1 / t)
    unrelated_remainder = field.from_sympy(1 / (t - 1) ** 2 + 1 / (t + 1))

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact) + unrelated_remainder,
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.exact_part == planted_exact
    assert result.remainder == unrelated_remainder
    assert result.remaining_pole_order == 0


def test_unsolved_local_rde_is_a_boundary_not_a_non_elementarity_verdict() -> None:
    tower = monomial_tower(t**2 + (1 / x - x) * t)
    field = tower.field()
    certificate = linear_certificate(tower, t - x)
    integrand = field.from_sympy(1 / (t - x))

    result = reduce_certified_linear_special_poles(
        integrand,
        tower.derivation(),
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.UNRESOLVED_RDE
    assert not result.is_fully_reduced
    assert result.reconstructs
    assert result.exact_part == field.zero
    assert result.remainder == integrand
    assert result.remaining_pole_order == 1
    assert result.unsupported_detail is None
    assert result.unresolved_step is not None
    assert result.unresolved_step.rde_result is not None
    assert not result.unresolved_step.rde_result.has_solution
    assert (
        result.unresolved_step.rde_result.obstruction
        is RationalRDEObstruction.SPDE_DEGREE
    )


def test_resonance_keeps_an_exact_integrand_as_an_unresolved_boundary() -> None:
    # The exact antiderivative has a higher p-pole whose leading terms cancel:
    # D(x**2/t**2)=-2/t.  Thus a failed order-one local RDE cannot be treated
    # as a global exactness or elementary-integration obstruction.
    tower = monomial_tower(t / x + t**2 / x**2)
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)
    integrand = field.from_sympy(-2 / t)
    hidden_exact = field.from_sympy(x**2 / t**2)

    assert derivation(hidden_exact) == integrand
    result = reduce_certified_linear_special_poles(integrand, derivation, certificate)

    assert result.status is LinearSpecialPoleReductionStatus.UNRESOLVED_RDE
    assert result.reconstructs
    assert result.exact_part == field.zero
    assert result.remainder == integrand
    assert result.remaining_pole_order == 1
    assert result.unresolved_step is not None
    assert result.unresolved_step.rde_result is not None
    assert not result.unresolved_step.rde_result.has_solution
    assert (
        tower.field(0).to_sympy(result.unresolved_step.rde_result.coefficient) == -1 / x
    )
    assert tower.field(0).to_sympy(result.unresolved_step.principal_coefficient) == -2


def test_unsupported_lower_rde_is_retained_as_a_structured_boundary() -> None:
    base = rational_base()
    lower = base.extend(
        MonomialExtension(
            Poly(y**2 + x, y, domain=base.field()),
            ConstantFieldStatus.UNKNOWN,
        )
    )
    tower = lower.extend(
        MonomialExtension(
            Poly(u, u, domain=lower.field()),
            ConstantFieldStatus.UNKNOWN,
        )
    )
    certificate = certify_special_polynomial(
        Poly(u, u, domain=lower.field()),
        tower.derivation(),
    )
    integrand = tower.field().from_sympy(1 / u)

    result = reduce_certified_linear_special_poles(
        integrand,
        tower.derivation(),
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.UNSUPPORTED_LOWER_RDE
    assert result.reconstructs
    assert result.exact_part == tower.field().zero
    assert result.remainder == integrand
    assert result.remaining_pole_order == 1
    assert result.unsupported_detail is not None
    assert result.unresolved_step is not None
    assert result.unresolved_step.rde_result is None


def test_constant_lower_field_is_a_structured_rde_boundary() -> None:
    tower = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(t, t, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    derivation = tower.derivation()
    certificate = certify_special_polynomial(Poly(t, t, domain=QQ), derivation)
    integrand = tower.field().from_sympy(1 / t)

    result = reduce_certified_linear_special_poles(
        integrand,
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.UNSUPPORTED_LOWER_RDE
    assert result.reconstructs
    assert result.exact_part == tower.field().zero
    assert result.remainder == integrand
    assert result.unsupported_detail is not None
    assert "standard rational C(x) base" in result.unsupported_detail


def test_replayed_lower_constants_enable_linear_special_reduction() -> None:
    base = rational_base()
    lower = base.extend(
        MonomialExtension(
            Poly(1 / x, y, domain=base.field()),
            ConstantFieldStatus.UNKNOWN,
        )
    )
    tower = lower.extend(
        MonomialExtension(
            Poly(u, u, domain=lower.field()),
            ConstantFieldStatus.UNKNOWN,
        )
    )
    derivation = tower.derivation()
    certificate = certify_special_polynomial(
        Poly(u, u, domain=lower.field()),
        derivation,
    )
    integrand = tower.field().from_sympy(1 / u)

    result = reduce_certified_linear_special_poles(
        integrand,
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.exact_part == tower.field().from_sympy(-1 / u)
    assert result.remainder == tower.field().zero
    assert result.unsupported_detail is None


def test_nested_replay_certifies_constants_before_special_pole_reduction() -> None:
    base = rational_base()
    primitive = base.extend(
        MonomialExtension(
            Poly(1 / x, y, domain=base.field()),
            ConstantFieldStatus.UNKNOWN,
        )
    )
    hypertangent = primitive.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = hypertangent.extend(
        MonomialExtension(
            Poly(u, u, domain=hypertangent.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    derivation = tower.derivation()
    certificate = certify_special_polynomial(
        Poly(u, u, domain=hypertangent.field()),
        derivation,
    )
    integrand = tower.field().from_sympy(1 / u)

    result = reduce_certified_linear_special_poles(
        integrand,
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.exact_part == tower.field().from_sympy(-1 / u)
    assert result.remainder == tower.field().zero
    assert result.unsupported_detail is None
    assert result.unresolved_step is None


def test_solver_value_errors_are_not_reclassified_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tower = monomial_tower(t)
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t)

    def fail_rde_solver(*_args: object, **_kwargs: object) -> object:
        raise ValueError("deliberate lower-RDE invariant failure")

    monkeypatch.setattr(
        "sympy_risch.monomial.special_poles.solve_risch_differential_equation",
        fail_rde_solver,
    )

    with pytest.raises(ValueError, match="deliberate lower-RDE invariant failure"):
        reduce_certified_linear_special_poles(
            tower.field().from_sympy(1 / t),
            derivation,
            certificate,
        )


def test_contract_rejects_non_linear_and_foreign_certificates() -> None:
    tangent_tower = monomial_tower(1 + t**2)
    nonlinear_certificate = linear_certificate(tangent_tower, 1 + t**2)
    with pytest.raises(ValueError, match="linear factor"):
        reduce_certified_linear_special_poles(
            tangent_tower.field().from_sympy(1 / (1 + t**2)),
            tangent_tower.derivation(),
            nonlinear_certificate,
        )

    first = monomial_tower(t)
    second = monomial_tower(t * (t - 1))
    foreign_certificate = linear_certificate(first, t)
    with pytest.raises(ValueError, match="another derivation"):
        reduce_certified_linear_special_poles(
            second.field().from_sympy(1 / t),
            second.derivation(),
            foreign_certificate,
        )

    equal_second = monomial_tower(t)
    assert first == equal_second
    assert first is not equal_second
    with pytest.raises(ValueError, match="another derivation"):
        reduce_certified_linear_special_poles(
            equal_second.field().from_sympy(1 / t),
            equal_second.derivation(),
            foreign_certificate,
        )

    contradictory = monomial_tower(t * (t - 1))
    contradictory_certificate = linear_certificate(contradictory, t)
    result = reduce_certified_linear_special_poles(
        contradictory.field().from_sympy(1 / t),
        contradictory.derivation(),
        contradictory_certificate,
    )
    assert result.reconstructs


def test_missing_target_pole_is_a_checked_no_op() -> None:
    tower = monomial_tower(t * (t - 1))
    field = tower.field()
    integrand = field.from_sympy(1 / (t - 1))

    result = reduce_certified_linear_special_poles(
        integrand,
        tower.derivation(),
        linear_certificate(tower, t),
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.is_fully_reduced
    assert result.reconstructs
    assert result.steps == ()
    assert result.exact_part == field.zero
    assert result.remainder == integrand


@given(
    root=st.sampled_from((0, x, x + 1, 1 / x)),
    cofactor_offset=st.sampled_from((-x, -1, 0, 1, x)),
    numerator=st.sampled_from((1, x, x + 1, x**2 + 1)),
    pole_order=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=24, deadline=None)
def test_generated_linear_special_principal_parts_reconstruct(
    root,
    cofactor_offset,
    numerator,
    pole_order,
) -> None:
    base = rational_base()
    lower_derivation = base.derivation()
    root_derivative = base.field().to_sympy(
        lower_derivation(base.field().from_sympy(root))
    )
    generator_derivative = root_derivative + (t + cofactor_offset) * (t - root)
    tower = base.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    certificate = linear_certificate(tower, t - root)
    planted_exact = field.from_sympy(numerator / (t - root) ** pole_order)
    leading_coefficient = base.field().to_sympy(
        lower_derivation(base.field().from_sympy(numerator))
        - base.field().convert(pole_order)
        * base.field().from_sympy(root + cofactor_offset)
        * base.field().from_sympy(numerator)
    )
    assume(cancel(leading_coefficient) != 0)

    result = reduce_certified_linear_special_poles(
        derivation(planted_exact),
        derivation,
        certificate,
    )

    assert result.status is LinearSpecialPoleReductionStatus.REDUCED
    assert result.reconstructs
    assert result.remaining_pole_order == 0
    assert result.steps
    assert result.steps[0].pole_order == pole_order
    assert all(step.is_reduced for step in result.steps)
