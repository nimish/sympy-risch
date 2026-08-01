r"""Explicit algebraic integration after a compatible constant base change.

This module owns the orchestration boundary between direct algebraic
logarithmic reconstruction and certified algebraic-constant composita.
Keeping it separate means the direct logarithmic kernel has no reason to know
about target towers or split places.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import Poly

from ..algebraic_geometry.base_change import (
    AlgebraicConstantBaseChangeUnsupported,
    CertifiedAlgebraicConstantBaseChange,
    build_algebraic_constant_base_change,
    build_algebraic_constant_compositum_base_change,
)
from ..tower import Derivation
from .basis import IntegralBasis
from .logarithmic import (
    AlgebraicIntegrationResult,
    AlgebraicIntegrationStatus,
    AlgebraicNeedsConstantBaseChange,
    integrate_algebraic,
)
from .residue_profile import (
    AlgebraicResidueProfileStatus,
    build_algebraic_residue_profile,
)


@dataclass(frozen=True, slots=True)
class AlgebraicConstantBaseChangeStep:
    """One certified target-field integration in a constant-extension chain."""

    base_change: CertifiedAlgebraicConstantBaseChange
    target_result: AlgebraicIntegrationResult

    def __post_init__(self) -> None:
        if not isinstance(self.base_change, CertifiedAlgebraicConstantBaseChange):
            raise TypeError("a constant-extension step needs a certified base change")
        if not isinstance(self.target_result, AlgebraicIntegrationResult):
            raise TypeError("a constant-extension step needs a target result")
        target_extension = self.base_change.target_extension
        target_derivation = self.base_change.target_derivation
        if self.target_result.collection.basis.extension is not target_extension:
            raise ValueError("the target integration belongs to another field")
        if not self.target_result.collection.derivation.has_same_position(
            target_derivation
        ):
            raise ValueError("the target integration belongs to another derivation")


@dataclass(frozen=True, slots=True)
class AlgebraicConstantBaseChangeIntegrationResult:
    r"""An algebraic integration continued through explicit constant fields.

    ``source_result`` records the original differential over ``C(x, y)``.
    Every item in ``steps`` contains a certified map to the next compositum
    and a fresh all-place integration of the embedded differential there.
    A successful antiderivative therefore belongs to the final target field;
    it is never silently presented as an expression in the source field.
    """

    source_result: AlgebraicIntegrationResult
    steps: tuple[AlgebraicConstantBaseChangeStep, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_result, AlgebraicIntegrationResult):
            raise TypeError("a base-change integration needs a source result")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, AlgebraicConstantBaseChangeStep) for step in self.steps
        ):
            raise TypeError("base-change integration steps must be a tuple")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a base-change integration needs a nonempty detail")

        previous = self.source_result
        for step in self.steps:
            if not isinstance(previous, AlgebraicNeedsConstantBaseChange):
                raise ValueError(
                    "only a constant-base-change boundary may be continued"
                )
            source_model = previous.collection.model
            change = step.base_change
            if change.source_model is not source_model:
                raise ValueError("a base change belongs to another source model")
            if not change.source_derivation.has_same_position(
                previous.collection.derivation
            ):
                raise ValueError("a base change belongs to another source derivation")
            target_extension = change.target_extension
            embedded = change.embed(previous.collection.integrand)
            if (
                step.target_result.collection.integrand - embedded
                != target_extension.field.zero
            ):
                raise ValueError(
                    "a target integration did not use the preceding embedded input"
                )
            previous = step.target_result

    @property
    def active_result(self) -> AlgebraicIntegrationResult:
        """The integration result in the last constructed constant field."""
        if self.steps:
            return self.steps[-1].target_result
        return self.source_result

    @property
    def base_change(self) -> CertifiedAlgebraicConstantBaseChange | None:
        """The last certified base change, for convenient inspection."""
        if self.steps:
            return self.steps[-1].base_change
        return None

    @property
    def target_result(self) -> AlgebraicIntegrationResult | None:
        """The last target result, or ``None`` when no change was constructed."""
        if self.steps:
            return self.steps[-1].target_result
        return None

    @property
    def status(self) -> AlgebraicIntegrationStatus:
        """Return the active integration outcome."""
        return self.active_result.status

    @property
    def is_elementary(self) -> bool:
        """Whether the active result is constructive."""
        return self.active_result.is_elementary

    @property
    def complete_negative(self) -> bool:
        """Whether the active field carries a Liouville obstruction."""
        return self.active_result.complete_negative

    @property
    def reconstructs(self) -> bool:
        """Whether the active decomposition reconstructs its input."""
        return self.active_result.reconstructs

    @property
    def source_reconstructs(self) -> bool:
        """Whether the original-field reduction reconstructs its input."""
        return self.source_result.reconstructs

    @property
    def antiderivative(self) -> Expr | None:
        """Return the source or final explicitly base-changed answer."""
        return self.active_result.antiderivative


def _automatic_constant_factor_candidates(
    source_result: AlgebraicIntegrationResult,
) -> tuple[tuple[Poly, ...], str]:
    """Return rationally defined constant-field candidates from a profile.

    A root of an irreducible profile factor is only a *candidate* for a
    compatible constant extension.  The certified base-change builder still
    checks that its root field contains every conjugate needed by the selected
    factor.  Over algebraic constants, only factors whose coefficients descend
    exactly to ``QQ`` enter this constructive compositum cell.
    """
    profile = build_algebraic_residue_profile(
        source_result.collection,
        Dummy("risch_residue"),
    )
    if profile.status is not AlgebraicResidueProfileStatus.PROFILED:
        return (
            (),
            "automatic residue-profile selection is unavailable: " + profile.detail,
        )

    polynomial = profile.square_free_residue_polynomial
    if polynomial is None:
        return (
            (),
            "the residue profile has no square-free polynomial",
        )
    try:
        _content, factors = polynomial.factor_list()
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return (), "the residue profile could not be factored over QQ: " + str(exc)

    rational_factors: list[Poly] = []
    for candidate, _multiplicity in factors:
        if candidate.degree() < 2:
            continue
        try:
            rational = Poly.from_dict(
                {
                    monomial: QQ.convert(coefficient)
                    for monomial, coefficient in candidate.rep.to_dict().items()
                },
                candidate.gens[0],
                domain=QQ,
            ).monic()
        except (BasePolynomialError, NotImplementedError, TypeError, ValueError):
            continue
        if rational.is_irreducible is True:
            rational_factors.append(rational)

    candidates = tuple(
        sorted(
            rational_factors,
            key=lambda factor: (
                factor.degree(),
                tuple(
                    (int(QQ.numer(coefficient)), int(QQ.denom(coefficient)))
                    for coefficient in factor.all_coeffs()
                ),
            ),
        )
    )
    if not candidates:
        return (
            (),
            "the residue profile has no new rationally defined irreducible "
            "factor supported by the constant-compositum layer",
        )
    return (
        candidates,
        "selected rationally defined constant-field candidates from the "
        "square-free all-place residue profile",
    )


def _continue_source_result(
    source_result: AlgebraicNeedsConstantBaseChange,
    factors: tuple[Poly, ...],
    *,
    lazy: bool,
    torsion_order_bound: int | None,
) -> tuple[AlgebraicConstantBaseChangeStep | None, str]:
    """Reintegrate one boundary over a chosen common constant field."""
    source_model = source_result.collection.model
    change = (
        build_algebraic_constant_base_change(
            source_model,
            source_result.collection.derivation,
            factors[0],
        )
        if len(factors) == 1
        else build_algebraic_constant_compositum_base_change(
            source_model,
            source_result.collection.derivation,
            factors,
        )
    )
    match change:
        case AlgebraicConstantBaseChangeUnsupported(detail=detail):
            return (
                None,
                "the selected algebraic constant compositum is unsupported: " + detail,
            )
        case CertifiedAlgebraicConstantBaseChange():
            pass

    target_derivation = change.target_derivation
    target_model = change.target_model
    target_result = integrate_algebraic(
        change.embed(source_result.collection.integrand),
        target_derivation,
        target_model.order.basis,
        lazy=lazy,
        torsion_order_bound=torsion_order_bound,
    )
    return (
        AlgebraicConstantBaseChangeStep(change, target_result),
        "the source differential was integrated again after one certified "
        "algebraic constant-compositum base change",
    )


def integrate_algebraic_with_constant_base_change(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    factor: Poly | None = None,
    *,
    lazy: bool = False,
    torsion_order_bound: int | None = None,
) -> AlgebraicConstantBaseChangeIntegrationResult:
    r"""Continue an algebraic integral through compatible constant fields.

    The source is first integrated through the displayed-constant logarithmic
    slice. If and only if that ends at
    :attr:`AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE`, an explicit
    ``factor`` requests one selected extension. When ``factor`` is omitted,
    every unresolved target is profiled anew and its deterministic rationally
    defined factors are joined into one common constant field before the curve
    is rebuilt. If the common join is unsupported, individual factors are
    tried so one unavailable candidate cannot suppress valid progress. Each
    certified base change strictly enlarges the constants.

    Every step embeds the preceding differential and collects it again at all
    target places before logarithmic reconstruction. Merely transporting a
    source residue list would omit target infinity and split descendants.
    Factors with genuinely algebraic coefficients and nonsplitting root
    fields remain explicit unsupported boundaries.

    ``torsion_order_bound`` is forwarded unchanged at the source and every
    rebuilt target field. A found relation is exact; exhaustion is not a
    nontorsion certificate.
    """
    source_result = integrate_algebraic(
        integrand,
        derivation,
        basis,
        lazy=lazy,
        torsion_order_bound=torsion_order_bound,
    )
    if not isinstance(source_result, AlgebraicNeedsConstantBaseChange):
        return AlgebraicConstantBaseChangeIntegrationResult(
            source_result,
            (),
            "the source integration did not require algebraic constant base change",
        )

    if factor is not None:
        step, detail = _continue_source_result(
            source_result,
            (factor,),
            lazy=lazy,
            torsion_order_bound=torsion_order_bound,
        )
        return AlgebraicConstantBaseChangeIntegrationResult(
            source_result,
            () if step is None else (step,),
            detail,
        )

    steps: list[AlgebraicConstantBaseChangeStep] = []
    current = source_result
    attempted: set[tuple[int, tuple[object, ...]]] = set()
    final_detail = "automatic algebraic constant descent made no progress"
    while isinstance(current, AlgebraicNeedsConstantBaseChange):
        candidates, final_detail = _automatic_constant_factor_candidates(current)
        progressed = False
        pending: list[Poly] = []
        for candidate in candidates:
            key = (
                id(current.collection.basis.extension),
                tuple(candidate.all_coeffs()),
            )
            if key in attempted:
                continue
            attempted.add(key)
            pending.append(candidate)

        # Joining every currently visible residue factor before rebuilding the
        # curve is the common case and avoids repeating maximal-order and
        # all-place work. If one factor is outside the supported splitting-root
        # cell, fall back to individual factors so that it cannot mask an
        # independent supported extension.
        if len(pending) > 1:
            step, attempt_detail = _continue_source_result(
                current,
                tuple(pending),
                lazy=lazy,
                torsion_order_bound=torsion_order_bound,
            )
            if step is not None:
                steps.append(step)
                current = step.target_result
                final_detail = attempt_detail
                progressed = True

        if progressed:
            continue

        if not isinstance(current, AlgebraicNeedsConstantBaseChange):
            raise AssertionError("a non-boundary result cannot continue base change")
        for candidate in pending:
            step, attempt_detail = _continue_source_result(
                current,
                (candidate,),
                lazy=lazy,
                torsion_order_bound=torsion_order_bound,
            )
            if step is None:
                final_detail = attempt_detail
                continue
            steps.append(step)
            current = step.target_result
            final_detail = attempt_detail
            progressed = True
            break
        if not progressed:
            break

    if steps and isinstance(current, AlgebraicNeedsConstantBaseChange):
        final_detail = (
            "the certified compositum chain exhausted the supported residue factors: "
            + final_detail
        )
    elif steps:
        final_detail = (
            f"algebraic integration continued through {len(steps)} certified "
            "constant-field extension" + ("" if len(steps) == 1 else "s")
        )
    return AlgebraicConstantBaseChangeIntegrationResult(
        source_result,
        tuple(steps),
        final_detail,
    )
