r"""Explicit algebraic integration after a compatible constant base change.

This module owns the orchestration boundary between direct algebraic
logarithmic reconstruction and the certified ``QQ -> QQ(alpha)`` base-change
layer.  Keeping it separate means the direct logarithmic kernel has no reason
to know about target towers or split places.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from ..algebraic_geometry.base_change import (
    AlgebraicConstantBaseChange,
    AlgebraicConstantBaseChangeStatus,
    build_algebraic_constant_base_change,
)
from ..tower import Derivation
from .basis import IntegralBasis
from .logarithmic import (
    AlgebraicIntegrationResult,
    AlgebraicIntegrationStatus,
    integrate_algebraic,
)
from .residue_profile import (
    AlgebraicResidueProfileStatus,
    build_algebraic_residue_profile,
)


@dataclass(frozen=True, slots=True)
class AlgebraicConstantBaseChangeIntegrationResult:
    r"""An algebraic integration attempt continued in ``C'(x, y)``.

    ``source_result`` records the original differential over ``C(x, y)``.
    When its residues require a compatible finite constant extension,
    ``base_change`` records the certified map ``C(x, y) -> C'(x, y)`` and
    ``target_result`` is a fresh all-place integration result for the embedded
    *original* differential. Thus a successful ``antiderivative`` belongs to
    the target field; it is not silently presented as an expression in the
    source field.
    """

    source_result: AlgebraicIntegrationResult
    base_change: AlgebraicConstantBaseChange | None
    target_result: AlgebraicIntegrationResult | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_result, AlgebraicIntegrationResult):
            raise TypeError("a base-change integration needs a source result")
        if self.base_change is not None and not isinstance(
            self.base_change,
            AlgebraicConstantBaseChange,
        ):
            raise TypeError("a base-change integration needs a base change")
        if self.target_result is not None and not isinstance(
            self.target_result,
            AlgebraicIntegrationResult,
        ):
            raise TypeError("a base-change integration needs a target result")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a base-change integration needs a nonempty detail")

        change = self.base_change
        target = self.target_result
        if change is None:
            if target is not None:
                raise ValueError("a target result needs its certified base change")
            return

        source_model = self.source_result.collection.model
        if source_model is None or change.source_model is not source_model:
            raise ValueError("the base change belongs to another source model")
        if not change.source_derivation.has_same_position(
            self.source_result.collection.derivation,
        ):
            raise ValueError("the base change belongs to another source derivation")

        if change.status is AlgebraicConstantBaseChangeStatus.UNSUPPORTED:
            if target is not None:
                raise ValueError("an unsupported base change cannot carry a target")
            return
        if not change.certified:
            raise ValueError("a base change has an unknown status")
        if target is None:
            raise ValueError("a certified base change needs a target result")
        if (
            self.source_result.status
            is not AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
        ):
            raise ValueError("only a source base-change boundary may be continued")

        target_extension = change.target_extension
        target_derivation = change.target_derivation
        if target_extension is None or target_derivation is None:
            raise ValueError("a certified base change lacks its target field")
        if target.collection.basis.extension is not target_extension:
            raise ValueError("the target integration belongs to another field")
        if not target.collection.derivation.has_same_position(target_derivation):
            raise ValueError("the target integration belongs to another derivation")
        embedded_source = change.embed(self.source_result.collection.integrand)
        if target.collection.integrand - embedded_source != target_extension.field.zero:
            raise ValueError("the target integration did not use the embedded source")

    @property
    def status(self) -> AlgebraicIntegrationStatus:
        """Return the source or target integration outcome."""
        if self.target_result is not None:
            return self.target_result.status
        if self.base_change is not None:
            return AlgebraicIntegrationStatus.UNSUPPORTED
        return self.source_result.status

    @property
    def is_elementary(self) -> bool:
        """Whether the active source or target result is constructive."""
        if self.target_result is not None:
            return self.target_result.is_elementary
        return self.base_change is None and self.source_result.is_elementary

    @property
    def reconstructs(self) -> bool:
        """Whether the active decomposition reconstructs its input field."""
        if self.target_result is not None:
            return self.target_result.reconstructs
        return self.source_result.reconstructs

    @property
    def source_reconstructs(self) -> bool:
        """Whether the original-field reduction reconstructs its input."""
        return self.source_result.reconstructs

    @property
    def antiderivative(self) -> Expr | None:
        """Return the source or explicitly base-changed displayed answer."""
        if self.target_result is not None:
            return self.target_result.antiderivative
        if self.base_change is None:
            return self.source_result.antiderivative
        return None


def _automatic_constant_factor_candidates(
    source_result: AlgebraicIntegrationResult,
) -> tuple[tuple[Poly, ...], str]:
    """Return supported one-step constant-field candidates from a profile.

    A root of an irreducible profile factor is only a *candidate* for a
    compatible constant extension.  The certified base-change builder still
    checks that its root field contains every conjugate needed by the selected
    factor, and the target integration is recollected independently below.
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
    if polynomial is None or polynomial.domain != QQ:
        return (
            (),
            "automatic residue-profile selection currently starts only from "
            "a profile over QQ",
        )
    try:
        _content, factors = polynomial.factor_list()
    except (NotImplementedError, TypeError, ValueError) as exc:
        return (), "the residue profile could not be factored over QQ: " + str(exc)

    candidates = tuple(
        sorted(
            (
                factor.monic()
                for factor, _multiplicity in factors
                if factor.degree() >= 2
            ),
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
            "the residue profile has no non-rational irreducible factor for "
            "a one-step constant extension",
        )
    return (
        candidates,
        "selected compatible constant-field candidates from the square-free "
        "all-place residue profile",
    )


def _continue_source_result(
    source_result: AlgebraicIntegrationResult,
    factor: Poly,
    *,
    max_x_degree: int,
    lazy: bool,
) -> AlgebraicConstantBaseChangeIntegrationResult:
    """Reintegrate one source boundary over a caller- or profile-chosen factor."""
    source_model = source_result.collection.model
    if source_model is None:
        raise ValueError("a base-change boundary lacks its source function field")
    change = build_algebraic_constant_base_change(
        source_model,
        source_result.collection.derivation,
        factor,
    )
    if not change.certified:
        return AlgebraicConstantBaseChangeIntegrationResult(
            source_result,
            change,
            None,
            "the selected algebraic constant base change is unsupported: "
            + change.detail,
        )

    target_derivation = change.target_derivation
    target_model = change.target_model
    if target_derivation is None or target_model is None:
        raise ValueError("a certified base change lacks its target integration data")
    target_result = integrate_algebraic(
        change.embed(source_result.collection.integrand),
        target_derivation,
        target_model.order.basis,
        max_x_degree=max_x_degree,
        lazy=lazy,
    )
    return AlgebraicConstantBaseChangeIntegrationResult(
        source_result,
        change,
        target_result,
        "the source differential was integrated again after certified algebraic "
        "constant base change",
    )


def integrate_algebraic_with_constant_base_change(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    factor: Poly | None = None,
    *,
    max_x_degree: int = 1,
    lazy: bool = False,
) -> AlgebraicConstantBaseChangeIntegrationResult:
    r"""Continue a compatible algebraic integral in a selected constant field.

    The source is first integrated through the displayed-constant logarithmic
    slice. If and only if that ends at
    :attr:`AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE`, an explicit
    ``factor`` is used to build the currently supported ``QQ -> QQ(alpha)``
    extension. When ``factor`` is omitted, the square-free all-place residue
    profile supplies deterministic irreducible candidates over ``QQ``. The
    first certified target that no longer needs another constant base change
    is retained. The embedded original differential is then collected again
    at every target place before logarithmic reconstruction. This is
    essential: merely transporting source finite residues would omit target
    infinity and split descendants.

    Automatic selection is a one-step planner, not general constant descent:
    it only considers profile factors whose root fields meet the current
    certified splitting condition. If no such target is found, the source
    ``NEEDS_CONSTANT_BASE_CHANGE`` boundary remains explicit.
    """
    source_result = integrate_algebraic(
        integrand,
        derivation,
        basis,
        max_x_degree=max_x_degree,
        lazy=lazy,
    )
    if (
        source_result.status
        is not AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
    ):
        return AlgebraicConstantBaseChangeIntegrationResult(
            source_result,
            None,
            None,
            "the source integration did not require algebraic constant base change",
        )

    if factor is not None:
        return _continue_source_result(
            source_result,
            factor,
            max_x_degree=max_x_degree,
            lazy=lazy,
        )

    candidates, detail = _automatic_constant_factor_candidates(source_result)
    fallback: AlgebraicConstantBaseChangeIntegrationResult | None = None
    for candidate in candidates:
        continued = _continue_source_result(
            source_result,
            candidate,
            max_x_degree=max_x_degree,
            lazy=lazy,
        )
        if continued.base_change is not None and continued.base_change.certified:
            if fallback is None:
                fallback = continued
            if continued.status in {
                AlgebraicIntegrationStatus.EXACT,
                AlgebraicIntegrationStatus.ELEMENTARY,
                AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER,
                AlgebraicIntegrationStatus.INCOMPLETE,
            }:
                return continued

    if fallback is not None:
        return AlgebraicConstantBaseChangeIntegrationResult(
            fallback.source_result,
            fallback.base_change,
            fallback.target_result,
            "no supported one-step profile factor resolved the source "
            "constant-base-change boundary",
        )
    return AlgebraicConstantBaseChangeIntegrationResult(
        source_result,
        None,
        None,
        detail,
    )
