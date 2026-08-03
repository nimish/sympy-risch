r"""Expression-level integration over conservatively compiled towers.

The compiler accepts rational arithmetic, rational powers, and nested
``log``, ``exp``, and ``tan`` calls without applying branch-changing
identities. A successful elementary result is therefore a local differential
presentation, not a claim about globally chosen principal branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy import diff
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from ._expression_compilation import (
    AlgebraicSourcePresentation,
    ExpressionCompilation,
    ExpressionCompilationOutcome,
    ExpressionUnsupportedReason,
    _embed_native_element,
    _local_expression_equal,
    compile_expression,
)
from ._typing import FieldElement
from .admission import is_standard_rational_base
from .algebraic.logarithmic import (
    AlgebraicIntegrationResult,
    AlgebraicIntegrationStatus,
    _integrate_algebraic_with_model,
)
from .algebraic_geometry.function_field import AlgebraicFunctionField
from .elementary import ElementaryIntegralResult, integrate_elementary
from .expression_discovery import (
    ExpressionDiscoveryOutcome,
    ExpressionGeneratorKind,
    ExpressionTowerCandidate,
    ExpressionTowerDiscovery,
    discover_expression_tower,
)
from .tower.core import AlgebraicExtension, DifferentialTower, MonomialExtension


class ExpressionIntegrationOutcome(Enum):
    """Outcome of elementary integration after expression compilation."""

    ELEMENTARY = "elementary"
    NON_ELEMENTARY = "non_elementary"
    UNSUPPORTED = "unsupported"


NativeIntegrationResult = ElementaryIntegralResult | AlgebraicIntegrationResult


def _native_outcome(
    native_result: NativeIntegrationResult,
) -> ExpressionIntegrationOutcome:
    """Map one validated native result to the public expression outcome."""
    if isinstance(native_result, ElementaryIntegralResult):
        return (
            ExpressionIntegrationOutcome.ELEMENTARY
            if native_result.is_elementary
            else ExpressionIntegrationOutcome.NON_ELEMENTARY
        )
    if native_result.status in {
        AlgebraicIntegrationStatus.EXACT,
        AlgebraicIntegrationStatus.ELEMENTARY,
    }:
        return ExpressionIntegrationOutcome.ELEMENTARY
    if native_result.status is AlgebraicIntegrationStatus.NON_ELEMENTARY:
        return ExpressionIntegrationOutcome.NON_ELEMENTARY
    return ExpressionIntegrationOutcome.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class ExpressionIntegralResult:
    r"""A public expression-level elementary-integration decision.

    ``ELEMENTARY`` carries a local back-substituted antiderivative and the
    complete native Risch certificate.  ``NON_ELEMENTARY`` carries the
    native result and its Liouville obstruction.  ``UNSUPPORTED`` is reserved
    for the compiler/admission/backend boundary and never means that the
    source expression has no elementary antiderivative.
    """

    compilation: ExpressionCompilation
    outcome: ExpressionIntegrationOutcome
    reason: ExpressionUnsupportedReason | None
    detail: str
    native_result: NativeIntegrationResult | None = None
    antiderivative: Expr | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.compilation, ExpressionCompilation):
            raise TypeError("an expression integral result needs a compilation")
        if not isinstance(self.outcome, ExpressionIntegrationOutcome):
            raise TypeError("an expression integral result needs a valid outcome")
        if self.reason is not None and not isinstance(
            self.reason,
            ExpressionUnsupportedReason,
        ):
            raise TypeError("an expression integration reason is invalid")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an expression integral result needs a nonempty detail")
        if self.native_result is not None and not isinstance(
            self.native_result,
            (ElementaryIntegralResult, AlgebraicIntegrationResult),
        ):
            raise TypeError("an expression integral result needs a native result")
        if self.antiderivative is not None and not isinstance(
            self.antiderivative, Expr
        ):
            raise TypeError("an antiderivative presentation must be a SymPy Expr")

        if self.outcome is ExpressionIntegrationOutcome.ELEMENTARY:
            if (
                not self.compilation.compiled
                or self.reason is not None
                or self.native_result is None
                or self.antiderivative is None
            ):
                raise ValueError("an elementary result needs a successful native proof")
            _validate_native_integration_result(self)
            return

        if self.outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY:
            if (
                not self.compilation.compiled
                or self.reason is not None
                or self.native_result is None
                or self.antiderivative is not None
            ):
                raise ValueError(
                    "a non-elementary result needs a native Liouville obstruction"
                )
            _validate_native_integration_result(self)
            return

        if self.reason is None:
            raise ValueError("an unsupported result needs a boundary reason")
        if self.antiderivative is not None:
            raise ValueError("an unsupported result cannot carry an antiderivative")
        if self.compilation.unsupported:
            if self.native_result is not None:
                raise ValueError(
                    "an unsupported compilation cannot carry a native result"
                )
            if self.reason is not self.compilation.reason:
                raise ValueError(
                    "an unsupported compilation must retain its boundary reason"
                )
            return
        if self.reason is not ExpressionUnsupportedReason.INTEGRATION_BACKEND:
            raise ValueError(
                "a compiled expression can be unsupported only by the integration backend"
            )
        if self.native_result is not None:
            _validate_native_integration_result(self)

    @property
    def is_elementary(self) -> bool:
        """Whether the native Risch dispatcher proved elementarity."""
        return self.outcome is ExpressionIntegrationOutcome.ELEMENTARY

    @property
    def is_non_elementary(self) -> bool:
        """Whether the native Risch dispatcher produced an obstruction."""
        return self.outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY

    @property
    def unsupported(self) -> bool:
        """Whether the compiler or backend deliberately declined the input."""
        return self.outcome is ExpressionIntegrationOutcome.UNSUPPORTED


def _validate_native_integration_result(result: ExpressionIntegralResult) -> None:
    """Tie a public integration outcome to its compiled native certificate."""
    compilation = result.compilation
    tower, element = compilation._require_compiled()
    native_result = result.native_result
    if native_result is None:
        raise ValueError("a native integration validation needs a native result")
    expected_outcome = _native_outcome(native_result)
    if result.outcome is not expected_outcome:
        raise ValueError("the public outcome does not match its native result")

    if isinstance(native_result, ElementaryIntegralResult):
        _validate_elementary_native_result(native_result, tower, element)
    else:
        extension = native_result.collection.basis.extension
        if tower.levels[tower.top] is not extension:
            raise ValueError("the algebraic result belongs to a different extension")
        if not native_result.collection.derivation.has_same_position(
            tower.derivation()
        ):
            raise ValueError("the algebraic result belongs to a different tower")
        if native_result.collection.integrand != _embed_native_element(tower, element):
            raise ValueError("the algebraic result has a different native integrand")
    if not native_result.reconstructs:
        raise ValueError("the native result does not reconstruct its integrand")
    if result.outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY and isinstance(
        native_result, ElementaryIntegralResult
    ):
        obstruction = native_result.obstruction
        if obstruction is None:
            raise ValueError("a non-elementary result needs a Liouville obstruction")
        if type(obstruction.level) is not int:
            raise ValueError("a Liouville obstruction needs an integer tower level")
        if not -1 <= obstruction.level <= tower.top:
            raise ValueError("the Liouville obstruction has an invalid tower level")
        if obstruction.level == -1:
            if obstruction.generator is not None or obstruction.family is not None:
                raise ValueError("a constant-field obstruction cannot name a generator")
        else:
            extension = tower.levels[obstruction.level]
            if not isinstance(extension, MonomialExtension):
                raise ValueError("a Liouville obstruction needs a monomial tower level")
            if (
                obstruction.generator != extension.generator
                or obstruction.family is not extension.family
            ):
                raise ValueError(
                    "the Liouville obstruction does not match its tower level"
                )
        remainder = native_result.unintegrated_remainder
        if remainder == tower.field().zero:
            raise ValueError("a Liouville obstruction needs a nonzero remainder")
        if _embed_native_element(tower, obstruction.remainder) != remainder:
            raise ValueError(
                "the Liouville obstruction remainder does not match the native result"
            )
        return
    if result.outcome is not ExpressionIntegrationOutcome.ELEMENTARY:
        return
    if result.antiderivative is None:
        raise ValueError("an elementary presentation is missing")
    native_antiderivative = native_result.antiderivative
    if native_antiderivative is None:
        raise ValueError("a successful native result omitted its antiderivative")
    expected_presentation = compilation.back_substitute_expression(
        native_antiderivative,
    )
    if result.antiderivative != expected_presentation:
        raise ValueError("the public presentation does not match the native proof")
    # A primitive algebraic presentation is already checked by exact source
    # images and native reconstruction.  Re-running ``cancel`` after replacing
    # its generator by a sum of radicals cannot use those defining relations
    # and can reject a valid presentation merely because it is not syntactically
    # rational.
    if not compilation.algebraic_presentations and not _local_expression_equal(
        diff(result.antiderivative, compilation.variable), compilation.expression
    ):
        raise ValueError(
            "the back-substituted antiderivative fails local differentiation"
        )


def _validate_elementary_native_result(
    native_result: ElementaryIntegralResult,
    tower: DifferentialTower,
    element: FieldElement,
) -> None:
    """Validate the exact tower position of a P/E/T result."""
    if not native_result.derivation.has_same_position(tower.derivation()):
        raise ValueError("the elementary result belongs to a different tower")
    if native_result.integrand != _embed_native_element(tower, element):
        raise ValueError("the elementary result has a different native integrand")


def _integrate_direct_algebraic(
    compilation: ExpressionCompilation,
) -> ExpressionIntegralResult | None:
    """Integrate the direct ``C(x) -> algebraic`` expression cell."""
    tower, element = compilation._require_compiled()
    if len(tower.levels) != 2:
        return None
    base, extension = tower.levels
    if not (
        isinstance(base, MonomialExtension)
        and is_standard_rational_base(base)
        and isinstance(extension, AlgebraicExtension)
    ):
        return None

    try:
        model = AlgebraicFunctionField(extension)
        native_result = _integrate_algebraic_with_model(
            element,
            tower.derivation(),
            model.order.basis,
            lazy=False,
            torsion_order_bound=None,
            model=model,
        )
    except NotImplementedError as exc:
        return ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.UNSUPPORTED,
            ExpressionUnsupportedReason.INTEGRATION_BACKEND,
            str(exc),
        )

    outcome = _native_outcome(native_result)
    if outcome is ExpressionIntegrationOutcome.ELEMENTARY:
        native_antiderivative = native_result.antiderivative
        if native_antiderivative is None:
            raise RuntimeError("a successful algebraic result omitted its presentation")
        return ExpressionIntegralResult(
            compilation,
            outcome,
            None,
            "the certified algebraic function-field dispatcher found a local "
            "elementary presentation",
            native_result,
            compilation.back_substitute_expression(native_antiderivative),
        )
    if outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY:
        return ExpressionIntegralResult(
            compilation,
            outcome,
            None,
            "the certified algebraic function-field dispatcher returned a "
            "Liouville obstruction",
            native_result,
        )
    return ExpressionIntegralResult(
        compilation,
        outcome,
        ExpressionUnsupportedReason.INTEGRATION_BACKEND,
        native_result.detail,
        native_result,
    )


def integrate_expression(
    expression: Expr,
    variable: Symbol,
) -> ExpressionIntegralResult:
    r"""Integrate a conservative SymPy expression through the Risch core.

    The elementary presentation is back-substituted only syntactically.  In
    particular, it is a local derivative identity and does not assert global
    principal-branch identities for logarithms or tangents.
    """
    compilation = compile_expression(expression, variable)
    if compilation.unsupported:
        if compilation.reason is None:
            raise RuntimeError("an unsupported compilation omitted its reason")
        return ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.UNSUPPORTED,
            compilation.reason,
            compilation.detail,
        )

    tower, element = compilation._require_compiled()
    try:
        elementary_result = integrate_elementary(element, tower.derivation())
    except NotImplementedError as exc:
        algebraic_result = _integrate_direct_algebraic(compilation)
        if algebraic_result is not None:
            return algebraic_result
        return ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.UNSUPPORTED,
            ExpressionUnsupportedReason.INTEGRATION_BACKEND,
            str(exc),
        )
    if not elementary_result.is_elementary:
        return ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.NON_ELEMENTARY,
            None,
            "the certified P/E/T dispatcher returned a Liouville obstruction",
            elementary_result,
        )
    native_antiderivative = elementary_result.antiderivative
    if native_antiderivative is None:
        raise RuntimeError("a successful elementary result omitted its presentation")
    return ExpressionIntegralResult(
        compilation,
        ExpressionIntegrationOutcome.ELEMENTARY,
        None,
        "the certified differential dispatcher found a local elementary presentation",
        elementary_result,
        compilation.back_substitute_expression(
            native_antiderivative,
        ),
    )


__all__ = [
    "AlgebraicSourcePresentation",
    "ExpressionCompilation",
    "ExpressionCompilationOutcome",
    "ExpressionDiscoveryOutcome",
    "ExpressionGeneratorKind",
    "ExpressionIntegralResult",
    "ExpressionIntegrationOutcome",
    "ExpressionTowerCandidate",
    "ExpressionTowerDiscovery",
    "ExpressionUnsupportedReason",
    "compile_expression",
    "discover_expression_tower",
    "integrate_expression",
]
