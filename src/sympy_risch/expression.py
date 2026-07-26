r"""Expression-level integration over conservatively compiled P/E/T towers.

The compiler accepts rational arithmetic, integer powers, and nested ``log``,
``exp``, and ``tan`` calls without applying branch-changing identities.  A
successful elementary result is therefore a local differential presentation,
not a claim about globally chosen principal branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy import diff
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from ._expression_compilation import (
    ExpressionCompilation,
    ExpressionCompilationOutcome,
    ExpressionUnsupportedReason,
    _embed_native_element,
    _local_expression_equal,
    compile_expression,
)
from .elementary import ElementaryIntegralResult, integrate_elementary
from .tower.core import MonomialExtension


class ExpressionIntegrationOutcome(Enum):
    """Outcome of elementary integration after expression compilation."""

    ELEMENTARY = "elementary"
    NON_ELEMENTARY = "non_elementary"
    UNSUPPORTED = "unsupported"


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
    elementary_result: ElementaryIntegralResult | None = None
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
        if self.elementary_result is not None and not isinstance(
            self.elementary_result,
            ElementaryIntegralResult,
        ):
            raise TypeError("an expression integral result needs an elementary result")
        if self.antiderivative is not None and not isinstance(
            self.antiderivative, Expr
        ):
            raise TypeError("an antiderivative presentation must be a SymPy Expr")

        if self.outcome is ExpressionIntegrationOutcome.ELEMENTARY:
            if (
                not self.compilation.compiled
                or self.reason is not None
                or self.elementary_result is None
                or not self.elementary_result.is_elementary
                or self.antiderivative is None
            ):
                raise ValueError("an elementary result needs a successful native proof")
            _validate_native_integration_result(self)
            return

        if self.outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY:
            if (
                not self.compilation.compiled
                or self.reason is not None
                or self.elementary_result is None
                or self.elementary_result.is_elementary
                or self.antiderivative is not None
            ):
                raise ValueError(
                    "a non-elementary result needs a native Liouville obstruction"
                )
            _validate_native_integration_result(self)
            return

        if self.reason is None or self.elementary_result is not None:
            raise ValueError("an unsupported result needs only a boundary reason")
        if self.antiderivative is not None:
            raise ValueError("an unsupported result cannot carry an antiderivative")
        if self.compilation.unsupported:
            if self.reason is not self.compilation.reason:
                raise ValueError(
                    "an unsupported compilation must retain its boundary reason"
                )
            return
        if self.reason is not ExpressionUnsupportedReason.INTEGRATION_BACKEND:
            raise ValueError(
                "a compiled expression can be unsupported only by the integration backend"
            )

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
    native_result = result.elementary_result
    if native_result is None:
        raise ValueError("a native integration validation needs an elementary result")
    if not native_result.derivation.has_same_position(tower.derivation()):
        raise ValueError("the elementary result belongs to a different tower")
    if native_result.integrand != _embed_native_element(tower, element):
        raise ValueError("the elementary result has a different native integrand")
    if not native_result.reconstructs:
        raise ValueError("the elementary result does not reconstruct its integrand")
    if result.outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY:
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
    if not _local_expression_equal(
        diff(result.antiderivative, compilation.variable),
        compilation.expression,
    ):
        raise ValueError(
            "the back-substituted antiderivative fails local differentiation"
        )


def integrate_expression(
    expression: Expr,
    variable: Symbol,
) -> ExpressionIntegralResult:
    r"""Integrate a conservative SymPy P/E/T expression through the Risch core.

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
        "the certified P/E/T dispatcher found a local elementary presentation",
        elementary_result,
        compilation.back_substitute_expression(
            native_antiderivative,
        ),
    )


__all__ = [
    "ExpressionCompilation",
    "ExpressionCompilationOutcome",
    "ExpressionIntegralResult",
    "ExpressionIntegrationOutcome",
    "ExpressionUnsupportedReason",
    "compile_expression",
    "integrate_expression",
]
