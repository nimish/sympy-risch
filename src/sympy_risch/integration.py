r"""Stable public decisions for expression-level Risch integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from ._expression_compilation import ExpressionUnsupportedReason
from .algebraic.liouville import (
    AlgebraicExactnessLiouvilleObstruction,
    AlgebraicLiouvilleObstruction,
)
from .algebraic.logarithmic import (
    AlgebraicExactnessNonElementary,
    AlgebraicResidueNonElementary,
)
from .elementary import ElementaryIntegralResult, IntegrationObstruction
from .expression import (
    ExpressionIntegralResult,
    ExpressionIntegrationOutcome,
    integrate_expression,
)


@dataclass(frozen=True, slots=True, init=False)
class RischIntegrationResult:
    """Common source certificate for one public Risch decision."""

    certificate: ExpressionIntegralResult

    def __init__(self) -> None:
        raise TypeError("Risch integration results are created by integrate_risch")

    @classmethod
    def _create(cls, certificate: ExpressionIntegralResult) -> Self:
        result = object.__new__(cls)
        object.__setattr__(result, "certificate", certificate)
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.certificate, ExpressionIntegralResult):
            raise TypeError("a Risch decision needs an expression certificate")

    @property
    def expression(self) -> Expr:
        """The source expression being integrated."""
        return self.certificate.compilation.expression

    @property
    def variable(self) -> Symbol:
        """The source integration variable."""
        return self.certificate.compilation.variable

    @property
    def detail(self) -> str:
        """A concise explanation of the decision."""
        return self.certificate.detail


@dataclass(frozen=True, slots=True, init=False)
class ElementaryPrimitive(RischIntegrationResult):
    """A locally valid elementary primitive with an exact native proof."""

    def _validate(self) -> None:
        RischIntegrationResult._validate(self)
        if self.certificate.outcome is not ExpressionIntegrationOutcome.ELEMENTARY:
            raise ValueError("an elementary primitive needs an elementary certificate")

    @property
    def primitive(self) -> Expr:
        """The local, back-substituted elementary antiderivative."""
        return cast(Expr, self.certificate.antiderivative)

    @property
    def reconstructs(self) -> bool:
        """Whether the contained native certificate reconstructs the integrand."""
        native = self.certificate.native_result
        return native is not None and native.reconstructs


@dataclass(frozen=True, slots=True, init=False)
class NonElementary(RischIntegrationResult):
    """A theorem-backed proof that no elementary primitive exists."""

    def _validate(self) -> None:
        RischIntegrationResult._validate(self)
        if self.certificate.outcome is not ExpressionIntegrationOutcome.NON_ELEMENTARY:
            raise ValueError("a non-elementary decision needs a Liouville obstruction")

    @property
    def obstruction(
        self,
    ) -> (
        IntegrationObstruction
        | AlgebraicLiouvilleObstruction
        | AlgebraicExactnessLiouvilleObstruction
    ):
        """The replayable native Liouville obstruction."""
        native = self.certificate.native_result
        if isinstance(
            native,
            (AlgebraicResidueNonElementary, AlgebraicExactnessNonElementary),
        ):
            return native.liouville_obstruction
        if (
            isinstance(native, ElementaryIntegralResult)
            and native.obstruction is not None
        ):
            return native.obstruction
        raise RuntimeError("a non-elementary certificate lost its obstruction")


@dataclass(frozen=True, slots=True, init=False)
class NotSupported(RischIntegrationResult):
    """An explicit grammar, admission, or backend capability boundary."""

    def _validate(self) -> None:
        RischIntegrationResult._validate(self)
        if self.certificate.outcome is not ExpressionIntegrationOutcome.UNSUPPORTED:
            raise ValueError("an unsupported decision needs a capability boundary")

    @property
    def reason(self) -> ExpressionUnsupportedReason:
        """The structured unsupported reason."""
        return cast(ExpressionUnsupportedReason, self.certificate.reason)


def integrate_risch(expression: Expr, variable: Symbol) -> RischIntegrationResult:
    r"""Decide elementary integration for the currently supported expression cell.

    Unsupported grammar, tower, and backend cases return :class:`NotSupported`;
    they are never converted into mathematical non-elementarity claims.
    """
    certificate = integrate_expression(expression, variable)
    match certificate.outcome:
        case ExpressionIntegrationOutcome.ELEMENTARY:
            return ElementaryPrimitive._create(certificate)
        case ExpressionIntegrationOutcome.NON_ELEMENTARY:
            return NonElementary._create(certificate)
        case ExpressionIntegrationOutcome.UNSUPPORTED:
            return NotSupported._create(certificate)
    raise RuntimeError("expression integration returned an unknown outcome")


__all__ = [
    "ElementaryPrimitive",
    "NonElementary",
    "NotSupported",
    "RischIntegrationResult",
    "integrate_risch",
]
