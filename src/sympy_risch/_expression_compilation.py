r"""Compile SymPy expressions into certified differential towers.

The elementary dispatcher deliberately works with an explicit differential
tower.  This module supplies a small, sound discovery boundary for the useful
syntactic fragment generated from rational arithmetic, rational powers, and
``log``, ``exp``, and ``tan``.  It does *not* apply identities such as
``exp(log(a)) = a`` or ``log(a*b) = log(a) + log(b)``: doing so would silently
choose principal branches and can change constants.  Instead each accepted
function call receives a fresh formal generator and its derivative is checked
by the existing extension factories.

The resulting presentation is therefore a local differential identity after
back-substitution.  It is intentionally not a claim about a globally chosen
principal branch of a complex elementary function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, cast

from sympy import cancel
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import tan
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from ._expression_validation import (
    source_derivative_in_native,
    source_value_in_native,
    validate_compiled_extension_sources,
)
from ._typing import FieldElement, SupportsAsExpr
from .dependency import (
    TranscendentalExtensionAnalysis,
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
    certify_primitive_extension,
)
from .tower.core import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
)


class ExpressionCompilationOutcome(Enum):
    """Whether an expression was compiled into a certified native tower."""

    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"


class ExpressionUnsupportedReason(Enum):
    """Stable reason that the conservative expression boundary declined input."""

    EXTRA_SYMBOL = "extra_symbol"
    NON_RATIONAL_CONSTANT = "non_rational_constant"
    NONINTEGER_POWER = "noninteger_power"
    UNSUPPORTED_FUNCTION = "unsupported_function"
    BRANCH_SENSITIVE_FORM = "branch_sensitive_form"
    SINGULAR_FUNCTION_ARGUMENT = "singular_function_argument"
    CONSTANT_EXTENSION = "constant_extension"
    DEPENDENT_EXTENSION = "dependent_extension"
    UNCERTIFIED_EXTENSION = "uncertified_extension"
    INTEGRATION_BACKEND = "integration_backend"


@dataclass(frozen=True, slots=True)
class ExpressionCompilation:
    r"""A checked translation from a SymPy expression to one native tower.

    ``generator_substitutions`` maps the fresh native generator symbols to the
    exact source function calls from which they were introduced.  Its right
    sides never contain native generators, so simultaneous ``xreplace`` gives
    a direct, branch-preserving syntactic presentation.
    """

    expression: Expr
    variable: Symbol
    outcome: ExpressionCompilationOutcome
    reason: ExpressionUnsupportedReason | None
    detail: str
    tower: DifferentialTower | None = None
    element: FieldElement | None = None
    generator_substitutions: tuple[tuple[Symbol, Expr], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.expression, Expr):
            raise TypeError("an expression compilation needs a SymPy Expr")
        if not isinstance(self.variable, Symbol):
            raise TypeError("the integration variable must be a SymPy Symbol")
        if self.variable.is_commutative is not True:
            raise TypeError("the integration variable must be commutative")
        if self.variable.is_zero is True:
            raise ValueError("the integration variable cannot be known to be zero")
        if self.variable.is_finite is False:
            raise ValueError("the integration variable must not be known infinite")
        if not isinstance(self.outcome, ExpressionCompilationOutcome):
            raise TypeError("an expression compilation needs a valid outcome")
        if self.reason is not None and not isinstance(
            self.reason,
            ExpressionUnsupportedReason,
        ):
            raise TypeError("an expression compilation reason is invalid")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an expression compilation needs a nonempty detail")

        substitutions = tuple(self.generator_substitutions)
        object.__setattr__(self, "generator_substitutions", substitutions)
        generators: set[Symbol] = set()
        for generator, source in substitutions:
            if not isinstance(generator, Symbol) or not isinstance(source, Expr):
                raise TypeError("generator substitutions must map Symbols to Exprs")
            if generator in generators:
                raise ValueError(
                    "generator substitutions must have distinct generators"
                )
            generators.add(generator)

        if self.outcome is ExpressionCompilationOutcome.COMPILED:
            if self.reason is not None:
                raise ValueError(
                    "a compiled expression cannot have an unsupported reason"
                )
            if self.tower is None or self.element is None:
                raise ValueError("a compiled expression needs a tower and element")
            if not isinstance(self.tower, DifferentialTower):
                raise TypeError(
                    "a compiled expression tower must be a DifferentialTower"
                )
            if not isinstance(self.element, SupportsAsExpr):
                raise TypeError("a compiled expression element must expose as_expr()")
            _validate_compiled_expression(self)
            return

        if self.reason is None:
            raise ValueError("an unsupported expression needs a reason")
        if self.reason is ExpressionUnsupportedReason.INTEGRATION_BACKEND:
            raise ValueError(
                "an integration-backend reason belongs only to an integration result"
            )
        if self.tower is not None or self.element is not None or substitutions:
            raise ValueError(
                "an unsupported expression cannot carry a partial tower or rewrite"
            )

    @property
    def compiled(self) -> bool:
        """Whether this compilation carries a certified native tower."""
        return self.outcome is ExpressionCompilationOutcome.COMPILED

    @property
    def unsupported(self) -> bool:
        """Whether this is a boundary outcome rather than a mathematical result."""
        return self.outcome is ExpressionCompilationOutcome.UNSUPPORTED

    def native_expression(self) -> Expr:
        """Return the top-field expression in fresh native generator symbols."""
        tower, element = self._require_compiled()
        return tower.field().to_sympy(element)

    def back_substitute(self, element: FieldElement) -> Expr:
        r"""Present a native top-field element through the source function calls.

        The returned expression is a local differential presentation.  It is
        not a simplification across logarithm or inverse-trigonometric branch
        cuts.
        """
        tower, _source_element = self._require_compiled()
        native = tower.field().to_sympy(_embed_native_element(tower, element))
        return self.back_substitute_expression(native)

    def back_substitute_expression(self, expression: Expr) -> Expr:
        """Back-substitute native generators in a presentation expression.

        Elementary logarithmic presentations can contain algebraic constants
        (for example, ``I`` from a complex partial fraction) which do not
        belong to the native differential field.  They still need only a
        syntactic generator substitution, so this method deliberately does
        not try to coerce them through the field backend.
        """
        self._require_compiled()
        if not isinstance(expression, Expr):
            raise TypeError("a presentation substitution requires a SymPy Expr")
        return expression.xreplace(dict(self.generator_substitutions))

    def reconstructed_expression(self) -> Expr:
        """Return the input reconstructed from its native top-field element."""
        _tower, element = self._require_compiled()
        return self.back_substitute(element)

    def _require_compiled(self) -> tuple[DifferentialTower, FieldElement]:
        """Return required state after enforcing the compiled-result invariant."""
        if self.tower is None or self.element is None:
            raise ValueError("this operation requires a compiled expression")
        return self.tower, self.element


class _UnsupportedExpression(Exception):
    """Internal short-circuit carrying one public boundary explanation."""

    def __init__(self, reason: ExpressionUnsupportedReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


_BRANCH_SENSITIVE_FUNCTION_NAMES = frozenset(
    {
        "Abs",
        "Heaviside",
        "Max",
        "Min",
        "Piecewise",
        "arg",
        "ceiling",
        "conjugate",
        "floor",
        "im",
        "re",
        "sign",
    }
)


def _embed_native_element(
    tower: DifferentialTower,
    element: FieldElement,
) -> FieldElement:
    """Embed a prior flat fraction-field element at ``tower``'s current top.

    ``Domain.convert`` sometimes preserves the old ``FracElement`` wrapper
    when two flat SymPy fraction fields share generators.  Reconstructing
    through its exact expression forces the target native field and prevents
    sibling function calls from being combined across domain instances.
    """
    if not isinstance(element, SupportsAsExpr):
        raise TypeError("a native expression element must expose as_expr()")
    return cast(FieldElement, tower.field().from_sympy(element.as_expr()))


def _local_expression_equal(left: Expr, right: Expr) -> bool:
    """Decide equality in the accepted rational expression grammar.

    Function calls remain opaque to ``cancel``.  This is deliberate: the
    comparison cannot accidentally use a branch-changing logarithmic or
    exponential identity.
    """
    try:
        return cancel(left - right) == 0
    except (NotImplementedError, TypeError, ValueError):
        return False


def _unary_function_argument(expression: Expr) -> Expr:
    """Return the sole argument of a supported unary source function."""
    if len(expression.args) != 1:
        raise ValueError("a supported source function must be unary")
    argument = expression.args[0]
    if not isinstance(argument, Expr):
        raise ValueError("a supported source function argument must be an Expr")
    return argument


def _validate_compiled_expression(compilation: ExpressionCompilation) -> None:
    """Validate public certificate invariants independently of the builder."""
    tower, element = compilation._require_compiled()
    substitutions = dict(compilation.generator_substitutions)
    validate_compiled_extension_sources(compilation, substitutions)
    normalized_element = _embed_native_element(tower, element)
    source_generators = {
        source: generator for generator, source in substitutions.items()
    }
    source_value = source_value_in_native(
        compilation.expression,
        compilation,
        source_generators,
    )
    if normalized_element != source_value:
        raise ValueError("the native element does not represent the source expression")
    if not _local_expression_equal(
        compilation.reconstructed_expression(),
        compilation.expression,
    ):
        raise ValueError("the native element fails source round-trip validation")
    if tower.derivation()(normalized_element) != source_derivative_in_native(
        compilation.expression,
        compilation,
        source_generators,
    ):
        raise ValueError(
            "the native derivative does not agree with the source-tree chain rule"
        )


class _ExpressionCompiler:
    """Mutable construction state hidden behind immutable public certificates."""

    def __init__(self, variable: Symbol) -> None:
        constants = DifferentialTower(QQ)
        base_analysis = certify_primitive_extension(
            Poly(1, variable, domain=QQ),
            constants.derivation(),
        )
        if base_analysis.status is not ConstantFieldStatus.PRESERVED:
            raise RuntimeError(
                "the standard rational QQ(x) base unexpectedly failed "
                "primitive constant-field certification"
            )

        self.variable = variable
        self.tower = constants.extend(base_analysis.extension)
        self.source_elements: dict[Expr, FieldElement] = {}
        self.generator_substitutions: dict[Symbol, Expr] = {}
        self.next_generator_index = 0

    def compile(self, expression: Expr) -> FieldElement:
        """Compile one expression, embedding prior results at the current top."""
        if expression in self.source_elements:
            return self._embed_current(self.source_elements[expression])
        if expression == self.variable:
            return self._from_sympy(expression)
        if expression.is_Rational is True:
            return self._from_sympy(expression)
        if expression.func in {log, exp, tan}:
            return self._compile_function(expression)
        if expression.is_number:
            self._unsupported(
                ExpressionUnsupportedReason.NON_RATIONAL_CONSTANT,
                f"{expression} is not a rational constant in QQ",
            )
        if isinstance(expression, Add):
            value = self.tower.field().zero
            for term in expression.args:
                term_value = self.compile(term)
                value = self._embed_current(value) + self._embed_current(term_value)
            return self._embed_current(value)
        if isinstance(expression, Mul):
            value = self.tower.field().one
            for factor in expression.args:
                factor_value = self.compile(factor)
                value = self._embed_current(value) * self._embed_current(factor_value)
            return self._embed_current(value)
        if isinstance(expression, Pow):
            if isinstance(expression.exp, Integer):
                return self._embed_current(
                    self.compile(expression.base) ** int(expression.exp)
                )
            if isinstance(expression.exp, Rational):
                return self._compile_algebraic_power(expression)
            self._unsupported(
                ExpressionUnsupportedReason.NONINTEGER_POWER,
                f"{expression} has a non-rational power outside the "
                "algebraic-power frontend",
            )
        if expression.func.__name__ in _BRANCH_SENSITIVE_FUNCTION_NAMES:
            self._unsupported(
                ExpressionUnsupportedReason.BRANCH_SENSITIVE_FORM,
                f"{expression.func.__name__} is branch- or piecewise-sensitive "
                "and is outside the local P/E/T grammar",
            )
        self._unsupported(
            ExpressionUnsupportedReason.UNSUPPORTED_FUNCTION,
            f"{expression.func.__name__} is outside rational arithmetic and "
            "the supported log/exp/tan grammar",
        )

    def _compile_function(self, expression: Expr) -> FieldElement:
        """Adjoin one syntactic log, exponential, or tangent generator."""
        if len(expression.args) != 1:
            self._unsupported(
                ExpressionUnsupportedReason.UNSUPPORTED_FUNCTION,
                f"{expression.func.__name__} needs exactly one argument",
            )
        argument = self._embed_current(
            self.compile(_unary_function_argument(expression))
        )
        lower_derivation = self.tower.derivation()
        if expression.func is log and argument == self.tower.field().zero:
            self._unsupported(
                ExpressionUnsupportedReason.SINGULAR_FUNCTION_ARGUMENT,
                "log(0) is undefined and has no differential-field generator",
            )
        argument_derivative = lower_derivation(argument)
        if expression.func is log:
            eta = argument_derivative / argument
        else:
            eta = argument_derivative
        if eta == self.tower.field().zero:
            self._unsupported(
                ExpressionUnsupportedReason.CONSTANT_EXTENSION,
                f"{expression} has zero derivative and would introduce a "
                "constant outside QQ",
            )

        generator = self._fresh_generator(expression.func.__name__)
        parent_field = self.tower.field()
        eta_expression = parent_field.to_sympy(eta)
        if expression.func is log:
            analysis = certify_primitive_extension(
                Poly(eta_expression, generator, domain=parent_field),
                lower_derivation,
            )
        elif expression.func is exp:
            analysis = certify_hyperexponential_extension(
                Poly(eta_expression * generator, generator, domain=parent_field),
                lower_derivation,
            )
        else:
            analysis = certify_hypertangent_extension(
                Poly(
                    eta_expression * (1 + generator**2),
                    generator,
                    domain=parent_field,
                ),
                lower_derivation,
            )

        if analysis.status is ConstantFieldStatus.ENLARGED:
            if expression.func is exp:
                dependency = cast(
                    TranscendentalExtensionAnalysis,
                    analysis,
                ).dependency
                if (
                    dependency is not None
                    and dependency.is_algebraically_dependent
                    and dependency.minimal_power is not None
                ):
                    return self._compile_algebraic_exponential(
                        expression,
                        dependency.minimal_power,
                    )
            self._unsupported(
                ExpressionUnsupportedReason.DEPENDENT_EXTENSION,
                f"{expression} is dependent over the lower tower: "
                f"{analysis.detail or 'the certified relation introduces a new constant'}",
            )
        if analysis.status is not ConstantFieldStatus.PRESERVED:
            self._unsupported(
                ExpressionUnsupportedReason.UNCERTIFIED_EXTENSION,
                f"the existing structure theorem could not certify constants "
                f"for {expression}: {analysis.detail or 'unsupported lower tower'}",
            )

        self.tower = self.tower.extend(analysis.extension)
        element = self._from_sympy(generator)
        self.source_elements[expression] = element
        self.generator_substitutions[generator] = expression
        return element

    def _compile_algebraic_exponential(
        self,
        expression: Expr,
        relation_degree: int,
    ) -> FieldElement:
        r"""Represent a certified finite-power exponential relation algebraically."""
        argument = _unary_function_argument(expression)
        powered_source = exp(relation_degree * argument)
        native_powered_source = powered_source.xreplace(
            {
                source: generator
                for generator, source in self.generator_substitutions.items()
            }
        )
        parent_field = self.tower.field()
        try:
            relation_value = parent_field.from_sympy(native_powered_source)
        except (TypeError, ValueError) as exc:
            self._unsupported(
                ExpressionUnsupportedReason.UNCERTIFIED_EXTENSION,
                "the finite-power exponential relation was certified, but its "
                f"source normalization is not in the lower field: {exc}",
            )

        generator = self._fresh_generator("algebraic_exp")
        modulus = Poly(
            generator**relation_degree - parent_field.to_sympy(relation_value),
            generator,
            domain=parent_field,
        )
        try:
            extension = AlgebraicExtension(modulus)
        except (NotImplementedError, ValueError) as exc:
            self._unsupported(
                ExpressionUnsupportedReason.UNCERTIFIED_EXTENSION,
                "the certified finite-power exponential relation could not be "
                f"normalized as an algebraic extension: {exc}",
            )
        self.tower = self.tower.extend(extension)
        element = self._from_sympy(generator)
        self.source_elements[expression] = element
        self.generator_substitutions[generator] = expression
        return element

    def _compile_algebraic_power(self, expression: Pow) -> FieldElement:
        """Adjoin the local branch selected by a nonintegral rational power."""
        exponent = expression.exp
        if not isinstance(exponent, Rational) or isinstance(exponent, Integer):
            raise TypeError("algebraic power compilation needs a rational exponent")
        base = self._embed_current(self.compile(expression.base))
        if base == self.tower.field().zero and exponent.p < 0:
            self._unsupported(
                ExpressionUnsupportedReason.SINGULAR_FUNCTION_ARGUMENT,
                f"{expression} divides by a zero algebraic base",
            )

        generator = self._fresh_generator("algebraic")
        parent_field = self.tower.field()
        modulus = Poly(
            generator ** int(exponent.q)
            - parent_field.to_sympy(base ** int(exponent.p)),
            generator,
            domain=parent_field,
        )
        try:
            extension = AlgebraicExtension(modulus)
        except (NotImplementedError, ValueError) as exc:
            self._unsupported(
                ExpressionUnsupportedReason.UNCERTIFIED_EXTENSION,
                f"the algebraic relation proposed by {expression} could not "
                f"be normalized over the lower tower: {exc}",
            )
        self.tower = self.tower.extend(extension)
        element = self._from_sympy(generator)
        self.source_elements[expression] = element
        self.generator_substitutions[generator] = expression
        return element

    def _from_sympy(self, expression: Expr) -> FieldElement:
        """Coerce one permitted SymPy expression to the current native field."""
        return cast(FieldElement, self.tower.field().from_sympy(expression))

    def _embed_current(self, element: FieldElement) -> FieldElement:
        """Embed a prior-level native element after a sibling adjoined a level."""
        return _embed_native_element(self.tower, element)

    def _fresh_generator(self, family: str) -> Dummy:
        """Return a fresh symbol that cannot collide with source identifiers."""
        generator = Dummy(f"_risch_{family}_{self.next_generator_index}")
        self.next_generator_index += 1
        return generator

    def _unsupported(
        self,
        reason: ExpressionUnsupportedReason,
        detail: str,
    ) -> NoReturn:
        """Raise an internal structured boundary outcome."""
        raise _UnsupportedExpression(reason, detail)


def _unsupported_compilation(
    expression: Expr,
    variable: Symbol,
    reason: ExpressionUnsupportedReason,
    detail: str,
) -> ExpressionCompilation:
    """Build an invariant-preserving unsupported compilation result."""
    return ExpressionCompilation(
        expression,
        variable,
        ExpressionCompilationOutcome.UNSUPPORTED,
        reason,
        detail,
    )


def compile_expression(
    expression: Expr,
    variable: Symbol,
) -> ExpressionCompilation:
    r"""Compile a conservative SymPy expression into a native tower.

    Accepted expressions use only the supplied symbol, rational constants,
    addition, multiplication, rational powers, and nested unary ``log``,
    ``exp``, and ``tan`` calls. Rational powers propose finite algebraic
    extensions; each transcendental extension is independently certified by
    the existing primitive, hyperexponential, or hypertangent factory. No
    unsupported input is converted into a negative mathematical integration
    claim.
    """
    if not isinstance(expression, Expr):
        raise TypeError("expression compilation requires a SymPy Expr")
    if not isinstance(variable, Symbol):
        raise TypeError("the integration variable must be a SymPy Symbol")
    if variable.is_commutative is not True:
        raise TypeError("the integration variable must be commutative")
    if variable.is_zero is True:
        raise ValueError("the integration variable cannot be known to be zero")
    if variable.is_finite is False:
        raise ValueError("the integration variable must not be known infinite")
    foreign_symbols = expression.free_symbols - {variable}
    if foreign_symbols:
        names = ", ".join(str(symbol) for symbol in sorted(foreign_symbols, key=str))
        return _unsupported_compilation(
            expression,
            variable,
            ExpressionUnsupportedReason.EXTRA_SYMBOL,
            f"the expression contains symbols other than {variable}: {names}",
        )

    compiler = _ExpressionCompiler(variable)
    try:
        element = compiler.compile(expression)
    except _UnsupportedExpression as failure:
        return _unsupported_compilation(
            expression,
            variable,
            failure.reason,
            failure.detail,
        )

    return ExpressionCompilation(
        expression,
        variable,
        ExpressionCompilationOutcome.COMPILED,
        None,
        "the expression was compiled into a certified differential tower",
        compiler.tower,
        compiler._embed_current(element),
        tuple(compiler.generator_substitutions.items()),
    )


__all__ = [
    "ExpressionCompilation",
    "ExpressionCompilationOutcome",
    "ExpressionUnsupportedReason",
    "compile_expression",
]
