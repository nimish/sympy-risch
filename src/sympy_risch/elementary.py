r"""Recursive elementary integration over an explicitly supplied tower.

The reducers in :mod:`sympy_risch.monomial` implement one tower level at a
time.  This module composes their common invariant

.. math::

    f = D(v) + \ell + r,

where ``v`` belongs to the current field, ``ell`` is a logarithmic
derivative, and ``r`` either belongs to the next lower field or carries a
Liouville obstruction.  Recursion stops at the standard rational base
``QQ(x)``.

This is deliberately a dispatcher for a *supplied* differential tower.  It
does not infer generators from an arbitrary SymPy expression, and it rejects
tower shapes for which the required lower-field Risch equations have not yet
been implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import log
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import RootSum

from ._typing import FieldElement, is_field_zero
from .hyperbolic import (
    HyperbolicMobiusNormalForm,
    build_hyperbolic_mobius_normal_form,
)
from .monomial import (
    BronsteinResidueReduction,
    HyperexponentialMonomialReduction,
    HypertangentMonomialReduction,
    NonlinearNoSpecialReduction,
    NonlinearNoSpecialStatus,
    PrimitiveMonomialReduction,
    ReducedFunctionReduction,
    reduce_hyperexponential_monomial,
    reduce_hypertangent_monomial,
    reduce_nonlinear_no_special,
    reduce_primitive_monomial,
)
from .rational import hermite_reduce_rational, integrate_rational
from .tower import (
    AlgebraicExtension,
    Derivation,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
    _coerce,
    _domain_symbols,
)

MonomialLevelCertificate = (
    PrimitiveMonomialReduction
    | HyperexponentialMonomialReduction
    | HypertangentMonomialReduction
    | NonlinearNoSpecialReduction
)


class IntegrationObstructionKind(Enum):
    """Mathematical reason an elementary integral cannot exist."""

    CONSTANT_FIELD = "constant_field"
    NONCONSTANT_RESIDUE = "nonconstant_residue"
    INFINITY_RESIDUE = "infinity_residue"
    REDUCED_POLYNOMIAL = "reduced_polynomial"
    UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION = "unsolvable_risch_differential_equation"


@dataclass(frozen=True, slots=True)
class IntegrationObstruction:
    r"""A terminal Liouville obstruction at one exact tower level."""

    level: int
    generator: Symbol | None
    family: MonomialFamily | None
    kinds: tuple[IntegrationObstructionKind, ...]
    remainder: FieldElement
    certificate: MonomialLevelCertificate | None

    def __post_init__(self) -> None:
        if not self.kinds:
            raise ValueError("an integration obstruction needs at least one kind")


@dataclass(frozen=True, slots=True)
class LogarithmicTerm:
    r"""One encoded logarithmic antiderivative and its known derivative.

    ``antiderivative`` may contain :class:`~sympy.polys.rootoftools.RootSum`.
    Its ``derivative`` is retained as a native tower-field element because
    ordinary SymPy differentiation treats formal tower generators as
    independent symbols and therefore cannot reconstruct the tower
    derivation by itself.
    """

    antiderivative: Expr
    derivative: FieldElement
    level: int
    certificate: (
        RationalElementaryReduction
        | tuple[BronsteinResidueReduction, ReducedFunctionReduction | None]
    )


@dataclass(frozen=True, slots=True)
class RationalElementaryReduction:
    """Structured rational integration certificate at the base level."""

    exact_part: Expr
    logarithmic_antiderivative: Expr
    logarithmic_derivative: Expr


@dataclass(frozen=True, slots=True)
class _LevelReduction:
    """Normalized internal view of one family-specific reduction."""

    certificate: MonomialLevelCertificate
    exact_part: FieldElement
    logarithmic_derivative: FieldElement
    logarithmic_terms: tuple[LogarithmicTerm, ...]
    remainder: FieldElement
    lower_remainder: FieldElement | None
    obstruction: IntegrationObstruction | None


@dataclass(frozen=True, slots=True)
class ElementaryIntegralResult:
    r"""Recursive result of elementary integration in a supplied tower.

    The local fields satisfy

    ``integrand = D(exact_part) + logarithmic_derivative + remainder``.

    If ``lower_result`` is present, ``remainder`` is precisely its integrand
    embedded into the current field.  ``obstruction`` is propagated unchanged
    from the terminal level, making the failing generator directly visible at
    the top-level call.
    """

    derivation: Derivation
    integrand: FieldElement
    exact_part: FieldElement
    logarithmic_terms: tuple[LogarithmicTerm, ...]
    logarithmic_derivative: FieldElement
    remainder: FieldElement
    lower_result: ElementaryIntegralResult | None
    obstruction: IntegrationObstruction | None
    certificate: (
        RationalElementaryReduction
        | MonomialLevelCertificate
        | HyperbolicMobiusElementaryCertificate
        | None
    )

    @property
    def is_elementary(self) -> bool:
        """Whether every remainder was integrated down to the rational base."""
        return self.obstruction is None

    @property
    def total_exact_part(self) -> FieldElement:
        """Return exact parts from every recursive level in the top field."""
        total = self.exact_part
        if self.lower_result is not None:
            total += _coerce(
                self.derivation.field,
                self.lower_result.total_exact_part,
            )
        return total

    @property
    def all_logarithmic_terms(self) -> tuple[LogarithmicTerm, ...]:
        """Return logarithmic terms in top-to-bottom tower order."""
        if self.lower_result is None:
            return self.logarithmic_terms
        return self.logarithmic_terms + self.lower_result.all_logarithmic_terms

    @property
    def total_logarithmic_derivative(self) -> FieldElement:
        """Return logarithmic derivatives from every level in the top field."""
        total = self.logarithmic_derivative
        if self.lower_result is not None:
            total += _coerce(
                self.derivation.field,
                self.lower_result.total_logarithmic_derivative,
            )
        return total

    @property
    def unintegrated_remainder(self) -> FieldElement:
        """Return zero on success, or the terminal obstruction in the top field."""
        if self.lower_result is None:
            return self.remainder
        return _coerce(
            self.derivation.field,
            self.lower_result.unintegrated_remainder,
        )

    @property
    def reconstructs(self) -> bool:
        """Whether the flattened recursive result reconstructs the input."""
        return (
            self.derivation(self.total_exact_part)
            + self.total_logarithmic_derivative
            + self.unintegrated_remainder
            == self.integrand
        )

    @property
    def antiderivative(self) -> Expr | None:
        r"""Return an encoded elementary antiderivative, when one exists.

        To differentiate this presentation with ordinary SymPy ``diff``,
        callers must replace formal generators by the functions they
        represent.
        """
        if not self.is_elementary:
            return None
        exact_expression = self.derivation.field.to_sympy(self.total_exact_part)
        return exact_expression + Add(
            *(term.antiderivative for term in self.all_logarithmic_terms)
        )


@dataclass(frozen=True, slots=True)
class HyperbolicMobiusElementaryCertificate:
    r"""Record an elementary decision pulled back through a Möbius normal form.

    The target decision is a complete hyperexponential decision in the
    isomorphic coordinate ``u = (1 + t)/(1 - t)``.  Keeping it intact makes
    the transformed RDE and residue certificates inspectable without falsely
    presenting them as reductions performed in the source coordinate.
    """

    normal_form: HyperbolicMobiusNormalForm
    target_result: ElementaryIntegralResult


def _is_standard_rational_level(extension: MonomialExtension) -> bool:
    """Whether ``extension`` is the standard base ``QQ(x)``, ``Dx = 1``."""
    return extension.parent_field == QQ and extension.generator_derivative == Poly(
        1, extension.generator, domain=QQ
    )


def _root_sum_logarithm(logand: Poly, residue_factor: Poly) -> Expr:
    r"""Encode ``sum(q(alpha)=0, alpha*log(logand(alpha)))``."""
    residue = residue_factor.gens[0]
    return RootSum(
        residue_factor,
        Lambda(residue, residue * log(logand.as_expr())),
        quadratic=True,
    )


def _logarithmic_expression(
    residue_reduction: BronsteinResidueReduction,
    reduced_function_reduction: ReducedFunctionReduction | None = None,
) -> Expr:
    """Construct the encoded logarithms recorded by one monomial reduction."""
    terms: list[Expr] = [
        _root_sum_logarithm(logand, residue_factor)
        for logand, residue_factor in residue_reduction.logarithmic_parts
    ]
    if (
        reduced_function_reduction is not None
        and reduced_function_reduction.logarithmic_argument is not None
    ):
        coefficient = reduced_function_reduction.logarithmic_coefficient
        if coefficient is None:
            raise ValueError("a logarithmic argument needs a logarithmic coefficient")
        argument = reduced_function_reduction.logarithmic_argument
        terms.append(argument.domain.to_sympy(coefficient) * log(argument.as_expr()))
    return Add(*terms)


def _monomial_logarithmic_terms(
    residue_reduction: BronsteinResidueReduction,
    logarithmic_derivative: FieldElement,
    level: int,
    reduced_function_reduction: ReducedFunctionReduction | None = None,
) -> tuple[LogarithmicTerm, ...]:
    """Normalize one level's constructive logarithmic data."""
    expression = _logarithmic_expression(
        residue_reduction,
        reduced_function_reduction,
    )
    if is_field_zero(logarithmic_derivative):
        if expression != 0:
            raise ValueError("zero logarithmic derivative has a nonzero encoding")
        return ()
    if expression == 0:
        raise ValueError("nonzero logarithmic derivative has no encoded antiderivative")
    return (
        LogarithmicTerm(
            expression,
            logarithmic_derivative,
            level,
            (residue_reduction, reduced_function_reduction),
        ),
    )


def _obstruction(
    derivation: Derivation,
    kinds: list[IntegrationObstructionKind],
    remainder: FieldElement,
    certificate: MonomialLevelCertificate,
) -> IntegrationObstruction:
    """Construct a terminal obstruction for the current monomial level."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("a monomial obstruction needs a monomial level")
    return IntegrationObstruction(
        derivation.level,
        extension.generator,
        extension.family,
        tuple(kinds),
        remainder,
        certificate,
    )


def _rational_base(
    element: FieldElement,
    derivation: Derivation,
    extension: MonomialExtension,
) -> ElementaryIntegralResult:
    """Integrate the terminal ``QQ(x)`` level constructively."""
    field = derivation.field
    expression = field.to_sympy(element)
    exact_expression, reduced_expression = hermite_reduce_rational(
        expression,
        extension.generator,
    )
    exact_part = _coerce(field, exact_expression)
    logarithmic_derivative = _coerce(field, reduced_expression)
    logarithmic_antiderivative = (
        integrate_rational(reduced_expression, extension.generator)
        if reduced_expression != 0
        else reduced_expression
    )
    certificate = RationalElementaryReduction(
        exact_expression,
        logarithmic_antiderivative,
        reduced_expression,
    )
    terms = (
        (
            LogarithmicTerm(
                logarithmic_antiderivative,
                logarithmic_derivative,
                derivation.level,
                certificate,
            ),
        )
        if logarithmic_derivative != field.zero
        else ()
    )
    result = ElementaryIntegralResult(
        derivation,
        element,
        exact_part,
        terms,
        logarithmic_derivative,
        field.zero,
        None,
        None,
        certificate,
    )
    if not result.reconstructs:
        raise ValueError("rational integration reconstruction failed")
    return result


def _primitive_level(
    element: FieldElement,
    derivation: Derivation,
) -> _LevelReduction:
    reduction = reduce_primitive_monomial(element, derivation)
    terms = _monomial_logarithmic_terms(
        reduction.residue_reduction,
        reduction.logarithmic_derivative,
        derivation.level,
    )
    kinds: list[IntegrationObstructionKind] = []
    if reduction.has_finite_residue_obstruction:
        kinds.append(IntegrationObstructionKind.NONCONSTANT_RESIDUE)
    if reduction.has_polynomial_obstruction:
        kinds.append(IntegrationObstructionKind.REDUCED_POLYNOMIAL)
    obstruction = (
        _obstruction(derivation, kinds, reduction.remainder, reduction)
        if kinds
        else None
    )
    return _LevelReduction(
        reduction,
        reduction.exact_part,
        reduction.logarithmic_derivative,
        terms,
        reduction.remainder,
        reduction.lower_remainder,
        obstruction,
    )


def _hyperexponential_level(
    element: FieldElement,
    derivation: Derivation,
) -> _LevelReduction:
    reduction = reduce_hyperexponential_monomial(element, derivation)
    terms = _monomial_logarithmic_terms(
        reduction.residue_reduction,
        reduction.logarithmic_derivative,
        derivation.level,
    )
    kinds: list[IntegrationObstructionKind] = []
    if reduction.has_finite_residue_obstruction:
        kinds.append(IntegrationObstructionKind.NONCONSTANT_RESIDUE)
    if reduction.has_laurent_obstruction:
        kinds.append(IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION)
    obstruction = (
        _obstruction(derivation, kinds, reduction.remainder, reduction)
        if kinds
        else None
    )
    return _LevelReduction(
        reduction,
        reduction.exact_part,
        reduction.logarithmic_derivative,
        terms,
        reduction.remainder,
        reduction.lower_remainder,
        obstruction,
    )


def _hypertangent_level(
    element: FieldElement,
    derivation: Derivation,
) -> _LevelReduction:
    reduction = reduce_hypertangent_monomial(element, derivation)
    common = reduction.common_reduction
    terms = _monomial_logarithmic_terms(
        common.residue_reduction,
        reduction.logarithmic_derivative,
        derivation.level,
        common.reduced_function_reduction,
    )
    kinds: list[IntegrationObstructionKind] = []
    if common.has_finite_residue_obstruction:
        kinds.append(IntegrationObstructionKind.NONCONSTANT_RESIDUE)
    if common.has_infinity_obstruction:
        kinds.append(IntegrationObstructionKind.INFINITY_RESIDUE)
    if reduction.has_special_obstruction:
        kinds.append(IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION)
    obstruction = (
        _obstruction(derivation, kinds, reduction.remainder, reduction)
        if kinds
        else None
    )
    return _LevelReduction(
        reduction,
        reduction.exact_part,
        reduction.logarithmic_derivative,
        terms,
        reduction.remainder,
        reduction.lower_remainder,
        obstruction,
    )


def _nonlinear_no_special_level(
    element: FieldElement,
    derivation: Derivation,
) -> _LevelReduction:
    reduction = reduce_nonlinear_no_special(element, derivation)
    common = reduction.common_reduction
    terms = _monomial_logarithmic_terms(
        common.residue_reduction,
        common.logarithmic_derivative,
        derivation.level,
        common.reduced_function_reduction,
    )
    kind_by_status = {
        NonlinearNoSpecialStatus.FINITE_RESIDUE_OBSTRUCTION: (
            IntegrationObstructionKind.NONCONSTANT_RESIDUE
        ),
        NonlinearNoSpecialStatus.INFINITY_RESIDUE_OBSTRUCTION: (
            IntegrationObstructionKind.INFINITY_RESIDUE
        ),
        NonlinearNoSpecialStatus.NONCONSTANT_REDUCED_POLYNOMIAL: (
            IntegrationObstructionKind.REDUCED_POLYNOMIAL
        ),
    }
    kind = kind_by_status.get(reduction.status)
    obstruction = (
        _obstruction(derivation, [kind], common.remainder, reduction)
        if kind is not None
        else None
    )
    return _LevelReduction(
        reduction,
        common.exact_part,
        common.logarithmic_derivative,
        terms,
        common.remainder,
        reduction.lower_remainder,
        obstruction,
    )


def _is_hyperbolic_mobius_law(extension: MonomialExtension) -> bool:
    r"""Whether ``Dt`` has the narrow normalizable law ``eta*(1-t**2)``."""
    derivative = extension.generator_derivative
    if derivative.degree() != 2:
        return False
    parent = extension.parent_field
    eta = parent.convert(derivative.nth(0))
    return (
        not is_field_zero(eta)
        and is_field_zero(parent.convert(derivative.nth(1)))
        and is_field_zero(parent.convert(derivative.nth(2)) + eta)
    )


def _fresh_hyperbolic_target_generator(derivation: Derivation) -> Symbol:
    """Choose a fresh internal coordinate for a hyperbolic Möbius transform."""
    source_symbols = set(_domain_symbols(derivation.field))
    index = 0
    while True:
        candidate = Symbol(f"_risch_hyperbolic_{index}")
        if candidate not in source_symbols:
            return candidate
        index += 1


def _pullback_hyperbolic_logarithmic_terms(
    target_terms: tuple[LogarithmicTerm, ...],
    normal_form: HyperbolicMobiusNormalForm,
) -> tuple[LogarithmicTerm, ...]:
    """Map encoded target logarithms back to the source Möbius coordinate."""
    substitution = {
        normal_form.target_generator: normal_form.inverse_generator_expression,
    }
    return tuple(
        LogarithmicTerm(
            term.antiderivative.xreplace(substitution),
            normal_form.inverse(term.derivative),
            term.level,
            term.certificate,
        )
        for term in target_terms
    )


def _pullback_hyperbolic_obstruction(
    obstruction: IntegrationObstruction | None,
    derivation: Derivation,
    normal_form: HyperbolicMobiusNormalForm,
) -> IntegrationObstruction | None:
    """Re-express a target obstruction in the source tower coordinate."""
    if obstruction is None:
        return None
    if obstruction.level == derivation.level:
        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise TypeError("a hyperbolic source level must be a monomial extension")
        generator = extension.generator
        family = extension.family
    else:
        generator = obstruction.generator
        family = obstruction.family
    return IntegrationObstruction(
        obstruction.level,
        generator,
        family,
        obstruction.kinds,
        normal_form.inverse(obstruction.remainder),
        obstruction.certificate,
    )


def _integrate_hyperbolic_mobius_level(
    element: FieldElement,
    derivation: Derivation,
) -> ElementaryIntegralResult:
    """Integrate one exact hyperbolic Möbius law through its exponential form."""
    normal_form = build_hyperbolic_mobius_normal_form(
        derivation,
        _fresh_hyperbolic_target_generator(derivation),
    )
    target_result = integrate_elementary(
        normal_form.forward(element),
        normal_form.target_derivation,
    )
    result = ElementaryIntegralResult(
        derivation,
        element,
        normal_form.inverse(target_result.total_exact_part),
        _pullback_hyperbolic_logarithmic_terms(
            target_result.all_logarithmic_terms,
            normal_form,
        ),
        normal_form.inverse(target_result.total_logarithmic_derivative),
        normal_form.inverse(target_result.unintegrated_remainder),
        None,
        _pullback_hyperbolic_obstruction(
            target_result.obstruction,
            derivation,
            normal_form,
        ),
        HyperbolicMobiusElementaryCertificate(normal_form, target_result),
    )
    if not result.reconstructs:
        raise ValueError("hyperbolic Möbius elementary pullback failed to reconstruct")
    return result


def integrate_elementary(
    integrand: object,
    derivation: Derivation,
) -> ElementaryIntegralResult:
    r"""Integrate ``integrand`` recursively in an explicit differential tower.

    Implemented tower levels are the standard rational base ``QQ(x)`` with
    ``Dx = 1`` and any finite ordering of primitive, hyperexponential, and
    hypertangent levels above it, provided every non-base level carries a
    certified preserved-constants status. The exact hyperbolic law
    ``Dt = eta*(1 - t**2)`` is normalized immutably to a hyperexponential
    coordinate. The nonlinear no-special branch is additionally available
    when its required absence-of-special-polynomials certificate is recorded
    on the :class:`~sympy_risch.tower.MonomialExtension`.

    Algebraic levels, generalized monomials without the needed structure
    theorem, and automatic tower discovery remain deliberately unsupported.
    An unsupported shape raises :class:`NotImplementedError`; it is never
    presented as a proof of non-elementarity.
    """
    field = derivation.field
    element = _coerce(field, integrand)

    if derivation.level == -1:
        obstruction = None
        if element != field.zero:
            obstruction = IntegrationObstruction(
                -1,
                None,
                None,
                (IntegrationObstructionKind.CONSTANT_FIELD,),
                element,
                None,
            )
        return ElementaryIntegralResult(
            derivation,
            element,
            field.zero,
            (),
            field.zero,
            element,
            None,
            obstruction,
            None,
        )

    extension = derivation.tower.levels[derivation.level]
    if isinstance(extension, AlgebraicExtension):
        raise NotImplementedError(
            "recursive elementary integration does not yet support "
            "algebraic tower levels"
        )
    if not isinstance(extension, MonomialExtension):
        raise TypeError("elementary integration requires a differential extension")

    if _is_standard_rational_level(extension):
        return _rational_base(element, derivation, extension)
    if derivation.level == 0:
        raise NotImplementedError(
            "elementary integration requires the standard rational base "
            "QQ(x) with Dx = 1"
        )

    if extension.family is MonomialFamily.PRIMITIVE:
        level_data = _primitive_level(element, derivation)
    elif extension.family is MonomialFamily.HYPEREXPONENTIAL:
        level_data = _hyperexponential_level(element, derivation)
    elif extension.family is MonomialFamily.HYPERTANGENT:
        level_data = _hypertangent_level(element, derivation)
    elif _is_hyperbolic_mobius_law(extension):
        return _integrate_hyperbolic_mobius_level(element, derivation)
    elif (
        extension.generator_derivative.degree() >= 2
        and extension.special_polynomial_status is SpecialPolynomialStatus.ABSENT
    ):
        level_data = _nonlinear_no_special_level(element, derivation)
    else:
        raise NotImplementedError(
            "elementary integration has no complete reduction for this "
            "monomial family; a general polynomial level requires certified "
            "absence of nonconstant special polynomials"
        )

    lower_result = None
    obstruction = level_data.obstruction

    if level_data.obstruction is None:
        lower_remainder = level_data.lower_remainder
        if lower_remainder is None:
            raise ValueError(
                "a successful level reduction did not expose a lower remainder"
            )
        if level_data.remainder != _coerce(field, lower_remainder):
            raise ValueError(
                "level remainder does not equal the embedded lower remainder"
            )
        lower_result = integrate_elementary(
            lower_remainder,
            derivation.lower(),
        )
        obstruction = lower_result.obstruction

    result = ElementaryIntegralResult(
        derivation,
        element,
        level_data.exact_part,
        level_data.logarithmic_terms,
        level_data.logarithmic_derivative,
        level_data.remainder,
        lower_result,
        obstruction,
        level_data.certificate,
    )
    if not result.reconstructs:
        raise ValueError("recursive elementary integration reconstruction failed")
    return result
