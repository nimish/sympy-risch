r"""Conservative algebraic-constant base changes for closed finite places.

The residue-profile step can exhibit irreducible polynomials whose roots are
needed to express local residues. This module implements the first safe
common-field instance of that construction:

``C(x, y) -> C'(x, y)``, where ``C`` is ``QQ`` or an algebraic number field
and every selected rational polynomial splits completely over the exact
compositum ``C'``. The source relation is rebuilt coefficient by coefficient
over the target constants, its constants are re-analysed, and the source
embedding is checked against both derivations. A finite source place is then
split by factoring its rational-base prime and identifying descendants from
the transported prime ideal.

The constant-field layer does construct exact composita of successive
rationally defined number fields.  It does not yet adjoin a polynomial with
genuinely non-rational coefficients, split infinity, or accept a target whose
algebraic relation or constant field falls outside the certified scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from sympy.core.singleton import S
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import Poly

from ..algebraic._field import _base_polynomial_ring, _extension_element
from ..algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
    analyze_algebraic_constant_field,
)
from ..tower import AlgebraicExtension, Derivation, DifferentialTower, MonomialExtension
from .constant_extensions import (
    AlgebraicConstantFieldCompositum,
    extend_algebraic_constant_compositum,
)
from .finite_places import FiniteAlgebraicPlace
from .function_field import AlgebraicFunctionField, AlgebraicPlace
from .infinity import InfiniteAlgebraicPlace


class AlgebraicPlaceSplitStatus(Enum):
    """Whether a source closed place was split over certified constants."""

    SPLIT = "split"
    UNSUPPORTED = "unsupported"


def _embed_parent_element(
    element: object,
    source: FractionField,
    target: FractionField,
    constants: AlgebraicConstantFieldCompositum,
) -> object:
    """Lift the constant embedding coefficientwise through ``C(x)``."""
    source_element = source.convert(element)
    target_ring = target.field.ring
    target_type = target.dtype
    if target_type is None:
        raise TypeError("the target rational function field has no element type")
    numerator = target_ring.from_dict(
        {
            monomial: constants.embed(coefficient)
            for monomial, coefficient in source_element.numer.to_dict().items()
        }
    )
    denominator = target_ring.from_dict(
        {
            monomial: constants.embed(coefficient)
            for monomial, coefficient in source_element.denom.to_dict().items()
        }
    )
    return target_type(numerator, denominator)


def _validate_base_change_source(
    source_model: AlgebraicFunctionField,
    source_derivation: Derivation,
    factors: tuple[Poly, ...],
    detail: str,
) -> None:
    """Validate the source request retained by either builder outcome."""
    if not isinstance(source_model, AlgebraicFunctionField):
        raise TypeError("a base change needs a source function field")
    if not isinstance(source_derivation, Derivation):
        raise TypeError("a base change needs a source derivation")
    if not isinstance(factors, tuple) or not factors:
        raise ValueError("a base change needs at least one polynomial factor")
    if any(not isinstance(factor, Poly) for factor in factors):
        raise TypeError("base-change factors must be SymPy polynomials")
    if not isinstance(detail, str) or not detail:
        raise ValueError("a base change needs a nonempty detail")
    if source_derivation.level < 0 or (
        source_derivation.tower.levels[source_derivation.level]
        is not source_model.extension
    ):
        raise ValueError("the source derivation is not bound to the source field")


@dataclass(frozen=True, slots=True)
class CertifiedAlgebraicConstantBaseChange:
    """A derivative-commuting embedding into certified enlarged constants."""

    source_model: AlgebraicFunctionField
    source_derivation: Derivation
    factors: tuple[Poly, ...]
    detail: str
    constant_compositum: AlgebraicConstantFieldCompositum
    target_extension: AlgebraicExtension
    target_derivation: Derivation
    target_model: AlgebraicFunctionField
    target_constant_field_analysis: AlgebraicConstantFieldAnalysis

    def __post_init__(self) -> None:
        _validate_base_change_source(
            self.source_model,
            self.source_derivation,
            self.factors,
            self.detail,
        )
        if not isinstance(
            self.constant_compositum,
            AlgebraicConstantFieldCompositum,
        ):
            raise ValueError("a certified base change needs a constant-field map")
        if not isinstance(self.target_extension, AlgebraicExtension):
            raise ValueError("a certified base change needs a target extension")
        if not isinstance(self.target_derivation, Derivation):
            raise ValueError("a certified base change needs a target derivation")
        if not isinstance(self.target_model, AlgebraicFunctionField):
            raise ValueError("a certified base change needs a target function field")
        if not isinstance(
            self.target_constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise ValueError("a certified base change needs constant-field analysis")
        if self.target_model.extension is not self.target_extension:
            raise ValueError("the target model belongs to another extension")
        if self.target_derivation.level < 0 or (
            self.target_derivation.tower.levels[self.target_derivation.level]
            is not self.target_extension
        ):
            raise ValueError("the target derivation is not bound to the target field")
        if self.target_constant_field_analysis.extension is not self.target_extension:
            raise ValueError("the target constant analysis belongs to another field")
        if (
            self.target_constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError("a certified base change needs preserved target constants")
        _parent, source_constants, _variable = _base_polynomial_ring(
            self.source_extension
        )
        if self.constant_compositum.source != source_constants:
            raise ValueError("the constant-field map has another source")
        if self.constant_compositum.factors != self.factors:
            raise ValueError("the constant-field map certifies different factors")
        _target_parent, target_constants, _target_variable = _base_polynomial_ring(
            self.target_extension
        )
        if self.constant_compositum.target != target_constants:
            raise ValueError("the constant-field map has another target")

    @property
    def source_extension(self) -> AlgebraicExtension:
        """The source algebraic extension."""
        return self.source_model.extension

    @property
    def target_constants(self) -> AlgebraicField:
        """The certified target constant field."""
        target = self.constant_compositum.target
        if not isinstance(target, AlgebraicField):
            raise ValueError("a certified base change has no algebraic target")
        return target

    def embed(self, element: object) -> ExtensionElement:
        r"""Embed a source-field element into the certified target field."""
        source = _extension_element(self.source_extension, element)
        source_parent = self.source_extension.parent_field
        target_parent = self.target_extension.parent_field
        if not isinstance(source_parent, FractionField) or not isinstance(
            target_parent, FractionField
        ):
            raise NotImplementedError(
                "constant base change requires rational function parent fields"
            )
        coefficients = [
            _embed_parent_element(
                coefficient,
                source_parent,
                target_parent,
                self.constant_compositum,
            )
            for coefficient in source.rep.to_list()
        ]
        representative = DMP(coefficients, target_parent, 0)
        return self.target_extension.field.dtype(
            representative, self.target_extension.field
        )


@dataclass(frozen=True, slots=True)
class AlgebraicConstantBaseChangeUnsupported:
    """A precise boundary where no target embedding was certified."""

    source_model: AlgebraicFunctionField
    source_derivation: Derivation
    factors: tuple[Poly, ...]
    detail: str

    def __post_init__(self) -> None:
        _validate_base_change_source(
            self.source_model,
            self.source_derivation,
            self.factors,
            self.detail,
        )


AlgebraicConstantBaseChange: TypeAlias = (
    CertifiedAlgebraicConstantBaseChange | AlgebraicConstantBaseChangeUnsupported
)


@dataclass(frozen=True, slots=True)
class AlgebraicFinitePlaceSplit:
    r"""Finite descendants of one source place under a base change.

    :attr:`target_places` and :attr:`ramification_indices` have matching
    order.  :meth:`transport_residue` returns target residue values in that
    same order, without imposing a common representation on their residue
    fields.
    """

    base_change: AlgebraicConstantBaseChange
    source_place: AlgebraicPlace
    status: AlgebraicPlaceSplitStatus
    detail: str
    target_places: tuple[FiniteAlgebraicPlace, ...] = ()
    ramification_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.base_change,
            (
                CertifiedAlgebraicConstantBaseChange,
                AlgebraicConstantBaseChangeUnsupported,
            ),
        ):
            raise TypeError("a place split needs a base change")
        if not isinstance(
            self.source_place,
            (FiniteAlgebraicPlace, InfiniteAlgebraicPlace),
        ):
            raise TypeError("a place split needs an algebraic source place")
        if not isinstance(self.status, AlgebraicPlaceSplitStatus):
            raise TypeError("a place split has an invalid status")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a place split needs a nonempty detail")
        if self.source_place.extension is not self.base_change.source_model.extension:
            raise ValueError("the source place belongs to another function field")

        if self.status is AlgebraicPlaceSplitStatus.UNSUPPORTED:
            if self.target_places or self.ramification_indices:
                raise ValueError("an unsupported place split cannot carry descendants")
            return

        if self.status is not AlgebraicPlaceSplitStatus.SPLIT:
            raise ValueError("a place split has an unknown status")
        if not isinstance(self.base_change, CertifiedAlgebraicConstantBaseChange):
            raise ValueError("a split place needs a certified base change")
        if not isinstance(self.source_place, FiniteAlgebraicPlace):
            raise ValueError("only finite places can currently be split")
        if not self.target_places:
            raise ValueError("a split place needs at least one descendant")
        if len(self.target_places) != len(self.ramification_indices):
            raise ValueError("place descendants and ramification indices disagree")
        target_model = self.base_change.target_model
        if any(place.order is not target_model.order for place in self.target_places):
            raise ValueError("a target place belongs to another function field")
        if any(index != 1 for index in self.ramification_indices):
            raise ValueError("the finite base-change layer only certifies e = 1")
        if (
            sum(place.degree for place in self.target_places)
            != self.source_place.degree
        ):
            raise ValueError("the split-place degrees do not account for the source")

    @property
    def split(self) -> bool:
        """Whether finite descendant places were certified."""
        return self.status is AlgebraicPlaceSplitStatus.SPLIT

    def transport_residue(self, value: object) -> tuple[ExtensionElement, ...]:
        r"""Transport one source residue to every descendant finite place.

        A residue is first lifted at the source place, embedded into the
        target function field, then reduced at each descendant.  This is the
        canonical local transport; it never guesses an embedding between the
        source and target residue fields directly.
        """
        if not self.split:
            raise NotImplementedError(self.detail)
        if not isinstance(self.source_place, FiniteAlgebraicPlace):
            raise NotImplementedError("infinite-place transport is not implemented")
        if not isinstance(self.base_change, CertifiedAlgebraicConstantBaseChange):
            raise ValueError("a split place lacks a certified base change")
        source_residue = self.source_place.residue_field.convert(value)
        lifted = self.source_place.lift_residue(source_residue)
        embedded = self.base_change.embed(lifted)
        return tuple(place.reduce_local(embedded) for place in self.target_places)


def _unsupported_base_change(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    factors: tuple[Poly, ...],
    detail: str,
) -> AlgebraicConstantBaseChange:
    """Construct the non-decision result for a narrow base-change boundary."""
    return AlgebraicConstantBaseChangeUnsupported(
        model,
        derivation,
        factors,
        detail,
    )


def _generator_derivatives_commute(
    change: CertifiedAlgebraicConstantBaseChange,
) -> bool:
    """Check derivation commutation on the rational and algebraic generators."""
    target_extension = change.target_extension
    target_derivation = change.target_derivation
    _parent, _constants, variable = _base_polynomial_ring(change.source_extension)
    source_generators = (
        change.source_extension.field.from_sympy(variable),
        change.source_extension.field.generator,
    )
    for source in source_generators:
        mapped_derivative = change.embed(change.source_derivation(source))
        target_derivative = target_derivation(change.embed(source))
        if mapped_derivative - target_derivative != target_extension.field.zero:
            return False
    return True


def build_algebraic_constant_compositum_base_change(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    factors: tuple[Poly, ...],
) -> AlgebraicConstantBaseChange:
    r"""Build one certified base change through a common constant field.

    The source constants may be ``QQ`` or a SymPy algebraic number field.
    Every selected factor is currently required to be irreducible over
    ``QQ``. Their splitting root fields are joined before the curve is rebuilt,
    so several independent residue fields require only one function-field base
    change. The source relation must remain a supported irreducible relation
    over the compositum, and the target constant field must again be
    preserved. Every other case returns
    :class:`AlgebraicConstantBaseChangeUnsupported`.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("constant base change needs an algebraic function field")
    if not isinstance(derivation, Derivation):
        raise TypeError("constant base change needs a derivation")
    if not isinstance(factors, tuple):
        raise TypeError("constant base change needs a tuple of factors")
    if not factors:
        raise ValueError("constant base change needs at least one factor")
    if any(not isinstance(factor, Poly) for factor in factors):
        raise TypeError("constant-base-change factors must be SymPy polynomials")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not model.extension
    ):
        raise ValueError("the derivation must be bound to the model extension")

    try:
        source_parent, constants, variable = _base_polynomial_ring(model.extension)
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(model, derivation, factors, str(exc))
    if constants != QQ and not isinstance(constants, AlgebraicField):
        return _unsupported_base_change(
            model,
            derivation,
            factors,
            "constant base change requires QQ or algebraic-number constants",
        )
    normalized_factors: list[Poly] = []
    for factor in factors:
        if len(factor.gens) != 1:
            return _unsupported_base_change(
                model,
                derivation,
                factors,
                "each selected constant polynomial must be univariate",
            )
        factor_variable = factor.gens[0]
        if factor_variable in {variable, model.extension.generator}:
            return _unsupported_base_change(
                model,
                derivation,
                factors,
                "constant polynomials need generators distinct from x and y",
            )
        try:
            normalized = Poly(
                factor.as_expr(),
                factor_variable,
                domain=QQ,
            ).monic()
        except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
            return _unsupported_base_change(
                model,
                derivation,
                factors,
                "a selected constant polynomial cannot be normalized over QQ: "
                + str(exc),
            )
        if normalized.degree() < 2:
            return _unsupported_base_change(
                model,
                derivation,
                factors,
                "each selected constant polynomial must have degree at least two",
            )
        if normalized.is_irreducible is not True:
            return _unsupported_base_change(
                model,
                derivation,
                factors,
                "each selected constant polynomial must be irreducible over QQ",
            )
        normalized_factors.append(normalized)
    normalized_tuple = tuple(dict.fromkeys(normalized_factors))

    source_analysis = analyze_algebraic_constant_field(model.extension, derivation)
    if source_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_tuple,
            "the source algebraic constant field is not certified as preserved: "
            f"{source_analysis.detail}",
        )

    try:
        constant_compositum = extend_algebraic_constant_compositum(
            constants,
            normalized_tuple,
        )
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_tuple,
            f"the selected algebraic constant compositum is unavailable: {exc}",
        )
    if constant_compositum.is_identity:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_tuple,
            "every selected factor already splits over the source constants",
        )
    target_constants = constant_compositum.target
    if not isinstance(target_constants, AlgebraicField):
        raise ValueError("a nontrivial constant compositum must be algebraic")

    try:
        target_rational = DifferentialTower(target_constants).extend(
            MonomialExtension(Poly(1, variable, domain=target_constants)),
        )
        target_parent = target_rational.field()
        if not isinstance(target_parent, FractionField):
            raise TypeError("the rebuilt rational tower has no fraction field")
        target_modulus = Poly.from_dict(
            {
                monomial: _embed_parent_element(
                    coefficient,
                    source_parent,
                    target_parent,
                    constant_compositum,
                )
                for monomial, coefficient in (
                    model.extension.minimal_polynomial.rep.to_dict().items()
                )
            },
            model.extension.generator,
            domain=target_parent,
        )
        target_extension = AlgebraicExtension(target_modulus)
        target_tower = target_rational.extend(target_extension)
        target_derivation = target_tower.derivation()
        target_analysis = analyze_algebraic_constant_field(
            target_extension,
            target_derivation,
        )
        if target_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
            return _unsupported_base_change(
                model,
                derivation,
                normalized_tuple,
                "the target algebraic constant field is not certified as preserved: "
                f"{target_analysis.detail}",
            )
        target_model = AlgebraicFunctionField(target_extension)
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_tuple,
            f"the source algebraic relation cannot be rebuilt over the compositum: {exc}",
        )

    candidate = CertifiedAlgebraicConstantBaseChange(
        model,
        derivation,
        normalized_tuple,
        "the selected constant polynomial splits completely and the rebuilt "
        "algebraic function field preserves its constants",
        constant_compositum,
        target_extension,
        target_derivation,
        target_model,
        target_analysis,
    )
    try:
        derivatives_commute = _generator_derivatives_commute(candidate)
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_tuple,
            f"the explicit source-to-target embedding could not be verified: {exc}",
        )
    if not derivatives_commute:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_tuple,
            "the explicit source-to-target embedding does not commute with D",
        )
    return candidate


def build_algebraic_constant_base_change(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    factor: Poly,
) -> AlgebraicConstantBaseChange:
    """Build a one-factor algebraic-constant base change."""
    if not isinstance(factor, Poly):
        raise TypeError("constant base change needs a SymPy Poly factor")
    return build_algebraic_constant_compositum_base_change(
        model,
        derivation,
        (factor,),
    )


def _unsupported_place_split(
    base_change: AlgebraicConstantBaseChange,
    source_place: AlgebraicPlace,
    detail: str,
) -> AlgebraicFinitePlaceSplit:
    """Construct an explicit non-decision for a place-splitting boundary."""
    return AlgebraicFinitePlaceSplit(
        base_change,
        source_place,
        AlgebraicPlaceSplitStatus.UNSUPPORTED,
        detail,
    )


def _strictly_positive_finite_valuation(value: object) -> bool:
    """Recognize a positive finite valuation without coercing SymPy infinity."""
    return value is not S.Infinity and isinstance(value, int) and value > 0


def split_algebraic_place(
    base_change: AlgebraicConstantBaseChange,
    source_place: AlgebraicPlace,
) -> AlgebraicFinitePlaceSplit:
    r"""Split one source place after a certified constant extension.

    Finite places are identified from the transported source prime ideal.  A
    descendant ``Q`` is accepted only when every transported ideal generator
    has positive ``Q``-valuation, its ramification index over the source
    place is one, and the descendant degrees sum to the source degree.  The
    reciprocal/infinite chart is deliberately not part of this first layer.
    """
    if not isinstance(
        base_change,
        (
            CertifiedAlgebraicConstantBaseChange,
            AlgebraicConstantBaseChangeUnsupported,
        ),
    ):
        raise TypeError("place splitting needs an algebraic constant base change")
    if not isinstance(
        source_place,
        (FiniteAlgebraicPlace, InfiniteAlgebraicPlace),
    ):
        raise TypeError("place splitting needs an algebraic source place")
    if source_place.extension is not base_change.source_model.extension:
        raise ValueError("the source place belongs to another function field")
    if isinstance(base_change, AlgebraicConstantBaseChangeUnsupported):
        return _unsupported_place_split(base_change, source_place, base_change.detail)
    if isinstance(source_place, InfiniteAlgebraicPlace):
        return _unsupported_place_split(
            base_change,
            source_place,
            "splitting places at infinity is not implemented by the finite layer",
        )
    if source_place.order is not base_change.source_model.order:
        raise ValueError("the source finite place belongs to another function field")

    target_constants = base_change.target_constants
    target_model = base_change.target_model
    target_extension = base_change.target_extension
    _parent, _constants, target_variable = _base_polynomial_ring(target_extension)
    try:
        target_base_prime = Poly(
            source_place.base_prime.as_expr(),
            target_variable,
            domain=target_constants,
        )
        _unit, factors = target_base_prime.factor_list()
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_place_split(
            base_change,
            source_place,
            f"the source base prime cannot be factored over QQ(alpha): {exc}",
        )
    if any(multiplicity != 1 for _factor, multiplicity in factors):
        return _unsupported_place_split(
            base_change,
            source_place,
            "the source base prime is inseparable after the constant extension",
        )

    descendants: list[FiniteAlgebraicPlace] = []
    ramification: list[int] = []
    try:
        source_generators = source_place.prime_ideal.basis_elements
        mapped_generators = tuple(
            base_change.embed(generator) for generator in source_generators
        )
        mapped_uniformizer = base_change.embed(source_place.uniformizer)
        for factor, _multiplicity in factors:
            for target_place in target_model.finite_places(factor.monic()):
                valuations = tuple(
                    target_place.valuation(generator) for generator in mapped_generators
                )
                if not all(
                    _strictly_positive_finite_valuation(valuation)
                    for valuation in valuations
                ):
                    continue
                index = target_place.valuation(mapped_uniformizer)
                if not isinstance(index, int) or index < 1:
                    return _unsupported_place_split(
                        base_change,
                        source_place,
                        "a descendant has no finite positive ramification index",
                    )
                descendants.append(target_place)
                ramification.append(index)
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_place_split(
            base_change,
            source_place,
            f"finite-place descendants could not be certified: {exc}",
        )

    if not descendants:
        return _unsupported_place_split(
            base_change,
            source_place,
            "no target finite place was identified above the source place",
        )
    if any(index != 1 for index in ramification):
        return _unsupported_place_split(
            base_change,
            source_place,
            "the finite base-change layer only supports unramified descendants",
        )
    if sum(place.degree for place in descendants) != source_place.degree:
        return _unsupported_place_split(
            base_change,
            source_place,
            "target descendant degrees do not account for the source place",
        )
    return AlgebraicFinitePlaceSplit(
        base_change,
        source_place,
        AlgebraicPlaceSplitStatus.SPLIT,
        "finite descendants were certified from the transported prime ideal",
        tuple(descendants),
        tuple(ramification),
    )


def split_finite_algebraic_place(
    base_change: AlgebraicConstantBaseChange,
    source_place: FiniteAlgebraicPlace,
) -> AlgebraicFinitePlaceSplit:
    """Finite-only spelling of :func:`split_algebraic_place`."""
    if not isinstance(source_place, FiniteAlgebraicPlace):
        raise TypeError("finite place splitting needs a finite algebraic place")
    return split_algebraic_place(base_change, source_place)
