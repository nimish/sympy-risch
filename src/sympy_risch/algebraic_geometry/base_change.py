r"""Conservative algebraic-constant base changes for closed finite places.

The residue-profile step can exhibit an irreducible polynomial whose roots
are needed to express local residues.  This module implements only the first
safe instance of that construction:

``QQ(x, y) -> QQ(alpha)(x, y)``, where ``q(alpha) = 0`` and the selected
``q`` splits completely over ``QQ(alpha)``.  The source relation is rebuilt
over the target constants, its constants are re-analysed, and the source
embedding is checked against both derivations.  A finite source place is then
split by factoring its rational-base prime and identifying descendants from
the transported prime ideal.

This is intentionally not a generic constant-extension mechanism.  It does
not merge unrelated residue fields, does not split infinity, and declines a
target whose algebraic relation or constant field falls outside the certified
scope.  Those boundaries make the resulting transport suitable as a later
input to logarithmic residue descent without pretending to solve that descent
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.core.singleton import S
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import Poly

from ..algebraic._field import _base_polynomial_ring, _extension_element
from ..algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
    analyze_algebraic_constant_field,
)
from ..tower import AlgebraicExtension, Derivation, DifferentialTower, MonomialExtension
from .finite_places import FiniteAlgebraicPlace
from .function_field import AlgebraicFunctionField, AlgebraicPlace
from .infinity import InfiniteAlgebraicPlace


class AlgebraicConstantBaseChangeStatus(Enum):
    """Whether the narrow algebraic-constant base change was certified."""

    CERTIFIED = "certified"
    UNSUPPORTED = "unsupported"


class AlgebraicPlaceSplitStatus(Enum):
    """Whether a source closed place was split over certified constants."""

    SPLIT = "split"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicConstantBaseChange:
    r"""A derivative-commuting, QQ-to-``QQ(alpha)`` function-field map.

    On a certified result, :meth:`embed` is the explicit map from the source
    algebraic function field into the target.  The builder checks it on the
    rational and algebraic generators, which determines commutation with the
    derivation on the whole source field.  Unsupported results carry only the
    source data and an exact reason; they are not partial embeddings.
    """

    source_model: AlgebraicFunctionField
    source_derivation: Derivation
    factor: Poly
    status: AlgebraicConstantBaseChangeStatus
    detail: str
    target_constants: AlgebraicField | None = None
    target_extension: AlgebraicExtension | None = None
    target_derivation: Derivation | None = None
    target_model: AlgebraicFunctionField | None = None
    target_constant_field_analysis: AlgebraicConstantFieldAnalysis | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_model, AlgebraicFunctionField):
            raise TypeError("a base change needs a source function field")
        if not isinstance(self.source_derivation, Derivation):
            raise TypeError("a base change needs a source derivation")
        if not isinstance(self.factor, Poly):
            raise TypeError("a base change needs a polynomial factor")
        if not isinstance(self.status, AlgebraicConstantBaseChangeStatus):
            raise TypeError("a base change has an invalid status")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a base change needs a nonempty detail")
        if self.source_derivation.level < 0 or (
            self.source_derivation.tower.levels[self.source_derivation.level]
            is not self.source_model.extension
        ):
            raise ValueError("the source derivation is not bound to the source field")

        target_data = (
            self.target_constants,
            self.target_extension,
            self.target_derivation,
            self.target_model,
            self.target_constant_field_analysis,
        )
        if self.status is AlgebraicConstantBaseChangeStatus.UNSUPPORTED:
            if any(value is not None for value in target_data):
                raise ValueError("an unsupported base change cannot carry target data")
            return

        if self.status is not AlgebraicConstantBaseChangeStatus.CERTIFIED:
            raise ValueError("a base change has an unknown status")
        if not isinstance(self.target_constants, AlgebraicField):
            raise ValueError("a certified base change needs algebraic constants")
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

    @property
    def certified(self) -> bool:
        """Whether an explicit target embedding is available."""
        return self.status is AlgebraicConstantBaseChangeStatus.CERTIFIED

    @property
    def source_extension(self) -> AlgebraicExtension:
        """The source algebraic extension."""
        return self.source_model.extension

    def embed(self, element: object) -> ExtensionElement:
        r"""Embed a source-field element into the certified target field.

        The current scope starts over ``QQ`` and retains the rational and
        algebraic generator names, so the exact SymPy expression is an
        unambiguous presentation of this particular field homomorphism.  It
        is deliberately kept local to a certified base-change object rather
        than treating arbitrary domain coercion as an algebraic embedding.
        """
        if not self.certified:
            raise NotImplementedError(self.detail)
        target_extension = self.target_extension
        if target_extension is None:
            raise ValueError("a certified base change has no target extension")
        source = _extension_element(self.source_extension, element)
        expression = self.source_extension.field.to_sympy(source)
        return cast(ExtensionElement, target_extension.field.from_sympy(expression))


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
        if not isinstance(self.base_change, AlgebraicConstantBaseChange):
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
        if self.source_place.extension is not self.base_change.source_extension:
            raise ValueError("the source place belongs to another function field")

        if self.status is AlgebraicPlaceSplitStatus.UNSUPPORTED:
            if self.target_places or self.ramification_indices:
                raise ValueError("an unsupported place split cannot carry descendants")
            return

        if self.status is not AlgebraicPlaceSplitStatus.SPLIT:
            raise ValueError("a place split has an unknown status")
        if not self.base_change.certified:
            raise ValueError("a split place needs a certified base change")
        if not isinstance(self.source_place, FiniteAlgebraicPlace):
            raise ValueError("only finite places can currently be split")
        if not self.target_places:
            raise ValueError("a split place needs at least one descendant")
        if len(self.target_places) != len(self.ramification_indices):
            raise ValueError("place descendants and ramification indices disagree")
        target_model = self.base_change.target_model
        if target_model is None:
            raise ValueError("a certified base change has no target model")
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
        source_residue = self.source_place.residue_field.convert(value)
        lifted = self.source_place.lift_residue(source_residue)
        embedded = self.base_change.embed(lifted)
        return tuple(place.reduce_local(embedded) for place in self.target_places)


def _unsupported_base_change(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    factor: Poly,
    detail: str,
) -> AlgebraicConstantBaseChange:
    """Construct the non-decision result for a narrow base-change boundary."""
    return AlgebraicConstantBaseChange(
        model,
        derivation,
        factor,
        AlgebraicConstantBaseChangeStatus.UNSUPPORTED,
        detail,
    )


def _target_factor_splits_completely(
    factor: Poly,
    constants: AlgebraicField,
) -> bool:
    """Whether every root of ``factor`` is already in ``constants``."""
    target_factor = Poly(factor.as_expr(), factor.gens[0], domain=constants)
    _unit, factors = target_factor.factor_list()
    return len(factors) == factor.degree() and all(
        component.degree() == 1 and multiplicity == 1
        for component, multiplicity in factors
    )


def _generator_derivatives_commute(change: AlgebraicConstantBaseChange) -> bool:
    """Check derivation commutation on the rational and algebraic generators."""
    target_extension = change.target_extension
    target_derivation = change.target_derivation
    if target_extension is None or target_derivation is None:
        return False

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


def build_algebraic_constant_base_change(
    model: AlgebraicFunctionField,
    derivation: Derivation,
    factor: Poly,
) -> AlgebraicConstantBaseChange:
    r"""Build the first certified algebraic-constant base-change layer.

    The accepted input is deliberately narrow: a standard algebraic function
    field over ``QQ(x)``, a monic irreducible polynomial ``q`` of degree at
    least two, and a target ``QQ(alpha)`` in which ``q`` splits completely.
    The source relation must remain a supported irreducible relation over the
    target constants, and the target constant field must again be preserved.
    Every other case is returned as :attr:`~AlgebraicConstantBaseChangeStatus.UNSUPPORTED`.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("constant base change needs an algebraic function field")
    if not isinstance(derivation, Derivation):
        raise TypeError("constant base change needs a derivation")
    if not isinstance(factor, Poly):
        raise TypeError("constant base change needs a SymPy Poly factor")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not model.extension
    ):
        raise ValueError("the derivation must be bound to the model extension")

    try:
        _parent, constants, variable = _base_polynomial_ring(model.extension)
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(model, derivation, factor, str(exc))
    if constants != QQ:
        return _unsupported_base_change(
            model,
            derivation,
            factor,
            "constant base change currently starts only from QQ",
        )
    if len(factor.gens) != 1:
        return _unsupported_base_change(
            model,
            derivation,
            factor,
            "the selected constant polynomial must be univariate",
        )
    factor_variable = factor.gens[0]
    if factor_variable in {variable, model.extension.generator}:
        return _unsupported_base_change(
            model,
            derivation,
            factor,
            "the constant polynomial needs a generator distinct from x and y",
        )
    try:
        normalized_factor = Poly(factor.as_expr(), factor_variable, domain=QQ).monic()
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            factor,
            f"the selected constant polynomial cannot be normalized over QQ: {exc}",
        )
    if normalized_factor.degree() < 2:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            "the selected constant polynomial must have degree at least two",
        )
    if normalized_factor.is_irreducible is not True:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            "the selected constant polynomial must be irreducible over QQ",
        )

    source_analysis = analyze_algebraic_constant_field(model.extension, derivation)
    if source_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            "the source algebraic constant field is not certified as preserved: "
            f"{source_analysis.detail}",
        )

    try:
        target_constants = QQ.alg_field_from_poly(normalized_factor)
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            f"SymPy cannot construct the selected algebraic constants: {exc}",
        )
    if not isinstance(target_constants, AlgebraicField):
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            "SymPy did not construct an algebraic-number constant field",
        )
    try:
        splits_completely = _target_factor_splits_completely(
            normalized_factor,
            target_constants,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            f"the selected constant polynomial cannot be factored over its root field: {exc}",
        )
    if not splits_completely:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            "the selected constant polynomial does not split completely over QQ(alpha)",
        )

    try:
        target_rational = DifferentialTower(target_constants).extend(
            MonomialExtension(Poly(1, variable, domain=target_constants)),
        )
        target_modulus = Poly(
            model.extension.minimal_polynomial.as_expr(),
            model.extension.generator,
            domain=target_rational.field(),
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
                normalized_factor,
                "the target algebraic constant field is not certified as preserved: "
                f"{target_analysis.detail}",
            )
        target_model = AlgebraicFunctionField(target_extension)
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            f"the source algebraic relation cannot be rebuilt over QQ(alpha): {exc}",
        )

    candidate = AlgebraicConstantBaseChange(
        model,
        derivation,
        normalized_factor,
        AlgebraicConstantBaseChangeStatus.CERTIFIED,
        "the selected constant polynomial splits completely and the rebuilt "
        "algebraic function field preserves its constants",
        target_constants,
        target_extension,
        target_derivation,
        target_model,
        target_analysis,
    )
    try:
        derivatives_commute = _generator_derivatives_commute(candidate)
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            f"the explicit source-to-target embedding could not be verified: {exc}",
        )
    if not derivatives_commute:
        return _unsupported_base_change(
            model,
            derivation,
            normalized_factor,
            "the explicit source-to-target embedding does not commute with D",
        )
    return candidate


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
    if not isinstance(base_change, AlgebraicConstantBaseChange):
        raise TypeError("place splitting needs an algebraic constant base change")
    if not isinstance(
        source_place,
        (FiniteAlgebraicPlace, InfiniteAlgebraicPlace),
    ):
        raise TypeError("place splitting needs an algebraic source place")
    if source_place.extension is not base_change.source_extension:
        raise ValueError("the source place belongs to another function field")
    if not base_change.certified:
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
    if target_constants is None or target_model is None or target_extension is None:
        raise ValueError("a certified base change has incomplete target data")
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
