r"""Certified first Riemann--Roch spaces for algebraic function fields.

This is intentionally a *certificate kernel*, not yet a general
Riemann--Roch basis algorithm.  The geometry layer supplies exact orders,
places, valuations, and divisors; this module turns those local facts into a
small collection of globally complete spaces only when a standard theorem
proves completeness.

For a divisor ``D`` on a smooth function field ``L/C``,

``L(D) = {f in L : div(f) + D >= 0} union {0}``.

The implemented certificates are:

* ``deg(D) < 0``: ``L(D)`` is zero;
* ``D = 0``: ``L(D) = C``;
* a degree-zero principal witness ``f`` with ``div(f) + D = 0``:
  ``L(D) = C*f``;
* the nonspecial range ``deg(D) > 2*g - 2``: a caller-provided basis with
  ``deg(D) + 1 - g`` exact, constant-linearly independent members is complete
  by Riemann--Roch; and
* a rational parameter ``t`` with pole divisor of degree one:
  ``L(m*(t)_infinity) = span_C(1, t, ..., t**m)``.

The last case is an executable seed for the eventual lattice-intersection
solver.  A degree-one pole divisor implies ``L = C(t)``; it is not guessed
from the displayed plane equation.  Every successful result rechecks every
candidate through the full global principal divisor, so finite, ramified, and
infinite valuation bounds are all enforced.

The module requires the existing algebraic constant-field proof.  It never
treats a displayed coefficient field as the true constant field merely
because SymPy can perform arithmetic in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix

from .algebraic._field import (
    _base_polynomial_ring,
    _common_denominator,
    _extension_element,
    _parent_element_polynomial,
    _polynomial_parent_element,
)
from .algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from .algebraic_geometry import AlgebraicDivisor, AlgebraicFunctionField


class RiemannRochSpaceStatus(Enum):
    """Whether a requested Riemann--Roch space is complete and certified."""

    CERTIFIED = "certified"
    UNSUPPORTED = "unsupported"


class RiemannRochCertificate(Enum):
    """The theorem which proves a :class:`RiemannRochSpace` complete."""

    NEGATIVE_DEGREE = "negative_degree"
    ZERO_DIVISOR = "zero_divisor"
    DEGREE_ZERO_WITNESS = "degree_zero_witness"
    NONSPECIAL = "nonspecial"
    RATIONAL_PARAMETER = "rational_parameter"


def _constant_coordinate_matrix(
    model: AlgebraicFunctionField,
    elements: tuple[ExtensionElement, ...],
) -> DomainMatrix:
    r"""Return exact constant-coordinate columns for ``elements``.

    The native extension basis gives every element coordinates in ``C(x)``.
    Clearing their common ``x`` denominator and equating all powers of both
    the algebraic generator and ``x`` converts a ``C``-linear relation into a
    finite exact matrix over ``C``. Its column rank is their constant-field
    span dimension; it deliberately does *not* encode linear relations over
    ``C(x)``, which would be too weak for global-function bases.
    """
    if not elements:
        _parent, constants, _variable = _base_polynomial_ring(model.extension)
        return DomainMatrix([], (0, 0), constants)

    extension = model.extension
    parent, constants, variable = _base_polynomial_ring(extension)
    if not isinstance(parent, FractionField):
        raise TypeError("the algebraic parent must be a rational function field")

    coordinates = tuple(
        cast(FracElement, parent.convert(element.rep.nth(power)))
        for element in elements
        for power in range(extension.field.rank)
    )
    denominator = _common_denominator(coordinates, constants, variable)
    denominator_element = _polynomial_parent_element(denominator, parent)
    rows: dict[tuple[int, int], list[object]] = {}

    for column, element in enumerate(elements):
        for power in range(extension.field.rank):
            coordinate = cast(FracElement, parent.convert(element.rep.nth(power)))
            polynomial = _parent_element_polynomial(
                coordinate * denominator_element,
                parent,
                constants,
                variable,
            )
            for (degree,), coefficient in polynomial.terms():
                row = rows.setdefault(
                    (power, degree),
                    [constants.zero for _ in elements],
                )
                row[column] = constants.convert(coefficient)

    if not rows:
        return DomainMatrix([], (0, len(elements)), constants)
    return DomainMatrix(
        list(rows.values()),
        (len(rows), len(elements)),
        constants,
    )


def _constant_linear_rank(
    model: AlgebraicFunctionField,
    elements: tuple[ExtensionElement, ...],
) -> int:
    """Return the rank of ``elements`` over the proved constant field."""
    return _constant_coordinate_matrix(model, elements).rank()


def _basis_membership_is_verified(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    basis: tuple[ExtensionElement, ...],
) -> bool:
    """Check every local valuation bound through a complete principal divisor."""
    zero = model.extension.field.zero
    for element in basis:
        if element == zero:
            return False
        if not (model.principal_divisor(element) + divisor).is_effective:
            return False
    return True


def _rational_parameter_pole_divisor(
    model: AlgebraicFunctionField,
    parameter: ExtensionElement,
) -> AlgebraicDivisor:
    """Return the pole divisor of a nonconstant candidate parameter."""
    if parameter == model.extension.field.zero:
        raise ValueError("a rational parameter must be nonzero")
    pole_divisor = model.principal_divisor(parameter).negative_part
    if pole_divisor.degree != 1:
        raise ValueError(
            "a rational-parameter certificate needs a pole divisor of degree one",
        )
    return pole_divisor


@dataclass(frozen=True, slots=True)
class RiemannRochSpace:
    r"""An immutable complete basis of one certified ``L(D)``.

    ``basis`` is a basis over the *true* constant field certified by
    ``constant_field_analysis``.  ``UNSUPPORTED`` is intentionally
    information-poor: it carries no candidate basis that a downstream solver
    could accidentally mistake for a complete space.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    status: RiemannRochSpaceStatus
    basis: tuple[ExtensionElement, ...]
    certificate: RiemannRochCertificate | None
    detail: str
    parameter: ExtensionElement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a Riemann--Roch space needs an algebraic function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a Riemann--Roch space needs an algebraic divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if not isinstance(
            self.constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError("a Riemann--Roch space needs constant-field analysis")
        if self.constant_field_analysis.extension is not self.model.extension:
            raise ValueError("the constant-field analysis belongs to another field")
        if not isinstance(self.status, RiemannRochSpaceStatus):
            raise TypeError("the Riemann--Roch status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a Riemann--Roch space needs a nonempty detail")

        basis = tuple(
            _extension_element(self.model.extension, element) for element in self.basis
        )
        object.__setattr__(self, "basis", basis)

        parameter: ExtensionElement | None = None
        if self.parameter is not None:
            parameter = _extension_element(self.model.extension, self.parameter)
            object.__setattr__(self, "parameter", parameter)

        if self.status is RiemannRochSpaceStatus.UNSUPPORTED:
            if basis or self.certificate is not None or parameter is not None:
                raise ValueError(
                    "an unsupported Riemann--Roch result cannot carry partial data",
                )
            return

        if self.status is not RiemannRochSpaceStatus.CERTIFIED:
            raise ValueError("a Riemann--Roch result has an unknown status")
        if (
            self.constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError(
                "a complete Riemann--Roch space needs a preserved constant field",
            )
        if not isinstance(self.certificate, RiemannRochCertificate):
            raise ValueError(
                "a complete Riemann--Roch space needs a theorem certificate"
            )

        self._verify_certificate(basis, parameter)

    @property
    def dimension(self) -> int | None:
        """The proved constant-field dimension, or ``None`` if unsupported."""
        if self.status is RiemannRochSpaceStatus.UNSUPPORTED:
            return None
        return len(self.basis)

    @property
    def complete(self) -> bool:
        """Whether this object is a full, rather than partial, space."""
        return self.status is RiemannRochSpaceStatus.CERTIFIED

    @property
    def unsupported(self) -> bool:
        """Whether no complete space was certified in the present scope."""
        return self.status is RiemannRochSpaceStatus.UNSUPPORTED

    def contains(self, element: object) -> bool:
        """Verify the defining valuation inequalities for one field element."""
        value = _extension_element(self.model.extension, element)
        if value == self.model.extension.field.zero:
            return True
        return (self.model.principal_divisor(value) + self.divisor).is_effective

    def _verify_certificate(
        self,
        basis: tuple[ExtensionElement, ...],
        parameter: ExtensionElement | None,
    ) -> None:
        """Recompute the theorem hypotheses; public construction is unforgeable."""
        certificate = self.certificate
        if certificate is RiemannRochCertificate.NEGATIVE_DEGREE:
            if self.divisor.degree >= 0 or basis or parameter is not None:
                raise ValueError("negative-degree completeness has inconsistent data")
            return

        if certificate is RiemannRochCertificate.ZERO_DIVISOR:
            if not self.divisor.is_zero or parameter is not None:
                raise ValueError("the zero-divisor certificate has inconsistent data")
            if basis != (self.model.extension.field.one,):
                raise ValueError("L(0) must use the canonical constant basis (1,)")
            return

        if certificate is RiemannRochCertificate.DEGREE_ZERO_WITNESS:
            if self.divisor.degree != 0 or len(basis) != 1 or parameter is not None:
                raise ValueError("the degree-zero witness has inconsistent data")
            if not _basis_membership_is_verified(self.model, self.divisor, basis):
                raise ValueError("the degree-zero witness is not in L(D)")
            if not (self.model.principal_divisor(basis[0]) + self.divisor).is_zero:
                raise ValueError("the degree-zero witness does not trivialize D")
            return

        if certificate is RiemannRochCertificate.NONSPECIAL:
            if parameter is not None:
                raise ValueError("a nonspecial certificate cannot carry a parameter")
            genus = self.model.genus
            if self.divisor.degree <= 2 * genus - 2:
                raise ValueError("the divisor is outside the nonspecial range")
            expected_dimension = self.divisor.degree + 1 - genus
            if len(basis) != expected_dimension:
                raise ValueError("the nonspecial basis has the wrong dimension")
            self._verify_basis(basis)
            return

        if certificate is RiemannRochCertificate.RATIONAL_PARAMETER:
            if parameter is None:
                raise ValueError("a rational-parameter certificate needs its parameter")
            pole_divisor = _rational_parameter_pole_divisor(self.model, parameter)
            if self.divisor.is_zero:
                exponent = 0
            else:
                place, _pole_order = pole_divisor.terms[0]
                exponent = self.divisor.coefficient(place)
                if exponent < 0 or self.divisor != pole_divisor.scale(exponent):
                    raise ValueError(
                        "the divisor is not a nonnegative parameter-pole multiple",
                    )
            expected_basis = tuple(parameter**power for power in range(exponent + 1))
            if basis != expected_basis:
                raise ValueError(
                    "a rational-parameter space must use consecutive parameter powers",
                )
            self._verify_basis(basis)
            return

        raise ValueError("the Riemann--Roch certificate has an unknown kind")

    def _verify_basis(self, basis: tuple[ExtensionElement, ...]) -> None:
        if not _basis_membership_is_verified(self.model, self.divisor, basis):
            raise ValueError("a proposed basis contains an element outside L(D)")
        if _constant_linear_rank(self.model, basis) != len(basis):
            raise ValueError("a proposed basis is constant-linearly dependent")


def _unsupported_space(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    detail: str,
) -> RiemannRochSpace:
    """Build the deliberately empty boundary result."""
    return RiemannRochSpace(
        model,
        divisor,
        constant_field_analysis,
        RiemannRochSpaceStatus.UNSUPPORTED,
        (),
        None,
        detail,
    )


def compute_riemann_roch_space(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    *,
    parameter: object | None = None,
    candidate_basis: tuple[object, ...] | None = None,
) -> RiemannRochSpace:
    r"""Construct a complete ``L(D)`` only in one proved first-scope case.

    ``constant_field_analysis`` must be the existing ``PRESERVED`` proof for
    the same direct algebraic ``C(x)`` presentation.  The function returns an
    explicit empty ``UNSUPPORTED`` result rather than a partial candidate list
    whenever its finite/infinite lattice-intersection algorithm is not yet
    implemented.

    Supplying ``candidate_basis`` enables the theorem-backed nonspecial and
    degree-zero-witness certificates.  It never bypasses validation: every
    element is tested at all places, and the basis is rechecked for linear
    independence over the constant field.  ``parameter`` enables the fully
    automatic rational-parameter case only when its pole divisor has degree
    one.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("Riemann--Roch construction needs an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("Riemann--Roch construction needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("Riemann--Roch construction needs constant-field analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant-field analysis belongs to another field")
    if parameter is not None and candidate_basis is not None:
        raise ValueError(
            "parameter and candidate_basis are alternative completeness inputs",
        )

    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_space(
            model,
            divisor,
            constant_field_analysis,
            "Riemann--Roch spaces require a certified preserved constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    candidates = (
        None
        if candidate_basis is None
        else tuple(
            _extension_element(model.extension, element) for element in candidate_basis
        )
    )

    if divisor.degree < 0:
        if candidates not in (None, ()) or parameter is not None:
            return _unsupported_space(
                model,
                divisor,
                constant_field_analysis,
                "a negative-degree divisor has L(D) = 0; supplied candidate data "
                "does not describe that space",
            )
        return RiemannRochSpace(
            model,
            divisor,
            constant_field_analysis,
            RiemannRochSpaceStatus.CERTIFIED,
            (),
            RiemannRochCertificate.NEGATIVE_DEGREE,
            "deg(D) < 0, so no nonzero function can have div(f) + D effective",
        )

    if divisor.is_zero:
        if parameter is not None:
            raise ValueError("a rational parameter is unnecessary for the zero divisor")
        basis = (model.extension.field.one,) if candidates is None else candidates
        return RiemannRochSpace(
            model,
            divisor,
            constant_field_analysis,
            RiemannRochSpaceStatus.CERTIFIED,
            basis,
            RiemannRochCertificate.ZERO_DIVISOR,
            "global regular functions on the complete curve are exactly constants",
        )

    if parameter is not None:
        try:
            parameter_element = _extension_element(model.extension, parameter)
            pole_divisor = _rational_parameter_pole_divisor(model, parameter_element)
        except (NotImplementedError, TypeError, ValueError) as exc:
            return _unsupported_space(
                model,
                divisor,
                constant_field_analysis,
                f"the proposed rational parameter is not certified: {exc}",
            )
        place, _pole_order = pole_divisor.terms[0]
        exponent = divisor.coefficient(place)
        if exponent >= 0 and divisor == pole_divisor.scale(exponent):
            basis = (
                tuple(parameter_element**power for power in range(exponent + 1))
                if candidates is None
                else candidates
            )
            return RiemannRochSpace(
                model,
                divisor,
                constant_field_analysis,
                RiemannRochSpaceStatus.CERTIFIED,
                basis,
                RiemannRochCertificate.RATIONAL_PARAMETER,
                "a degree-one pole parameter identifies the function field with C(t)",
                parameter_element,
            )
        return _unsupported_space(
            model,
            divisor,
            constant_field_analysis,
            "the divisor is not a nonnegative multiple of the parameter pole divisor",
        )

    if candidates is not None and divisor.degree == 0 and len(candidates) == 1:
        return RiemannRochSpace(
            model,
            divisor,
            constant_field_analysis,
            RiemannRochSpaceStatus.CERTIFIED,
            candidates,
            RiemannRochCertificate.DEGREE_ZERO_WITNESS,
            "the supplied degree-zero principal witness spans L(D)",
        )

    if candidates is not None:
        try:
            genus = model.genus
        except (NotImplementedError, TypeError, ValueError) as exc:
            return _unsupported_space(
                model,
                divisor,
                constant_field_analysis,
                f"the genus needed for the nonspecial theorem is unavailable: {exc}",
            )
        if divisor.degree > 2 * genus - 2:
            return RiemannRochSpace(
                model,
                divisor,
                constant_field_analysis,
                RiemannRochSpaceStatus.CERTIFIED,
                candidates,
                RiemannRochCertificate.NONSPECIAL,
                "Riemann--Roch gives l(D) = deg(D) + 1 - genus in this range",
            )

    return _unsupported_space(
        model,
        divisor,
        constant_field_analysis,
        "general finite/infinite lattice intersection for Riemann--Roch bases "
        "is not implemented yet",
    )
