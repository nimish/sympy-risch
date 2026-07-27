r"""Shared RDE certificate and polynomial helpers."""

from __future__ import annotations

from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement


def _field_element_is_zero(value: FieldElement, field: Domain) -> bool:
    """Test exact zero despite noncanonical Gaussian fraction-field units."""
    return value == field.zero or cancel(field.to_sympy(value)) == 0


def _field_elements_equal(
    left: FieldElement,
    right: FieldElement,
    field: Domain,
) -> bool:
    """Test exact field equality through the zero test for ``left - right``."""
    return _field_element_is_zero(left - right, field)


def _validate_decided_outcome(
    solution: object | None,
    obstruction: object | None,
    *,
    label: str,
) -> None:
    """Require exactly one of a solution and a proved obstruction."""
    if solution is not None and obstruction is not None:
        raise ValueError(f"{label} cannot contain both a solution and an obstruction")
    if solution is None and obstruction is None:
        raise ValueError(f"{label} needs either a solution or an obstruction")


def _validate_paired_outcome(
    real_solution: object | None,
    imaginary_solution: object | None,
    obstruction: object | None,
    *,
    label: str,
) -> None:
    """Require a complete real solution pair or one proved obstruction."""
    if (real_solution is None) != (imaginary_solution is None):
        raise ValueError(f"{label} must contain both solution coordinates or neither")
    _validate_decided_outcome(real_solution, obstruction, label=label)


def _validate_failed_division(
    polynomial: Poly | None,
    remainder: Poly | None,
    *,
    required: bool,
    label: str,
) -> None:
    """Validate optional exact-division obstruction data."""
    if (polynomial is None) != (remainder is None):
        raise ValueError(f"{label} needs both obstruction polynomial and remainder")
    if required and polynomial is None:
        raise ValueError(f"{label} needs a failed exact-division certificate")
    if remainder is not None and remainder.is_zero:
        raise ValueError(f"{label} obstruction remainder must be nonzero")


def _integer_constant(value: object, domain: Domain) -> int | None:
    """Return an integer represented in an exact constant domain, if any."""
    expression = domain.to_sympy(domain.convert(value))
    if expression.is_Integer is not True:
        return None
    return int(expression)


def _maximum_polynomial_degree(*polynomials: Poly) -> int | None:
    """Return the largest degree among nonzero polynomials, if one exists."""
    degrees = [
        polynomial.degree() for polynomial in polynomials if not polynomial.is_zero
    ]
    return max(degrees) if degrees else None
