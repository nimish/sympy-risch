from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from sympy_risch.algebraic.presentation import normalize_primitive_element
from sympy_risch.tower import (
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
)

x, a, b, z = symbols("x a b z")


def rational_function_field():
    return (
        DifferentialTower(QQ).extend(MonomialExtension(Poly(1, x, domain=QQ))).field()
    )


def test_primitive_presentation_recovers_independent_radicals() -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**3 - x, b, domain=parent)),
    )

    presentation = normalize_primitive_element(sources, z)

    assert presentation.tensor_rank == 6
    assert presentation.target.field.generator == sum(presentation.source_images)
    embedded_x = presentation.target.from_power_basis_coordinates(
        (
            parent.from_sympy(x),
            *(parent.zero for _ in range(presentation.tensor_rank - 1)),
        )
    )
    assert presentation.source_images[0] ** 2 == embedded_x
    assert presentation.source_images[1] ** 3 == embedded_x


def test_primitive_presentation_uses_one_global_characteristic_polynomial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
    )

    calls = 0
    original = DomainMatrix.charpoly

    def counting_characteristic_polynomial(matrix: DomainMatrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(DomainMatrix, "charpoly", counting_characteristic_polynomial)
    presentation = normalize_primitive_element(sources, z)

    assert presentation.tensor_rank == 4
    assert calls == 1


def test_reducible_tensor_presentation_stops_after_one_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x, b, domain=parent)),
    )

    calls = 0
    original = DomainMatrix.charpoly

    def counting_characteristic_polynomial(matrix: DomainMatrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(DomainMatrix, "charpoly", counting_characteristic_polynomial)
    with pytest.raises(NotImplementedError, match="does not certify"):
        normalize_primitive_element(sources, z)
    assert calls == 1


def test_primitive_presentation_rejects_a_forged_source_image() -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
    )
    presentation = normalize_primitive_element(sources, z)

    with pytest.raises(ValueError, match="defining relation"):
        replace(
            presentation,
            source_images=(
                presentation.target.field.one,
                presentation.source_images[1],
            ),
        )


def test_primitive_presentation_rejects_a_foreign_parent() -> None:
    parent = rational_function_field()
    other_parent = QQ.frac_field(symbols("u"))
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - 2, b, domain=other_parent)),
    )

    with pytest.raises(ValueError, match="same parent field"):
        normalize_primitive_element(sources, z)
