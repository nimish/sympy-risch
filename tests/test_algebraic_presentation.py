from dataclasses import replace

import pytest
from sympy import Poly, resultant, symbols
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

import sympy_risch.algebraic.presentation as presentation_module
from sympy_risch.algebraic.presentation import normalize_primitive_element
from sympy_risch.tower import (
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
)

x, a, b, c, z = symbols("x a b c z")


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


def test_primitive_presentation_tries_another_linear_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (
        AlgebraicExtension(Poly(a**2 - 2, a, domain=QQ)),
        AlgebraicExtension(Poly(b**3 - 3, b, domain=QQ)),
    )
    attempts = 0
    original = presentation_module._try_primitive_target

    def reject_first_candidate(modulus: Poly) -> AlgebraicExtension | None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return None
        return original(modulus)

    monkeypatch.setattr(
        presentation_module,
        "_try_primitive_target",
        reject_first_candidate,
    )

    presentation = normalize_primitive_element(sources, z)

    assert attempts == 2
    assert presentation.primitive_coefficients == (QQ.one, QQ(2))
    assert presentation.target.field.generator == (
        presentation.source_images[0] + 2 * presentation.source_images[1]
    )
    eliminated_b = resultant(b**3 - 3, z - a - 2 * b, b)
    expected_modulus = Poly(
        resultant(a**2 - 2, eliminated_b, a),
        z,
        domain=QQ,
    ).monic()
    assert presentation.target.minimal_polynomial == expected_modulus


def test_primitive_presentation_allocates_only_two_full_square_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
        AlgebraicExtension(Poly(c**2 - x - 2, c, domain=parent)),
    )
    shapes: list[tuple[int, int]] = []
    original = DomainMatrix

    def recording_domain_matrix(rows, shape, domain):
        shapes.append(shape)
        return original(rows, shape, domain)

    monkeypatch.setattr(
        presentation_module,
        "DomainMatrix",
        recording_domain_matrix,
    )
    presentation = normalize_primitive_element(sources, z)

    assert presentation.tensor_rank == 8
    assert shapes.count((presentation.tensor_rank, presentation.tensor_rank)) == 2


def test_reducible_tensor_presentation_exhausts_the_theorem_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - 4 * x, b, domain=parent)),
    )

    calls = 0
    original = DomainMatrix.charpoly

    def counting_characteristic_polynomial(matrix: DomainMatrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(DomainMatrix, "charpoly", counting_characteristic_polynomial)
    with pytest.raises(NotImplementedError, match="tensor product is reducible"):
        normalize_primitive_element(sources, z)
    # There are at most binomial(4, 2) = 6 bad parameters if this rank-four
    # tensor product is a field, so seven exact failures prove that it is not.
    assert calls == 7


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


def test_primitive_presentation_rejects_a_parent_generator() -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
    )

    with pytest.raises(ValueError, match="already occurs in the parent field"):
        normalize_primitive_element(sources, x)
