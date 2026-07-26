from importlib.metadata import version
from typing import get_type_hints

import sympy_risch
import sympy_risch.algebraic as algebraic
import sympy_risch.algebraic_geometry as algebraic_geometry
from sympy_risch import __version__


def test_package_import() -> None:
    assert __version__ == version("sympy-risch")


def test_public_exports_are_defined_once() -> None:
    assert len(sympy_risch.__all__) == len(set(sympy_risch.__all__))
    assert all(hasattr(sympy_risch, name) for name in sympy_risch.__all__)


def test_algebraic_facade_has_an_intentional_advanced_surface() -> None:
    assert set(algebraic.__all__) == {
        "AlgebraicAllPlaceResidueResult",
        "AlgebraicAllPlaceResidueStatus",
        "AlgebraicConstantEmbedding",
        "AlgebraicConstantFieldAnalysis",
        "AlgebraicConstantFieldStatus",
        "AlgebraicFraction",
        "AlgebraicLocalDifferentialResidue",
        "AlgebraicPlaceResidue",
        "IntegralBasis",
        "SimpleRadicalData",
        "TragerAlgebraicResidueResult",
        "analyze_algebraic_constant_field",
        "collect_algebraic_all_place_residues",
        "hermite_reduce_algebraic",
        "lazy_hermite_reduce_algebraic",
        "recognize_simple_radical",
        "simple_radical_integral_basis",
        "trager_algebraic_place_residues",
        "trager_algebraic_residue_polynomial",
    }
    assert all(hasattr(algebraic, name) for name in algebraic.__all__)
    assert not hasattr(algebraic, "UnramifiedFinitePlace")
    assert not hasattr(algebraic, "unramified_finite_places")


def test_algebraic_geometry_facade_has_an_intentional_advanced_surface() -> None:
    assert set(algebraic_geometry.__all__) == {
        "AlgebraicDifferential",
        "AlgebraicDivisor",
        "AlgebraicFractionalIdeal",
        "AlgebraicFunctionField",
        "AlgebraicOrder",
        "FiniteAlgebraicPlace",
        "InfiniteAlgebraicPlace",
        "IntegralClosureResult",
        "LocalLaurentExpansion",
        "integral_closure",
        "polynomial_order",
    }
    assert all(hasattr(algebraic_geometry, name) for name in algebraic_geometry.__all__)
    assert not hasattr(algebraic_geometry, "AlgebraicPlace")
    assert not hasattr(algebraic_geometry, "IntegralClosureStep")
    assert not hasattr(algebraic_geometry, "OrderMaximalityCertificate")
    assert not hasattr(algebraic_geometry, "ReciprocalFunctionField")


def test_algebraic_geometry_public_annotations_are_runtime_resolvable() -> None:
    for name in algebraic_geometry.__all__:
        value = getattr(algebraic_geometry, name)
        if isinstance(value, type):
            get_type_hints(value)


def test_admission_exports_are_public() -> None:
    expected = {
        "TowerAdmission",
        "TowerAdmissionOutcome",
        "TowerAdmissionReason",
        "assess_elementary_admission",
        "assess_scalar_rde_admission",
    }

    assert expected <= set(sympy_risch.__all__)


def test_expression_frontend_exports_are_public() -> None:
    expected = {
        "ExpressionCompilation",
        "ExpressionCompilationOutcome",
        "ExpressionIntegrationOutcome",
        "ExpressionIntegralResult",
        "ExpressionUnsupportedReason",
        "compile_expression",
        "integrate_expression",
    }

    assert expected <= set(sympy_risch.__all__)


def test_riccati_mobius_exports_are_public() -> None:
    expected = {
        "RiccatiMobiusElementaryCertificate",
        "RiccatiMobiusNormalForm",
        "build_riccati_mobius_normal_form",
        "integrate_riccati_mobius_normal_form",
    }

    assert expected <= set(sympy_risch.__all__)


def test_certified_special_pole_exports_are_public() -> None:
    expected = {
        "LinearSpecialPoleReduction",
        "LinearSpecialPoleReductionStatus",
        "LinearSpecialPoleStep",
        "reduce_certified_linear_special_poles",
    }

    assert expected <= set(sympy_risch.__all__)
