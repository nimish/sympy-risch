from importlib.metadata import version

import sympy_risch
import sympy_risch.algebraic as algebraic
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
