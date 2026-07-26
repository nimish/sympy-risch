import sympy_risch
from sympy_risch import __version__


def test_package_import() -> None:
    assert __version__ == "0.1.0"


def test_public_exports_are_defined_once() -> None:
    assert len(sympy_risch.__all__) == len(set(sympy_risch.__all__))
    assert all(hasattr(sympy_risch, name) for name in sympy_risch.__all__)


def test_admission_exports_are_public() -> None:
    expected = {
        "TowerAdmission",
        "TowerAdmissionOutcome",
        "TowerAdmissionReason",
        "assess_elementary_admission",
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
