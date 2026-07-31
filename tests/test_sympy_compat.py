"""Temporary compatibility behavior needed from released SymPy."""

from sympy import sqrt
from sympy.polys.domains import QQ

from sympy_risch._sympy_compat import install_field_no_ring_fast_path


def test_algebraic_field_gcd_and_lcm_do_not_request_a_ring() -> None:
    install_field_no_ring_fast_path()
    constants = QQ.algebraic_field(sqrt(2))

    def unexpected_get_ring():
        raise AssertionError("field gcd/lcm should not request an unavailable ring")

    constants.get_ring = unexpected_get_ring
    generator = constants.unit

    assert constants.gcd(generator, constants.one) == constants.one
    assert constants.lcm(generator, constants.one) == generator


def test_fields_with_an_associated_ring_keep_denominator_content() -> None:
    assert QQ.gcd(QQ(2, 3), QQ(4, 9)) == QQ(2, 9)
    assert QQ.lcm(QQ(2, 3), QQ(4, 9)) == QQ(4, 3)
