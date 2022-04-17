import pytest

from factor import modular_inverse, continued_fraction, approximate_fraction


def test_modular_inverse():
    assert modular_inverse(a=1, N=2) == 1
    assert modular_inverse(a=1, N=5) == 1
    assert modular_inverse(a=2, N=5) == 3
    assert modular_inverse(a=3, N=5) == 2
    assert modular_inverse(a=4, N=5) == 4
    with pytest.raises(ValueError):
        modular_inverse(a=1, N=1)
    with pytest.raises(ValueError):
        modular_inverse(a=1, N=0)
    with pytest.raises(ValueError):
        modular_inverse(a=5, N=5)


def test_continued_fraction():
    assert continued_fraction(0, 1) == [0]
    assert continued_fraction(1, 1) == [1]
    assert continued_fraction(2, 1) == [2]
    assert continued_fraction(2, 3) == [0, 1, 2]
    assert continued_fraction(7, 13) == [0, 1, 1, 6]
    assert continued_fraction(100, 41) == [2, 2, 3, 1, 1, 2]


def test_approximate_fraction():
    assert approximate_fraction(0, 1, 2) == (0, 1)
    assert approximate_fraction(1, 1, 2) == (1, 1)
    assert approximate_fraction(0, 13, 7) == (0, 1)
    assert approximate_fraction(1, 13, 7) == (1, 7)
    assert approximate_fraction(2, 13, 7) == (1, 7)
    assert approximate_fraction(3, 13, 7) == (1, 4)
    assert approximate_fraction(4, 13, 7) == (2, 7)
    assert approximate_fraction(5, 13, 7) == (2, 5)
    assert approximate_fraction(6, 13, 7) == (3, 7)
    assert approximate_fraction(7, 13, 7) == (4, 7)
    assert approximate_fraction(8, 13, 7) == (3, 5)
    assert approximate_fraction(9, 13, 7) == (5, 7)
    assert approximate_fraction(10, 13, 7) == (3, 4)
    assert approximate_fraction(11, 13, 7) == (6, 7)
    assert approximate_fraction(12, 13, 7) == (6, 7)
    assert approximate_fraction(13, 13, 7) == (1, 1)
