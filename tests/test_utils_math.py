import pytest

from factor import modular_inverse


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
