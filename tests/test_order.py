import pytest
import numpy as np

from factor import OrderFinder, QuantumOrderFinder, FakeQuantumOrderFinder

TEST_RNG = np.random.default_rng(seed=2022)
TEST_N_TRIALS = 3


def test_order_finder():
    assert OrderFinder(1, 5).find() == 1
    assert OrderFinder(2, 5).find() == 4
    assert OrderFinder(3, 5).find() == 4
    assert OrderFinder(4, 5).find() == 2
    # Check exceptions raised on invalid arguments.
    with pytest.raises(ValueError):
        OrderFinder(5, 5)
    with pytest.raises(ValueError):
        OrderFinder(-1, 5)
    with pytest.raises(ValueError):
        OrderFinder(1, 1)


def test_quantum_order_finder():
    assert QuantumOrderFinder(2, 5).find() == 4
    # Check timeout.
    with pytest.raises(RuntimeError):
        QuantumOrderFinder(2, 5, max_iters=1).find()


def test_fake_quantum_order_finder():
    assert FakeQuantumOrderFinder(2, 5).find() == 4
    assert FakeQuantumOrderFinder(7, 13).find() == 12
    # Check other arguments
    assert FakeQuantumOrderFinder(7, 13, n_qubits=8).find() == 12
    assert FakeQuantumOrderFinder(7, 13, threshold=1).find() == 12


def test_fake_quantum_order_finder_multiple():
    for n in range(2, 8):
        for _ in range(TEST_N_TRIALS):
            N = max(3, int(TEST_RNG.integers(2 ** (n - 1), 2**n)))
            a = N
            while np.gcd(a, N) != 1:
                a = int(TEST_RNG.integers(2, N))

            order_finder = FakeQuantumOrderFinder(a, N)
            assert order_finder.find() == order_finder._order
