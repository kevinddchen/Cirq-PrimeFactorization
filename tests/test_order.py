import logging

import pytest
import numpy as np

from factor import QuantumOrderFinder, FakeQuantumOrderFinder

TEST_RNG = np.random.default_rng(seed=2022)
TEST_N_TRIALS = 10


def test_quantum_order_finder():
    order_finder = QuantumOrderFinder(3, 7)
    assert order_finder.find() == 6


def test_fake_quantum_order_finder():
    order_finder = FakeQuantumOrderFinder(3, 7)
    assert order_finder.find() == 6


@pytest.mark.skip(reason="Too slow")
def test_order_finders():
    for _ in range(TEST_N_TRIALS):
        N = int(np.random.randint(3, 2**4))
        a = N
        while np.gcd(a, N) != 1:
            a = int(np.random.randint(2, N))

        logging.info(f"Finding order of {a} mod {N}")

        assert FakeQuantumOrderFinder(a, N).find() == QuantumOrderFinder(a, N).find()
