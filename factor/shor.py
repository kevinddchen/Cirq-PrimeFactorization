import numpy as np

from typing import Type

from factor import OrderFinder, QuantumOrderFinder, is_prime


def shor(
    N: int,
    order_finder: Type[OrderFinder] = QuantumOrderFinder,
    max_iters_shor: int = 100,
    **kwargs,
) -> int:
    """Given an integer N > 1, return a non-trivial factor of N, or 1 if N is a
    (probable) prime.

    Args:
        N: The number to find a factor of.
        order_finder: The class to use for finding the order of the factor.
            Use `FakeQuantumOrderFinder` for testing.
        max_iters_shor: The maximum number of iterations to run.
        kwargs: Passed to `order_finder`.

    Raises:
        RuntimeError: if a factor was not found after `max_iters` iterations.
    """

    # check that N is odd
    if N % 2 == 0:
        return 2

    # check that N is composite
    if is_prime(N):
        return 1

    # check that N is not an integer power
    for k in range(2, int(np.log(N) / np.log(3)) + 1):
        x = round(pow(N, 1.0 / k))
        if pow(x, k) == N:
            return x

    for _ in range(max_iters_shor):
        a = np.random.randint(2, N)
        d = np.gcd(a, N)
        if d != 1:
            return d  # got lucky!

        # find period of a modulo N
        r = order_finder(a, N, **kwargs).find()

        if r % 2 == 0:
            d = np.gcd(pow(a, r // 2, N) - 1, N)
            if d != 1:
                return d

    msg = f"shor: failed to find factor in {max_iters_shor} iterations."
    raise RuntimeError(msg)
