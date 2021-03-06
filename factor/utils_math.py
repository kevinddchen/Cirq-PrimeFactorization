from __future__ import annotations

import numpy as np


def modular_inverse(a: int, N: int) -> int:
    """Return the inverse of `a` modulo `N`.

    Args:
        a: The number to invert.
        N: The modulus. Must be > 1.

    Returns:
        The inverse of `a` modulo `N`.

    Raises:
        ValueError: if `N` <= 1 or the inverse does not exist.
    """
    if not N > 1:
        msg = f"modular_inverse: `N` ({N}) must be > 1."
        raise ValueError(msg)
    try:
        return pow(a, -1, N)
    except ValueError:
        msg = f"modular_inverse: invalid arguments a={a} and N={N}."
        raise ValueError(msg)


def continued_fraction(p: int, q: int) -> list[int]:
    """Given p/q, return its continued fraction.

    Args:
        p: The numerator.
        q: The denominator.

    Returns:
        The continued fraction expansion of p/q, [a0; a1, a2, ..., an].
    """
    coeffs = list()
    while q != 0:
        a = p // q
        coeffs.append(a)
        p, q = q, p - q * a
    return coeffs


def approximate_fraction(p: int, q: int, N: int) -> tuple[int, int]:
    """Given p/q, find the closest fraction a/b where b <= N.

    Args:
        p: The numerator.
        q: The denominator.

    Returns:
        The tuple (a, b).
    """
    # truncate sequence of continued fraction approximations when denominator >= N
    a1, a2 = 1, 0
    b1, b2 = 0, 1
    truncated = False
    for k in continued_fraction(p, q):
        if k * b1 + b2 >= N:
            truncated = True
            break
        a1, a2 = k * a1 + a2, a1
        b1, b2 = k * b1 + b2, b1

    if truncated:
        # use largest j where k/2 <= j < k and j*b1 + b2 < N.
        j = (N - b2) // b1
        if j >= k:
            pass
        elif k < 2 * j:  # found good j
            a1 = j * a1 + a2
            b1 = j * b1 + b2
        elif k == 2 * j:  # if k even, j = k/2 only admissible if the approximation is better
            next_a = j * a1 + a2
            next_b = j * b1 + b2
            if abs((p * 1.0) / q - (next_a * 1.0) / next_b) < abs((p * 1.0) / q - (a1 * 1.0) / b1):
                a1, b1 = next_a, next_b

    # else, no better approximation
    return a1, b1


def is_prime(n: int, num_iters: int = 40) -> bool:
    """Returns True if n is a probable prime. Implementation uses a
    Miller-Rabin primality test modified from
    https://gist.github.com/Ayrx/5884790.

    Args:
        n: The number to test.
        num_iters: The number of iterations to run.

    Returns:
        True if n is a probable prime.
    """

    if n == 2 or n == 3:
        return True

    if n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(num_iters):
        a = np.random.randint(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
