from __future__ import annotations


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
