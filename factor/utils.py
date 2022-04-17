from __future__ import annotations

import logging

import cirq
import numpy as np

# Note: an integer a = 2^(n-1) a_(n-1) + ... + 2 a_1 + a_0 is represented
# by n bits with the convention that the ith bit is a_i.


def size_in_bits(x: int) -> int:
    """Return the minimum number of bits required to represent an integer.

    Args:
        x: The integer to represent. Must be non-negative.

    Returns:
        The number of bits required to represent `x`.

    Raises:
        ValueError: If `x` is negative.
    """
    if x < 0:
        msg = f"size_in_bits: `x` ({x}) must be non-negative."
        raise ValueError(msg)
    elif x == 0:
        return 1
    else:
        return int(np.log2(x)) + 1


def prepare_state(qubits: list[cirq.Qid], x: int) -> list[cirq.Gate]:
    """Prepare qubits into an initial state.

    Args:
        qubits: The qubits to prepare.
        x: The initial state of the qubits. Must be non-negative.

    Returns:
        A list of gates to prepare the qubits.

    Raises:
        ValueError: If `x` is negative.
    """
    gates = list()
    if size_in_bits(x) > len(qubits):
        logging.warning(f"prepare_state: `x` ({x}) cannot fit into {len(qubits)} qubits; some bits will be dropped.")
    for q in qubits:
        if x % 2:
            gates.append(cirq.X(q))
        x >>= 1
    return gates


def bits_to_integer(bits: list[int]) -> int:
    """Return the integer representation of a string of bits.

    Args:
        bits: The bits to convert.

    Returns:
        The integer representation of the bits.
    """
    x = 0
    for b in bits[::-1]:
        x <<= 1
        x += b
    return x


def integer_to_bits(n_bits: int, x: int) -> list[int]:
    """Return the representation of an integer as a string of `n` bits.

    Args:
        n_bits: The number of bits used to represent the integer.
        x: The integer to convert. Must be non-negative.

    Returns:
        The bits representing `x`.

    Raises:
        ValueError: If `x` is negative.
    """
    bits = list()
    if size_in_bits(x) > n_bits:
        logging.warning(f"integer_to_bits: `x` ({x}) cannot fit into {n_bits} bits; some bits will be dropped.")
    for _ in range(n_bits):
        bits.append(x % 2)
        x >>= 1
    return bits
