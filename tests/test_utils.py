import cirq
import pytest

from factor import size_in_bits, prepare_state, bits_to_integer, integer_to_bits


def test_size_in_bits():
    assert size_in_bits(0) == 1
    assert size_in_bits(1) == 1
    assert size_in_bits(2) == 2
    assert size_in_bits(3) == 2
    assert size_in_bits(4) == 3
    assert size_in_bits(255) == 8
    assert size_in_bits(256) == 9
    with pytest.raises(ValueError):
        size_in_bits(-1)


def test_prepare_state():
    qubits = cirq.GridQubit.rect(1, 8)
    assert prepare_state(qubits=qubits, x=0) == list()
    assert prepare_state(qubits=qubits, x=1) == [cirq.X(qubits[0])]
    assert prepare_state(qubits=qubits, x=2) == [cirq.X(qubits[1])]
    assert prepare_state(qubits=qubits, x=3) == [cirq.X(qubits[0]), cirq.X(qubits[1])]
    assert prepare_state(qubits=qubits, x=4) == [cirq.X(qubits[2])]
    assert prepare_state(qubits=qubits, x=255) == [cirq.X(qubits[i]) for i in range(8)]
    with pytest.raises(ValueError):
        prepare_state(qubits=qubits, x=-1)


def test_bits_to_integer():
    assert bits_to_integer([]) == 0
    assert bits_to_integer([0]) == 0
    assert bits_to_integer([0, 0]) == 0
    assert bits_to_integer([1]) == 1
    assert bits_to_integer([1, 0]) == 1
    assert bits_to_integer([0, 1]) == 2
    assert bits_to_integer([1, 1]) == 3
    assert bits_to_integer([1, 0, 1, 0, 1]) == 21


def test_integer_to_bits():
    assert integer_to_bits(n_bits=1, x=0) == [0]
    assert integer_to_bits(n_bits=2, x=0) == [0, 0]
    assert integer_to_bits(n_bits=1, x=1) == [1]
    assert integer_to_bits(n_bits=2, x=1) == [1, 0]
    assert integer_to_bits(n_bits=2, x=2) == [0, 1]
    assert integer_to_bits(n_bits=2, x=3) == [1, 1]
    assert integer_to_bits(n_bits=5, x=21) == [1, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        integer_to_bits(n_bits=0, x=-1)
