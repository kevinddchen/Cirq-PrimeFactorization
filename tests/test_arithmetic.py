from __future__ import annotations

import logging

import cirq
import numpy as np

from factor import Add, MAdd, MMult, Ua, MExp, utils

TEST_RNG = np.random.default_rng(seed=2022)
TEST_N_TRIALS = 3


def _eval(
  gate: cirq.Gate, 
  register_lengths: list[int], 
  inputs: list[int], 
  expected_outputs: list[int],
):
  """ Evaluate a gate on a given input and check the output. 
  
  Args:
    gate: The gate to evaluate.
    register_lengths: The number of bits in each register.
    inputs: The input values to each register
    expected_outputs: The expected output values for each register.

  Raises:
    AssertionError: If the output does not match the expected output.
  """

  # check lengths
  assert len(register_lengths) == len(inputs) == len(expected_outputs)
  # declare qubits
  total_qubits = sum(register_lengths)
  qubits = cirq.LineQubit.range(total_qubits)
  # initialize qubits
  circuit = cirq.Circuit()
  cum_n = 0
  for i, n in enumerate(register_lengths):
    circuit.append(utils.prepare_state(qubits[cum_n:cum_n + n], inputs[i]))
    cum_n += n
  # initialize circuit
  circuit.append(gate(*qubits))
  circuit.append(cirq.measure(*qubits))
  # perform one measurement
  result = cirq.Simulator().run(circuit, repetitions=1)
  _, raw_output = result.measurements.popitem()
  # validate output
  cum_n = 0
  for i, n in enumerate(register_lengths):
    output_i = utils.bits_to_integer(raw_output[0, cum_n:cum_n + n])
    assert output_i == expected_outputs[i]
    cum_n += n


def test_add():
  for n in range(1, 9):
    for _ in range(TEST_N_TRIALS):
      a = int(TEST_RNG.integers(2 ** n))
      b = int(TEST_RNG.integers(2 ** n))

      logging.info(f"Add - {n} bits - {a} + {b}")

      _eval(
        gate=Add(n, a), 
        register_lengths=[n+1, n], 
        inputs=[b, 0], 
        expected_outputs=[b+a, 0])


def test_madd():
  for n in range(2, 9):
    for _ in range(TEST_N_TRIALS):
      N = int(TEST_RNG.integers(2 ** (n-1), 2 ** n))
      a = int(TEST_RNG.integers(N))
      b = int(TEST_RNG.integers(N))

      logging.info(f"MAdd - {n} bits - ({a} + {b}) % {N}")

      _eval(
        gate=MAdd(n, a, N),
        register_lengths=[n, n+2],
        inputs=[b, 0],
        expected_outputs=[(b + a) % N, 0])


def test_mmult():
  for n in range(2, 5):
    for _ in range(TEST_N_TRIALS):
      N = TEST_RNG.integers(2 ** (n-1), 2 ** n)
      a = TEST_RNG.integers(N)
      b = TEST_RNG.integers(N)
      x = TEST_RNG.integers(N)

      logging.info(f"MMult - {n} bits - ({b} + {x} * {a}) % {N}")

      _eval(
        gate=MMult(n, a, N),
        register_lengths=[n, n, n+2],
        inputs=[x, b, 0],
        expected_outputs=[x, (b + x * a) % N, 0])


def test_ua():
  for n in range(2, 5):
    for _ in range(TEST_N_TRIALS):
      N = max(3, int(TEST_RNG.integers(2 ** (n-1), 2 ** n)))
      a = N
      while np.gcd(a, N) != 1:
        a = int(TEST_RNG.integers(2, N))
      x = int(TEST_RNG.integers(N))

      logging.info(f"Ua - {n} bits - ({x} * {a}) % {N}")

      _eval(
        gate=Ua(n, a, N),
        register_lengths=[n, 2*n+2],
        inputs=[x, 0],
        expected_outputs=[(x * a) % N, 0])


def test_mexp():
  for n in range(2, 5):
    m = n
    for _ in range(TEST_N_TRIALS):
      N = max(3, int(TEST_RNG.integers(2 ** (n-1), 2 ** n)))
      a = N
      while np.gcd(a, N) != 1:
        a = int(TEST_RNG.integers(2, N))
      x = int(TEST_RNG.integers(N))
      k = int(TEST_RNG.integers(2 ** m))

      logging.info(f"MExp - {n} bits - ({x} * {a} ^ {k}) % {N}")

      _eval(
        gate=MExp(m, n, a, N),
        register_lengths=[m, n, 2*n+2],
        inputs=[k, x, 0],
        expected_outputs=[k, (x*pow(a, k, N))%N, 0])
