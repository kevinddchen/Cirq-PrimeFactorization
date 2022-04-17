from __future__ import annotations

import logging
from functools import cached_property

import cirq
import numpy as np

from factor import MExp, approximate_fraction, size_in_bits, prepare_state, bits_to_integer


class OrderFinder(object):
    def __init__(self, a: int, N: int):
        """Class that finds the multiplicative order of `a` modulo `N`.

        Args:
            a: The number to find the order of. Must satisfy 0 < a < N and
                gcd(a, N) = 1.
            N: The modulus. Must be > 1.

        Raises:
            ValueError: if any of the conditions above are not met.
        """
        if not (0 < a and a < N and np.gcd(a, N) == 1):
            msg = f"OrderFinder: invalid arguments a={a} and N={N}."
            raise ValueError(msg)
        self.a = a
        self.N = N

    @cached_property
    def _order(self) -> int:
        """Find the order by brute force. Runs in O(N) time."""
        d, r = self.a, 1
        while d != 1:
            d, r = (d * self.a) % self.N, r + 1
        return r

    def find(self) -> int:
        """Find the order."""
        return self._order


class QuantumOrderFinder(OrderFinder):
    def __init__(
        self,
        a: int,
        N: int,
        n_qubits: int | None = None,
        threshold: int = 2,
        max_iters: int = 100,
    ):
        """Order finding in O((log N)^3) by quantum phase estimation.

        Let `r` denote the order we are trying to find. The circuit is run
        multiple times to generate candidate measurements for the order. To
        avoid mistaking measurements of 2*r, 3*r, ... for the actual order, any
        candidate must be measured at least `threshold` times before we
        verify that it is the order. Having `threshold=2` should be sufficient.

        Args:
            a: The number to find the order of. Must satisfy 0 < a < N and
                gcd(a, N) = 1.
            N: The modulus. Must be > 1.
            n_qubits: The number of qubits to use. If None, will be inferred.
            threshold: The number of times a candidate must be measured before
                it is verified.
            max_iters: The maximum number of times to run the circuit before
                raising a `RuntimeError`.

        Raises:
            RuntimeError: if the order was not found after `max_iters`
                iterations.
        """
        super().__init__(a, N)
        self.threshold = threshold
        self.max_iters = max_iters
        self.n = n_qubits or size_in_bits(N)
        self.m = 2 * self.n
        self.candidates = dict()  # tracks candidate orders and their counts

        # Prepare quantum circuit that performs phase estimation on MExp.
        k = cirq.GridQubit.rect(1, self.m, top=0)
        x = cirq.GridQubit.rect(1, self.n, top=1)
        anc = cirq.GridQubit.rect(1, 2 * self.n + 2, top=2)
        self.circuit = cirq.Circuit()
        self.circuit.append(prepare_state(x, 1))
        self.circuit.append(cirq.H(ki) for ki in k)
        self.circuit.append(MExp(self.m, self.n, a, N).on(*k, *x, *anc))
        self.circuit.append(cirq.qft(*k[::-1], inverse=True))
        self.circuit.append(cirq.measure(*k))

    def _sample(self) -> int:
        """Runs quantum circuit once and returns a measurement."""
        result = cirq.Simulator().run(self.circuit, repetitions=1)
        _, raw_output = result.measurements.popitem()
        bits = [int(x) for x in raw_output[0]]  # convert all to ints
        return bits_to_integer(bits)

    def find(self) -> int:
        """Find the order."""
        logging.debug(f"Finding order of {self.a} modulo {self.N} ...")
        logging.debug(f"Running on {self.n} qubits ...")
        logging.debug(f"- threshold={self.threshold}")
        logging.debug(f"- max_iters={self.max_iters}")

        for iter in range(self.max_iters):

            j = self._sample()  # j drawn from [0, 2^m)
            _, q = approximate_fraction(j, 2**self.m, self.N)  # _/q drawn uniformly from {0, 1/r, 2/r, ..., (r-1)/r}
            logging.debug(f"\tMeasured candidate order q={q}")

            self.candidates[q] = self.candidates.get(q, 0) + 1
            # if a q is observed `threshold` times, check if it is the order.
            if self.candidates[q] == self.threshold:
                if pow(self.a, q, self.N) == 1:
                    logging.debug(f"Found order r={q} in {iter+1} iterations.")
                    return q
                else:
                    logging.debug(f"\tq={q} is not the order. Continuing ...")

        msg = f"QuantumOrderFinder: failed to find order in {self.max_iters} iterations."
        raise RuntimeError(msg)


class FakeQuantumOrderFinder(QuantumOrderFinder):
    """Since the output distribution for the function `_sample()` is
    theoretically known, we can avoid simulating the quantum circuit by
    directly sampling the distribution. This is fast for small inputs on a
    classical computer and allows us to quickly debug the other classical parts
    of the order-finding algorithm.

    It is important to emphasize that the runtime complexity of this method is
    O(N) and it does not scale well to large inputs. This is because the
    formulating the distribution requires knowing the order in the first place.
    So we need to first compute the order by brute force.
    """

    def _distribution(self, M: int, x: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
        """f(x) = sin^2(M * x) / [M * sin(x)]^2."""
        x = np.maximum(np.abs(x), eps)  # regularize behavior at x=0
        return (np.sin(M * x) ** 2) / ((M * np.sin(x)) ** 2)

    def _sample(self):
        """Sample directly from known distribution."""
        logging.debug("\tSampling from known distribution ...")
        M = 2**self.m
        k = np.random.randint(self._order)
        deltas = np.pi * (k * 1.0 / self._order - np.linspace(0, 1, M, endpoint=False))
        probs = self._distribution(M, deltas)
        return np.random.choice(M, p=probs)
