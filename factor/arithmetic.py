from __future__ import annotations

import cirq

from factor import integer_to_bits, modular_inverse


# Quantum gates for arithmetic. Implementations are based on
# https://arxiv.org/abs/quant-ph/9511018.

# Note: an integer a = 2^(n-1) a_(n-1) + ... + 2 a_1 + a_0 is represented
# by n bits with the convention that the ith bit is a_i.


class Add(cirq.Gate):
    def __init__(self, n: int, a: int):
        """Add classical integer a to qubit b. Uses O(n) elementary gates.

        |b> --> |b+a>

        The input to the gate is 2n+1 qubits split into two registers:
        - n+1 qubits for b, 0 <= b < 2 ** n. The most significant digit must be
          initialized to 0. b+a is saved here.
        - n qubits initialized to 0. These are unchanged by the gate.

        Args:
            n: number of qubits.
            a: 0 <= a < 2 ** n.
        """
        super().__init__()
        self.n = n
        self.a_bits = integer_to_bits(n_bits=n, x=a)

    def _num_qubits_(self):
        return 2 * self.n + 1

    def _circuit_diagram_info_(self, args):
        return ["Add_b"] * (self.n + 1) + ["Add_anc"] * self.n

    def _decompose_(self, qubits):
        b = qubits[: self.n]
        anc = qubits[self.n + 1 :] + (qubits[self.n],)  # internally, b[n] is placed in anc[n]

        # In forward pass, store carried bits in ancilla.
        for i in range(self.n):
            if self.a_bits[i]:
                yield cirq.CNOT(b[i], anc[i + 1])
                yield cirq.X(b[i])
            yield cirq.TOFFOLI(anc[i], b[i], anc[i + 1])
        yield cirq.CNOT(anc[self.n - 1], b[self.n - 1])
        # In backward pass, undo carries, then add a and carries to b.
        for i in range(self.n - 2, -1, -1):
            yield cirq.TOFFOLI(anc[i], b[i], anc[i + 1])
            if self.a_bits[i]:
                yield cirq.X(b[i])
                yield cirq.CNOT(b[i], anc[i + 1])
                yield cirq.X(b[i])
            yield cirq.CNOT(anc[i], b[i])


class MAdd(cirq.Gate):
    def __init__(self, n: int, a: int, N: int):
        """Add classical integer a to qubit b, modulo N. Integers a and b must
        be less than N for correct behavior. Uses O(n) elementary gates.

        |b> --> |b+a mod N>

        Input to the gate is 2n+2 qubits split into two registers:
        - n qubits for b, 0 <= b < N. a+b mod N is saved here.
        - n+2 qubits initialized to 0. These are unchanged by the gate.

        Args:
            n: number of qubits.
            a: 0 <= a < N.
            N: 1 < N < 2 ** n.
        """
        super().__init__()
        self.n = n
        self.a = a
        self.N = N

    def _num_qubits_(self):
        return 2 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MAdd_b"] * self.n + ["MAdd_anc"] * (self.n + 2)

    def _decompose_(self, qubits):
        b = qubits[: self.n + 1]  # extra qubit for overflow
        anc = qubits[self.n + 1 : 2 * self.n + 1]
        t = qubits[2 * self.n + 1]

        Add_a = Add(self.n, self.a)
        Add_N = Add(self.n, self.N)
        yield Add_a.on(*b, *anc)
        yield cirq.inverse(Add_N).on(*b, *anc)
        # Second register is a+b-N. The most significant digit indicates underflow from subtraction.
        yield cirq.CNOT(b[self.n], t)
        yield Add_N.controlled(1).on(t, *b, *anc)
        # To reset t, subtract a from second register. If underflow again, means that t=0 previously.
        yield cirq.inverse(Add_a).on(*b, *anc)
        yield cirq.X(b[self.n])
        yield cirq.CNOT(b[self.n], t)
        yield cirq.X(b[self.n])
        yield Add_a.on(*b, *anc)


class MMult(cirq.Gate):
    def __init__(self, n: int, a: int, N: int):
        """Multiply qubit x by classical integer a, modulo N. Integers a, b,
        and x must be less than N for correct behavior. Uses O(n^2)
        elementary gates.

        |x; b> --> |x; b + x*a mod N>

        Input to the gate is 3n+2 qubits split into three registers:
        - n qubits for x, 0 <= x < N. These are unchanged by the gate.
        - n qubits for b, 0 <= b < N. b + x*a mod N is saved here.
        - n+2 qubits initialized to 0. These are unchanged by the gate.

        Args:
            n: number of qubits.
            a: 0 <= a < N.
            N: 1 < N < 2 ** n.
        """
        super().__init__()
        self.n = n
        self.a = a
        self.N = N

    def _num_qubits_(self):
        return 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MMult_x"] * self.n + ["MMult_b"] * self.n + ["MMult_anc"] * (self.n + 2)

    def _decompose_(self, qubits):
        x = qubits[: self.n]
        b = qubits[self.n : 2 * self.n]
        anc = qubits[2 * self.n :]

        # x*a = 2^(n-1) x_(n-1) a + ... + 2 x_1 a + x_0 a
        # so the bits of x control the addition of a * 2^i
        d = self.a  # stores a * 2^i mod N
        for i in range(self.n):
            yield MAdd(self.n, d, self.N).controlled(1).on(x[i], *b, *anc)
            d = (d << 1) % self.N


class Ua(cirq.Gate):
    def __init__(self, n: int, a: int, N: int, inv_a: int | None = None):
        """Multiply qubit x by classical integer a, modulo N, where
        gcd(a, N) = 1. Similar to `MMult`, but acts on the x qubits directly.
        Integers a and x must be less than N for correct behavior. Uses O(n^2)
        elementary gates.

        |x> --> |x*a mod N>

        Input to the gate is 3n+2 qubits split into two registers:
        - n qubits for x, 0 <= x < N. x*a mod N is saved here.
        - 2n+2 qubits initialized to 0. These are unchanged by the gate.

        Args:
            n: number of qubits.
            a: 0 < a < N and gcd(a, N) = 1.
            N: 1 < N < 2 ** n.
            inv_a: inverse of a mod N. If None, will be computed.
        """
        super().__init__()
        self.n = n
        self.a = a
        self.N = N
        if inv_a:
            self.inv_a = inv_a
        else:
            self.inv_a = modular_inverse(a=a, N=N)

    def _num_qubits_(self):
        return 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["Ua_x"] * self.n + ["Ua_anc"] * (2 * self.n + 2)

    def _decompose_(self, qubits):
        x = qubits[: self.n]
        anc_mult = qubits[self.n : 2 * self.n]
        anc_add = qubits[2 * self.n :]

        yield MMult(self.n, self.a, self.N).on(*x, *anc_mult, *anc_add)
        for i in range(self.n):
            yield cirq.SWAP(x[i], anc_mult[i])
        yield cirq.inverse(MMult(self.n, self.inv_a, self.N)).on(*x, *anc_mult, *anc_add)


class MExp(cirq.Gate):
    def __init__(self, m: int, n: int, a: int, N: int):
        """Multiply qubit x by a^k, modulo N, where a is classical integer and
        gcd(a, N) = 1. Integers a and x must be less than N for correct
        behavior. Uses O(m * n^2) elementary gates.

        |k; x> --> |k; x * a^k mod N>

        Input to the gate is m+3n+2 qubits split into three registers:
        - m qubits for k, 0 <= k < 2^m. These are unchanged by the gate.
        - n qubits for x, 0 <= x < N. x * a^k mod N is saved here.
        - 2n+2 qubits initialized to 0. These are unchanged by the gate.

        Args:
            m: number of qubits for k.
            n: number of qubits for x.
            a: 0 < a < N and gcd(a, N) = 1.
            N: 1 < N < 2 ** n.
        """
        super().__init__()
        self.m = m
        self.n = n
        self.a = a
        self.inv_a = modular_inverse(a=a, N=N)
        self.N = N

    def _num_qubits_(self):
        return self.m + 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MExp_k"] * self.m + ["MExp_x"] * self.n + ["MExp_anc"] * (2 * self.n + 2)

    def _decompose_(self, qubits):
        k = qubits[: self.m]
        x = qubits[self.m : self.m + self.n]
        anc = qubits[self.m + self.n :]

        d = self.a  # stores a^(2^i)
        inv_d = self.inv_a  # stores a^(-2^i)
        for i in range(self.m):
            yield Ua(self.n, d, self.N, inv_a=inv_d).controlled(1).on(k[i], *x, *anc)
            d = (d * d) % self.N
            inv_d = (inv_d * inv_d) % self.N
