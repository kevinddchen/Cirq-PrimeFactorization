import cirq

from factor import utils


# Quantum gates for arithmetic.
# Implementation based on https://arxiv.org/abs/quant-ph/9511018.

# Note: an integer a = 2^(n-1) a_(n-1) + ... + 2 a_1 + a_0 is represented
# by n qubits with the convention that the nth qubit represents a_n.


class Add(cirq.Gate):
    """Add classical integer a to qubit b. To account for possible overflow, an
    extra qubit (initialized to zero) must be supplied for b. Uses O(n) elementary
    gates.

      |b> --> |b+a>

    Parameters:
      n: number of qubits.
      a: integer, 0 <= a < 2^n.

    Input to gate is 2n+1 qubits split into:
      n+1 qubits for b, 0 <= b < 2^n. The most significant digit is initialized to 0. b+a is saved here.
      n ancillary qubits initialized to 0. Unchanged by operation.
    """

    def __init__(self, n, a):
        super().__init__()
        self.n = n
        self.a = a

    def _num_qubits_(self):
        return 2 * self.n + 1

    def _circuit_diagram_info_(self, args):
        return ["Add_b"] * (self.n + 1) + ["Add_anc"] * self.n

    def _decompose_(self, qubits):
        n = self.n
        a = utils.integer_to_bits(n, self.a)
        b = qubits[:n]
        anc = qubits[n + 1 :] + (qubits[n],)  # internally, b[n] is placed in anc[n]

        # In forward pass, store carried bits in ancilla.
        for i in range(n):
            if a[i]:
                yield cirq.CNOT(b[i], anc[i + 1])
                yield cirq.X(b[i])
            yield cirq.TOFFOLI(anc[i], b[i], anc[i + 1])
        yield cirq.CNOT(anc[n - 1], b[n - 1])
        # In backward pass, undo carries, then add a and carries to b.
        for i in range(n - 2, -1, -1):
            yield cirq.TOFFOLI(anc[i], b[i], anc[i + 1])
            if a[i]:
                yield cirq.X(b[i])
                yield cirq.CNOT(b[i], anc[i + 1])
                yield cirq.X(b[i])
            yield cirq.CNOT(anc[i], b[i])


class MAdd(cirq.Gate):
    """Add classical integer a to qubit b, modulo N. Integers a and b must be less
    than N for correct behavior. Uses O(n) elementary gates.

      |b> --> |b+a mod N>

    Parameters:
      n: number of qubits.
      a: integer, 0 <= a < N.
      N: integer, 1 < N < 2^n.

    Input to gate is 2n+2 qubits split into:
      n qubits for b, 0 <= b < N. a+b mod N is saved here.
      n+2 ancillary qubits initialized to 0. Unchanged by operation.
    """

    def __init__(self, n, a, N):
        super().__init__()
        self.n = n
        self.a = a
        self.N = N

    def _num_qubits_(self):
        return 2 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MAdd_b"] * self.n + ["MAdd_anc"] * (self.n + 2)

    def _decompose_(self, qubits):
        n = self.n
        b = qubits[: n + 1]  # extra qubit for overflow
        anc = qubits[n + 1 : 2 * n + 1]
        t = qubits[2 * n + 1]

        Add_a = Add(n, self.a)
        Add_N = Add(n, self.N)
        yield Add_a.on(*b, *anc)
        yield cirq.inverse(Add_N).on(*b, *anc)
        # Second register is a+b-N. The most significant digit indicates underflow from subtraction.
        yield cirq.CNOT(b[n], t)
        yield Add_N.controlled(1).on(t, *b, *anc)
        # To reset t, subtract a from second register. If underflow again, means that t=0 previously.
        yield cirq.inverse(Add_a).on(*b, *anc)
        yield cirq.X(b[n])
        yield cirq.CNOT(b[n], t)
        yield cirq.X(b[n])
        yield Add_a.on(*b, *anc)


class MMult(cirq.Gate):
    """Multiply qubit x by classical integer a, modulo N. Exact map is:

      |x; b> --> |x; b + x*a mod N>

    Integers a, b, and x must be less than N for correct behavior. Uses O(n^2)
    elementary gates.

    Parameters:
      n: number of qubits.
      a: integer, 0 <= a < N.
      N: integer, 1 < N < 2^n.

    Input to gate is 3n+2 qubits split into:
      n qubits for x, 0 <= x < N. Unchanged by operation.
      n qubits for b, 0 <= b < N. b + x*a mod N is saved here.
      n+2 ancillary qubits initialized to 0. Unchanged by operation.
    """

    def __init__(self, n, a, N):
        super().__init__()
        self.n = n
        self.a = a
        self.N = N

    def _num_qubits_(self):
        return 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MMult_x"] * self.n + ["MMult_b"] * self.n + ["MMult_anc"] * (self.n + 2)

    def _decompose_(self, qubits):
        n = self.n
        N = self.N
        x = qubits[:n]
        b = qubits[n : 2 * n]
        anc = qubits[2 * n :]

        # x*a = 2^(n-1) x_(n-1) a + ... + 2 x_1 a + x_0 a
        # so the bits of x control the addition of a * 2^i
        d = self.a  # stores a * 2^i mod N
        for i in range(n):
            yield MAdd(n, d, N).controlled(1).on(x[i], *b, *anc)
            d = (d << 1) % N


class Ua(cirq.Gate):
    """Applies the unitary n-qubit operation,

      |x> --> |x*a mod N>

    where gcd(a, N) = 1. Integers a and x must be less than N for correct
    behavior. Uses O(n^2) elementary gates.

    Parameters:
      n: number of qubits.
      a: integer, 0 < a < N and gcd(a, N) = 1.
      N: integer, 1 < N < 2^n.
      inv_a: (optional) integer, inverse of a mod N. Skips recalculation of this if provided.

    Input to gate is 3n+2 qubits split into:
      n qubits for x, 0 <= x < N. x*a mod N is saved here.
      2n+2 ancillary qubits initialized to 0. Unchanged by operation.
    """

    def __init__(self, n, a, N, inv_a=None):
        super().__init__()
        self.n = n
        self.a = a
        self.N = N
        if inv_a:
            self.inv_a = inv_a
        else:
            self.inv_a = pow(a, -1, N)

    def _num_qubits_(self):
        return 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["Ua_x"] * self.n + ["Ua_anc"] * (2 * self.n + 2)

    def _decompose_(self, qubits):
        n = self.n
        N = self.N
        x = qubits[:n]
        anc_mult = qubits[n : 2 * n]
        anc_add = qubits[2 * n :]

        yield MMult(n, self.a, N).on(*x, *anc_mult, *anc_add)
        for i in range(n):
            yield cirq.SWAP(x[i], anc_mult[i])
        yield cirq.inverse(MMult(n, self.inv_a, N)).on(*x, *anc_mult, *anc_add)


class MExp(cirq.Gate):
    """Multiply qubit x by a^k, modulo N, where a is a classical integer and
    gcd(a, N) = 1. Integers a and x must be less than N for correct behavior. Uses
    O(m * n^2) elementary gates.

      |k; x> --> |k; x * a^k mod N>

    Parameters:
      m: number of qubits for k.
      n: number of qubits for x.
      a: integer, 0 < a < N and gcd(a, N) = 1.
      N: integer, 1 < N < 2^n.

    Input to gate is m+3n+2 qubits split into:
      m qubits for k, 0 <= k < 2^m. Unchanged by operation.
      n qubits for x, 0 <= x < N. x * a^k mod N is saved here.
      2n+2 ancillary qubits initialized to 0. Unchanged by operation.
    """

    def __init__(self, m, n, a, N):
        super().__init__()
        self.m = m
        self.n = n
        self.a = a
        self.inv_a = pow(a, -1, N)  # inverse of a mod N
        self.N = N

    def _num_qubits_(self):
        return self.m + 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MExp_k"] * self.m + ["MExp_x"] * self.n + ["MExp_anc"] * (2 * self.n + 2)

    def _decompose_(self, qubits):
        m = self.m
        n = self.n
        N = self.N
        k = qubits[:m]
        x = qubits[m : m + n]
        anc = qubits[m + n :]

        d = self.a  # stores a^(2^i)
        inv_d = self.inv_a  # stores a^(-2^i)
        for i in range(m):
            yield Ua(n, d, N, inv_d).controlled(1).on(k[i], *x, *anc)
            d = (d * d) % N
            inv_d = (inv_d * inv_d) % N
