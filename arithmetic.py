import cirq
import numpy as np
import utils


## Quantum gates for arithmetic.
## Implementation based on https://arxiv.org/abs/quant-ph/9511018.
## TODO: try QFT-based gates in https://arxiv.org/abs/quant-ph/0205095. May be more efficient.


class Add(cirq.Gate):
  '''Add classical integer a to qubit b. To account for possible overflow, an
  extra qubit (initialized to zero) must be supplied for b. 
  
    |b> --> |b+a>
      
  Parameters:
    n: number of qubits.
    a: integer, 0 <= a < 2^n.
  
  Input to gate is 2n+1 qubits split into:
    n+1 qubits for b, 0 <= b < 2^n. The most significant digit is initialized to 0. b+a is saved here.
    n ancillary qubits initialized to 0. Unchanged by operation.
  '''
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
    anc = qubits[n+1:] + (qubits[n],) # internally, b[n] is placed in anc[n]

    ## In forward pass, store carried bits in ancilla.
    for i in range(n):
      if a[i]:
        yield cirq.CNOT(b[i], anc[i+1])
        yield cirq.X(b[i])
      yield cirq.TOFFOLI(anc[i], b[i], anc[i+1])
    yield cirq.CNOT(anc[n-1], b[n-1])
    ## In backward pass, undo carries, then add a and carries to b.
    for i in range(n-2, -1, -1):
      yield cirq.TOFFOLI(anc[i], b[i], anc[i+1])
      if a[i]:
        yield cirq.X(b[i])
        yield cirq.CNOT(b[i], anc[i+1])
        yield cirq.X(b[i])
      yield cirq.CNOT(anc[i], b[i])


class MAdd(cirq.Gate):
  '''Add classical integer a to qubit b, modulo N. Integers a and b must be less 
  than N for correct behavior.
  
    |b> --> |b+a mod N>
      
  Parameters:
    n: number of qubits.
    a: integer, 0 <= a < N.
    N: integer, 1 < N < 2^n.
  
  Input to gate is 2n+2 qubits split into:
    n qubits for b, 0 <= b < N. a+b mod N is saved here. 
    n+2 ancillary qubits initialized to 0. Unchanged by operation.
  '''
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
    b = qubits[:n+1] # extra qubit for overflow
    anc = qubits[n+1:2*n+1]
    t = qubits[2*n+1]
    
    Add_a = Add(n, self.a)
    Add_N = Add(n, self.N)
    yield Add_a.on(*b, *anc)
    yield cirq.inverse(Add_N).on(*b, *anc)
    ## Second register is a+b-N. The most significant digit indicates underflow from subtraction.
    yield cirq.CNOT(b[n], t)
    yield Add_N.controlled(1).on(t, *b, *anc)
    ## To reset t, subtract a from second register. If underflow again, means that t=0 previously.
    yield cirq.inverse(Add_a).on(*b, *anc)
    yield cirq.X(b[n])
    yield cirq.CNOT(b[n], t)
    yield cirq.X(b[n])
    yield Add_a.on(*b, *anc)


class CMMult(cirq.Gate):
  '''Controlled multiplication of qubit b by classical integer a, modulo N.
  Integers a and b must be less than N for correct behavior.

    |c; b; 0> --> |c; b; b*a mod N>  if c = 1
                  |c; b; b>          if c = 0

  Parameters: 
    n: number of qubits. 
    a: integer, 0 <= a < N.
    N: integer, 1 < N < 2^n.

  Input to gate is 3n+3 qubits split into: 
    1 qubit for c. Unchanged by operation.
    n qubits for b, 0 <= b < N. Unchanged by operation. 
    n qubits initialized to 0. b*a mod N is saved here. 
    n+2 ancillary qubits initialized to 0. Unchanged by operation.
  '''
  def __init__(self, n, a, N):
    super().__init__()
    self.n = n
    self.a = a
    self.N = N
  
  def _num_qubits_(self):
    return 3 * self.n + 3
  
  def _circuit_diagram_info_(self, args):
    return ["CMMult_c"] + ["CMMult_b"] * self.n + ["CMMult_out"] * self.n + ["CMMult_anc"] * (self.n + 2)
  
  def _decompose_(self, qubits):
    n = self.n
    N = self.N
    c = qubits[0]
    b = qubits[1:n+1]
    prod = qubits[n+1:2*n+1]
    anc = qubits[2*n+1:]

    d = self.a # stores a * 2^i mod N
    for i in range(n):
      yield MAdd(n, d, N).controlled(2).on(c, b[i], *prod, *anc)
      d = (d << 1) % N
    ## If c=0, just copy b to prod.
    yield cirq.X(c)
    for i in range(n):
      yield cirq.TOFFOLI(c, b[i], prod[i])
    yield cirq.X(c)


class MExp(cirq.Gate):
  '''Multiply qubit b by a^k, where a is a classical integer. Integers a and b 
  must be less than N for correct behavior.

    |k; b> --> |k; b * a^k mod N>

  Parameters: 
    m: number of qubits for k.
    n: number of qubits for x. 
    a: integer, 1 <= a < N and gcd(a, N) = 1.
    N: integer, 1 < N < 2^n.

  Input to gate is m+3n+2 qubits split into: 
    m qubits for k, 0 <= k < 2^m. Unchanged by operation.
    n qubits for b, 0 <= b < N. b * a^k mod N is saved here.
    2n+2 ancillary qubits initialized to 0. Unchanged by operation.
  '''
  def __init__(self, m, n, a, N):
    super().__init__()
    assert 0 < a, "a must be positive. (a={})".format(a)
    assert np.gcd(a, N) == 1, "a and N must be coprime. (a={}, N={})".format(a, N)
    self.m = m
    self.n = n
    self.a = a
    self.ia = pow(a, -1, N) # inverse of a mod N
    self.N = N
  
  def _num_qubits_(self):
    return self.m + 3 * self.n + 2
  
  def _circuit_diagram_info_(self, args):
    return ["MExp_k"] * self.m + ["MExp_b"] * self.n + ["MExp_anc"] * (2 * self.n + 2)
  
  def _decompose_(self, qubits):
    m = self.m
    n = self.n
    N = self.N
    k = qubits[:m]
    b = qubits[m:m+n]
    zeros = qubits[m+n:m+2*n]
    anc = qubits[m+2*n:]

    d = self.a # stores a^(2^i)
    id = self.ia # stores a^(-2^i)
    for i in range(m):
      yield CMMult(n, d, N).on(k[i], *b, *zeros, *anc)
      yield cirq.inverse(CMMult(n, id, N)).on(k[i], *zeros, *b, *anc)
      b, zeros = zeros, b
      ## b is multiplied by a^(2^i). zeros is 0.
      d = (d * d) % N
      id = (id * id) % N
    ## if m is odd, then "b" and "zeros" are swapped
    if (m % 2):
      for i in range(n):
        yield cirq.SWAP(b[i], zeros[i])


## ========================
## UNIT TESTS =============
## ========================


def add_unit_test(n_tests=5, n_bits=8):
    
  print("Add unit test")
  print("Number of bits: {0:d}".format(n_bits))
  
  n = n_bits
  b = cirq.GridQubit.rect(1, n+1, top=0)
  anc = cirq.GridQubit.rect(1, n, top=1)
  
  for i_test in range(n_tests):
      
    ## Pick two random integers.
    a_int = np.random.randint(2 ** n)
    b_int = np.random.randint(2 ** n)

    circuit = cirq.Circuit()
    circuit.append(utils.prepare_state(b, b_int))
    circuit.append(Add(n, a_int).on(*b, *anc))
    circuit.append(cirq.measure(*b, *anc))

    ## Run one measurement and interpret result.
    result = cirq.Simulator().run(circuit, repetitions=1)
    for key in result.measurements:
      out_array = result.measurements[key][0]
      b_out = utils.bits_to_integer(out_array[:n+1])
      anc_out = utils.bits_to_integer(out_array[n+1:])

      b_exp = a_int + b_int
      assert b_out == b_exp, "Incorrect addition. (output={}, expected={})".format(b_out, b_exp)
      assert anc_out == 0, "Ancillary qubits must be unchanged."
      print("Test {:2d}/{:d} PASSED: {:3d} + {:3d} = {:3d}".format(
        i_test+1, n_tests, a_int, b_int, b_out
      ))

  print()   
  return True


def madd_unit_test(n_tests=5, n_bits=8):
    
  print("MAdd unit test")
  print("Number of bits: {0:d}".format(n_bits))
    
  n = n_bits
  b = cirq.GridQubit.rect(1, n, top=0)
  anc = cirq.GridQubit.rect(1, n+2, top=1)
  
  for i_test in range(n_tests):
      
    ## Pick random N.
    N_int = np.random.randint( 2 ** (n-1), 2 ** n)
    ## Pick two random integers.
    a_int = np.random.randint(N_int)
    b_int = np.random.randint(N_int)

    circuit = cirq.Circuit()
    circuit.append(utils.prepare_state(b, b_int))
    circuit.append(MAdd(n, a_int, N_int).on(*b, *anc))
    circuit.append(cirq.measure(*b, *anc))

    ## Run one measurement and interpret result.
    result = cirq.Simulator().run(circuit, repetitions=1)
    for key in result.measurements:
      out_array = result.measurements[key][0]
      b_out = utils.bits_to_integer(out_array[:n])
      anc_out = utils.bits_to_integer(out_array[n:])

      b_exp = (a_int + b_int) % N_int
      assert b_out == b_exp, "Incorrect addition. (output={}, expected={})".format(b_out, b_exp)
      assert anc_out == 0, "Ancillary qubits must be unchanged."
      print("Test {:2d}/{:d} PASSED: {:3d} + {:3d} = {:3d} (mod {:3d})".format(
        i_test+1, n_tests, a_int, b_int, b_out, N_int
      ))

  print()   
  return True


def mmult_unit_test(n_tests=5, n_bits=4):
    
  print("CMMult unit test")
  print("Number of bits: {0:d}".format(n_bits))
    
  n = n_bits
  c = cirq.GridQubit.rect(1, 1, top=0)
  b = cirq.GridQubit.rect(1, n, top=1)
  prod = cirq.GridQubit.rect(1, n, top=2)
  anc = cirq.GridQubit.rect(1, n+2, top=3)
  
  for i_test in range(n_tests):
      
    ## Pick random N.
    N_int = np.random.randint( 2 ** (n-1), 2 ** n)
    ## Pick two random integers.
    a_int = np.random.randint(N_int)
    b_int = np.random.randint(N_int)

    circuit = cirq.Circuit()
    circuit.append(utils.prepare_state(b, b_int))
    circuit.append(utils.prepare_state(c, 1))
    circuit.append(CMMult(n, a_int, N_int).on(*c, *b, *prod, *anc))
    circuit.append(cirq.measure(*c, *b, *prod, *anc))

    ## Run one measurement and interpret result.
    result = cirq.Simulator().run(circuit, repetitions=1)
    for key in result.measurements:
      out_array = result.measurements[key][0]
      c_out = utils.bits_to_integer(out_array[:1])
      b_out = utils.bits_to_integer(out_array[1:n+1])
      prod_out = utils.bits_to_integer(out_array[n+1:2*n+1])
      anc_out = utils.bits_to_integer(out_array[2*n+1:])

      prod_exp = (a_int * b_int) % N_int
      assert c_out == 1, "control qubit must be unchanged."
      assert b_out == b_int, "b qubits must be unchanged."
      assert prod_out == prod_exp, "Incorrect product. (output={}, expected={})".format(prod_out, prod_exp)
      assert anc_out == 0, "Ancillary qubits must be unchanged."
      print("Test {:2d}/{:d} PASSED: {:2d} * {:2d} = {:2d} (mod {:2d})".format(
        i_test+1, n_tests, a_int, b_int, prod_out, N_int
      ))
      
  print()
  return True


def mexp_unit_test(n_tests=5, n_bits=4):
    
  print("MExp unit test")
  print("Number of bits: {0:d}".format(n_bits))
    
  m = n_bits
  n = n_bits
  k = cirq.GridQubit.rect(1, m, top=0)
  b = cirq.GridQubit.rect(1, n, top=1)
  anc = cirq.GridQubit.rect(1, 2*n+2, top=2)
  
  for i_test in range(n_tests):
      
    ## Pick random integers.
    N_int = np.random.randint( 2 ** (n-1), 2 ** n)
    k_int = np.random.randint( 2 ** (m-1), 2 ** m)
    a_int = N_int
    while np.gcd(a_int, N_int) != 1:
      a_int = np.random.randint(2, N_int)
    b_int = np.random.randint(N_int)

    circuit = cirq.Circuit()
    circuit.append(utils.prepare_state(k, k_int))
    circuit.append(utils.prepare_state(b, b_int))
    circuit.append(MExp(m, n, a_int, N_int).on(*k, *b, *anc))
    circuit.append(cirq.measure(*k, *b, *anc))

    ## Run one measurement and interpret result.
    result = cirq.Simulator().run(circuit, repetitions=1)
    for key in result.measurements:
      out_array = result.measurements[key][0]
      k_out = utils.bits_to_integer(out_array[:m])
      b_out = utils.bits_to_integer(out_array[m:m+n])
      anc_out = utils.bits_to_integer(out_array[m+n:])

      b_exp = (b_int * pow(a_int, k_int, N_int)) % N_int
      assert k_out == k_int, "k qubits must be unchanged."
      assert b_out == b_exp, "Incorrect product. (output={}, expected={})".format(b_out, b_exp)
      assert anc_out == 0, "Ancillary qubits must be unchanged."
      print("Test {:2d}/{:d} PASSED: {:2d} * ({:2d} ** {:2d}) = {:2d} (mod {:2d})".format(
        i_test+1, n_tests, b_int, a_int, k_int, b_out, N_int
      ))
      
  print()
  return True
        

if __name__ == "__main__":
  
  assert add_unit_test()
  assert madd_unit_test()
  assert mmult_unit_test()
  assert mexp_unit_test()
