import cirq
import numpy as np
import utils


class Add(cirq.Gate):
  '''Add classical integer a to qubit b. To account for potential overflow, an
  extra qubit (initialized to zero) must be supplied for b. 
  
    |b> --> |b+a>
      
  Parameters:
    n: number of qubits.
    a: integer, 0 <= a < 2**n.
  
  Input to gate is 2n+1 qubits split into:
    n+1 qubits for b, with most significant digit initialized to 0. b+a is saved here.
    n ancillary qubits initialized to 0. Remain 0 after operation.
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
    ## In backward pass, undo carries and add a and carries to b.
    for i in range(n-2, -1, -1):
      yield cirq.TOFFOLI(anc[i], b[i], anc[i+1])
      if a[i]:
        yield cirq.X(b[i])
        yield cirq.CNOT(b[i], anc[i+1])
        yield cirq.X(b[i])
      yield cirq.CNOT(anc[i], b[i])


class MAdd(cirq.Gate):
  '''Add classical integer a to qubit b, modulo N. Integers must be less than N.
  
    |b> --> |b+a mod N>
      
  Parameters:
    n: number of qubits.
    a: integer, 0 <= a < N
    N: integer, 0 <= N < 2**n
  
  Input to gate is 2n+2 qubits split into:
    n qubits for b < N. a+b mod N is saved here.
    n+2 ancillary qubits initialized to 0. Remain 0 after operation.
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

      assert b_out == a_int + b_int, "Incorrect addition."
      assert anc_out == 0, "Ancillary qubits must be unchanged."
      print("Test {0:2d} PASSED: {1:3d} + {2:3d} = {3:3d}".format(i_test, a_int, b_int, b_out))
      
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

      assert b_out == (a_int + b_int) % N_int, "Incorrect addition."
      assert anc_out == 0, "Ancillary qubits must be unchanged."
      print("Test {0:2d} PASSED: {1:3d} + {2:3d} = {3:3d} (mod {4:d})".format(i_test, a_int, b_int, b_out, N_int))
      
  return True
        

if __name__ == "__main__":
  
  assert add_unit_test()
  assert madd_unit_test()
