import cirq
import numpy as np
import utils


class Add(cirq.Gate):
  '''Add two integers a and b, each encoded in n qubits. To account for
  potential overflow, an extra qubit, initialized to zero, must be supplied for b. 
  
    |a; b> --> |a; a+b>
      
  Parameters:
    n: number of qubits.
  
  Input to gate is 3n+1 qubits split into:
    n qubits for a. 
    n+1 qubits for b, with most significant digit initialized to 0. a+b is saved here.
    n ancillary qubits initialized to 0.
  '''
  def __init__(self, n):
    super().__init__()
    self.n = n
  
  def _num_qubits_(self):
    return 3*self.n + 1
  
  def _circuit_diagram_info_(self, args):
    return ["ADD_a"] * self.n + ["ADD_b"] * (self.n + 1) + ["ADD_anc"] * self.n 
  
  def _decompose_(self, qubits):
    n = self.n
    a = qubits[:n]
    b = qubits[n:2*n]
    anc = qubits[2*n+1:] + (qubits[2*n],) # b[n] is placed in anc[n] for simplicity
    ## In forward pass, store carried bits in ancilla.
    for i in range(n):
      yield cirq.TOFFOLI(a[i], b[i], anc[i+1])
      yield cirq.CNOT(a[i], b[i])
      yield cirq.TOFFOLI(anc[i], b[i], anc[i+1])
    yield cirq.CNOT(anc[n-1], b[n-1])
    ## In backward pass, undo carries and add a and carries to b.
    for i in range(n-2, -1, -1):
      ## inverse carry
      yield cirq.TOFFOLI(anc[i], b[i], anc[i+1])
      yield cirq.CNOT(a[i], b[i])
      yield cirq.TOFFOLI(a[i], b[i], anc[i+1])
      ## sum
      yield cirq.CNOT(a[i], b[i])
      yield cirq.CNOT(anc[i], b[i])


class AddMod(cirq.Gate):
  '''Add two integers mod N. Integers must be less than N.
  
    |a; b> --> |a; a+b mod N>
      
  Parameters:
    n: number of qubits.
  
  Input to gate is 5n+2 qubits split into:
    n qubits for a < N.
    n qubits for b < N. a+b mod N is saved to these qubits.
    n qubits for N.
    2n+2 ancillary qubits initialized to 0.
  '''
  def __init__(self, n):
    super().__init__()
    self.n = n
  
  def _num_qubits_(self):
    return 5 * self.n + 2
  
  def _circuit_diagram_info_(self, args):
    return ["ADDMOD_a"] * self.n + ["ADDMOD_b"] * self.n + ["ADDMOD_N"] * self.n + \
           ["ADDMOD_anc"] * (2 * self.n + 2)
  
  def _decompose_(self, qubits):
    n = self.n
    a = qubits[:n]
    b = qubits[n:2*n] + (qubits[5*n],) # extra qubit for overflow
    N = qubits[2*n:3*n]
    anc1 = qubits[3*n:4*n]
    anc2 = qubits[4*n:5*n]
    t = qubits[5*n+1]
    
    yield Add(n).on(*a, *b, *anc1)
    yield cirq.inverse(Add(n)).on(*N, *b, *anc1)
    ## Second register is a+b-N. The most significant digit indicates underflow from subtraction.
    yield cirq.CNOT(b[n], t)
    ## If t=1 (underflow), copy N to anc2. 
    for i in range(n):
      yield cirq.TOFFOLI(t, N[i], anc2[i])
    ## Add anc2 to second register.
    yield Add(n).on(*anc2, *b, *anc1)
    ## Reset anc2.
    for i in range(n):
      yield cirq.TOFFOLI(t, N[i], anc2[i])
    ## Second register is a+b mod N.
    ## To reset t, subtract a from second register. If underflow again, means that t=1 previously.
    yield cirq.inverse(Add(n)).on(*a, *b, *anc1)
    yield cirq.CNOT(b[n], t)
    yield Add(n).on(*a, *b, *anc1)
        


## ========================
## UNIT TESTS =============
## ========================

def add_unit_test(n_tests=5, n_bits=6):
    
  print("ADD unit test")
  print("Number of bits: {0:d}".format(n_bits))
  
  n = n_bits
  a = cirq.GridQubit.rect(1, n, top=0)
  b = cirq.GridQubit.rect(1, n+1, top=1)
  anc = cirq.GridQubit.rect(1, n, top=2)
  
  for i_test in range(n_tests):
      
    ## Pick two random integers.
    a_int = np.random.randint(2 ** n)
    b_int = np.random.randint(2 ** n)

    circuit = cirq.Circuit()
    circuit.append(utils.prepare_state(a, a_int))
    circuit.append(utils.prepare_state(b, b_int))
    circuit.append(Add(n).on(*a, *b, *anc))
    circuit.append(cirq.measure(*b))

    ## Run one measurement and interpret result.
    result = cirq.Simulator().run(circuit, repetitions=1)
    for key in result.measurements:
      out_array = result.measurements[key][0]
      out = 0
      for x in out_array[::-1]:
        out <<= 1
        out += x

      success = a_int + b_int == out
      print("Test {0:2d}: {1:3d} + {2:3d} = {3:3d}\t{4:s}".format
        (
          i_test, a_int, b_int, out,
          "PASS" if success else "FAIL"
        )
      )
      
      if not success:
        return False
      
  return True


def addmod_unit_test(n_tests=5, n_bits=4):
    
  print("ADDMOD unit test")
  print("Number of bits: {0:d}".format(n_bits))
    
  n = n_bits
  a = cirq.GridQubit.rect(1, n, top=0)
  b = cirq.GridQubit.rect(1, n, top=1)
  N = cirq.GridQubit.rect(1, n, top=2)
  anc = cirq.GridQubit.rect(1, 2*n+2, top=3)
  
  for i_test in range(n_tests):
      
    ## Pick random N.
    N_int = np.random.randint( 2 ** (n-1), 2 ** n)
    ## Pick two random integers.
    a_int = np.random.randint(N_int)
    b_int = np.random.randint(N_int)

    circuit = cirq.Circuit()
    circuit.append(utils.prepare_state(a, a_int))
    circuit.append(utils.prepare_state(b, b_int))
    circuit.append(utils.prepare_state(N, N_int))
    circuit.append(AddMod(n).on(*a, *b, *N, *anc))
    circuit.append(cirq.measure(*b))

    ## Run one measurement and interpret result.
    result = cirq.Simulator().run(circuit, repetitions=1)
    for key in result.measurements:
      out_array = result.measurements[key][0]
      out = 0
      for x in out_array[::-1]:
        out <<= 1
        out += x

      success = (a_int + b_int) % N_int == out
      print("Test {0:2d}: {1:2d} + {2:2d} = {3:2d} (mod {4:d})\t{5:s}".format
        (
          i_test, a_int, b_int, out, N_int,
          "PASS" if success else "FAIL"
        )
      )
      
      if not success:
        return False
      
  return True
        
        

if __name__ == "__main__":
  
  assert add_unit_test()
  assert addmod_unit_test()
