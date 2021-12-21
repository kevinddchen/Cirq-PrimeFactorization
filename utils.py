import cirq


def prepare_state(qubits, i):
  '''Prepare qubits with state given by integer i.'''
  for q in qubits:
    if (i % 2):
      yield cirq.X(q)
    i >>= 1