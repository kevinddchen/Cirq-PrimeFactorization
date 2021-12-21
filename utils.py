import cirq


def prepare_state(qubits, i):
  '''Prepare qubits with state given by integer i.'''
  for q in qubits:
    if (i % 2):
      yield cirq.X(q)
    i >>= 1


def bits_to_integer(bits):
  '''From a string of bits, return the integer representation.'''
  i = 0
  for b in bits[::-1]:
    i <<= 1
    i += b
  return i


def integer_to_bits(n, i):
  '''From integer, return string of bits representation. n is total number of bits.'''
  bits = []
  for j in range(n):
    bits.append(i % 2)
    i >>= 1
  return bits
