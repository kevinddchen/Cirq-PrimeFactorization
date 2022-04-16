import cirq
import numpy as np

from factor import MExp, utils


class OrderFinder(object):
  '''Brute force order finding in O(N).  
  
  Parameters:
    a: integer, 1 < a < N and gcd(a, N) = 1.
    N: integer. 
  '''

  def __init__(self, a, N):
    assert np.gcd(a, N) == 1
    self.a = a
    self.N = N
      
  def find(self):
    '''Find the order.'''
    d, r = self.a, 1
    while d != 1:
      d, r = (d * self.a) % self.N, r + 1
    return r
    

class QuantumOrderFinder(OrderFinder):
  '''Order finding in O((log N)^3) by quantum phase estimation.
  
  Since the measurement for the order is probabilistic, in order to avoid
  mistaking 2*r, 3*r, ... for the actual order, any candidate must be measured
  at least `THRESHOLD` times before it is considered. But since this is very
  unlikely, having `THRESHOLD=2` should be sufficient.
  '''
  
  def __init__(self, a, N, THRESHOLD=2):
    super().__init__(a, N)
    self.THRESHOLD = THRESHOLD
    self.n = int(np.log2(N)) + 1 # number of bits to represent N
    self.m = 2 * self.n
    self._q_dict = {}
    
    ## Prepare quantum circuit that performs phase estimation on MExp.
    k = cirq.GridQubit.rect(1, self.m, top=0)
    x = cirq.GridQubit.rect(1, self.n, top=1)
    anc = cirq.GridQubit.rect(1, 2*self.n+2, top=2)
    self.circuit = cirq.Circuit()
    self.circuit.append(utils.prepare_state(x, 1))
    self.circuit.append(cirq.H(ki) for ki in k)
    self.circuit.append(MExp(self.m, self.n, a, N).on(*k, *x, *anc))
    self.circuit.append(cirq.qft(*k[::-1], inverse=True))
    self.circuit.append(cirq.measure(*k))
      
  def sample(self):
    '''Runs quantum circuit once and returns a measurement.'''
    result = cirq.Simulator().run(self.circuit, repetitions=1)
    _, raw_output = result.measurements.popitem()
    return utils.bits_to_integer(raw_output[0])
      
  def find(self):
    print("Finding order of {:d} modulo {:d} ...".format(self.a, self.N))
    print("Running on n={:d} qubits ...".format(self.n))
    print("(THRESHOLD={:d})".format(self.THRESHOLD))
    i = 0
    while True:
      i += 1
      j = self.sample() # j drawn from [0, 2^m)
      _, q = _approximate_fraction(j, 2 ** self.m, self.N)
      ## _/q is drawn uniformly from k/r where k=0, 1, ..., r-1.

      print("Iteration {:d}: q={:d}".format(i, q))

      ## _q_dict stores all qs.
      if q in self._q_dict:
        self._q_dict[q] += 1
      else:
        self._q_dict[q] = 1

      ## if a q is observed [THRESHOLD] times, check if it is the order.
      if self._q_dict[q] == self.THRESHOLD:
        if pow(self.a, q, self.N) == 1:
          print("Found order r={:d}!\n".format(q))
          return q
        else:
          print("q={:d} is not the order. Continuing ...".format(q))


class FakeQuantumOrderFinder(QuantumOrderFinder):
  '''Order finding by sampling the known output distribution of the quantum
  phase estimation circuit. This mimics the exact output of `QuantumOrderFinder`
  without simulating the quantum circuit, which is slow on a classical computer.
  However, this method is no faster than brute force.'''
    
  def __init__(self, a, N, THRESHOLD=2):
    super().__init__(a, N, THRESHOLD)
    of = OrderFinder(a, N)
    self.r = of.find()
      
  def distribution(self, M, x, eps=1e-12):
    '''f(x) = sin^2(Mx) / [M sin(x)]^2. '''
    x = np.maximum(np.abs(x), eps) # regularize behavior at x=0
    return (np.sin(M*x) ** 2) / ((M*np.sin(x)) ** 2)
      
  def sample(self):
    '''Sample directly from known distribution.'''
    M = 2 ** self.m
    k = np.random.randint(self.r)
    deltas = np.pi * (k*1./self.r - np.linspace(0, 1, M, endpoint=False))
    probs = self.distribution(M, deltas)
    return np.random.choice(M, p=probs)

  def find(self):
    print("FAKE QUANTUM CIRCUIT")
    return super().find()


def _continued_fraction(p, q):
  '''Given p/q, return its continued fraction as a sequence.'''
  while q != 0:
    a = p // q
    yield a
    p, q = q, p - q*a


def _approximate_fraction(p, q, N):
  '''Given p/q, find the closest fraction a/b where b < N. Returns a tuple (a, b).'''
  ## truncate continued fraction expansion when denominator >= N
  a1, a2 = 1, 0
  b1, b2 = 0, 1
  truncated = False
  for k in _continued_fraction(p, q):
    if k*b1 + b2 >= N:
      truncated = True
      break
    a1, a2 = k*a1 + a2, a1
    b1, b2 = k*b1 + b2, b1
  
  if truncated:
    ## use largest j where k/2 <= j < k and j*b1 + b2 < N.
    j = (N - b2) // b1
    if j >= k:
      pass
    elif k < 2*j: # found good j
      a1 = j*a1 + a2
      b1 = j*b1 + b2
    elif k == 2*j: # if k even, j = k/2 only admissible if the approximation is better
      next_a = j*a1 + a2
      next_b = j*b1 + b2
      if abs( (p*1.)/q - (next_a*1.)/next_b ) < abs( (p*1.)/q - (a1*1.)/b1 ):
          a1, b1 = next_a, next_b
    ## else, no better approximation
  return a1, b1
