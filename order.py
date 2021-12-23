import cirq
import numpy as np

from arithmetic import MExp
import utils


class OrderFinder(object):
  '''Brute force order finder in O(N). Must have gcd(a, N) = 1. '''

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
  '''Order finding in O((log N)^3) by simulating quantum circuit. Since the
  measurement is probabilistic, in order to avoid mistaking 2*r, 3*r, ... for
  the actual order, any candidate for r must be measured at least [THRESHOLD]
  times before it is considered. '''
  
  def __init__(self, a, N, THRESHOLD=2):
    super().__init__(a, N)
    self.THRESHOLD = THRESHOLD
    self.n = int(np.log2(N)) + 1 
    self.m = 2 * self.n
    self._r_dict = {}
    
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
    results = cirq.Simulator().run(self.circuit, repetitions=1)
    for key in results.measurements:
      out_array = results.measurements[key][0]
    return utils.bits_to_integer(out_array)
      
  def find(self):
    print("Finding order of {:d} modulo {:d} ...".format(self.a, self.N))
    print("Running on n={:d} qubits ...".format(self.n))
    print("(THRESHOLD={:d})".format(self.THRESHOLD))
    i = 0
    while True:
      i += 1
      j = self.sample()
      _, q = approximate_fraction(j, 2 ** self.m, self.N)
      print("Iteration {:d}: {:d}".format(i, q))

      ## _r_dict stores all denominators.
      ## if a denominator is observed [THRESHOLD] times, check if it is the order.
      if q in self._r_dict:
        self._r_dict[q] += 1
      else:
        self._r_dict[q] = 1

      if self._r_dict[q] == self.THRESHOLD:
        if pow(self.a, q, self.N) == 1:
          print("Found order r={:d}!".format(q))
          return q
        else:
          print("r={:d} is not the order. Continuing ...".format(q))


class FakeQuantumOrderFinder(QuantumOrderFinder):
  '''Order finding by sampling the known output distribution of the quantum
  circuit. This mimics the behavior of QuantumOrderFinder without simulating the
  quantum circuit, which is slow on a classical computer. This method is O(N)
  and is no faster than brute force.'''
    
  def __init__(self, a, N, THRESHOLD=2):
    super().__init__(a, N, THRESHOLD)
    of = OrderFinder(a, N)
    self.r = of.find()
      
  def f(self, C, x):
    '''f(x) = [sin^2(Cx)] / [C sin(x)]^2. '''
    if np.sin(x) == 0:
      return 1
    else:
      return (np.sin(C*x) ** 2) / ((C*np.sin(x)) ** 2)
      
  def sample(self):
    '''Sample directly from known distribution.'''
    M = 2 ** self.m
    k = np.random.randint(self.r)
    probs = np.zeros(M)
    for j in range(M):
      delta = np.pi * (k*1./self.r - j*1./M)
      probs[j] = self.f(M, delta)
    return np.random.choice(M, p=probs)

  def find(self):
    print("FAKE QUANTUM CIRCUIT")
    super().find()


## ========================
## HELPER FUNCTIONS =======
## ========================


def continued_fraction(p, q):
  '''Given p/q, return its continued fraction as a sequence.'''
  while q != 0:
    a = p // q
    yield a
    p, q = q, p - q*a


def approximate_fraction(p, q, N):
  '''Given p/q, find the closest fraction a/b where b < N. Returns a tuple (a, b).'''
  ## truncate continued fraction expansion when denominator >= N
  a1, a2 = 1, 0
  b1, b2 = 0, 1
  truncated = False
  for k in continued_fraction(p, q):
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


## ========================
## UNIT TESTS =============
## ========================


if __name__ == "__main__":
  
  ## Test quantum order finder.
  of = QuantumOrderFinder(3, 7)
  of.find()
  print()

  ## Test classical parts of quantum order finder
  for i in range(3):

    N = np.random.randint( 2 ** 7, 2 ** 8 )
    a = N
    while np.gcd(a, N) != 1:
      a = np.random.randint(2, N)
    of = FakeQuantumOrderFinder(a, N)
    of.find()
    print()
