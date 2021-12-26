import numpy as np

import order


def shor(N, debug=False):
  '''Given an integer N > 1, return a non-trivial factor of N, or 1 if N is a
  (probable) prime.
  
  Debug mode uses `FakeQuantumOrderFinder` instead of `QuantumOrderFinder`. See
  `order.py` for more details.
  '''

  ## check that N is odd
  if N % 2 == 0:
    return 2

  ## check that N is composite
  if miller_rabin(N):
    return 1

  ## check that N is not an integer power
  for k in range(2, int(np.log(N)/np.log(3))+1):
    x = round(pow(N, 1./k))
    if pow(x, k) == N:
      return x

  while True:
    a = np.random.randint(2, N)
    d = np.gcd(a, N)
    if d != 1:
      return d
    
    ## if debug mode, use FakeQuantumOrderFinder
    if debug:
      of = order.FakeQuantumOrderFinder(a, N)
    else:
      of = order.QuantumOrderFinder(a, N)

    ## find period of a modulo N
    r = of.find()

    if r % 2 == 0:
      d = np.gcd(pow(a, r // 2, N) - 1, N)
      if d != 1:
        return d


def miller_rabin(n, k=40):
  '''Returns True if n is a probable prime. Implementation uses the Miller-Rabin
  primality test. Modified from https://gist.github.com/Ayrx/5884790.
  '''

  if n == 2 or n == 3:
    return True

  if n % 2 == 0:
    return False

  r, s = 0, n - 1
  while s % 2 == 0:
    r += 1
    s //= 2
  for _ in range(k):
    a = np.random.randint(2, n - 1)
    x = pow(a, s, n)
    if x == 1 or x == n - 1:
      continue
    for _ in range(r - 1):
      x = pow(x, 2, n)
      if x == n - 1:
        break
    else:
      return False
  return True
