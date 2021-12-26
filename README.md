# Cirq-PrimeFactorization

This is an implementation of Shor's algorithm in Cirq (version 0.13.1).

[QuantumComputing.pdf](QuantumComputing.pdf) contains details on the algorithm itself.
[demo.ipynb](demo.ipynb) contains examples on using the code.

Simulating the quantum circuit is very slow on a classical computer.
Even the simplest case of factoring `15 = 3 * 5` took ~30 minutes to run.
However, since the probability distribution for the outputs of the quantum circuit in Shor's algorithm is known, we can draw from this distribution to mimic the output from the quantum circuit.
Doing so helps test the classical portions of the algorithm, since we can skip waiting for the simulation of the quantum circuit to complete.

This shortcut is implemented as the `FakeQuantumOrderFinder` in [order.py](order.py), in contrast to `QuantumOrderFinder` which uses the full quantum circuit.
Do note that `FakeQuantumOrderFinder` is no faster than finding the order by brute force, since we actually need to calculate the order to know the probability distribution.

In the end, Shor's algorithm must be run on a quantum computer.
I hope to have the opportunity to do so sometime in the future (although at the time of writing, I am not very confident that my implementation in its current state will be able to run on current hardware).
Nevertheless, I hope this repository can serve as useful reference for others.
