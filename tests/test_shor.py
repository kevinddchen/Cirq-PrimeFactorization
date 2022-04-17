from factor import shor, FakeQuantumOrderFinder


def test_shor():
    for N in [3 * 5, 7 * 13, 5 * 101]:
        f = shor(N, order_finder=FakeQuantumOrderFinder)
        assert N == f * (N // f)
