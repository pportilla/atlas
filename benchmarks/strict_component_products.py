"""Time construction of the strict-transform ideal on fresh native charts."""
import statistics
from time import perf_counter

from engine import native_polynomial as np
from engine.local_blowups import Chart, _hypersurface


def chart(index, case):
    x, y, z = (np.Dummy(name) for name in "xyz")
    if case == "one principal component":
        ideals = [(x**3 + y**2 + index*z,)]
    elif case == "four principal components":
        ideals = [(x**3 + y**2 + index*z,), (y-z,), (x+index*y,), (z+index*x,)]
    else:
        ideals = [(x+index*z, y), (y+index*x, z), (z+index*y, x)]
    components = tuple((f"S{j}", 1, 1, ideal) for j, ideal in enumerate(ideals))
    return Chart((x,y,z), (), components, (), (), (), ())


def main():
    for case in ("one principal component", "four principal components", "three two-generator components"):
        trials = []
        for _ in range(8):
            charts = [chart(i, case) for i in range(1, 21)]
            start = perf_counter()
            for item in charts:
                _hypersurface(item)
            trials.append((perf_counter() - start) * 1000 / len(charts))
        samples = ", ".join(f"{t:.3f}" for t in trials[1:])
        print(f"{case}: {samples} ms/chart (median {statistics.median(trials[1:]):.3f})")


if __name__ == "__main__":
    main()
