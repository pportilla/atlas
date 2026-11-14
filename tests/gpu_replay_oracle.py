"""Independent NumPy interpreter for checking modular replay."""
import numpy as np

from engine.gpu_replay import instructions


def replay_reference(code, pool_size, work_size, primes, pools, status):
    """Run the program for all the primes. `pools` (primes x pool_size,
    uint32) is updated in place and `status` (uint32) receives 1 for the
    primes that agree with the program."""
    primes = np.asarray(primes, dtype=np.uint64)
    p = primes[:, None]
    square = p*p
    work = np.zeros((len(primes), work_size), dtype=np.uint64)
    status[:] = 1

    def axpy(factor, source, index):
        # factor < p and pool < p: the product is below p^2, the sum below 2 p^2 < 2^63.
        w = work[:, index] + factor*pools[:, source:source + len(index)].astype(np.uint64)
        work[:, index] = np.where(w >= square, w - square, w)

    for op, scalars, start, end in instructions(code):
        index = np.asarray(code[start:end], dtype=np.intp)
        if op == 1:
            work[:, :scalars[0]] = 0
        elif op == 2:
            axpy(np.ones_like(p), scalars[0], index)
        elif op == 3:
            axpy(p - np.uint64(1), scalars[0], index)
        elif op == 4:
            column, source = scalars
            f = work[:, column] % primes
            work[:, column] = 0
            axpy(np.where(f == 0, np.uint64(0), primes - f)[:, None], source, index)
        elif op == 5:
            status[((work[:, index] % p) != 0).any(axis=1)] = 0
        else:
            target = scalars[0]
            base = work[:, index[0]] % primes
            status[base == 0] = 0
            inverse = np.array([pow(int(b), int(q) - 2, int(q)) if b else 0 for b, q in zip(base, primes)],
                               dtype=np.uint64)[:, None]
            pools[:, target] = 1
            pools[:, target + 1:target + len(index)] = ((work[:, index[1:]] % p)*inverse % p).astype(np.uint32)
    return 0
