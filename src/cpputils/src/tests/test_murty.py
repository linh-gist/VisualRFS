from murty import Murty
import numpy as np

mgen = Murty(np.eye(2))

ok, cost, sol = mgen.draw()

assert ok
assert cost == 0.
assert (sol == np.array([1, 0])).all()

ok, cost, sol = mgen.draw()

assert ok
assert cost == 2.
assert (sol == np.array([0, 1])).all()

ok, cost, sol = mgen.draw()

assert not ok
