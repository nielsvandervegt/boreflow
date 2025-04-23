import numpy as np

from boreflow import BCWOS, Geometry, Simulation


def test_simulation():
    # Create simulation
    bc = BCWOS(2.0)
    geometry = Geometry([0, 2, 11], [3, 3, 0], [0.0175, 0.0175])
    sim = Simulation(t_end=10.0, cfl=0.2, max_dt=0.01, nx=110)

    # Run simulation and at results at s = 10m
    results = sim.run(geometry, bc)
    _, h, u = results.get_st(10.0)

    # Test
    assert np.isclose(np.max(h), 0.123, atol=0.1)
    assert np.isclose(np.max(u), 7.246, atol=0.1)


if __name__ == "__main__":
    test_simulation()
