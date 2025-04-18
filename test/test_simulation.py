import numpy as np

from boreflow import BCWOS, Geometry, Simulation, Solver


def test_simulation():
    # Create simulation
    bc = BCWOS(2.0)
    geometry = Geometry([0, 2, 11], [3, 3, 0], [0.0175, 0.0175])
    sim = Simulation(t_end=2.0, cfl=0.5, max_dt=0.01, dx=0.1)

    # Run simulation and at results at s = 10m
    results = sim.run(geometry, bc, Solver.EF_LLF)
    _, h, u = results.get_st(10.0)

    # Test
    assert np.isclose(np.max(h), 0.132, atol=0.1)
    assert np.isclose(np.max(u), 7.209, atol=0.1)
