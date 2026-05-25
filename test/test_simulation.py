import numpy as np

from boreflow import BCArray, BCWOS, Geometry, Simulation
from boreflow.enum import Flux, Limiter, TimeIntegration
from boreflow.finite_volume_method import FVM


def test_simulation():
    # Create simulation
    bc = BCWOS(2.0, 0.0, 0.0)
    geometry = Geometry([0, 2, 11], [3, 3, 0], [0.0175, 0.0175])
    sim = Simulation(t_end=10.0, cfl=0.2, max_dt=0.01, nx=110)

    # Run simulation and at results at s = 10m
    results = sim.run(geometry, bc)
    _, h, u = results.get_st(10.0)

    # Test
    assert np.isclose(np.max(h), 0.132, atol=0.1)
    assert np.isclose(np.max(u), 7.265, atol=0.1)


def _lake_at_rest_setup(nx=120):
    geometry = Geometry(
        [0.0, 3.0, 8.0, 14.0, 20.0],
        [0.40, 0.10, 0.55, 0.20, 0.45],
        [0.02, 0.02, 0.02, 0.02],
    )
    fvm = FVM(boundary_condition=BCArray([0.0, 1.0], [1.0, 1.0], [0.0, 0.0]), t_end=0.25, max_dt=5e-4, cfl=0.25, verbose=False)
    fvm.discretise(geometry, nx=nx)

    eta0 = float(np.max(fvm.z_cells) + 0.35)
    U = np.zeros((2, nx + 2))
    U[0, :] = np.maximum(eta0 - fvm.z_cells, fvm.h_min)

    h_left = float(max(eta0 - fvm.z_cells[0], fvm.h_min))
    fvm.bc_left = BCArray([0.0, fvm.t_end], [h_left, h_left], [0.0, 0.0])
    return fvm, eta0, U


def test_still_water_over_nonuniform_bed_remains_quiescent():
    fvm, eta0, U = _lake_at_rest_setup(nx=140)

    max_u = 0.0
    max_eta_dev = 0.0
    prev_eta_dev = 0.0

    # Time step is tied to model CFL condition.
    for _ in range(60):
        max_speed = max(fvm.compute_max_velocity(U), 1e-8)
        dt = min(fvm.cfl * fvm.dx / max_speed, fvm.max_dt)
        rhs = fvm.compute_rhs(0.0, U, Limiter.minmod, Flux.HLL)
        U = U + dt * rhs

        h = U[0, 1:-1]
        u = np.divide(U[1, 1:-1], h, out=np.zeros_like(h), where=h > fvm.h_wet)
        eta = h + fvm.z_cells[1:-1]

        max_u = max(max_u, float(np.max(np.abs(u))))
        eta_dev = float(np.max(np.abs(eta - eta0)))
        max_eta_dev = max(max_eta_dev, eta_dev)

        # non-growing check with a small numerical slack
        growth_slack = 5.0 * dt / fvm.dx
        assert eta_dev <= prev_eta_dev + growth_slack
        prev_eta_dev = eta_dev

    # Tolerances explicitly tied to dx and dt scale.
    u_tol = 50.0 * fvm.max_dt * np.sqrt(fvm.g * (eta0 - np.min(fvm.z_cells))) / fvm.dx
    eta_tol = 25.0 * fvm.max_dt * np.sqrt(fvm.g * (eta0 - np.min(fvm.z_cells)))

    assert max_u <= u_tol
    assert max_eta_dev <= eta_tol


def test_shallow_wet_dry_edge_preserves_nonnegative_depth():
    nx = 120
    geometry = Geometry([0.0, 10.0, 20.0], [0.0, 0.05, 0.0], [0.02, 0.02])
    t = np.array([0.0, 0.05, 0.10, 0.20])
    h = np.array([0.0, 0.02, 0.0, 0.0])
    u = np.array([0.0, 0.5, 0.0, 0.0])
    bc = BCArray(t, h, u)

    fvm = FVM(boundary_condition=bc, t_end=0.2, max_dt=2e-4, cfl=0.2, verbose=False)
    fvm.discretise(geometry, nx=nx)
    fvm.run(Limiter.minmod, Flux.HLL, TimeIntegration.RK2)

    depth = fvm.geometry.h
    assert np.all(np.isfinite(depth))

    depth_floor_tol = 10.0 * fvm.h_min
    assert float(np.min(depth)) >= -depth_floor_tol
