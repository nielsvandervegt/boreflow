from copy import deepcopy
from time import time

from .boundary_conditions.bc_base import BCBase
from .enum import Solver
from .finite_volume_method import FVM
from .geometry import Geometry


class Simulation:
    """
    A class to simulate a model using the Finite Volume Method (FVM).

    Attributes
    ----------
    bc : BCBase
        Boundary conditions object applied at the first interface.
    geometry: Geometry
        The Geometry object of the domain.
    t_end : float
        Total simulation time (default: 10s)
    cfl : float
        CFL (Courant–Friedrichs–Lewy) number for time step stability (default: 0.5)
    max_dt : float
        Maximum time step size (default: 0.01s)
    dx : float
        Spatial grid resolution (default: 0.1m)
    """

    # Default
    bc: BCBase
    geometry: Geometry
    t_end: float
    cfl: float
    max_dt: float
    dx: float

    def __init__(self, t_end: float = 10, cfl: float = 0.5, max_dt: float = 0.01, dx: float = 0.1) -> None:
        """
        Initialize simulation time and spatial parameters
        """
        self.t_end = t_end
        self.max_dt = max_dt
        self.dx = dx
        self.cfl = cfl

    def run(self, geometry: Geometry, bc: BCBase, solver: Solver = Solver.EF_LLF) -> Geometry:
        """
        Run the simulation using the provided geometry, boundary conditions, and solver.

        The simulation involves discretizing the geometry, applying the Finite Volume Method (FVM),
        and computing the wetting front velocity for each geometry part. The simulation runs until
        the end time is reached or the conditions specified by the solver are satisfied.

        Parameters
        ----------
        geometry : Geometry
            Geometry object representing the physical domain for the simulation.
        bc : BCBase
            Boundary conditions for the simulation.
        solver : Solver, optional
            The solver to use for the simulation. The default is `Solver.EF_LLF` (Euler Forward with Local Lax-Friedrichs).

        Returns
        -------
        Geometry
            A new copy of the geometry object with the simulation results.
        """
        # Save
        geometry = deepcopy(geometry)
        bc = deepcopy(bc)
        geometry.boundary_condition = bc

        # If geometry is not initialised, give error
        if geometry is None:
            raise ValueError("Geometry not initialised")

        # Start time
        start_time = time()

        # Run for each geometry part
        for i, _geometrypart in enumerate(geometry):
            _bc = bc if i == 0 else geometry.geometry_parts[i - 1]
            FVM.run_fvm(_geometrypart, _bc, solver, self.t_end, self.cfl, self.max_dt, self.dx)
            _geometrypart.derive_front_velocity()

        # Finish
        geometry.simulated = True
        geometry.simulation_time = time() - start_time
        print(f"Simulation done in {geometry.simulation_time:.2f} sec")

        return geometry
