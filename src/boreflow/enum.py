from enum import Enum


class Solver(Enum):
    """
    Enumeration of solvers used for solving the Shallow Water Equations (SSSWE).

    This Enum defines different numerical solvers that can be used in the simulation, each with its specific time-stepping scheme.

    Attributes
    ----------
    EF_LLF
        Euler Forward with Local Lax-Friedrichs flux.
    RK4_LLF
        Runge-Kutta 4th order with Local Lax-Friedrichs flux.
    """

    EF_LLF = 0
    RK4_LLF = 1
