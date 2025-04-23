from enum import Enum


class TimeIntegration(Enum):
    """
    Enumeration of time integration methods used for solving the Shallow Water Equations (SSSWE).

    Attributes
    ----------
    EF_LLF : int
        Euler Forward (1st order) time-stepping method.
    RK2_LLF : int
        Runge-Kutta (2nd order) time-stepping method.
    """

    EF = 0
    RK2 = 1


class Flux(Enum):
    """
    Enumeration of flux types used for the calculation of fluxes in shallow water equations.

    Attributes
    ----------
    HLL : int
        Harten-Lax-van Leer (HLL) flux
    """

    HLL = 0


class Limiter(Enum):
    """
    Enumeration of limiter types used for slope limiting in the numerical solution of the shallow water equations.

    Attributes
    ----------
    minmod : int
        Minmod limiter
    vanLeer : int
        Van Leer limiter
    """

    minmod = 0
    vanLeer = 1
