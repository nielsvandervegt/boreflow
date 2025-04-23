from enum import Enum


class TimeIntegration(Enum):
    """
    Enumeration of time integration methods used for solving the Shallow Water Equations (SSSWE).

    Attributes
    ----------
    EF : int
        Euler Forward (1st order) time-stepping method.
    RK2 : int
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
    MC : int
        Monotonized Central (MC) limiter
    minmod : int
        Minmod limiter
    superbee : int
        Superbee limiter
    vanLeer : int
        Van Leer limiter
    """

    MC = 0
    minmod = 1
    superbee = 2
    vanLeer = 3
