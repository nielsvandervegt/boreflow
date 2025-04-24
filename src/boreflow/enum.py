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
    Koren : int
        Koren limiter
    MC : int
        Monotonized Central (MC) limiter
    minmod : int
        Minmod limiter
    superbee : int
        Superbee limiter
    vanAlbada : int
        Van Albada symmetric limiter
    vanLeer : int
        Van Leer limiter
    """

    Koren = 0
    MC = 1
    minmod = 2
    superbee = 3
    vanAlbada = 4
    vanLeer = 5
