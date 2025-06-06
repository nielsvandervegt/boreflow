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
    Rusanov : int
        Rusanov flux
    HLL : int
        Harten-Lax-van Leer (HLL) flux
    """

    Rusanov = 0
    HLL = 1


class Limiter(Enum):
    """
    Enumeration of limiter types used for slope limiting in the numerical solution of the shallow water equations.

    Attributes
    ----------
    Koren : int
        Koren limiter
    MC : int
        Monotonized Central (MC) limiter
    MC_minmod : int
        Hybrid Monotonized Central (MC) and minmod limiter with minmod applied to low flow thicknesses (<1cm)
    minmod : int
        Minmod limiter
    superbee : int
        Superbee limiter
    vanAlbada : int
        Van Albada symmetric limiter
    vanLeer : int
        Van Leer limiter
    vanLeer_minmod : int
        Hybrid Van Leer and minmod limiter with minmod applied to low flow thicknesses (<1cm)
    """

    Koren = 0
    MC = 1
    minmod = 2
    superbee = 3
    vanAlbada = 4
    vanLeer = 5
    vanLeer_minmod = 6
    MC_minmod = 7
