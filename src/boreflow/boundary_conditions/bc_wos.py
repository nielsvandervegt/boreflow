from .bc_base import BCBaseOvertopping


class BCWOS(BCBaseOvertopping):
    """
    Boundary condition for overtopping flow, based on empirical formulas
    from van der Meer et al. (2011) and Hughes et al. (2012).

    Attributes
    ----------
    volume : float
        The to be simulated individual overtopping volume
    u_peak : float
        Peak flow velocity
    h_peak : float
        Peak flow thickness
    t_ovt : float
        The total time of the overtopping event
    tru_tovt : float
        Ratio between the time of upeak and the overtopping time (tovt) (default: 0.01)
    trh_tovt : float
        Ratio between the time of hpeak and the overtopping time (tovt) (default: 0.07)
    coef : float
        Coefficient optimized such that the integrated u(t) and h(t) equal the given volume
    """

    def __init__(self, volume: float, tru_tovt: float = 0.01, trh_tovt: float = 0.07) -> None:
        """
        Initialize the boundary condition.
        """
        # Calculate peak flow characteristics
        self.volume = volume
        self.tru_tovt = tru_tovt
        self.trh_tovt = trh_tovt
        self.u_peak = 5.0 * volume**0.34
        self.h_peak = 0.133 * volume**0.5
        self.t_ovt = 4.4 * volume**0.3

        # Optimize flow
        self.coef = self.optimize_flow()
