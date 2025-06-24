from .bc_base import BCBaseOvertopping


class BCWOSTholen(BCBaseOvertopping):
    """
    Boundary condition for overtopping flow, based on fit observed at the first
    measurement location and the temporal flow profile of Hughes et al. (2012).

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
        Ratio between the time of upeak and the overtopping time (tovt) (default: 0.02)
    trh_tovt : float
        Ratio between the time of hpeak and the overtopping time (tovt) (default: 0.08)
    coef : float
        Coefficient optimized such that the integrated u(t) and h(t) equal the given volume
    """

    def __init__(self, volume: float, tru_tovt: float = 0.018, trh_tovt: float = 0.066) -> None:
        """
        Initialize the boundary condition.
        """
        # Calculate peak flow characteristics
        self.volume = volume
        self.tru_tovt = tru_tovt
        self.trh_tovt = trh_tovt
        self.u_peak = 4.618 * volume**0.258
        self.h_peak = 0.096 * volume**0.872
        self.t_ovt = 4.4 * volume**0.3

        # Optimize flow
        self.coef = self.optimize_flow()
