from .bc_base import BCBaseOvertopping


class BCWOS(BCBaseOvertopping):
    """
    Boundary condition for overtopping flow, based on empirical formulas
    from van der Meer et al. (2011) and Hughes et al. (2012).
    """
    
    def __init__(self, volume: float, tru_tovt: float = 0.0, trh_tovt: float = 0.0) -> None:
        """
        Initialize the boundary condition.
        
        Parameters
        ----------
        volume : float
            Total overtopping volume [m³/m].
        tru_tovt : float
            Time ratio for velocity peak (default: 0.05).
        trh_tovt : float
            Time ratio for depth peak (default: 0.13).
        """
        # Calculate peak flow characteristics
        self.volume = volume
        self.tru_tovt = tru_tovt
        self.trh_tovt = trh_tovt
        self.u_peak = 5.0 * volume ** 0.34 #4.22 * volume ** 0.26
        self.h_peak = 0.133 * volume ** 0.5 #0.16 * volume ** 0.49
        self.t_ovt = 4.4 * volume ** 0.3

        # Optimize flow
        self.optimize_flow()
