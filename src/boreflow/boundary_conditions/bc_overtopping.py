import numpy as np

from scipy.stats import norm

from .bc_base import BCBaseOvertopping


class BCOvertopping(BCBaseOvertopping):
    """
    Boundary condition for overtopping flow, based on empirical formulas
    from van der Meer et al. (2011), van Damme (2016), and Hughes et al. (2012),
    with uncertainty from van der Vegt et al. (2025).
    """

    def __init__(self, volume, cota, tru_tovt: float = 0.05, trh_tovt: float = 0.13, unc_ppf: float = 0.5) -> None:
        """
        Initialize the boundary condition.
        
        Parameters
        ----------
        volume : float
            Total overtopping volume [m3/m].
        cota : float
            Waterside slope of the dike (1:cota).
        tru_tovt : float
            Time ratio for velocity peak (default: 0.05).
        trh_tovt : float
            Time ratio for depth peak (default: 0.13).
        unc_ppf : float
            Percent point function (quantile) for uncertainty (default: 0.5).
        """
        # Store base parameters
        self.volume = volume
        self.tru_tovt = tru_tovt
        self.trh_tovt = trh_tovt

        # Uncertainty (van der Vegt et al., 2025)
        sigma_qu = np.interp(cota, [3, 6], [0.219, 0.129])
        qu = norm(0, sigma_qu).ppf(unc_ppf)
        qh = -0.538 * qu + 0.0658

        # Geometry and relation between V and Ru - Rc (van Damme, 2016)
        alpha = np.arctan(1 / cota)
        RuRc = np.sqrt(2 * np.sin(alpha)**2 * volume / (np.cos(alpha) * 0.055))

        # Peak flow values (van der Meer, 2011; EurOtop, 2018)
        cu = np.interp(cota, [3, 6], [1.4, 1.5])
        ch = np.interp(cota, [4, 6], [0.2, 0.3])
        self.u_peak = cu * np.sqrt(9.81 * RuRc) * np.exp(qu)
        self.h_peak = ch * RuRc * np.exp(qh)

        # Overtopping time (Hughes et al., 2012)
        self.t_ovt = 4.0 * volume ** 0.41

        # Fit flow shape (Hughes et al., 2012)
        self.optimize_flow()
