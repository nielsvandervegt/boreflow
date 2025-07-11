import numpy as np

from abc import ABC, abstractmethod
from scipy.optimize import root_scalar
from typing import Union


class BCBase(ABC):
    """
    Base class for boundary conditions
    Requires subclasses to define get_flow(t)
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_flow(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute (h, u) for time t.

        Parameters
        ----------
        t: Union[float, np.ndarray]
            Time as a float or array

        Returns
        -------
        np.ndarray
            2D array with (h, u) for t
        """
        pass


class BCBaseOvertopping(BCBase):
    """
    Base class for overtopping flow with an optimized shape coefficient based on the given volume

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
        Ratio between the time of hpeak and the overtopping time (tovt) (default: 0.08)
    coef : float
        Coefficient optimized such that the integrated u(t) and h(t) equal the given volume
    """

    volume: float
    u_peak: float
    h_peak: float
    t_ovt: float
    tru_tovt: float = 0.01
    trh_tovt: float = 0.08
    coef: float

    @abstractmethod
    def __init__(self) -> None:
        pass

    def optimize_flow(self) -> float:
        """
        Optimize coefficient until time series of u(t) and h(t) match the given volume

        Returns
        -------
        float
            The optimized coefficient
        """

        def optimize_volume(_coef):
            _t = np.arange(0, self.t_ovt, 0.001)
            h, u = self.get_flow(_t, _coef)
            V = np.sum(h * u * np.diff(np.append(_t, _t[-1])))  # Append last value for same length
            return V - self.volume

        sol = root_scalar(optimize_volume, method="brentq", x0=1, bracket=(0, 150), xtol=0.0001)
        return sol.root

    def get_flow(self, t: Union[float, np.ndarray], coef: float = None) -> np.ndarray:
        """
        Compute (h, u) for time t based on current or given coef.

        Parameters
        ----------
        t: Union[float, np.ndarray]
            Time as a float or array
        coef: float, optional
            Coefficient, if None use the coefficient determined by the optimize_flow function (default: None)

        Returns
        -------
        np.ndarray
            2D array with (h, u) for t
        """
        t = np.array([t]) if isinstance(t, float) else np.array(t)
        _coef = self.coef if coef is None else coef

        # Flow thickness
        tpeak = self.trh_tovt * self.t_ovt
        _h = np.interp(t, [0, tpeak, self.t_ovt], [0, 1, 0]) ** _coef * self.h_peak
        _h[t < tpeak] = t[t < tpeak] * self.h_peak / tpeak

        # Flow velocity
        tpeak = self.tru_tovt * self.t_ovt
        _u = np.interp(t, [0, tpeak, self.t_ovt], [0, 1, 0]) ** _coef * self.u_peak
        _u[t < tpeak] = t[t < tpeak] * self.u_peak / tpeak

        return np.array([_h, _u])
