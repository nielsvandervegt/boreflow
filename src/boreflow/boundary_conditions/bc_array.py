import numpy as np

from typing import Union

from .bc_base import BCBase


class BCArray(BCBase):
    """
    Boundary condition based on pre-defined (t, h, u) arrays.

    Attributes
    ----------
    t: np.ndarray
        Array with the time signal
    h: np.ndarray
        Array with the flow thickness signal
    u: np.ndarray
        Array with the flow velocity signal
    """

    t: np.ndarray
    h: np.ndarray
    u: np.ndarray

    def __init__(self, t: np.ndarray, h: np.ndarray, u: np.ndarray) -> None:
        """
        Initialize the boundary condition.
        """
        self.t = np.array(t)
        self.h = np.array(h)
        self.u = np.array(u)

        # Mask
        mask = ~(np.isnan(self.h) | np.isnan(self.u) | (self.h < 0) | (self.u < 0))
        self.t = self.t[mask]
        self.h = self.h[mask]
        self.u = self.u[mask]

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
        t = np.array([t]) if isinstance(t, float) else np.array(t)
        _h = np.interp(t, self.t, self.h)
        _u = np.interp(t, self.t, self.u)
        return np.array([_h, _u])
