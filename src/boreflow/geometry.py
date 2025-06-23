import numpy as np

from typing import Union

from .boundary_conditions.bc_base import BCBase


class Geometry:
    """
    Represents a discretized 1D geometry composed of multiple connected geometry parts,
    each defined by a pair of x and z coordinates, and associated Manning roughness values.

    Attributes
    ----------
    geometry_x : np.ndarray
        Array of x-coordinates.
    geometry_s : np.ndarray
        Arrau of s-coordinates (distance along the geometry)
    geometry_z : np.ndarray
        Array of corresponding z-coordinates (elevation).
    geometry_n : np.ndarray
        Array of Manning's n values, one per segment (length = len(x) - 1).
    x : np.ndarray
        Discretisation cell centers x
    s : np.ndarray
        Discretisation cell centers s
    z : np.ndarray
        Discretisation cell centers z
    t : np.ndarray
        Time steps of the results
    u : np.ndarray
        Results flow velocity [t, x]
    h : np.ndarray
        Results flow thickness along z-coordinates [t, x]
    h_s : np.ndarray
        Results flow thickness perpendicular to slope [t, x]
    t_front : np.ndarray
        Time of passing of the wetting front [x]
    u_front : np.ndarray
        Velocity of the wetting front [x]
    simulated : bool
        Flag whether the model is simulated
    simulation_time : float
        Time it took to simulate the model
    boundary_condition : BCBase
        The boundary condition applied when simulated
    """

    # Geometry
    geometry_x: np.ndarray
    geometry_s: np.ndarray
    geometry_z: np.ndarray
    geometry_n: np.ndarray

    # Discretisation
    x: np.ndarray
    s: np.ndarray
    z: np.ndarray

    # Results
    t: np.ndarray
    u: np.ndarray
    h: np.ndarray
    h_s: np.ndarray
    t_front: np.ndarray
    u_front: np.ndarray

    # Other
    simulated: bool
    simulation_time: float
    boundary_condition: BCBase

    def __init__(self, geometry_x: np.ndarray, geometry_z: np.ndarray, geometry_n: np.ndarray) -> None:
        """
        Initialize a new Geometry object by discretizing the input profile.
        """
        # Check and save the input
        self.geometry_x = np.array(geometry_x)
        self.geometry_z = np.array(geometry_z)
        self.geometry_n = np.array(geometry_n)
        self.check_geometry()

        # Calculate alpha and slope coordinate
        dx = self.geometry_x[1:] - self.geometry_x[:-1]
        dz = self.geometry_z[1:] - self.geometry_z[:-1]
        self.geometry_alpha = -np.arctan(dz / dx)
        self.geometry_s = np.concatenate(([self.geometry_x[0]], np.sqrt(dx**2 + dz**2)))
        self.geometry_s = np.cumsum(self.geometry_s)

    def check_geometry(self):
        """
        Validates the consistency of the geometry input arrays.
        """
        # x and z should be of equal length
        if len(self.geometry_x) != len(self.geometry_z):
            raise ValueError("Arrays x and z should be of equal length.")

        # n_manning should be equal to x/z minus 1
        if len(self.geometry_n) != (len(self.geometry_x) - 1):
            raise ValueError("Array n_manning should be equal to the length of x minus 1.")

    def derive_front_velocity(self, threshold: float = 0.01) -> None:
        """
        Derive the time and velocity of the wetting front for this geometry part.

        The wetting front is considered as the point where the water depth exceeds a given threshold.
        The time of passing (t_front) is calculated based on linear interpolation.
        The velocity (u_front) of the wetting front is calculated based on second-order differences.

        Parameters
        ----------
        threshold : float, optional
            The threshold for determining the wetting front location (default: 0.01m)
        """
        # Wavefront
        self.t_front = np.empty((len(self.x)))
        self.u_front = np.empty((len(self.x)))

        # Determine the time of passing of the wetting front (t_front)
        for j, _h in enumerate(self.h_s.T):
            idx = np.where(_h > threshold)[0]
            if len(idx) > 0:
                self.t_front[j] = np.interp(0.01, _h[idx[0] - 1 : idx[0] + 1], self.t[idx[0] - 1 : idx[0] + 1])
            else:
                self.t_front[j] = None

        # Determine the velocity of the wetting front (u_front) using second-order differences
        for j in range(len(self.t_front)):
            if j == 0:
                dt = self.t_front[1] - self.t_front[0]
                dx = self.s[1] - self.s[0]
            elif j == len(self.t_front) - 1:
                dt = self.t_front[-1] - self.t_front[-2]
                dx = self.s[-1] - self.s[-2]
            else:
                dt = (self.t_front[j + 1] - self.t_front[j - 1]) / 2
                dx = (self.s[j + 1] - self.s[j - 1]) / 2
            self.u_front[j] = dx / dt

    def get_xt(self, x: float, get_h_perpendicular: bool = True) -> Union[np.ndarray, None]:
        """
        Get the time series of flow variables at a specific x-location.

        Parameters
        ----------
        x : float
            The x-coordinate at which to retrieve the time series.
        get_h_perpendicular : bool, optional
            Whether to compute the perpendicular water depth (default is True).

        Returns
        -------
        np.ndarray or None
            A np.ndarray containing the time series [t, h, u] at location x, or None if x is outside the modeled domain.
        """
        # Check if this geometry part is simulated
        if not self.simulated:
            raise ValueError("Model not simulated")

        # Check if x is in discretisation
        if x < self.x[0] or self.x[-1] < x:
            print(f"Cannot get data for x={x}. Is the location outside the grid?")
            return None

        # Search for lower x and upper x
        idx_lower = np.array([np.abs(_x - x) for _x in self.x]).argmin()
        idx_upper = idx_lower + 1

        # Interpolate
        _u = np.array(self.u[:, idx_lower] + (x - self.x[idx_lower]) / (self.x[idx_upper] - self.x[idx_lower]) * (self.u[:, idx_upper] - self.u[:, idx_lower]))
        if get_h_perpendicular:
            _h = np.array(
                self.h_s[:, idx_lower] + (x - self.x[idx_lower]) / (self.x[idx_upper] - self.x[idx_lower]) * (self.h_s[:, idx_upper] - self.h_s[:, idx_lower])
            )
        else:
            _h = np.array(
                self.h[:, idx_lower] + (x - self.x[idx_lower]) / (self.x[idx_upper] - self.x[idx_lower]) * (self.h[:, idx_upper] - self.h[:, idx_lower])
            )

        return np.array([self.t, _h, _u])

    def get_st(self, s: float, get_h_perpendicular: bool = True) -> Union[np.ndarray, None]:
        """
        Get the time series of flow variables at a specific s-location along the slope.

        Parameters
        ----------
        s : float
            The slope-based coordinate (distance along the geometry).
        get_h_perpendicular : bool, optional
            Whether to compute the perpendicular flow thickness (default is True).

        Returns
        -------
        np.ndarray or None
            A np.ndarray containing the time series at location x, or None if x is outside the modeled domain.
        """
        # Transform s into x
        _x = np.interp(s, self.geometry_s, self.geometry_x)

        # Return
        return self.get_xt(_x, get_h_perpendicular)

    def get_peak_flow(self, get_h_perpendicular: bool = True, quantile: float = 1.0) -> np.ndarray:
        """
        Get the peak flow characteristics along the x-coordinate.

        Parameters:
        ----------
        get_h_perpendicular : bool
            Whether to compute the perpendicular flow thickness (default is True).
        quantile : float, optional
            Get the quantiles, quantile=1 is the maximum values (default: 1.0)

        Returns:
        -------
        np.ndarray
            Numpy 2D array with [hpeak, upeak, qpeak]
        """
        # Check if this geometry part is simulated
        if not self.simulated:
            raise ValueError("Model not simulated")

        # Choose the height array (perpendicular or horizontal)
        _h = self.h_s if get_h_perpendicular else self.h
        _hpeak = np.quantile(_h, quantile, axis=0)
        _upeak = np.quantile(self.u, quantile, axis=0)
        _qpeak = np.quantile(self.u * _h, quantile, axis=0)

        return np.array([_hpeak, _upeak, _qpeak])
