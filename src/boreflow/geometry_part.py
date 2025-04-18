import numpy as np


class GeometryPart:
    """
    Represents a discretized segment of a geometry with associated parameters and wavefront data.
    Contains methods for initialization, simulation, and obtaining time-series data for flow variables.
    """

    # Geometry
    id: int
    geometry_x: np.ndarray
    geometry_z: np.ndarray
    geometry_n: float
    geometry_alpha: float
    simulated: bool = False

    # Discretise
    x: np.ndarray
    s: np.ndarray
    t: np.ndarray
    u: np.ndarray
    h_x: np.ndarray
    h_s: np.ndarray

    # Wavefront
    t_front: np.ndarray
    u_front: np.ndarray

    def __init__(self, id: int, x: np.ndarray, z: np.ndarray, n_manning: float) -> None:
        """
        Initialize a GeometryPart object with geometry data and Manning's roughness.

        Parameters
        ----------
        id : int
            Unique identifier for this geometry part.
        x : np.ndarray
            Array of two x-coordinates defining the start and end of the geometry part.
        z : np.ndarray
            Array of two z-coordinates (elevations) corresponding to the start and end.
        n_manning : float
            Manning's roughness coefficient for the geometry part.
        """
        # Check the input
        self.__check_geometry(x, z, n_manning)

        # Save the input
        self.id = id
        self.geometry_x = x
        self.geometry_z = z
        self.geometry_n = n_manning

        # Calculate the angle
        self.geometry_alpha = np.arctan((z[1] - z[0]) / (x[0] - x[1]))

    def __check_geometry(self, x: np.ndarray, z: np.ndarray, n_manning: float) -> None:
        """
        Check the consistency of the input geometry data.

        Parameters
        ----------
        x : np.ndarray
            Array of two x-coordinates for the geometry part.
        z : np.ndarray
            Array of two z-coordinates (elevations) for the geometry part.
        n_manning : float
            Manning's roughness coefficient.
        """
        # Check if x and z consists of two values, e.g. [xstart, xend]
        if len(x) != 2 or len(z) != 2:
            raise ValueError("Length of x, z, or both is not equal to 2.")

        # Check if the roughness is a float (Mannings coefficient)
        if not isinstance(n_manning, float):
            raise ValueError("Roughness n (Mannings coefficent) should be a float.")

    def init_simulation(self, dx: float) -> None:
        """
        Initialize the simulation for this geometry part by discretizing the x-coordinate.

        Parameters
        ----------
        dx : float
            The spacing between discretized x-coordinates.
        """
        # Discretise
        self.x = np.arange(self.geometry_x[0] + dx / 2, self.geometry_x[1], dx)
        self.s = (self.x - self.geometry_x[0]) / np.cos(self.geometry_alpha) + self.geometry_x[0]

        # Initial conditions
        self.t = np.array([])
        self.u = np.empty((0, len(self.x)))
        self.h_x = np.empty((0, len(self.x)))
        self.h_s = np.empty((0, len(self.x)))

        # Wavefront
        self.t_front = np.empty((len(self.x)))
        self.u_front = np.empty((len(self.x)))

    def get_xt(self, x: float, get_h_perpendicular: bool = True) -> np.ndarray:
        """
        Return the time series at a specific location x.

        Parameters
        ----------
        x : float
            The x-coordinate at which to retrieve the time series.
        get_h_perpendicular : bool, optional
            Whether to compute the flow depth perpendicular to the slope (default: True).

        Returns
        -------
        np.ndarray
            An array containing the time series at location x. The array includes time (t), water depth (h), and velocity (u).
            If the location x is outside the geometry, returns None.
        """
        # Check if this geometry part is simulated
        if not self.simulated:
            raise ValueError("Model not simulated")

        # Check if x is in discretisation
        if not (np.min(self.x) <= x and x <= np.max(self.x)):
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
                self.h_x[:, idx_lower] + (x - self.x[idx_lower]) / (self.x[idx_upper] - self.x[idx_lower]) * (self.h_x[:, idx_upper] - self.h_x[:, idx_lower])
            )

        return np.array([self.t, _h, _u])

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
        # Determine the time of passing of the wetting front (t_front)
        for j, _h in enumerate(self.h_s.T):
            idx = np.where(_h > threshold)[0]
            if len(idx) > 0:
                self.t_front[j] = np.interp(0.01, _h[idx[0] - 1 : idx[0] + 1], self.t[idx[0] - 1 : idx[0] + 1])
            else:
                self.t_front[j] = None

        # Determine the velocity of the wetting front (u_front) using second-order differences
        dx = self.x[1] - self.x[0]
        for j in range(len(self.t_front)):
            if j == 0:
                dt = self.t_front[1] - self.t_front[0]
            elif j == len(self.t_front) - 1:
                dt = self.t_front[-1] - self.t_front[-2]
            else:
                dt = (self.t_front[j + 1] - self.t_front[j - 1]) / 2
            self.u_front[j] = (dx / np.cos(self.geometry_alpha)) / dt
