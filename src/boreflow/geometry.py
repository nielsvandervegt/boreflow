import numpy as np

from typing import Iterator, Union

from .boundary_conditions.bc_base import BCBase
from .geometry_part import GeometryPart


class Geometry():
    """
    Represents a discretized 1D geometry composed of multiple connected geometry parts,
    each defined by a pair of x and z coordinates, and associated Manning roughness values.
    """
    
    geometry_x: list
    geometry_z: list
    geometry_n: list
    geometry_s: list
    geometry_parts: list = []
    simulated: bool = False
    simulation_time: float
    boundary_condition: BCBase

    def __init__(self, x: np.ndarray, z: np.ndarray, n_manning: np.ndarray) -> None:
        """
        Initialize a new Geometry object by discretizing the input profile.

        Parameters
        ----------
        x : np.ndarray
            Array of x-coordinates.
        z : np.ndarray
            Array of corresponding z-coordinates (elevation).
        n_manning : np.ndarray
            Array of Manning's n values, one per segment (length = len(x) - 1).
        """
        # Init
        self.__iteration_index = 0

        # Check and save the input
        self.check_geometry(x, z, n_manning)
        self.geometry_x = np.array(x)
        self.geometry_z = np.array(z)
        self.geometry_n = np.array(n_manning)
        self.geometry_s = np.concatenate(([self.geometry_x[0]], np.sqrt((self.geometry_x[1:] - self.geometry_x[:-1])**2 + (self.geometry_z[1:] - self.geometry_z[:-1])**2)))
        self.geometry_s = np.cumsum(self.geometry_s)

        # Init the geometry parts
        self.geometry_parts = []
        for i in range(len(x) - 1):
            self.geometry_parts.append(GeometryPart(i+1, x[i:i+2], z[i:i+2], n_manning[i]))
    
    def __len__(self) -> int:
        """
        Returns the number of geometry parts.

        Returns
        -------
        int
            Number of discretized geometry segments.
        """
        return len(self.geometry_parts)

    def __iter__(self) -> Iterator[GeometryPart]:
        """
        Initializes an iterator over the geometry parts.
        """
        self.__iteration_index = 0
        return self

    def __next__(self) -> GeometryPart:
        """
        Returns the next GeometryPart in the iteration.

        Returns
        -------
        GeometryPart
            The next part in the geometry sequence.
        """
        if self.__iteration_index < len(self.geometry_parts):
            self.__iteration_index += 1
            return self.geometry_parts[self.__iteration_index - 1]
        else:
            self.__iteration_index = 0
            raise StopIteration
    
    def check_geometry(self, x: np.ndarray, z: np.ndarray, n_manning: np.ndarray):
        """
        Validates the consistency of the geometry input arrays.

        Parameters
        ----------
        x : np.ndarray
            Array of x-coordinates.
        z : np.ndarray
            Array of z-coordinates.
        n_manning : np.ndarray
            Array of Manning's n values.
        """
        # x and z should be of equal length
        if len(x) != len(z):
            raise ValueError("Arrays x and z should be of equal length.")
        
        # n_manning should be equal to x/z minus 1
        if len(n_manning) != (len(x) - 1):
            raise ValueError("Array n_manning should be equal to the length of x minus 1.")

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
        # Check if model is simulated
        if self.simulated == False:
            raise ValueError("Model not simulated")
        
        # Search the right geometry part
        _data = None
        for _geometry_part in self:
            _data = _geometry_part.get_xt(x, get_h_perpendicular)
            if _data is not None:
                break
        
        # Error if no geometrypart can be identified (outside the grid or between two SSSWE)
        if _data is None:
            print(f"Cannot get data for x={x}. Is the location outside the grid or around between two SSSWE boundaries?")
        
        # Return data
        return _data
    
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

    def get_peak_flow_x(self, get_h_perpendicular: bool = True) -> np.ndarray:
        """
        Get the peak flow characteristics along the x-coordinate.

        Parameters:
        ----------
        get_h_perpendicular : bool
            Whether to compute the perpendicular flow thickness (default is True).
        
        Returns:
        -------
        np.ndarray
            Numpy 2D array with [x, hpeak, upeak, ufront]
        """
        # Initialize empty lists for storing the results
        _x = np.array([])
        _hpeak = np.array([])
        _upeak = np.array([])
        _ufront = np.array([])

        # Loop over geometry parts and collect data
        for geometry_part in self:
            
            # Add the x-coordinates and front velocities
            _x = np.concatenate((_x, geometry_part.x))
            _ufront = np.concatenate((_ufront, geometry_part.u_front))
            
            # Choose the height array (perpendicular or horizontal)
            _h = geometry_part.h_s if get_h_perpendicular else geometry_part.h_x
            
            # Get the peak height and velocity
            _hpeak = np.concatenate((_hpeak, np.max(_h, axis=0)))
            _upeak = np.concatenate((_upeak, np.max(geometry_part.u, axis=0)))

        return np.array([_x, _hpeak, _upeak, _ufront])

    def get_peak_flow_s(self, get_h_perpendicular: bool = True) -> np.ndarray:
        """
        Get the peak flow characteristics along the geometry (s-coordinate)

        Parameters:
        ----------
        get_h_perpendicular : bool
            Whether to compute the perpendicular flow thickness (default is True).
        
        Returns:
        -------
        np.ndarray
            Numpy 2D array with [s, hpeak, upeak, ufront]
        """
        # Get flow
        _x, _hpeak, _upeak, _ufront = self.get_peak_flow_x(get_h_perpendicular)
        
        # Transform s into x
        _s = np.interp(_x, self.geometry_x, self.geometry_s)

        # Return
        return np.array([_s, _hpeak, _upeak, _ufront])
