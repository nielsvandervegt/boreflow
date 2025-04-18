import numpy as np

from tqdm import tqdm
from typing import Union

from .boundary_conditions.bc_base import BCBase
from .enum import Solver
from .geometry_part import GeometryPart


class FVM:
    """
    A class implementing the Finite Volume Method (FVM) for solving one-dimensional Shallow Water Equations (Maranzoni and Tomirotti, 2022).

    The FVM is used for time integration (Euler Forward, Runge-Kutta 4) and flux calculation (local Lax-Friedrichs).
    """

    # Parameters
    g = 9.81

    @staticmethod
    def run_fvm(
        geometry_part: GeometryPart, boundary_condition: Union[BCBase, GeometryPart], solver: Solver, t_end: float, cfl: float, max_dt: float, dx: float
    ):
        """
        Solve the one-dimensional Shallow Water Equations using a Finite Volume Method.

        Updates the solution within the provided GeometryPart object.

        Parameters
        ----------
        geometry_part : GeometryPart
            The domain to be simulated
        boundary_condition : Union[BCBase, GeometryPart]
            Boundary condition definition or the upstream GeometryPart
        solver : Solver
            Solver enum containing the numerical integration scheme (e.g., RK4 with LLF flux).
        t_end : float
            End time of the simulation.
        cfl : float
            Courant–Friedrichs–Lewy number for time step stability.
        max_dt : float
            Maximum allowed timestep
        dx : float
            Spatial resolution of the computational grid.
        """
        # Progress bar
        pbar = tqdm(total=t_end, desc=f"Part #{geometry_part.id}", bar_format="{l_bar}{bar}| Simulated: {n:.2f}/{total:.2f} sec")

        # Initial conditions
        geometry_part.init_simulation(dx)
        U = np.zeros((2, len(geometry_part.x)))

        # Finite volume
        t = 0.0
        while t < t_end:
            # Determine timestep, use as maximum 'max_dt'
            max_speed = np.maximum(FVM.max_wave_speed(U), 1e-8)
            dt = np.min([cfl * dx / max_speed, max_dt])
            if t + dt > t_end:
                dt = t_end - t
            if t == 0.0:
                dt = 1e-6

            # Update cells
            if solver == Solver.EF_LLF:
                U = FVM.euler_step(U, t, dt, dx, geometry_part, boundary_condition)
            elif solver == Solver.RK4_LLF:
                U = FVM.rk4_step(U, t, dt, dx, geometry_part, boundary_condition)
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            # Save
            geometry_part.t = np.append(geometry_part.t, float(t))
            geometry_part.h_x = np.concatenate((geometry_part.h_x, [U[0, :]]), axis=0)
            _div = np.divide(U[1, :], U[0, :], out=np.zeros_like(U[0, :]), where=U[0, :] > 1e-6)
            geometry_part.u = np.concatenate((geometry_part.u, [_div]), axis=0)

            # Update progress bar
            pbar.update(dt if t <= t_end else t_end - (t - dt))

            # Increase timestep
            t += dt

        # Determine the flow thickness perpendicular to the slope
        geometry_part.h_s = geometry_part.h_x * np.cos(geometry_part.geometry_alpha)

        # Flag the object as done
        geometry_part.simulated = True

        # Close the progress bar
        pbar.close()

    @staticmethod
    def euler_step(U, t, dt, dx, geometry_part, boundary_condition) -> np.ndarray:
        """
        Perform a single Euler Forward time step.

        Parameters
        ----------
        U : np.ndarray
            The current state of the solution (height and momentum).
        t : float
            Current time.
        dt : float
            Time step size.
        dx : float
            Spatial resolution.
        geometry_part : GeometryPart
            The part of the domain being simulated.
        boundary_condition : Union[BCBase, GeometryPart]
            Boundary condition or upstream geometry part.

        Returns
        -------
        np.ndarray
            Updated state after the Euler step.
        """
        rhs = FVM.compute_rhs(U, t, dx, geometry_part, boundary_condition)
        return U + dt * rhs

    @staticmethod
    def rk4_step(U, t, dt, dx, geometry_part, boundary_condition) -> np.ndarray:
        """
        Perform a single Runge-Kutta 4 time step.

        Parameters
        ----------
        U : np.ndarray
            The current state of the solution (height and momentum).
        t : float
            Current time.
        dt : float
            Time step size.
        dx : float
            Spatial resolution.
        geometry_part : GeometryPart
            The part of the domain being simulated.
        boundary_condition : Union[BCBase, GeometryPart]
            Boundary condition or upstream geometry part.

        Returns
        -------
        np.ndarray
            Updated state after the RK4 step.
        """
        k1 = FVM.compute_rhs(U, t, dx, geometry_part, boundary_condition)
        k2 = FVM.compute_rhs(U + 0.5 * dt * k1, t, dx, geometry_part, boundary_condition)
        k3 = FVM.compute_rhs(U + 0.5 * dt * k2, t, dx, geometry_part, boundary_condition)
        k4 = FVM.compute_rhs(U + dt * k3, t, dx, geometry_part, boundary_condition)
        return U + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def compute_rhs(U, t, dx, geometry_part: GeometryPart, bc: Union[BCBase, GeometryPart]) -> np.ndarray:
        """
        Compute the right-hand side (RHS) of the Shallow Water Equations for the finite volume method.

        The RHS is calculated as the flux differences between neighboring cells, along with the source terms
        (such as friction and gravitational forces). The fluxes are computed using the local Lax-Friedrichs method.

        Parameters
        ----------
        U : np.ndarray
            The current state of the solution at all grid points (2 x Nx), where:
            - U[0, :] represents the water height (h)
            - U[1, :] represents the water momentum (h * u)
        t : float
            The current time of the simulation.
        dx : float
            The spatial resolution of the computational grid (distance between grid points).
        geometry_part : GeometryPart
            The specific geometry being simulated, which includes grid information, slope angle, and roughness.
        bc : Union[BCBase, GeometryPart]
            The boundary condition object or the upstream geometry part providing boundary conditions.

        Returns
        -------
        np.ndarray
            The right-hand side of the Shallow Water Equations (2 x Nx array).
            This will be used for updating the solution in the finite volume method.
        """
        # Compute fluxes
        Nx = U.shape[1]
        F_interface = np.zeros((2, Nx + 1))
        for i in range(Nx + 1):
            if i == 0:
                UL = np.array(FVM.boundary_conditions(t, geometry_part, bc))
                UR = U[:, 0]
            elif i == Nx:
                UL = U[:, Nx - 1]
                UR = U[:, Nx - 1]
            else:
                UL = U[:, i - 1]
                UR = U[:, i]
            F_interface[:, i] = FVM.rusanov_flux(UL, UR, geometry_part.geometry_alpha)

        S = FVM.compute_source_friction(U, geometry_part)

        # Compute the spatial operator
        rhs = np.zeros_like(U)
        for i in range(Nx):
            rhs[:, i] = -(F_interface[:, i + 1] - F_interface[:, i]) / dx + S[:, i]

        # Return the right hand side
        return rhs

    @staticmethod
    def boundary_conditions(t: float, geometry_part: GeometryPart, bc: Union[BCBase, GeometryPart]):
        """
        Return the boundary condition at the upstream interface. The boundary condition can be derived
        from either a Boundary Condition (BCBase) object or from the downstream interface of the upstream GeometryPart.

        Parameters
        ----------
        t : float
            Current time of the simulation.
        geometry_part : GeometryPart
            Geometry part for which the boundary condition is being applied.
        bc : Union[BCBase, GeometryPart]
            Boundary condition, which can be a BCBase object or the upstream GeometryPart providing the boundary condition.

        Returns
        -------
        list
            The boundary condition at the interface as a list:
            [h, h * u] where `h` is the water height and `u` is the velocity (momentum / height).
        """
        if isinstance(bc, BCBase):
            _h, _u = bc.get_flow(t)
            _h = _h[0] / np.cos(geometry_part.geometry_alpha)
            return [_h, _h * _u[0]]
        elif isinstance(bc, GeometryPart):
            _bc_h = np.array(bc.h_x[:, -1]) * np.cos(bc.geometry_alpha)
            _h = np.interp(t, bc.t, _bc_h)
            _h = _h / np.cos(geometry_part.geometry_alpha)
            _u = np.interp(t, bc.t, bc.u[:, -1])
            return [_h, _h * _u]
        else:
            raise NotImplementedError("Unknown object to derive the boundary conditions from")

    @staticmethod
    def flux_state(U_vec, alpha):
        """
        Compute the flux for a given state (h, hu) using the shallow water equations.

        Parameters
        ----------
        U_vec : np.ndarray
            The state vector containing [h, hu] where `h` is the water height and `hu` is the water momentum (h * u).
        alpha : float
            The slope angle of the geometry.

        Returns
        -------
        np.ndarray
            The flux vector [F0, F1] where:
            - F0 is the flux related to the water height (h)
            - F1 is the flux related to the momentum (hu)
        """
        h = U_vec[0]
        u = U_vec[1] / h if h > 1e-6 else 0.0
        F0 = u * h * np.cos(alpha)
        F1 = (u**2 * h + 0.5 * FVM.g * h**2 * (np.cos(alpha) ** 2)) * np.cos(alpha)
        return np.array([F0, F1])

    @staticmethod
    def rusanov_flux(UL, UR, alpha) -> np.ndarray:
        """
        Compute the local Lax-Friedrichs flux at the interface between the left (UL) and right (UR) states.

        This method estimates the maximum wave speed and uses it to compute the flux at the interface between two adjacent cells
        using the Lax-Friedrichs method.

        Parameters
        ----------
        UL : np.ndarray
            The state vector [h, hu] at the left cell of the interface.
        UR : np.ndarray
            The state vector [h, hu] at the right cell of the interface.
        alpha : float
            The slope angle of the geometry.

        Returns
        -------
        np.ndarray
            The flux vector at the interface [F0, F1].
        """
        FL = FVM.flux_state(UL, alpha)
        FR = FVM.flux_state(UR, alpha)

        # Estimate wave speeds for left and right states:
        uL = UL[1] / UL[0] if UL[0] > 1e-6 else 0.0
        uR = UR[1] / UR[0] if UR[0] > 1e-6 else 0.0
        cL = np.sqrt(FVM.g * UL[0])
        cR = np.sqrt(FVM.g * UR[0])
        smax = max(abs(uL) + cL, abs(uR) + cR)

        return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)

    @staticmethod
    def compute_source_friction(U, geometry_part: GeometryPart, h_threshold: float = 0.001) -> np.ndarray:
        """
        Compute the source term related to friction for the shallow water equations.

        The source term accounts for the frictional forces due to the roughness of the terrain (Manning's roughness)
        and gravitational forces acting on the water surface.

        Parameters
        ----------
        U : np.ndarray
            The state vector [h, hu] at each grid point where:
            - U[0, :] represents the water height (h)
            - U[1, :] represents the water momentum (h * u)
        geometry_part : GeometryPart
            The geometry part being simulated, including slope angle and roughness.
        h_threshold : float, optional
            A threshold for the water height below which the friction is considered negligible (default: 0.001).

        Returns
        -------
        np.ndarray
            The source term for the friction (2 x Nx array).
        """
        S = np.zeros_like(U)
        h = U[0, :]

        # Avoid division by zero by setting u to 0 when h is below the threshold
        u = U[1, :] / np.maximum(h, h_threshold)

        # Ensure the friction term does not cause overflow by limiting h and using np.maximum
        ft = (
            FVM.g
            * h
            * (geometry_part.geometry_n**2 * u**2)
            / (np.maximum(h ** (4 / 3), h_threshold ** (4 / 3)))
            * np.sqrt(1 + np.tan(geometry_part.geometry_alpha) ** 2)
        )
        friction_term = np.where(h > h_threshold, ft, 0.0)

        # Update source term
        S[1, :] = FVM.g * h * np.sin(geometry_part.geometry_alpha) - friction_term
        return S

    @staticmethod
    def max_wave_speed(U: np.ndarray) -> float:
        """
        Compute the maximum wave speed over all cells in the domain.

        The maximum wave speed is the sum of the velocity and the wave speed (due to gravity) at each grid point,
        and this is used to determine the appropriate time step size in the simulation.

        Parameters
        ----------
        U : np.ndarray
            The state vector [h, hu] at each grid point where:
            - U[0, :] represents the water height (h)
            - U[1, :] represents the water momentum (h * u)

        Returns
        -------
        float
            The maximum wave speed across all grid points.
        """
        h = U[0, :]
        hu = U[1, :]
        u = np.zeros(len(h))
        u[h > 1e-6] = hu[h > 1e-6] / h[h > 1e-6]
        c = np.sqrt(FVM.g * h)
        return np.max(np.abs(u) + c)
