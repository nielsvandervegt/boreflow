import numpy as np
from tqdm import tqdm

from .boundary_conditions.bc_base import BCBase
from .geometry import Geometry
from .enum import Flux, Limiter, TimeIntegration


class FVM:
    """
    Class to solve the SSSWE
    """

    # Objects
    geometry: Geometry
    bc_left: BCBase

    # Discretisation
    dx: float
    x_cells: np.ndarray
    z_cells: np.ndarray
    n_cells: np.ndarray
    x_interfaces: np.ndarray

    # Simulation parameters
    t_end: float
    max_dt: float
    cfl: float

    # Other
    g: float = 9.81
    first_dt: float = 1e-6
    h_wet: float = 1e-3  # h < h_min is assumed to be dry
    h_min: float = 1e-6

    def __init__(self, boundary_condition: BCBase, t_end: float, max_dt: float, cfl: float):
        """
        Initialize FVM solver
        """
        # Save parameters
        self.bc_left = boundary_condition
        self.t_end = t_end
        self.max_dt = max_dt
        self.cfl = cfl

    def discretise(self, geometry: Geometry, nx: int):
        # Save
        self.geometry = geometry
        self.nx = nx

        # x_cells and z_cells (add ghost cells to both ends)
        self.dx = (geometry.geometry_x[-1] - geometry.geometry_x[0]) / self.nx
        self.x_cells = np.linspace(-self.dx / 2, geometry.geometry_x[-1] + self.dx / 2, self.nx + 2)
        self.z_cells = np.interp(self.x_cells, geometry.geometry_x, geometry.geometry_z)
        self.z_cells[0] = (self.z_cells[1] - self.z_cells[2]) + self.z_cells[1]
        self.z_cells[-1] = (self.z_cells[-2] - self.z_cells[-3]) + self.z_cells[-2]

        # Roughness and slope in all interior cells (no ghost cells)
        self.n_cells = np.zeros_like(self.x_cells)
        self.n_cells[1:-1] = geometry.geometry_n[np.searchsorted(geometry.geometry_x, self.x_cells[1:-1], side="left") - 1]
        self.alpha_cells = np.zeros_like(self.x_cells)
        self.alpha_cells[1:-1] = np.arctan((self.z_cells[:-2] - self.z_cells[2:]) / (2 * self.dx))

        # Init results
        geometry.t = np.array([])
        geometry.x = self.x_cells[1:-1]
        geometry.s = np.interp(geometry.x, geometry.geometry_x, geometry.geometry_s)
        geometry.u = np.empty((0, self.nx))
        geometry.h = np.empty((0, self.nx))
        geometry.h_s = np.empty((0, self.nx))

    def run(self, limiter: Limiter, flux: Flux, timeintegration: TimeIntegration):
        """"""
        # Progress bar
        pbar = tqdm(total=self.t_end, desc="Simulating", bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} s")

        # Initial conditions: Empty cells + ghost cells
        U = np.zeros((2, self.nx + 2))

        # Time stepping
        t = 0.0
        while t < self.t_end:
            # CFL to determine dt
            max_speed = np.maximum(self.compute_max_velocity(U), 1e-8)
            dt = np.min([self.cfl * self.dx / max_speed, self.max_dt])
            if t + dt > self.t_end:
                dt = self.t_end - t

            # If t = 0, use self.first_dt as timestep
            if t == 0.0:
                dt = self.first_dt

            # Time step
            if timeintegration == TimeIntegration.EF:
                rhs = self.compute_rhs(t, U, limiter, flux)
                U = U + dt * rhs
            elif timeintegration == TimeIntegration.RK2:
                k1 = self.compute_rhs(t, U, limiter, flux)
                k2 = self.compute_rhs(t + dt, U + 0.5 * dt * k1, limiter, flux)
                U = U + dt * k2
            else:
                raise NotImplementedError()

            # Save results
            self.geometry.t = np.concatenate((self.geometry.t, [t]))
            self.geometry.h = np.concatenate((self.geometry.h, [U[0, 1:-1]]), axis=0)
            u = np.divide(U[1, 1:-1], U[0, 1:-1], out=np.zeros_like(U[0, 1:-1]), where=U[0, 1:-1] > self.h_wet)
            self.geometry.u = np.concatenate((self.geometry.u, [u]), axis=0)

            # Update progress bar and increase time
            pbar.update(dt)
            t += dt

        # Calculate water depth perpendicular to slope
        self.geometry.h_s = self.geometry.h * np.cos(self.alpha_cells[1:-1])

        # Close progress bar
        pbar.close()

    def compute_rhs(self, t: float, U: np.ndarray, limiter: Limiter, flux: Flux):
        # 1) Add boundary conditions
        _h, _u = self.bc_left.get_flow(t).T[0]
        U[:, 0] = np.array([_h, _h * _u]) if _h > self.h_min else np.array([0.0, 0.0])
        U[:, -1] = U[:, -2]

        # 2) Apply limiter to interior cells
        deltaU = self.limiter(U, limiter)
        deltaAlpha = self.limiter(np.array([self.alpha_cells]), limiter)[0]

        # 3) Reconstruct fluxes and alpha at left and right of interface
        UL = U[:, :-1] + 0.5 * deltaU[:, :-1]
        UR = U[:, 1:] - 0.5 * deltaU[:, 1:]
        alphaL = self.alpha_cells[:-1] + 0.5 * deltaAlpha[:-1]
        alphaR = self.alpha_cells[1:] - 0.5 * deltaAlpha[1:]

        # 4) Calculate flux
        F = np.zeros_like(UL)
        if flux == Flux.HLL:
            for i in range(len(F[0])):
                F[:, i] = self.hll_flux(UL[:, i], UR[:, i], alphaL[i], alphaR[i])
        else:
            raise NotImplementedError()

        # 5) Calculate source
        S = self.source_term(U)

        # 6) Calculate and return RHS
        rhs = np.zeros_like(U)
        rhs[:, 1:-1] = -(F[:, 1:] - F[:, :-1]) / self.dx + S[:, 1:-1]
        return rhs

    def limiter(self, U: np.ndarray, limiter: Limiter):
        """"""
        # Create an empty deltaU
        deltaU = np.zeros_like(U)

        # Apply limiter at each interface (2, ..., N-1)
        for i in range(len(U)):
            dL = U[i, 1:-1] - U[i, :-2]
            dR = U[i, 2:] - U[i, 1:-1]
            if limiter == Limiter.minmod:
                deltaU[i, 1:-1] = np.where(dL * dR > 0, np.sign(dL) * np.minimum(np.abs(dL), np.abs(dR)), 0.0)
            elif limiter == Limiter.vanLeer:
                _limiter = np.zeros_like(deltaU[i, 1:-1])
                _mask = dL * dR > 0
                _limiter[_mask] = 2 * dL[_mask] * dR[_mask] / (dL[_mask] + dR[_mask])
                deltaU[i, 1:-1] = _limiter
            else:
                raise NotImplementedError()

        # Return
        return deltaU

    def hll_flux(self, UL, UR, alphaL, alphaR):
        # Get h and u
        hL, huL = UL
        hR, huR = UR

        # If the water depth is too shallow, assume dry cell
        if hL < self.h_min:
            hL = 0.0
            huL = 0.0
            uL = 0.0
        else:
            uL = huL / hL
        if hR < self.h_min:
            hR = 0.0
            huR = 0.0
            uR = 0.0
        else:
            uR = huR / hR

        # If both cells have no water depth, return no flux
        if hL < self.h_min and hR < self.h_min:
            return np.array([0, 0])

        # Reconstruct
        UL = np.array([hL, huL])
        UR = np.array([hR, huR])

        # Calculate wave speeds
        cL = np.sqrt(self.g * hL) * np.cos(alphaL)
        cR = np.sqrt(self.g * hR) * np.cos(alphaR)
        sL = np.minimum(uL * np.cos(alphaL) - cL, uR * np.cos(alphaR) - cR)
        sR = np.maximum(uL * np.cos(alphaL) + cL, uR * np.cos(alphaR) + cR)

        # Left and right flux
        FL = np.array([hL * uL * np.cos(alphaL), (hL * uL**2 + 0.5 * self.g * hL**2 * np.cos(alphaL) ** 2) * np.cos(alphaL)])
        FR = np.array([hR * uR * np.cos(alphaR), (hR * uR**2 + 0.5 * self.g * hR**2 * np.cos(alphaR) ** 2) * np.cos(alphaR)])

        # HLL
        if sL > 0:
            return FL
        elif sR < 0:
            return FR
        else:
            return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)

    def source_term(self, U):
        # Dry
        h, hu = U

        # Calculate Source Terms (Momentum only)
        source_term = np.zeros((2, self.nx + 2))
        for i in range(1, self.nx + 1):  # Avoid ghost cells
            # Apply dry conditions
            if h[i] <= self.h_wet:
                source_term[1, i] = 0.0
                continue

            # Friction term for source
            u_i = hu[i] / h[i]
            friction_term = (self.n_cells[i] ** 2 * u_i * abs(u_i)) / h[i] ** (4 / 3) * np.sqrt(1 + np.tan(self.alpha_cells[i]) ** 2)
            source_term[1, i] = FVM.g * h[i] * (np.sin(self.alpha_cells[i]) - friction_term)

        return source_term

    def compute_max_velocity(self, U: np.ndarray):
        """"""
        # Init empty array
        u = np.zeros_like(U[0])

        # Avoid division by zero for dry cells
        u[U[0] > FVM.h_min] = U[1][U[0] > FVM.h_min] / U[0][U[0] > FVM.h_min]
        return np.max(np.abs(u))
