#!/usr/bin/env python3
"""
Vertical grid structure for SPEEDY model.

Defines sigma coordinate levels and reference atmospheric profiles.
Based on SPEEDY vertical_grid.f90 and mod_radcon.f90
"""

import jax
import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple

from .constants import Constants

class VerticalGrid:
    """
    Vertical grid structure using sigma coordinates.
    From SPEEDY geometry.f90
    
    Sigma coordinate: σ = p/p_s (pressure normalized by surface pressure)
    
    """
    
    def __init__(self, kx: int, constants: Optional[Constants] = None):
        self.kx = kx
        
        if constants is None:
            constants = Constants()
        self.constants = constants
        
        # Define sigma levels exactly as in SPEEDY Fortran
        if kx == 8:
            # Half levels (interfaces) - from SPEEDY geometry.f90
            hsg_array = [0.000, 0.050, 0.140, 0.260, 0.420, 0.600, 0.770, 0.900, 1.000]
            self.hsg = np.array(hsg_array)
        elif kx == 7:
            hsg_array = [0.020, 0.140, 0.260, 0.420, 0.600, 0.770, 0.900, 1.000]
            self.hsg = np.array(hsg_array)
        elif kx == 5:
            hsg_array = [0.000, 0.150, 0.350, 0.650, 0.900, 1.000]
            self.hsg = np.array(hsg_array)
        else:
            # Generic distribution for other kx
            self.hsg = np.linspace(0.0, 1.0, kx + 1)
        
        # Full levels (where T, u, v are defined) - midpoints of half levels
        # VECTORIZED: fsg[k] = 0.5 * (hsg[k+1] + hsg[k]) for all k
        self.fsg = 0.5 * (self.hsg[1:] + self.hsg[:-1])  # [kx]
        
        # fsgr = akap / (2 * fsg) - for temperature equation - VECTORIZED
        self.fsgr = constants.akap / (2.0 * self.fsg)  # [kx]

        # Layer thickness - VECTORIZED
        self.dhs = np.diff(self.hsg)  # [kx]
        
        # Reciprocal layer thickness - VECTORIZED
        self.dhsr = 0.5 / self.dhs  # [kx]
        
        # Reference temperature profile (for semi-implicit scheme)
        self.tref, self.tref1, self.tref2, self.tref3 = self._setup_reference_temperature()
        
        # Geopotential constants
        self.xgeop1, self.xgeop2 = self._setup_geopotential_constants()

        # Physics constants
        self.sigl, self.grdsig, self.grdscp, self.wvi = self._setup_physics_constants()
        
    def _setup_reference_temperature(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Setup reference temperature profile for semi-implicit scheme.
        From SPEEDY implicit.f90
        
        Reference atmosphere is a function of sigma only:
        T_ref(σ) = 288 K * max(0.2, σ)^(R*γ/(1000*g))
        
        where γ is the lapse rate (K/km)
        """
        gamma = self.constants.gamma  # K/km (dynamical constant)
        rgas = self.constants.rgas
        grav = self.constants.grav
        akap = self.constants.akap
        
        # Polytropic exponent
        rgam = rgas * gamma / (1000.0 * grav)
        
        tref = 288.0 * np.maximum(0.2, self.fsg)**rgam  # [kx]
        
        # VECTORIZED: Derived reference temperature arrays
        tref1 = rgas * tref   # R * T_ref (for geopotential) [kx]
        tref2 = akap * tref   # akap * T_ref (for temperature equation) [kx]
        tref3 = self.fsgr * tref  # (fsg * R) * T_ref [kx]
        
        return tref, tref1, tref2, tref3
    
    def _setup_geopotential_constants(self) -> Tuple[jax.Array, jax.Array]:
        """
        Setup constants for geopotential calculation.
        From SPEEDY geopotential.f90
        """
        rgas = self.constants.rgas
        
        # VECTORIZED: xgeop1[k] = R * log(hsg[k+1] / fsg[k]) for all k
        # hsg[1:] is hsg[k+1] for k=0..kx-1
        # fsg is fsg[k] for k=0..kx-1
        xgeop1 = rgas * np.log(self.hsg[1:] / self.fsg)  # [kx]
        
        # VECTORIZED: xgeop2[k] = R * log(fsg[k] / hsg[k]) for k=1..kx-1
        # xgeop2[0] = 0 (not used)
        xgeop2 = np.zeros(self.kx)
        xgeop2[1:] = rgas * np.log(self.fsg[1:] / self.hsg[1:-1])  # [kx-1]
        
        return xgeop1, xgeop2
    
    def _setup_physics_constants(self):
        """
        Setup physics-related vertical coordinate arrays.
        
        Based on SPEEDY physics.f90:initialize_physics()
        
        Computes:
        - sigl: Log of full sigma levels
        - sigh: Half sigma levels (shifted for physics)
        - grdsig: Flux to tendency conversion (momentum/humidity)
        - grdscp: Flux to tendency conversion (temperature)
        - wvi: Weights for vertical interpolation at half-levels
        """
        kx = self.kx
        grav = self.constants.grav
        cp = self.constants.cp
        p0 = self.constants.p0
        
        hsg = self.hsg  # Half sigma levels [kx+1]
        fsg = self.fsg  # Full sigma levels [kx]
        dhs = self.dhs  # Layer thickness [kx]
        
        # Log of full sigma levels
        sigl = np.log(fsg)  # [kx]
        
        # Half sigma levels for physics (same as hsg)
        sigh = hsg.copy()  # [kx+1]
        
        # Flux to tendency conversion factors
        grdsig = grav / (dhs * p0)  # [kx]
        grdscp = grdsig / cp   # [kx]
        
        # Weights for vertical interpolation at half-levels
        wvi = np.zeros((kx, 2))
        
        # Interior levels (k = 0..kx-2)
        # wvi[k, 0] = 1 / (sigl[k+1] - sigl[k])
        # wvi[k, 1] = (log(sigh[k+1]) - sigl[k]) * wvi[k, 0]
        wvi[:kx-1, 0] = 1.0 / (sigl[1:] - sigl[:-1])
        wvi[:kx-1, 1] = (np.log(sigh[1:kx]) - sigl[:kx-1]) * wvi[:kx-1, 0]
        
        # Surface level (k = kx-1)
        wvi[kx-1, 0] = 0.0
        wvi[kx-1, 1] = (np.log(0.99) - sigl[kx-1]) * wvi[kx-2, 0]

        return sigl, grdsig, grdscp, wvi[:,1]

def create_vertical_grid(kx: int = 8, **kwargs) -> VerticalGrid:
    """
    Create VerticalGrid instance with optional constants override.
    
    Args:
        kx: Number of vertical levels
        **kwargs: Optional arguments to pass to VerticalGrid
        
    Returns:
        VerticalGrid instance
        
    Example:
        # Use defaults
        vgrid = create_vertical_grid(kx=8)
        
        # With custom constants
        from constants import create_constants
        const = create_constants(gamma=5.5)
        vgrid = create_vertical_grid(kx=8, constants=const)
    """
    return VerticalGrid(kx, **kwargs)

if __name__ == "__main__":
    # Print vertical grid for verification
    vgrid = VerticalGrid(kx=8)
    
    print("=" * 70)
    print("SPEEDY Vertical Grid Structure")
    print("=" * 70)
    
    print(f"\nNumber of levels: {vgrid.kx}")
    print(f"\nSigma coordinate levels:")
    print(f"{'Level':<8} {'σ_half':<10} {'σ_full':<10} {'Δσ':<10} {'T_ref (K)':<12}")
    print("-" * 70)
    
    for k in range(vgrid.kx):
        sigma_half = float(vgrid.fsg[k])
        sigma_full = float(vgrid.hsg[k])
        delta_sigma = float(vgrid.dhs[k])
        temp_ref = float(vgrid.tref[k])
        
        print(f"{k:<8} {sigma_half:<10.3f} {sigma_full:<10.3f} {delta_sigma:<10.3f} "
              f"{temp_ref:<12.1f}")
    
    # Print bottom boundary
    print(f"{'Surface':<8} {float(vgrid.hsg[-1]):<10.3f}")
    
    print("\n" + "=" * 70)
