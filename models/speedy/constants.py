#!/usr/bin/env python3
"""
Physical and dynamical constants for SPEEDY model.

Based on SPEEDY Fortran modules:
- physical_constants.f90
- dynamical_constants.f90
- mod_radcon.f90

Note: Some constants that are better treated as tunable parameters
are in Config and Param (state.py) rather than here. These include:
- Surface albedo values (albsea, albice, albsn, sd2sc)
- Coupling flags
- Diffusion timescales (thd, thdd, thds, tdrs)
"""

from typing import NamedTuple

class Constants(NamedTuple):
    """
    Physical and dynamical constants.
    """
    
    # ========================================================================
    # Fundamental Physical Constants
    # ========================================================================
    
    rearth: float = 6.371e6    # Earth radius (m)
    omega: float = 7.292e-5    # Earth angular velocity (rad/s)
    grav: float = 9.81         # Gravitational acceleration (m/s^2)
    
    # ========================================================================
    # Thermodynamic Constants
    # ========================================================================
    
    p0: float = 1.0e5          # Reference pressure (Pa)
    cp: float = 1004.0         # Specific heat at constant pressure (J/kg/K)
    akap: float = 2.0/7.0      # R/Cp (kappa = 2/7 for diatomic gas)
    rgas: float = 2.0/7.0 * 1004.0  # Gas constant for dry air (J/kg/K) = akap*cp
    
    alhc: float = 2501.0       # Latent heat of condensation
    alhs: float = 2801.0       # Latent heat of sublimation
    sbc: float = 5.67e-8       # Stefan-Boltzmann constant (W/m^2/K^4)
    
    # ========================================================================
    # Reference Atmosphere Constants
    # ========================================================================
    
    gamma: float = 6.0         # Reference lapse rate (K/km)
    hscale: float = 7.5        # Reference scale height for pressure (km)
    hshum: float = 2.5         # Reference scale height for specific humidity (km)
    refrh1: float = 0.7        # Reference relative humidity of near-surface air
    
    # ========================================================================
    # Radiation Constants
    # ========================================================================
    
    # Solar radiation
    solc: float = 342.0        # Solar constant (area-averaged) (W/m^2)

def create_constants(**kwargs) -> Constants:
    """
    Create Constants instance with optional overrides.
    
    Args:
        **kwargs: Any constant to override from default values
        
    Returns:
        Constants instance
        
    Example:
        # Use defaults
        constants = create_constants()
        
        # Override specific values
        constants = create_constants(grav=9.80665, ablco2=8.0)
    """
    return Constants(**kwargs)

# ============================================================================
# Default Instance
# ============================================================================

DEFAULT_CONSTANTS = Constants()

if __name__ == "__main__":
    # Print constants for verification
    const = Constants()
    
    print("=" * 70)
    print("SPEEDY Physical and Dynamical Constants")
    print("=" * 70)
    
    print("\n--- Fundamental Physical Constants ---")
    print(f"  Earth radius:     {const.rearth/1e6:.3f} × 10⁶ m")
    print(f"  Angular velocity: {const.omega:.4e} rad/s")
    print(f"  Gravity:          {const.grav:.2f} m/s²")
    
    print("\n--- Thermodynamic Constants ---")
    print(f"  Reference pressure p0: {const.p0/1e5:.0f} × 10⁵ Pa")
    print(f"  Gas constant R:        {const.rgas:.1f} J/kg/K")
    print(f"  Specific heat Cp:      {const.cp:.1f} J/kg/K")
    print(f"  Kappa (R/Cp):          {const.akap:.4f}")
    print(f"  Latent heat (cond):    {const.alhc:.0f} J/g")
    print(f"  Stefan-Boltzmann:      {const.sbc:.2e} W/m²/K⁴")
    
    print("\n--- Reference Atmosphere ---")
    print(f"  Lapse rate γ:     {const.gamma:.1f} K/km")
    print(f"  Scale height H:   {const.hscale:.1f} km")
    print(f"  Humidity scale:   {const.hshum:.1f} km")
    print(f"  Reference RH:     {const.refrh1:.1f}")
    
    print("\n--- Radiation Constants ---")
    print(f"  Solar constant:   {const.solc:.0f} W/m²")
    
    print("=" * 70)