#!/usr/bin/env python3
"""
Type definitions for SPEEDY-JAX model.

This module contains all the NamedTuple types used throughout the model
to avoid circular import dependencies.
"""

import jax
from typing import NamedTuple, Optional
from .util import TimeInfo
from dajax.models.base import add_operators

# ============================================================================
# Spectral and Grid State Classes
# ============================================================================

@add_operators
class SpectralState(NamedTuple):
    """Single time level of spectral variables"""
    vor: jax.Array  # Vorticity [mx,nx,kx]
    div: jax.Array  # Divergence [mx,nx,kx]
    t: jax.Array    # Temperature [mx,nx,kx]
    q: jax.Array    # Humidity [mx,nx,kx]
    ps: jax.Array   # Log surface pressure [mx,nx]

@add_operators    
class PhyState(NamedTuple):
    """Physical space variables for output/analysis"""
    u: jax.Array     # Zonal wind [ix,il,kx]
    v: jax.Array     # Meridional wind [ix,il,kx]
    t: jax.Array     # Temperature [ix,il,kx]
    q: jax.Array     # Specific humidity [ix,il,kx]
    ps: jax.Array    # Log surface pressure [ix,il]

@add_operators
class ObsState(NamedTuple):
    u: jax.Array # [nlev,nlat,nlon] pressure not sigma levels
    v: jax.Array
    t: jax.Array
    q: jax.Array 
    ps: jax.Array 
    u2: jax.Array
    v2: jax.Array
    wind: jax.Array

# ============================================================================
# Physics State Classes
# ============================================================================

class LandState(NamedTuple):
    """
    Land surface model prognostic state.
    
    Based on SPEEDY land_model.f90
    
    Fields with suffix:
    - _am: Used by atmospheric model
    - _lm: From land model
    """
    stl_am: jax.Array      # [ix, il] Land surface temperature for atmosphere
    snowd_am: jax.Array    # [ix, il] Snow depth (water equivalent, mm)
    soilw_am: jax.Array    # [ix, il] Soil water availability (0-1)
    stl_lm: jax.Array      # [ix, il] Land model surface temperature

class SeaState(NamedTuple):
    """
    Sea/ice model prognostic state.
    
    Based on SPEEDY sea_model.f90
    
    Fields with suffix:
    - _am: Used by atmospheric model
    - _om: From ocean model
    """
    sst_am: jax.Array      # [ix, il] Sea surface temperature for atmosphere
    sice_am: jax.Array     # [ix, il] Sea ice fraction (0-1)
    tice_am: jax.Array     # [ix, il] Sea ice temperature
    sst_om: jax.Array      # [ix, il] Ocean model SST
    sice_om: jax.Array     # [ix, il] Ocean model sea ice fraction
    tice_om: jax.Array     # [ix, il] Ocean model ice temperature
    ssti_om: jax.Array     # [ix, il] SST for sea-ice interface

class SolarState(NamedTuple):
    """
    Solar and ozone fields (updated every nstrad steps).
    
    Based on SPEEDY shortwave_radiation.f90:get_zonal_average_fields()
    """
    fsol: jax.Array        # [ix, il] Incoming solar flux at TOA (W/m²)
    ozupp: jax.Array       # [ix, il] Ozone absorption (upper stratosphere)
    ozone: jax.Array       # [ix, il] Ozone absorption (lower stratosphere)
    zenit: jax.Array       # [ix, il] Zenith angle correction factor
    stratz: jax.Array      # [ix, il] Stratospheric correction for polar night

class ForcingState(NamedTuple):
    """
    Time-dependent forcing fields.
    
    Updated daily for land/sea/albedo/co2/qcorh.
    Solar updated every nstrad steps.
    
    Note: tcorh is NOT included because it is time-independent
    and computed once during setup in dynamics._setup_orographic_corrections().
    
    Based on SPEEDY forcing.f90
    """
    ablco2: float          # CO2 absorptivity for longwave radiation
    qcorh: jax.Array       # [mx, nx] Humidity diffusion correction (complex)
    albsfc: jax.Array      # [ix, il] Combined surface albedo
    alb_l: jax.Array       # [ix, il] Land albedo (bare + snow)
    alb_s: jax.Array       # [ix, il] Sea albedo (open + ice)
    snowc: jax.Array       # [ix, il] Effective snow cover fraction
    land: LandState        # Land surface state
    sea: SeaState          # Sea surface state
    solar: SolarState      # Solar/ozone fields

class CachedState(NamedTuple):
    """
    Cached fields from shortwave radiation (updated every nstrad steps).
    
    These values persist between radiation calls and are used by:
    - tau2, stratc: Longwave radiation (every step)
    - ssrd: Surface fluxes (every step)
    - tsr, ssr: Diagnostics
    
    Based on SPEEDY shortwave_radiation.f90
    """
    tau2: jax.Array        # [ix, il, kx, 4] LW transmissivity (4 bands)
    stratc: jax.Array      # [ix, il, 2] Stratospheric correction
    ssrd: jax.Array        # [ix, il] Surface downward SW (W/m²)
    dfabs_sw: jax.Array    # [ix, il, kx] SW absorbed flux (W/m²)
    tsr: jax.Array         # [ix, il] TOA net SW (W/m²)
    ssr: jax.Array         # [ix, il] Surface net SW (W/m²)

class DiagState(NamedTuple):
    """Diagnostic outputs from physics (computed fresh each step)."""
    # Surface fluxes (indexed: 0=land, 1=sea, 2=weighted average)
    hfluxn: jax.Array      # [ix, il, 2] Net heat flux into surface (land, sea only)
    shf: jax.Array         # [ix, il, 3] Sensible heat flux
    evap: jax.Array        # [ix, il, 3] Evaporation (kg/m²/s)
    ustr: jax.Array        # [ix, il, 3] U-stress
    vstr: jax.Array        # [ix, il, 3] V-stress
    slru: jax.Array        # [ix, il, 3] Surface upward LW emission
    
    # Radiation diagnostics
    ssrd: jax.Array        # [ix, il] Surface downward SW (copy from cached)
    slrd: jax.Array        # [ix, il] Surface downward LW
    slr: jax.Array         # [ix, il] Surface net LW
    olr: jax.Array         # [ix, il] Outgoing LW
    
    # Precipitation
    precnv: jax.Array      # [ix, il] Convective precipitation (g/m²/s)
    precls: jax.Array      # [ix, il] Large-scale precipitation (g/m²/s)
    
    # Cloud diagnostics
    #cloudc: jax.Array      # [ix, il] Total cloud cover (0-1)
    #clstr: jax.Array       # [ix, il] Stratiform cloud cover (0-1)
    
    # Other
    #cbmf: jax.Array        # [ix, il] Cloud-base mass flux
    ts: jax.Array          # [ix, il] Surface temperature
    tskin: jax.Array       # [ix, il] Skin temperature
    u0: jax.Array          # [ix, il] Near-surface u-wind
    v0: jax.Array          # [ix, il] Near-surface v-wind
    t0: jax.Array          # [ix, il] Near-surface temperature

# ============================================================================
# Main Model State
# ============================================================================

class State(NamedTuple):
    """
    Complete model state for leapfrog time stepping with physics.
    
    Contains:
    - filt: Filtered old state F(1) - used for physics
    - curr: Current state F(2) - used for dynamics
    - forcing: All forcing fields (land, sea, solar, albedo, CO2, qcorh)
    - cached: Cached shortwave outputs for longwave and surface fluxes
    - time: Current model time
    
    The leapfrog scheme computes:
        new = filt + 2*dt * tendencies
        filt_new = RAW_filter(filt, curr, new)
        curr_new = new
    """
    filt: SpectralState    # Filtered old state (F₁)
    curr: SpectralState    # Current state (F₂)
    forcing: ForcingState  # Forcing fields
    cached: CachedState    # Cached radiation fields
    diag: DiagState        # Diagnostics fields
    time: TimeInfo         # Current model time

@add_operators
class Param(NamedTuple):
    """
    Physical parameters organized by category:
    1. Surface parameters
    2. Radiation parameters
    3. Convection parameters
    4. Large-scale condensation parameters
    5. Surface flux parameters
    6. Vertical diffusion parameters
    """
    # Surface properties
    albsea: float = 0.07   # Sea surface albedo
    albice: float = 0.60   # Sea ice albedo
    albsn: float = 0.60    # Snow albedo
    sd2sc: float = 60.0    # Snow depth to cover conversion
    emisfc: float = 0.98   # Surface emissivity
    
    # Convection
    psmin: float = 0.8         # Min normalized surface pressure for convection
    trcnv: float = 6.0         # Relaxation time towards reference state (hours)
    rhbl: float = 0.9          # RH threshold in boundary layer
    rhil: float = 0.7          # RH threshold in intermediate layers
    entmax: float = 0.5        # Max entrainment as fraction of cloud-base mass flux
    smf: float = 0.8           # Ratio of secondary to primary mass flux at cloud base
    
    # Large-Scale Condensation
    trlsc: float = 4.0         # Relaxation time for specific humidity (hours)
    rhlsc: float = 0.9         # Max RH threshold (at sigma=1)
    drhlsc: float = 0.1        # Vertical range of RH threshold
    rhblsc: float = 0.95       # RH threshold for boundary layer
    
    # Cloud parameters
    rhcl1: float = 0.30        # RH threshold for cloud cover = 0
    rhcl2: float = 1.00        # RH threshold for cloud cover = 1
    qacl: float = 0.20         # Specific humidity threshold for clouds
    wpcl: float = 0.2          # Cloud cover weight for sqrt(precip)
    pmaxcl: float = 10.0       # Max precip (mm/day) for cloud contribution
    clsmax: float = 0.60       # Maximum stratiform cloud cover
    clsminl: float = 0.15      # Minimum stratiform cloud cover over land
    gse_s0: float = 0.25       # DSE gradient for cloud cover = 0
    gse_s1: float = 0.40       # DSE gradient for cloud cover = 1
    albcl: float = 0.43        # Cloud albedo (for cloud cover = 1)
    albcls: float = 0.50       # Stratiform cloud albedo
    
    # Shortwave radiation
    epssw: float = 0.020   # Ozone fraction of solar constant
    absdry: float = 0.033  # Dry air absorption
    absaer: float = 0.033  # Aerosol absorption
    abswv1: float = 0.022  # Water vapor absorption (visible)
    abswv2: float = 15.0   # Water vapor absorption (near-IR)
    abscl1: float = 0.015  # Cloud absorption coefficient 1
    abscl2: float = 0.15   # Cloud absorption coefficient 2
    
    # Longwave radiation
    epslw: float = 0.05    # Fraction in "black" band
    ablwin: float = 0.3    # Window band absorption
    ablco2: float = 6.0    # CO2 band absorption
    ablwv1: float = 0.7    # H2O weak band absorption
    ablwv2: float = 50.0   # H2O strong band absorption
    ablcl1: float = 12.0   # Cloud LW absorption 1
    ablcl2: float = 0.6    # Cloud LW absorption 2
    
    # Surface fluxes
    fwind0: float = 0.95   # Ratio of near-sfc wind to lowest-level wind
    ftemp0: float = 1.00   # Weight for near-sfc extrapolation of temp
    fhum0: float = 0.80    # Weight for near-sfc extrapolation of humidity
    cdl: float = 2.4e-3    # Land drag coefficient
    cds: float = 1.0e-3    # Sea drag coefficient
    chl: float = 1.2e-3    # Land heat exchange coefficient
    chs: float = 0.9e-3    # Sea heat exchange coefficient
    vgust: float = 5.0     # Gustiness velocity (m/s)
    ctday: float = 1.0e-2  # Daily cycle correction
    dtheta: float = 3.0    # Theta gradient for stability (K)
    fstab: float = 0.67    # Stability factor weight
    hdrag: float = 2000.0  # Height factor for drag
    clambda: float = 7.0   # Heat conductivity in ice (W/m/K)
    clambsn: float = 7.0   # Heat conductivity in snow (W/m/K)
    
    # Vertical diffusion
    trvdi: float = 24.0    # Relaxation time (hours)
    trvds: float = 6.0     # Relaxation time over stable PBL

class Config(NamedTuple):
    """Model configuration"""
    # Spectral resolution
    trunc: int = 30     # Spectral truncation (T30)
    # Grid resolution (derived from trunc)
    ix: int = 96        # Number of longitudes
    iy: int = 24        # Number of latitudes per hemisphere
    # Vertical levels
    kx: int = 8         # Number of vertical levels

    # Time stepping
    dt: float = 2400.0     # Time step in seconds (default: 40 minutes)
    dtrad: float = 7200.0  # Radiation time step in seconds (default: 120 minutes)
    # Time stepping parameters
    rob: float = 0.05   # Robert filter coefficient
    wil: float = 0.53   # Williams filter coefficient
    alph: float = 0.5   # Semi-implicit coefficient

    # Diffusion parameters
    npowhd: int = 4
    thd: float = 2.4        # Vorticity/temp diffusion timescale (hours)
    thdd: float = 2.4       # Divergence diffusion timescale (hours)
    thds: float = 12.0      # Stratospheric diffusion timescale (hours)
    tdrs: float = 24.0*30.0 # Stratospheric drag timescale (hours)

    # Physics flags (Phase 2)
    sppt_on: bool = False         # Stochastic physics perturbations
    land_coupling_flag: int = 1   # 0=climatology, 1=slab model
    sea_coupling_flag: int = 0    # 0=climatology, 1=anomaly, 2=slab
    ice_coupling_flag: int = 1    # 0=climatology, 1=slab model
    increase_co2: bool = False    # Enable CO2 increase over time
    
    # Derived parameters (computed in __init__)
    il: Optional[int] = None   # Total latitudes = 2*iy
    nx: Optional[int] = None   # trunc + 2
    mx: Optional[int] = None   # trunc + 1
    nsteps: Optional[int] = None    # Number of steps per day
    nstrad: Optional[int] = None    # Shortwave radiation period (number of steps)

    @classmethod
    def create(cls, trunc: int = 30, dt: float = 2400.0, dtrad: float = 7200.0, kx: int = 8, **kwargs) -> 'Config':
        """
        Create configuration with automatic computation of derived parameters.
        
        Args:
            trunc: Spectral truncation (default: T30)
            dt: Model time step in seconds (default: 2400s = 40 min)
            dtrad: Radiation time step in seconds (default: 7200s = 120 min)
            kx: Number of vertical levels (default: 8)
            **kwargs: Override any other Config parameters
            
        Returns:
            Config with all derived parameters filled in
            
        Example:
            # Basic usage
            config = Config.create(trunc=30, dt=2400.0, kx=8)
            
            # With physics flags
            config = Config.create(trunc=30, dt=2400.0, land_coupling_flag=0)
            
            # Higher resolution
            config = Config.create(trunc=42, dt=1800.0, kx=8)
        """
        # Derived horizontal resolution from truncation
        # Rule: ix should be at least 3*trunc and divisible by 4
        ix = trunc * 3 + 6
        iy = ix // 4
        il = 2 * iy
        
        # Derived spectral dimensions
        nx = trunc + 2
        mx = trunc + 1
        
        # Derived time stepping parameters
        seconds_per_day = 86400.0
        nsteps = int(seconds_per_day / dt)
        nstrad = int(dtrad / dt)
        
        # Validate
        if seconds_per_day % dt != 0:
            raise ValueError(f"dt={dt}s must divide evenly into 86400s (one day)")
        if dtrad % dt != 0:
            raise ValueError(f"dtrad={dtrad}s must be a multiple of dt={dt}s")
        if nsteps % 2 != 0:
            raise ValueError(f"nsteps={nsteps} must be even for leapfrog scheme")
        
        return cls(
            trunc=trunc,
            ix=ix,
            iy=iy,
            kx=kx,
            dt=dt,
            dtrad=dtrad,
            il=il,
            nx=nx,
            mx=mx,
            nsteps=nsteps,
            nstrad=nstrad,
            **kwargs
        )
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        return (
            f"Config(\n"
            f"  Resolution: T{self.trunc}, {self.ix}x{self.il} grid, {self.kx} levels\n"
            f"  Spectral: mx={self.mx}, nx={self.nx}\n"
            f"  Time step: dt={self.dt}s ({self.dt/60:.1f} min), "
            f"nsteps={self.nsteps}/day\n"
            f"  Radiation: dtrad={self.dtrad}s ({self.dtrad/60:.1f} min), "
            f"nstrad={self.nstrad} steps\n"
            f"  Semi-implicit: α={self.alph}, Robert={self.rob}, Williams={self.wil}\n"
            f"  Diffusion: thd={self.thd}h, thdd={self.thdd}h, thds={self.thds}h\n"
            f"  Physics: land_coupling={self.land_coupling_flag}, "
            f"sea_coupling={self.sea_coupling_flag}\n"
            f")"
        )







