#!/usr/bin/env python3
"""
Static/boundary condition fields for SPEEDY model.

Implements:
- Reading boundary condition files (NetCDF)
- Field storage and access
- Derived field computation (masks, spectral truncation)

Based on SPEEDY Fortran modules:
- boundaries.f90
- land_model.f90
- sea_model.f90

"""

import pathlib
import jax
import jax.numpy as jnp
import xarray as xr
from typing import Dict, Optional
from functools import partial

from .state import Config
from .util import Utility
from .constants import Constants
from .transformer import Transformer

class StaticFields:
    """
    Manages static/boundary condition fields for SPEEDY model.
    
    Provides:
    - NetCDF file reading with xarray
    - Field storage and access
    """
    
    def __init__(self, config: Config, constants: Constants, transformer: Transformer, data_dir: Optional[str] = None):
        """
        Initialize static fields manager.
        
        Args:
            config: Config instance from speedy_jax
            constants: Constants instance from speedy_jax
            transformer: Transformer instance
            data_dir: Path to data directory (default: './data/speedy')
        """
        self.config = config
        self.constants = constants
        self.transformer = transformer
        
        # Set data directory
        if data_dir is None:
            # Try to find data directory relative to project root
            dajax_root = pathlib.Path(__file__).parent.parent.parent
            self.data_dir = dajax_root / 'data' / 'speedy'
        else:
            self.data_dir = pathlib.Path(data_dir)
        
        # Field storage (similar to shqg.py)
        self.fields: Dict[str, jax.Array] = {}
        
        #print(f"Initializing static fields from {self.data_dir}")
        
        # Read and process fields
        self._read_boundary_fields()
        self._process_boundary_fields()
        self._read_land_fields()
        self._read_sea_fields()
        
        #print(f"  Loaded {len(self.fields)} static fields")
    
    # ========================================================================
    # Field Access Methods
    # ========================================================================
    
    def get_field(self, name: str) -> jax.Array:
        """Get a field by name."""
        if name not in self.fields:
            raise KeyError(f"Field '{name}' not found. Available: {list(self.fields.keys())}")
        return self.fields[name]
    
    def set_field(self, name: str, data: jax.Array) -> None:
        """Set a field by name."""
        self.fields[name] = data
    
    def has_field(self, name: str) -> bool:
        """Check if a field exists."""
        return name in self.fields
    
    # ========================================================================
    # NetCDF Reading Boundary Fields
    # ========================================================================
    
    def _read_boundary_fields(self) -> None:
        """
        Read boundary condition fields from surface.nc.
        
        Fields read:
        - orog: Orography (m)
        - lsm: Land-sea mask (fractional)
        - alb: Albedo
        
        Creates:
        - phi0: Unfiltered surface geopotential (m²/s²)
        - fmask: Land-sea mask
        - alb0: Albedo
        
        Note: Input file is assumed to be on the same grid as the model.
        """
        surface_file = self.data_dir / 'surface.nc'
        
        if not surface_file.exists():
            raise FileNotFoundError(
                f"Boundary file not found: {surface_file}\n"
                f"Please ensure data files are in {self.data_dir}"
            )
        
        #print(f"  Reading boundary fields from {surface_file.name}")
        
        # Read NetCDF file
        ds = xr.open_dataset(surface_file)
        
        # Read and process orography
        if 'orog' in ds.variables:
            orog = ds['orog'].values.T[:, ::-1]  # [ix,il] or [lon,lat]
            
            # Convert to geopotential: phi = g * h
            phi0 = self.constants.grav * orog
            self.set_field('phi0', jnp.array(phi0))
            #print(f"    phi0: shape={phi0.shape}, min={phi0.min():.1f}, max={phi0.max():.1f} m²/s²")
        else:
            raise KeyError("'orog' field not found in surface.nc")
        
        # Read land-sea mask
        if 'lsm' in ds.variables:
            fmask = ds['lsm'].values.T[:, ::-1]
            
            self.set_field('fmask', jnp.array(fmask))
            #print(f"    fmask: shape={fmask.shape}, min={fmask.min():.3f}, max={fmask.max():.3f}")
        else:
            raise KeyError("'lsm' field not found in surface.nc")
        
        # Read albedo
        if 'alb' in ds.variables:
            alb0 = ds['alb'].values.T[:, ::-1]
            
            self.set_field('alb0', jnp.array(alb0))
            #print(f"    alb0: shape={alb0.shape}, min={alb0.min():.3f}, max={alb0.max():.3f}")
        else:
            raise KeyError("'alb' field not found in surface.nc")
        
        ds.close()

    # ========================================================================
    # Derived Field Computation
    # ========================================================================
    
    def _process_boundary_fields(self) -> None:
        """
        Compute derived fields from basic boundary conditions.
        
        Computes:
        - phis0: Spectrally-filtered surface geopotential
        - fmask_l, fmask_s: Land and sea masks
        - bmask_l, bmask_s: Binary land and sea masks
        """
        #print("  Computing derived fields")
        
        # 1. Spectral truncation of surface geopotential
        phi0 = self.get_field('phi0')
        phis0 = self.transformer.spectral_truncation(phi0)
        self.set_field('phis0', phis0)
        #print(f"    phis0 (truncated): min={phis0.min():.1f}, max={phis0.max():.1f} m²/s²")
        # Convert to spectral space for dynamics
        phis_spec = self.transformer.grid_to_spec(phis0)
        self.set_field('phis_spec', phis_spec)
        #print(f"    phis_spec (spectral): shape={phis_spec.shape}, max={jnp.abs(phis_spec).max():.2e}")
        
        # 2. Land and sea masks
        fmask = self.get_field('fmask')
        threshold = 0.1  # From SPEEDY land_model.f90:65 and sea_model.f90:122
        
        # Land mask
        fmask_l = fmask.copy()
        # Where mask >= threshold, keep it; where > 1-threshold, set to 1
        fmask_l = jnp.where(fmask >= threshold, fmask, 0.0)
        fmask_l = jnp.where(fmask > (1.0 - threshold), 1.0, fmask_l)
        self.set_field('fmask_l', fmask_l)
        
        # Binary land mask
        bmask_l = jnp.where(fmask >= threshold, 1.0, 0.0)
        self.set_field('bmask_l', bmask_l)
        
        # Sea mask
        fmask_s = 1.0 - fmask
        fmask_s = jnp.where(fmask_s >= threshold, fmask_s, 0.0)
        fmask_s = jnp.where(fmask_s > (1.0 - threshold), 1.0, fmask_s)
        self.set_field('fmask_s', fmask_s)
        
        # Binary sea mask
        bmask_s = jnp.where(fmask_s >= threshold, 1.0, 0.0)
        self.set_field('bmask_s', bmask_s)
        
        #print(f"    Land fraction: {fmask_l.mean():.3f}")
        #print(f"    Sea fraction: {fmask_s.mean():.3f}")
    
    # ========================================================================
    # NetCDF Reading Land Sea Fields
    # ========================================================================
    
    def _read_land_fields(self) -> None:
        """
        Read land surface fields from land.nc, snow.nc, soil.nc, surface.nc.
        Based on SPEEDY land_model.f90:land_model_init
        
        Assumes: NetCDF files are already on model grid with correct dimension ordering [ix, il, 12]
        
        Fields read and processed:
        - stl12: Land surface temperature (12 months)
        - snowd12: Snow depth (12 months)
        - veg_high, veg_low: Vegetation fractions → combined
        - swl1, swl2: Soil moisture (12 months) → soilw12
        
        Creates:
        - stl12, snowd12, soilw12: Monthly climatologies [ix, il, 12]
        """
        bmask_l = self.get_field('bmask_l')
        
        #print("  Reading land surface fields")
        
        # ========================================================================
        # 1. Land Surface Temperature (12 months)
        # ========================================================================
        ds_land = xr.open_dataset(self.data_dir / 'land.nc')
        stl12_raw = jnp.array(ds_land['stl'].values.transpose(2, 1, 0)[:, ::-1, :])  # [ix, il, 12]
        ds_land.close()
        
        # Fill missing values for each month
        stl12 = jnp.zeros_like(stl12_raw)
        for month in range(12):
            stl12 = stl12.at[:, :, month].set(Utility.fillsf(stl12_raw[:, :, month], 0.0))
        
        # Check consistency with land mask
        stl12 = Utility.forchk(bmask_l, 273.0, stl12)
        self.set_field('stl12', stl12)
        #print(f"    stl12: range=[{stl12.min():.1f}, {stl12.max():.1f}] K")
        
        # ========================================================================
        # 2. Snow Depth (12 months)
        # ========================================================================
        ds_snow = xr.open_dataset(self.data_dir / 'snow.nc')
        snowd12 = jnp.array(ds_snow['snowd'].values.transpose(2, 1, 0)[:, ::-1, :])  # [ix, il, 12]
        ds_snow.close()
        
        # Check consistency with land mask
        snowd12 = Utility.forchk(bmask_l, 0.0, snowd12)
        self.set_field('snowd12', snowd12)
        #print(f"    snowd12: range=[{snowd12.min():.1f}, {snowd12.max():.1f}] mm")
        
        # ========================================================================
        # 3. Vegetation and Soil Moisture → Soil Water Availability (12 months)
        # ========================================================================
        
        # Read vegetation fractions
        ds_surface = xr.open_dataset(self.data_dir / 'surface.nc')
        veg_high = jnp.array(ds_surface['vegh'].values.T[:, ::-1])  # [ix, il]
        veg_low = jnp.array(ds_surface['vegl'].values.T[:, ::-1])   # [ix, il]
        ds_surface.close()
        
        # Combine vegetation fractions
        veg = jnp.maximum(0.0, veg_high + 0.8 * veg_low)
        #print(f"    veg: range=[{veg.min():.3f}, {veg.max():.3f}]")
        
        # Read soil moisture
        ds_soil = xr.open_dataset(self.data_dir / 'soil.nc')
        swl1 = jnp.array(ds_soil['swl1'].values.transpose(2, 1, 0)[:, ::-1, :])  # [ix, il, 12]
        swl2 = jnp.array(ds_soil['swl2'].values.transpose(2, 1, 0)[:, ::-1, :])  # [ix, il, 12]
        ds_soil.close()
        
        # Compute soil water availability (from land_model.f90:105-133)
        swcap = 0.30   # Soil wetness at field capacity
        swwil = 0.17   # Soil wetness at wilting point
        idep2 = 3      # Depth ratio
        
        swwil2 = idep2 * swwil
        rsw = 1.0 / (swcap + idep2 * (swcap - swwil))
        
        # Vectorized computation for all months
        swroot = idep2 * swl2  # [ix, il, 12]
        soilw12 = jnp.minimum(1.0, rsw * (swl1 + veg[:, :, jnp.newaxis] * jnp.maximum(0.0, swroot - swwil2)))
        
        # Check consistency with land mask
        soilw12 = Utility.forchk(bmask_l, 0.0, soilw12)
        self.set_field('soilw12', soilw12)
        #print(f"    soilw12: range=[{soilw12.min():.3f}, {soilw12.max():.3f}]")
        
    def _read_sea_fields(self) -> None:
        """
        Read sea surface fields from sea.nc, sice.nc.
        Based on SPEEDY sea_model.f90:sea_model_init
        
        Assumes: NetCDF files are already on model grid with correct dimension ordering [ix, il, 12]
        
        Fields read:
        - sst12: Sea surface temperature (12 months)
        - sice12: Sea ice fraction (12 months)
        
        Creates:
        - sst12, sice12: Monthly climatologies [ix, il, 12]
        
        Note: SST anomalies (sstan3) not implemented - only climatological mode supported
        """
        bmask_s = self.get_field('bmask_s')
        
        #print("  Reading sea surface fields")
        
        # ========================================================================
        # 1. Sea Surface Temperature (12 months)
        # ========================================================================
        ds_sea = xr.open_dataset(self.data_dir / 'sea_surface_temperature.nc')
        sst12_raw = jnp.array(ds_sea['sst'].values.transpose(2, 1, 0)[:, ::-1, :])  # [ix, il, 12]
        ds_sea.close()
        
        # Fill missing values for each month
        sst12 = jnp.zeros_like(sst12_raw)
        for month in range(12):
            sst12 = sst12.at[:, :, month].set(Utility.fillsf(sst12_raw[:, :, month], 0.0))
        
        # Check consistency with sea mask
        # SST range: 271.4 K (freezing) to ~303 K (tropical)
        sst12 = Utility.forchk(bmask_s, 273.0, sst12)
        self.set_field('sst12', sst12)
        #print(f"    sst12: range=[{sst12.min():.1f}, {sst12.max():.1f}] K")
        
        # ========================================================================
        # 2. Sea Ice Fraction (12 months)
        # ========================================================================
        ds_sice = xr.open_dataset(self.data_dir / 'sea_ice.nc')
        sice12 = jnp.array(ds_sice['icec'].values.transpose(2, 1, 0)[:, ::-1, :])  # [ix, il, 12]
        ds_sice.close()
        
        # Check consistency with sea mask
        # Sea ice fraction: 0.0 (no ice) to 1.0 (full ice cover)
        sice12 = Utility.forchk(bmask_s, 0.0, sice12)
        self.set_field('sice12', sice12)
        #print(f"    sice12: range=[{sice12.min():.3f}, {sice12.max():.3f}]")

