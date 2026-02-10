#!/usr/bin/env python3
"""
Solar and ozone fields for SPEEDY radiation.

Computes daily-average solar insolation and ozone absorption fields
updated every radiation timestep (nstrad).

Based on SPEEDY shortwave_radiation.f90:get_zonal_average_fields()
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from .util import TimeInfo
from .state import Config, Param, SolarState
from .constants import Constants
from .transformer import Transformer

class Solar:
    """
    Solar and ozone field computation.
    
    Computes zonally-averaged fields for radiation:
    - Solar insolation at TOA (with seasonal cycle)
    - Ozone absorption (upper and lower stratosphere)
    - Zenith angle correction
    - Stratospheric correction for polar night
    
    Based on SPEEDY shortwave_radiation.f90:get_zonal_average_fields()
    """
    
    # Solar constant (area-averaged)
    SOLC = 342.0  # W/m²
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        transformer: Transformer,  # Transformer instance for lat/lon
    ):
        """
        Initialize solar module.
        
        Args:
            config: Model configuration
            constants: Physical constants
            transformer: Transformer instance for geometry
        """
        self.config = config
        self.constants = constants
        self.transformer = transformer
        
        self._setup()
    
    def _setup(self):
        """
        Initialize solar constants.
        
        Precomputes latitude-dependent quantities.
        """
        # Get sin/cos of latitude from transformer
        # transformer.lat is half grid (NH), need full grid
        lat_half = self.transformer.lat  # [iy] degrees
        lat_full = jnp.concatenate([-lat_half, lat_half[::-1]])  # [il] S to N
        
        self.sia = jnp.sin(lat_full * jnp.pi / 180.0)  # [il]
        self.coa = jnp.cos(lat_full * jnp.pi / 180.0)  # [il]
    
    def initialize(self, param: Param, time: TimeInfo) -> SolarState:
        """
        Initialize solar state.
        
        Called once at model startup.
        
        Args:
            tyear: Time as fraction of year (0-1)
            
        Returns:
            Initial SolarState
        """
        return self.forward(param, time)
    
    @partial(jax.jit, static_argnames=['self'])
    def forward(self, param: Param, time: TimeInfo) -> SolarState:
        """
        Compute solar and ozone fields for current date.
        
        Called every radiation timestep (nstrad).
        
        Based on SPEEDY shortwave_radiation.f90:get_zonal_average_fields()
        
        Args:
            tyear: Time as fraction of year (0-1, 0 = Jan 1 00:00)
            
        Returns:
            SolarState with fsol, ozupp, ozone, zenit, stratz
        """
        ix, il = self.config.ix, self.config.il
        tyear = time.tyear
        
        # ====================================================================
        # Year phase (0 = winter solstice = Dec 22)
        # ====================================================================
        pi = jnp.pi
        alpha = 2.0 * pi * (tyear + 10.0 / 365.0)
        
        # Ozone seasonal factors
        coz1 = jnp.maximum(0.0, jnp.cos(alpha))
        coz2 = 1.8
        
        # Zenith angle parameters
        azen = 1.0
        nzen = 2
        rzen = -jnp.cos(alpha) * 23.45 * pi / 180.0
        
        # Polar night threshold
        fs0 = 6.0
        
        # ====================================================================
        # Daily-average solar radiation at TOA
        # ====================================================================
        topsr = self._compute_solar(tyear)  # [il]
        
        # ====================================================================
        # Compute fields for each latitude
        # ====================================================================
        # Legendre polynomial P2(sin(lat))
        flat2 = 1.5 * self.sia**2 - 0.5  # [il]
        
        # Solar flux (broadcast to full grid)
        fsol = jnp.broadcast_to(topsr[jnp.newaxis, :], (ix, il))  # [ix, il]
        
        # Ozone depth
        ozupp = jnp.full((ix, il), 0.5 * param.epssw)
        ozone_lat = 0.4 * param.epssw * (1.0 + coz1 * self.sia + coz2 * flat2)
        ozone = jnp.broadcast_to(ozone_lat[jnp.newaxis, :], (ix, il))
        
        # Zenith angle correction
        zenit_lat = 1.0 + azen * (1.0 - (self.coa * jnp.cos(rzen) + self.sia * jnp.sin(rzen))) ** nzen
        zenit = jnp.broadcast_to(zenit_lat[jnp.newaxis, :], (ix, il))
        
        # Apply zenith correction to ozone absorption
        ozupp = fsol * ozupp * zenit
        ozone = fsol * ozone * zenit
        
        # Polar night cooling correction
        stratz = jnp.maximum(fs0 - fsol, 0.0)
        
        return SolarState(
            fsol=fsol,
            ozupp=ozupp,
            ozone=ozone,
            zenit=zenit,
            stratz=stratz
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_solar(self, tyear: float) -> jax.Array:
        """
        Compute daily-average solar insolation at TOA.
        
        Based on Hartmann (1994).
        
        Args:
            tyear: Time as fraction of year
            
        Returns:
            topsr: Daily-average insolation [il] (W/m²)
        """
        csol = 4.0 * self.SOLC  # Full solar constant
        
        pi = jnp.pi
        alpha = 2.0 * pi * tyear
        
        # Fourier expansion for declination and distance
        ca1 = jnp.cos(alpha)
        sa1 = jnp.sin(alpha)
        ca2 = jnp.cos(2 * alpha)
        sa2 = jnp.sin(2 * alpha)
        ca3 = jnp.cos(3 * alpha)
        sa3 = jnp.sin(3 * alpha)
        
        # Solar declination (radians)
        decl = (0.006918 - 0.399912 * ca1 + 0.070257 * sa1
                - 0.006758 * ca2 + 0.000907 * sa2
                - 0.002697 * ca3 + 0.001480 * sa3)
        
        # Earth-Sun distance factor
        fdis = (1.000110 + 0.034221 * ca1 + 0.001280 * sa1
                + 0.000719 * ca2 + 0.000077 * sa2)
        
        cdecl = jnp.cos(decl)
        sdecl = jnp.sin(decl)
        tdecl = sdecl / cdecl
        
        # Hour angle at sunrise/sunset (vectorized over latitudes)
        csolp = jnp.clip(-tdecl * self.sia / self.coa, -1.0, 1.0)
        hlim = jnp.arccos(csolp)
        
        # Daily-average insolation
        topsr = csol * fdis * (hlim * self.sia * sdecl + self.coa * cdecl * jnp.sin(hlim)) / pi
        
        return topsr  # [il]
