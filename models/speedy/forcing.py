#!/usr/bin/env python3
"""
Time-dependent forcing orchestrator for SPEEDY model.

Orchestrates all forcing modules:
- Land surface model (land.py)
- Sea surface model (sea.py)  
- Solar/ozone fields (solar.py)
- Surface albedo and CO2 (this module)

ForcingState contains:
- ablco2: CO2 absorptivity
- qcorh: Humidity correction for diffusion
- albsfc: Combined surface albedo
- land: LandState
- sea: SeaState
- solar: SolarState

Based on SPEEDY forcing.f90, physics.f90
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .util import Utility, TimeInfo
from .state import Config, Param, ForcingState, DiagState
from .constants import Constants
from .transformer import Transformer
from .static import StaticFields
from .land import Land
from .sea import Sea
from .solar import Solar

class Forcing:
    """
    Forcing orchestrator for SPEEDY.
    
    Coordinates updates to all time-dependent forcing fields:
    - Land surface temperature (daily)
    - Sea surface temperature and ice (daily)
    - Solar and ozone fields (every nstrad steps)
    - Surface albedo (daily)
    - CO2 absorptivity (daily, optional trend)
    - Humidity correction for diffusion (daily)
    
    Based on SPEEDY forcing.f90 and physics.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        transformer: Transformer,  # Transformer instance
        static_fields: StaticFields,  # StaticFields instance
    ):
        """
        Initialize forcing orchestrator.
        
        Args:
            config: Model configuration
            constants: Physical constants
            transformer: Transformer instance for spectral transforms
            static_fields: StaticFields instance for boundary data
        """
        self.config = config
        self.constants = constants
        self.transformer = transformer
        self.static = static_fields
        
        # Initialize sub-modules
        self.land = Land(config, constants, transformer, static_fields)
        self.sea = Sea(config, constants, transformer, static_fields)
        self.solar = Solar(config, constants, transformer)
        
        # Setup forcing-specific constants
        self._setup()
    
    def _setup(self):
        """
        Setup time-independent forcing quantities.
        
        Based on SPEEDY forcing.f90:set_forcing(imode=0) and setgam()
        """
        # ====================================================================
        # Reference lapse rate for humidity correction
        # ====================================================================
        # gamlat = gamma / (1000 * grav) in K/m
        gamlat = self.constants.gamma / (1000.0 * self.constants.grav)
        self.gamlat = gamlat
        
        # Temperature correction in grid space: tcorh_grid = gamlat * phis0
        phis0 = self.static.get_field('phis0')  # [ix, il]
        self.tcorh_grid = gamlat * phis0  # [ix, il]
        
        # Pressure exponent for humidity correction
        self.pexp = 1.0 / (self.constants.rgas * gamlat)
        
        # ====================================================================
        # Static fields for albedo computation
        # ====================================================================
        self.alb0 = self.static.get_field('alb0')  # Bare land albedo [ix, il]
        self.fmask_l = self.static.get_field('fmask_l')  # Land fraction [ix, il]
        self.fmask_s = self.static.get_field('fmask_s')  # Sea fraction [ix, il]
    
    def initialize(self, param: Param, time: TimeInfo) -> ForcingState:
        """
        Initialize all forcing fields.
        
        Called once at model startup.
        
        Args:
            time: Initial model time
            
        Returns:
            Initial ForcingState
        """
        # Initialize sub-modules
        land_state = self.land.initialize(param, time)
        sea_state = self.sea.initialize(param, time)
        solar_state = self.solar.initialize(param, time)
        
        # Compute surface albedo
        albsfc, alb_l, alb_s, snowc = self._compute_surface_albedo(param, land_state.snowd_am, sea_state.sice_am)
        
        # Compute CO2 absorptivity
        ablco2 = self._compute_co2_absorptivity(param, time)
        
        # Compute humidity correction
        qcorh = self._compute_qcorh(param, land_state.stl_am, sea_state.sst_am)
        
        return ForcingState(
            ablco2=ablco2,
            qcorh=qcorh,
            albsfc=albsfc,
            alb_l=alb_l,
            alb_s=alb_s,
            snowc=snowc,
            land=land_state,
            sea=sea_state,
            solar=solar_state
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def forward(self, param: Param, state: ForcingState, diag: DiagState, time: TimeInfo) -> ForcingState:
        """
        Update all forcing fields.
        
        Called daily for land/sea/albedo/co2/qcorh.
        Solar is updated every nstrad steps (controlled by update_solar flag).
        
        Args:
            state: Current forcing state
            time: Current model time
            diag: Physics diagnostics (contains hfluxn, shf, evap, ssrd)
            update_solar: Whether to update solar fields (every nstrad steps)
            
        Returns:
            Updated ForcingState
        """
        # ====================================================================
        # 1. Update land surface model
        # ====================================================================
        land_state = self.land.forward(param, state.land, diag, time)
        
        # ====================================================================
        # 2. Update sea surface model
        # ====================================================================
        sea_state = self.sea.forward(param, state.sea, diag, time)
        
        # ====================================================================
        # 3. Update solar fields (only daily)
        # 4. Update CO2 absorptivity
        # ====================================================================
        def compute_daily():
            ablco2 = self._compute_co2_absorptivity(param, time)
            solar_state = self.solar.forward(param, time)
            return ablco2, solar_state
        def use_cached():
            return state.ablco2, state.solar
        daily = (time.step % self.config.nsteps) == 0
        ablco2, solar_state = jax.lax.cond(daily, compute_daily, use_cached)
        
        # ====================================================================
        # 5. Compute surface albedo
        # ====================================================================
        albsfc, alb_l, alb_s, snowc = self._compute_surface_albedo(param, land_state.snowd_am, sea_state.sice_am)
        
        # ====================================================================
        # 6. Compute humidity correction
        # ====================================================================
        qcorh = self._compute_qcorh(param, land_state.stl_am, sea_state.sst_am)
        
        return ForcingState(
            ablco2=ablco2,
            qcorh=qcorh,
            albsfc=albsfc,
            alb_l=alb_l,
            alb_s=alb_s,
            snowc=snowc,
            land=land_state,
            sea=sea_state,
            solar=solar_state
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_surface_albedo(self, param: Param, snowd_am: jax.Array, sice_am: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute surface albedo from snow and ice coverage.
        
        Based on SPEEDY forcing.f90 lines 55-61
        
        Args:
            snowd_am: Snow depth [ix, il] (mm water equivalent)
            sice_am: Sea ice fraction [ix, il] (0-1)
            
        Returns:
            albsfc: Combined surface albedo [ix, il]
            alb_l: Land albedo [ix, il]
            alb_s: Sea albedo [ix, il]
            snowc: Snow cover fraction [ix, il]
        """
        # Snow cover fraction (0-1)
        snowc = jnp.minimum(1.0, snowd_am / param.sd2sc)
        
        # Land albedo: bare land + snow effect
        alb_l = self.alb0 + snowc * (param.albsn - self.alb0)
        
        # Sea albedo: open sea + ice effect
        alb_s = param.albsea + sice_am * (param.albice - param.albsea)
        
        # Combined surface albedo (weighted by land fraction)
        albsfc = alb_s + self.fmask_l * (alb_l - alb_s)
        
        return albsfc, alb_l, alb_s, snowc
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_co2_absorptivity(self, param: Param, time: TimeInfo) -> float:
        """
        Compute CO2 absorptivity with optional time trend.
        
        Based on SPEEDY forcing.f90 lines 64-71
        
        Args:
            time: Current model time
            
        Returns:
            ablco2: CO2 absorptivity
        """
        # Reference year and rate of change
        iyear_ref = 1950
        del_co2 = 0.005  # Rate of change per year
        
        # Compute time-dependent CO2 absorptivity
        ablco2 = jax.lax.cond(
            self.config.increase_co2,
            lambda _: param.ablco2 * jnp.exp(del_co2 * (time.year + time.tyear - iyear_ref)),
            lambda _: param.ablco2,
            operand=None
        )
        
        return ablco2
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_qcorh(self, param: Param, stl_am: jax.Array, sst_am: jax.Array) -> jax.Array:
        """
        Compute humidity correction for horizontal diffusion.
        
        Based on SPEEDY forcing.f90 lines 84-99
        
        Args:
            stl_am: Land surface temperature [ix, il]
            sst_am: Sea surface temperature [ix, il]
            
        Returns:
            qcorh: Humidity correction in spectral space [mx, nx]
        """
        # Surface temperature (weighted average of land and sea)
        tsfc = self.fmask_l * stl_am + self.fmask_s * sst_am  # [ix, il]
        
        # Reference temperature (surface + orographic correction)
        tref = tsfc + self.tcorh_grid  # [ix, il]
        
        # Surface pressure factor
        psfc = (tsfc / tref) ** self.pexp  # [ix, il]
        
        # Saturation specific humidity at reference and surface conditions
        qref = Utility.get_qsat(tref, jnp.ones_like(psfc), -1.0)
        qsfc = Utility.get_qsat(tsfc, psfc, 1.0)
        
        # Humidity correction in grid space
        corh_grid = self.constants.refrh1 * (qref - qsfc)  # [ix, il]
        
        # Transform to spectral space
        qcorh = self.transformer.grid_to_spec(corh_grid)  # [mx, nx]
        
        # Apply truncation if needed
        qcorh = jax.lax.cond(
            self.config.ix == self.config.iy * 4,
            lambda x: self.transformer.apply_truncation(x),
            lambda x: x,
            qcorh
        )
        
        return qcorh
