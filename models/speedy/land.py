#!/usr/bin/env python3
"""
Slab land surface model for SPEEDY.

Implements a simple slab model for land surface temperature evolution:
- Heat capacity depends on soil vs ice sheet (based on albedo)
- Temperature anomaly evolves based on net heat flux
- Relaxation towards climatology

Snow depth and soil water availability are always from climatology.

Based on SPEEDY land_model.f90
"""

import jax
import jax.numpy as jnp
from functools import partial

from .util import Utility, TimeInfo
from .state import Config, LandState, Param, DiagState
from .constants import Constants
from .transformer import Transformer
from .static import StaticFields

class Land:
    """
    Slab land surface model.
    
    The land model evolves surface temperature based on:
    - Net heat flux from atmosphere (hfluxn)
    - Relaxation towards climatology (cdland)
    - Heat capacity (different for soil vs ice sheet)
    
    Snow depth and soil water are always from climatology.
    
    Based on SPEEDY land_model.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        transformer: Transformer,  # Transformer instance
        static_fields: StaticFields,  # StaticFields instance
    ):
        """
        Initialize land model.
        
        Args:
            config: Model configuration
            constants: Physical constants
            transformer: Transformer instance (for grid info)
            static_fields: StaticFields instance for boundary data
        """
        self.config = config
        self.constants = constants
        self.transformer = transformer
        self.static_fields = static_fields
        
        # Setup model parameters (vectorized)
        self._setup()
    
    def _setup(self):
        """
        Initialize land model constants.
        
        Computes heat capacity and dissipation coefficient arrays.
        All masks and climatology are from StaticFields.
        
        Based on SPEEDY land_model.f90:land_model_init()
        """
        dt = self.config.dt
        
        # ====================================================================
        # Get masks and climatology from StaticFields
        # ====================================================================
        fmask_l = self.static_fields.get_field('fmask_l')  # [ix, il]
        alb0 = self.static_fields.get_field('alb0')        # [ix, il]
        self.stl12 = self.static_fields.get_field('stl12')      # [ix, il, 12]
        self.snowd12 = self.static_fields.get_field('snowd12')  # [ix, il, 12]
        self.soilw12 = self.static_fields.get_field('soilw12')  # [ix, il, 12]
        
        # ====================================================================
        # Model parameters
        # ====================================================================
        depth_soil = 1.0     # Soil layer depth (m)
        depth_lice = 5.0     # Land-ice depth (m)
        tdland = 40.0        # Dissipation time (days) for temp anomalies
        flandmin = 1.0/3.0   # Min land fraction for anomaly definition
        
        # Heat capacities per m^2 (depth * heat_capacity/m^3)
        hcapl = depth_soil * 2.50e6   # Soil
        hcapli = depth_lice * 1.93e6  # Ice sheet
        
        # ====================================================================
        # Compute heat capacity array (vectorized)
        # Ice sheet identified by high albedo (>= 0.4)
        # ====================================================================
        rhcapl = jnp.where(alb0 < 0.4, dt / hcapl, dt / hcapli)
        self.rhcapl = rhcapl  # [ix, il] dt/heat_capacity
        
        # ====================================================================
        # Compute dissipation coefficient (vectorized)
        # dmask blanks out points with insufficient land fraction
        # cdland = dmask * tdland / (1 + dmask * tdland)
        # ====================================================================
        dmask = jnp.where(fmask_l >= flandmin, 1.0, 0.0)
        self.cdland = dmask * tdland / (1.0 + dmask * tdland)  # [ix, il]
    
    def initialize(self, param: Param, time: TimeInfo) -> LandState:
        """
        Initialize land surface state from climatology.
        
        Called once at model startup.
        
        Args:
            time: Initial model time
            
        Returns:
            Initial LandState
        """
        # Interpolate climatology to initial date
        stl_cl = Utility.forin5(self.stl12, time.imont1, time.tmonth)
        snowd_cl = Utility.forint(self.snowd12, time.imont1, time.tmonth)
        soilw_cl = Utility.forint(self.soilw12, time.imont1, time.tmonth)
        
        # Initialize from climatology
        return LandState(stl_am=stl_cl, snowd_am=snowd_cl, soilw_am=soilw_cl, stl_lm=stl_cl)
    
    @partial(jax.jit, static_argnames=['self'])
    def forward(self, param: Param, state: LandState, diag: DiagState, time: TimeInfo) -> LandState:
        """
        Advance land model one timestep.
        
        Based on SPEEDY land_model.f90:couple_land_atm()
        
        Args:
            state: Current land model state
            time: Current model time (for climatology interpolation)
            diag: Physics diagnostics containing hfluxn
            
        Returns:
            Updated LandState
        """
        # ====================================================================
        # 1. Interpolate climatology to current date
        # ====================================================================
        stlcl_ob = Utility.forin5(self.stl12, time.imont1, time.tmonth)
        snowdcl_ob = Utility.forint(self.snowd12, time.imont1, time.tmonth)
        soilwcl_ob = Utility.forint(self.soilw12, time.imont1, time.tmonth)
        
        # ====================================================================
        # 2. Update land surface temperature
        # ====================================================================
        # Run land model or use climatology based on coupling flag
        stl_lm_new, stl_am = jax.lax.cond(
            self.config.land_coupling_flag == 1,
            lambda _: self._run(param, state.stl_lm, stlcl_ob, diag),
            lambda _: (stlcl_ob, stlcl_ob),  # Use climatology
            operand=None
        )
        
        # ====================================================================
        # 3. Snow and soil water always from climatology
        # ====================================================================
        return LandState(stl_am=stl_am, snowd_am=snowdcl_ob, soilw_am=soilwcl_ob, stl_lm=stl_lm_new)
    
    @partial(jax.jit, static_argnames=['self'])
    def _run(self, param: Param, stl_lm: jax.Array, stlcl_ob: jax.Array, diag: DiagState) -> tuple:
        """
        Integrate slab land model for one timestep.
        
        Based on SPEEDY land_model.f90:run_land_model()
        
        The temperature anomaly evolves as:
            tanom_new = cdland * (tanom + rhcapl * hfluxn)
            stl_lm_new = tanom_new + stlcl_ob
        
        Args:
            stl_lm: Current land model temperature [ix, il]
            stlcl_ob: Climatological temperature [ix, il]
            diag: Physics diagnostics containing hfluxn
            
        Returns:
            (stl_lm_new, stl_am): Updated temperatures
        """
        # Extract land heat flux from diagnostics
        hfluxn_land = diag.hfluxn[:, :, 0]  # [ix, il]

        # Temperature anomaly w.r.t. climatology
        tanom = stl_lm - stlcl_ob
        
        # Time evolution of temperature anomaly
        tanom_new = self.cdland * (tanom + self.rhcapl * hfluxn_land)
        
        # Full surface temperature
        stl_lm_new = tanom_new + stlcl_ob
        
        # Atmosphere sees the land model temperature
        stl_am = stl_lm_new
        
        return stl_lm_new, stl_am
