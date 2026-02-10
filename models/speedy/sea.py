#!/usr/bin/env python3
"""
Slab ocean and sea-ice model for SPEEDY.

Implements:
- Mixed-layer ocean model with latitude-dependent depth
- Sea-ice temperature evolution
- Multiple SST coupling modes (climatology, anomaly, full slab)

Based on SPEEDY sea_model.f90
"""

import jax
import jax.numpy as jnp
from functools import partial

from .util import Utility, TimeInfo
from .state import Config, SeaState, Param, DiagState
from .constants import Constants
from .transformer import Transformer
from .static import StaticFields

class Sea:
    """
    Slab ocean and sea-ice model.
    
    The sea model evolves:
    - SST based on net heat flux and relaxation to climatology
    - Sea ice temperature with non-linear damping
    - Sea ice fraction (currently persistent from climatology)
    
    Coupling modes (sea_coupling_flag):
    - 0: Prescribed SST from climatology
    - 1: Prescribed SST + anomaly
    - 2: Full SST from coupled ocean model
    
    Based on SPEEDY sea_model.f90
    """
    
    # Freezing point of seawater
    SSTFR = 273.2 - 1.8  # K
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        transformer: Transformer,  # Transformer instance
        static_fields: StaticFields,  # StaticFields instance
    ):
        """
        Initialize sea model.
        
        Args:
            config: Model configuration
            constants: Physical constants
            transformer: Transformer instance (for grid/latitude info)
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
        Initialize sea model constants.
        
        Computes latitude-dependent heat capacities and dissipation coefficients.
        All masks and climatology are from StaticFields.
        
        Based on SPEEDY sea_model.f90:sea_model_init()
        """
        ix, il = self.config.ix, self.config.il
        dt = self.config.dt
        
        # ====================================================================
        # Get masks and climatology from StaticFields
        # ====================================================================
        fmask_s = self.static_fields.get_field('fmask_s')  # [ix, il]
        self.sst12 = self.static_fields.get_field('sst12')      # [ix, il, 12]
        self.sice12 = self.static_fields.get_field('sice12')    # [ix, il, 12]
        
        # ====================================================================
        # Model parameters
        # ====================================================================
        # Ocean mixed layer depth: d + (d0-d)*(cos_lat)^3
        depth_ml = 60.0      # High-latitude depth (m)
        dept0_ml = 40.0      # Minimum depth at tropics (m)
        
        # Sea-ice depth: d + (d0-d)*(cos_lat)^2
        depth_ice = 2.5      # High-latitude depth (m)
        dept0_ice = 1.5      # Minimum depth (m)
        
        # Dissipation times (days)
        tdsst = 90.0         # For SST anomalies
        tdice = 30.0         # For sea ice temp anomalies
        
        # Minimum fraction of sea for anomaly definition
        fseamin = 1.0 / 3.0
        
        # Heat flux coefficient at sea/ice interface [(W/m^2)/deg]
        self.beta = 1.0
        
        # ====================================================================
        # Compute heat capacities (vectorized over latitude)
        # ====================================================================
        coslat = self.transformer.cosg  # [il]
        
        # Ocean heat capacity: depth varies with cos^3(lat)
        hcaps = 4.18e6 * (depth_ml + (dept0_ml - depth_ml) * coslat**3)  # [il]
        
        # Ice heat capacity: depth varies with cos^2(lat)
        hcapi = 1.93e6 * (depth_ice + (dept0_ice - depth_ice) * coslat**2)  # [il]
        
        # Broadcast to full grid [ix, il]
        self.rhcaps = dt / hcaps[jnp.newaxis, :]  # [1, il] -> broadcasts to [ix, il]
        self.rhcapi = dt / hcapi[jnp.newaxis, :]
        
        # ====================================================================
        # Compute domain mask and dissipation coefficients (vectorized)
        # ====================================================================
        # Using global domain
        dmask = jnp.ones((ix, il))
        
        # Smooth latitudinal boundaries (vectorized)
        dmask_padded = jnp.pad(dmask, ((0, 0), (1, 1)), mode='edge')
        dmask = 0.25 * (dmask_padded[:, :-2] + 2*dmask_padded[:, 1:-1] + dmask_padded[:, 2:])
        
        # Blank out points with insufficient sea fraction
        dmask = jnp.where(fmask_s >= fseamin, dmask, 0.0)
        
        # Dissipation coefficients
        self.cdsea = dmask * tdsst / (1.0 + dmask * tdsst)  # [ix, il]
        self.cdice = dmask * tdice / (1.0 + dmask * tdice)  # [ix, il]
        
        # Climatological heat flux (annual mean, currently zero)
        self.hfseacl = jnp.zeros((ix, il))
    
    def initialize(self, param: Param, time: TimeInfo) -> SeaState:
        """
        Initialize sea surface state from climatology.
        
        Called once at model startup.
        
        Args:
            time: Initial model time
            
        Returns:
            Initial SeaState
        """
        # Interpolate climatology to initial date
        sstcl_ob, sicecl_ob, ticecl_ob = self._interpolate_climatology(time)
        
        # Initialize ocean model from climatology
        # Setting sst_om = 0 is not a good practice in the original model
        #sst_om = jax.lax.cond(
            #self.config.sea_coupling_flag <= 0,
            #lambda x: jnp.zeros_like(x),
            #lambda x: x,
            #sstcl_ob
        #)
        sst_om = sstcl_ob
        tice_om = ticecl_ob
        sice_om = sicecl_ob
        
        # Compute SST for atmosphere
        sst_am, sice_am, tice_am, ssti_om = self._compute_atm_fields(
            sstcl_ob, sicecl_ob, ticecl_ob, sst_om, tice_om, sice_om
        )
        
        return SeaState(sst_am=sst_am, sice_am=sice_am, tice_am=tice_am,
            sst_om=sst_om, sice_om=sice_om, tice_om=tice_om, ssti_om=ssti_om
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def forward(self, param: Param, state: SeaState, diag: DiagState, time: TimeInfo) -> SeaState:
        """
        Advance sea model one timestep.
        
        Based on SPEEDY sea_model.f90:couple_sea_atm()
        
        Args:
            state: Current sea model state
            time: Current model time
            diag: Physics diagnostics containing hfluxn, shf, evap, ssrd
            
        Returns:
            Updated SeaState
        """
        # ====================================================================
        # 1. Interpolate climatology to current date
        # ====================================================================
        sstcl_ob, sicecl_ob, ticecl_ob = self._interpolate_climatology(time)
        
        # ====================================================================
        # 2. Update ocean/ice model if coupling enabled
        # ====================================================================
        sst_om, sice_om, tice_om = jax.lax.cond(
            self.config.sea_coupling_flag > 0 or self.config.ice_coupling_flag > 0,
            lambda _: self._run(param, state, sstcl_ob, sicecl_ob, ticecl_ob, diag),
            #lambda _: (state.sst_om, state.sice_om, state.tice_om), # this logic is not good in the original model
            lambda _: (sstcl_ob, sicecl_ob, ticecl_ob),
            operand=None
        )
        
        # ====================================================================
        # 3. Compute fields for atmospheric model
        # ====================================================================
        sst_am, sice_am, tice_am, ssti_om = self._compute_atm_fields(
            sstcl_ob, sicecl_ob, ticecl_ob, sst_om, tice_om, sice_om)
        
        return SeaState(sst_am=sst_am, sice_am=sice_am, tice_am=tice_am,
            sst_om=sst_om, sice_om=sice_om, tice_om=tice_om, ssti_om=ssti_om)
    
    @partial(jax.jit, static_argnames=['self'])
    def _interpolate_climatology(self, time: TimeInfo) -> tuple:
        """
        Interpolate climatological fields to current date.
        
        Also adjusts fields over sea ice for consistency.
        
        Args:
            time: Current model time
            
        Returns:
            (sstcl_ob, sicecl_ob, ticecl_ob): Climatological fields
        """
        # Interpolate SST (5-point) and sea ice (linear)
        sstcl_raw = Utility.forin5(self.sst12, time.imont1, time.tmonth)
        sicecl_raw = Utility.forint(self.sice12, time.imont1, time.tmonth)
        
        # Adjust fields over sea ice for consistency
        sstfr = self.SSTFR
        
        # Warm water case: SST > freezing
        sicecl_warm = jnp.minimum(0.5, sicecl_raw)
        ticecl_warm = jnp.full_like(sstcl_raw, sstfr)
        sstcl_warm = jnp.where(
            sicecl_raw > 0.0,
            sstfr + (sstcl_raw - sstfr) / jnp.maximum(1.0 - sicecl_warm, 1e-10),
            sstcl_raw
        )
        
        # Cold water case: SST <= freezing
        sicecl_cold = jnp.maximum(0.5, sicecl_raw)
        ticecl_cold = sstfr + (sstcl_raw - sstfr) / sicecl_cold
        sstcl_cold = jnp.full_like(sstcl_raw, sstfr)
        
        # Select based on SST
        is_warm = sstcl_raw > sstfr
        sstcl_ob = jnp.where(is_warm, sstcl_warm, sstcl_cold)
        sicecl_ob = jnp.where(is_warm, sicecl_warm, sicecl_cold)
        ticecl_ob = jnp.where(is_warm, ticecl_warm, ticecl_cold)
        
        return sstcl_ob, sicecl_ob, ticecl_ob
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_atm_fields(
        self, sstcl_ob: jax.Array, sicecl_ob: jax.Array, ticecl_ob: jax.Array,
        sst_om: jax.Array, tice_om: jax.Array, sice_om: jax.Array) -> tuple:
        """
        Compute sea surface fields for atmospheric model.
        
        Args:
            sstcl_ob: Climatological SST
            sicecl_ob: Climatological sea ice fraction
            ticecl_ob: Climatological sea ice temperature
            sst_om: Ocean model SST
            tice_om: Ocean model ice temperature
            sice_om: Ocean model ice fraction
            
        Returns:
            (sst_am, sice_am, tice_am, ssti_om)
        """
        # SST for atmosphere depends on coupling mode
        sst_am = jax.lax.cond(
            self.config.sea_coupling_flag <= 1,
            lambda _: sstcl_ob,  # Use climatology
            lambda _: sst_om,    # Use ocean model
            operand=None
        )
        
        # Sea ice depends on coupling mode
        sice_am, tice_am = jax.lax.cond(
            self.config.ice_coupling_flag <= 0,
            lambda _: (sicecl_ob, ticecl_ob), # Use climatology
            lambda _: (sice_om, tice_om),     # Use ocean model
            operand=None
        )
        
        # Blend SST and ice temperature based on ice fraction
        sst_am = sst_am + sice_am * (tice_am - sst_am)
        
        # Combined field for ocean model
        ssti_om = sst_om + sice_am * (tice_am - sst_om)
        
        return sst_am, sice_am, tice_am, ssti_om
    
    @partial(jax.jit, static_argnames=['self'])
    def _run(
        self,
        param: Param,
        state: SeaState,
        sstcl_ob: jax.Array,
        sicecl_ob: jax.Array,
        ticecl_ob: jax.Array,
        diag: DiagState
    ) -> tuple:
        """
        Integrate slab ocean and sea-ice models for one timestep.
        
        Based on SPEEDY sea_model.f90:run_sea_model()
        
        Args:
            state: Current sea model state
            sstcl_ob: Climatological SST
            sicecl_ob: Climatological sea ice fraction
            ticecl_ob: Climatological sea ice temperature
            diag: Physics diagnostics
            
        Returns:
            (sst_om, tice_om, sice_om): Updated ocean model fields
        """
        sstfr = self.SSTFR
        
        # Extract relevant fields from state and diagnostics
        sst_om = state.sst_om
        tice_om = state.tice_om
        tice_am = state.tice_am
        sice_am = state.sice_am
        
        hfluxn_sea = diag.hfluxn[:, :, 1]  # [ix, il]
        shf_sea = diag.shf[:, :, 1]        # [ix, il]
        evap_sea = diag.evap[:, :, 1]      # [ix, il]
        ssrd = diag.ssrd                    # [ix, il]
        
        # ====================================================================
        # 1. Ocean mixed layer
        # ====================================================================
        
        # Difference in heat flux between ice and sea surface
        difice = (
            (param.albsea - param.albice) * ssrd
            + param.emisfc * self.constants.sbc * (sstfr**4 - tice_am**4)
            + shf_sea + evap_sea * self.constants.alhc
        )
        
        # Net heat flux into sea-ice surface
        hflux_i = hfluxn_sea + difice * (1.0 - sice_am)
        
        # Net heat flux into ocean
        hflux = hfluxn_sea - self.hfseacl - sicecl_ob * (hflux_i + self.beta * (sstfr - tice_om))
        
        # Temperature anomaly evolution
        tanom = sst_om - sstcl_ob
        tanom = self.cdsea * (tanom + self.rhcaps * hflux)
        sst_om_new = tanom + sstcl_ob
        
        # ====================================================================
        # 2. Sea-ice slab model
        # ====================================================================
        
        # Net heat flux into ice
        hflux = hflux_i + self.beta * (sstfr - tice_om)
        
        # Temperature anomaly with non-linear damping
        tanom = tice_om - ticecl_ob
        
        # Non-linear damping coefficient
        anom0 = 20.0
        cdis = self.cdice * (anom0 / (anom0 + jnp.abs(tanom)))
        
        # Time evolution
        tanom = cdis * (tanom + self.rhcapi * hflux)
        tice_om_new = tanom + ticecl_ob
        
        # Sea ice fraction persists from climatology
        sice_om_new = sicecl_ob
        
        return sst_om_new, sice_om_new, tice_om_new
