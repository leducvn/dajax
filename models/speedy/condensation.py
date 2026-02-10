#!/usr/bin/env python3
"""
Large-scale condensation parameterization for SPEEDY.

Large-scale condensation is modelled as a relaxation of humidity to a
sigma-dependent threshold value RH(σ):

    ∂q/∂t = -(q - RH(σ) * q_sat) / τ_lsc

where τ_lsc is the relaxation timescale. Temperature tendency is the
resultant latent heating:

    ∂T/∂t = -(L/cp) * ∂q/∂t

Precipitation is diagnosed as moisture lost to condensation.

Based on SPEEDY large_scale_condensation.f90
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .state import Config, Param
from .constants import Constants
from .vertical import VerticalGrid

class Condensation:
    """
    Large-scale condensation scheme.
    
    Condenses moisture when humidity exceeds a sigma-dependent
    threshold. The threshold increases with sigma (lower in upper
    troposphere, higher near surface).
    
    Based on SPEEDY large_scale_condensation.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
    ):
        """
        Initialize condensation scheme.
        
        Args:
            config: Model configuration
            constants: Physical constants
            vertical_grid: Vertical grid structure
        """
        self.config = config
        self.constants = constants
        self.vertical_grid = vertical_grid
        
        # Setup constants
        self._setup()
    
    def _setup(self):
        """
        Initialize condensation constants.
        
        Computes:
        - Relaxation rate
        - RH threshold profile
        - Conversion factors
        """
        dhs = self.vertical_grid.dhs
        self.fsg = jnp.array(self.vertical_grid.fsg)

        # ====================================================================
        # Basic constants
        # ====================================================================
        self.qsmax = 10.0  # Maximum humidity for stability limit
        
        # Latent heating factor: L/cp
        self.tfact = self.constants.alhc / self.constants.cp
        
        # Pressure-to-precipitation factor: p0/g
        self.prg = self.constants.p0 / self.constants.grav
        
        # Precipitation factor for each layer
        self.pfact = jnp.array(dhs * self.prg)  # [kx]
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        psa: jax.Array,
        qa: jax.Array,
        qsat: jax.Array,
        itop: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute large-scale condensation tendencies.
        Note that we use 0-based Python index for itop, itop_new
        Args:
            psa: Normalized surface pressure [ix, il] (p/p0)
            qa: Specific humidity [ix, il, kx] (g/kg)
            qsat: Saturation specific humidity [ix, il, kx] (g/kg)
            itop: Cloud top from convection [ix, il] (updated in place)
            
        Returns:
            Tuple of:
            - itop_new: Updated cloud top [ix, il]
            - precls: Large-scale precipitation [ix, il] (g/m²/s)
            - tt_lsc: Temperature tendency [ix, il, kx] (K/s)
            - qt_lsc: Humidity tendency [ix, il, kx] (g/kg/s)
        """
        kx = self.config.kx
        
        # Relaxation rate (1/s)
        rtlsc = 1.0 / (param.trlsc * 3600.0)
        # ====================================================================
        # RH threshold profile (sigma-dependent)
        # rhref = rhlsc + drhlsc * (σ² - 1)
        # At surface (σ=1): rhref = rhlsc
        # At top (σ→0): rhref = rhlsc - drhlsc
        # In boundary layer: max(rhref, rhblsc)
        # ====================================================================
        sig2 = jnp.array(self.fsg ** 2)  # [kx]
        rhref = param.rhlsc + param.drhlsc * (sig2 - 1.0)
        rhref = rhref.at[kx-1].set(jnp.maximum(rhref[kx-1], param.rhblsc))
        # Maximum condensation rate (for stability)
        dqmax = self.qsmax * sig2 * rtlsc  # [kx]

        # Surface pressure squared (for stability limit)
        psa2 = psa ** 2  # [ix, il]
        
        # ====================================================================
        # Compute tendencies for each layer (vectorized)
        # ====================================================================
        # Humidity deficit: dqa = rhref * qsat - qa
        # Negative dqa means supersaturation -> condensation
        rhref_3d = rhref[jnp.newaxis, jnp.newaxis, :]  # [1, 1, kx]
        dqa = rhref_3d * qsat - qa  # [ix, il, kx]
        
        # Condensation occurs where dqa < 0
        condensing = dqa < 0  # [ix, il, kx]
        
        # Humidity tendency: dq/dt = dqa * rtlsc (only where condensing)
        dqlsc = jnp.where(condensing, dqa * rtlsc, 0.0)  # [ix, il, kx]
        
        # Temperature tendency with stability limit
        # dtlsc = tfact * min(-dqlsc, dqmax * psa²)
        dqmax_3d = dqmax[jnp.newaxis, jnp.newaxis, :]  # [1, 1, kx]
        psa2_3d = psa2[:, :, jnp.newaxis]  # [ix, il, 1]
        dtlsc = jnp.where(condensing, self.tfact * jnp.minimum(-dqlsc, dqmax_3d * psa2_3d), 0.0)  # [ix, il, kx]
        
        # Top layer has no condensation (k=0)
        dqlsc = dqlsc.at[:, :, 0].set(0.0)
        dtlsc = dtlsc.at[:, :, 0].set(0.0)
        
        # ====================================================================
        # Update cloud top
        # itop = min over k where condensing
        # ====================================================================
        # Create layer indices
        layer_idx = jnp.arange(kx)[jnp.newaxis, jnp.newaxis, :]  # [1, 1, kx]
        
        # Where condensing, use layer index; otherwise use large value
        itop_candidates = jnp.where(condensing, layer_idx, kx)
        
        # Minimum over layers (starting from k=1)
        itop_lsc = jnp.min(itop_candidates[:, :, 1:], axis=2)  # [ix, il]
        
        # Update itop: take minimum of convective and large-scale
        itop_new = jnp.minimum(itop, itop_lsc)
        
        # ====================================================================
        # Compute precipitation
        # precls = -Σ(pfact * dqlsc) * psa
        # ====================================================================
        pfact_3d = self.pfact[jnp.newaxis, jnp.newaxis, :]  # [1, 1, kx]
        
        # Sum over layers k=1 to kx (skip k=0)
        precls = -jnp.sum(pfact_3d[:, :, 1:] * dqlsc[:, :, 1:], axis=2)  # [ix, il]
        precls = precls * psa  # Scale by surface pressure
        
        return itop_new, precls, dtlsc, dqlsc
