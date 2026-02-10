#!/usr/bin/env python3
"""
Convection parameterization for SPEEDY.

Implements a simplified version of the Tiedtke (1993) mass-flux
convection scheme.

Based on SPEEDY convection.f90
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .state import Config, Param
from .constants import Constants
from .vertical import VerticalGrid

class Convection:
    """
    Mass-flux convection scheme.
    
    Convection is diagnosed by checking for conditional instability
    (saturation moist static energy decreases with height) and
    boundary layer humidity exceeding a threshold.
    
    Based on SPEEDY convection.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
    ):
        """
        Initialize convection scheme.
        
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
        Initialize convection constants.
        
        Computes:
        - Entrainment profile
        - Vertical interpolation weights
        - Mass flux scaling
        """
        kx = self.config.kx
        self.fsg = jnp.array(self.vertical_grid.fsg)
        self.dhs = jnp.array(self.vertical_grid.dhs)
        self.wvi = jnp.array(self.vertical_grid.wvi)
        
        # ====================================================================
        # Basic constants
        # ====================================================================
        # note: for nlp we use 0-based Python index, so be careful
        self.nlp = kx  # kx (no convection marker)
        self.fqmax = 5.0   # Maximum humidity ratio
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        psa: jax.Array,
        se: jax.Array,
        qa: jax.Array,
        qsat: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute convective tendencies.
        Note that we use 0-based Python index for itop
        Args:
            psa: Normalized surface pressure [ix, il] (p/p0)
            se: Dry static energy [ix, il, kx] (cp*T + g*z)
            qa: Specific humidity [ix, il, kx] (g/kg)
            qsat: Saturation specific humidity [ix, il, kx] (g/kg)
            
        Returns:
            Tuple of:
            - itop: Top of convection (layer index) [ix, il]
            - precnv: Convective precipitation [ix, il] (g/m^2/s)
            - dfse: Net flux of dry static energy [ix, il, kx] (K/s)
            - dfqa: Net flux of specific humidity [ix, il, kx] (g/kg/s)
        """
        # Diagnose convection
        itop, qdif = self._diagnose_convection(param, psa, se, qa, qsat)
        
        # Compute convective fluxes
        precnv, dfse, dfqa = self._compute_fluxes(param, psa, se, qa, qsat, itop, qdif)
        
        return itop, precnv, dfse, dfqa
    
    @partial(jax.jit, static_argnames=['self'])
    def _diagnose_convection(
        self,
        param: Param,
        psa: jax.Array,
        se: jax.Array,
        qa: jax.Array,
        qsat: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Diagnose convectively unstable gridboxes.
        
        Convection is activated where:
        1. Surface pressure > psmin
        2. Conditional instability exists (MSS decreases with height)
        3. Either convective instability or RH > threshold in boundary layer
        
        Args:
            psa: Normalized surface pressure [ix, il]
            se: Dry static energy [ix, il, kx]
            qa: Specific humidity [ix, il, kx]
            qsat: Saturation specific humidity [ix, il, kx]
            
        Returns:
            Tuple of:
            - itop: Top of convection (layer index) [ix, il]
            - qdif: Excess humidity in convective gridboxes [ix, il]
        """
        kx = self.config.kx
        alhc = self.constants.alhc
        rhbl = param.rhbl
        
        # Saturation moist static energy: mss = se + alhc * qsat
        mss = se + alhc * qsat  # [ix, il, kx]
        
        # Moist static energy in boundary layer
        mse_sfc = se[:, :, kx-1] + alhc * qa[:, :, kx-1]  # Bottom layer
        mse_nl1 = se[:, :, kx-2] + alhc * qa[:, :, kx-2]  # Second layer
        mse1 = jnp.minimum(mse_sfc, mse_nl1)  # Min of lowest two
        
        # Saturation (or super-saturated) MSE in boundary layer
        mss0 = jnp.maximum(mse_sfc, mss[:, :, kx-1])
        
        # RH thresholds
        qthr0 = rhbl * qsat[:, :, kx-1]
        qthr1 = rhbl * qsat[:, :, kx-2]
        lqthr = (qa[:, :, kx-1] > qthr0) & (qa[:, :, kx-2] > qthr1)
        
        # Initialize outputs
        itop = jnp.full((self.config.ix, self.config.il), self.nlp, dtype=jnp.int32)
        qdif = jnp.zeros((self.config.ix, self.config.il))
        msthr = jnp.zeros((self.config.ix, self.config.il))
        
        # Check for conditional and convective instability
        # Scan from k=kx-4 to k=2 (top of troposphere)
        ktop1 = jnp.full_like(itop, kx-1)  # Conditional instability top
        ktop2 = jnp.full_like(itop, kx-1)  # Convective instability top
        
        def check_layer(carry, k):
            ktop1, ktop2, msthr = carry
            
            # Interpolate MSS to layer interface
            mss2 = mss[:, :, k] + self.wvi[k] * (mss[:, :, k+1] - mss[:, :, k])
            
            # Check 1: conditional instability (mss0 > mss2)
            ktop1 = jnp.where(mss0 > mss2, k, ktop1)
            
            # Check 2: convective instability (mse1 > mss2)
            update_ktop2 = mse1 > mss2
            ktop2 = jnp.where(update_ktop2, k, ktop2)
            msthr = jnp.where(update_ktop2, mss2, msthr)
            
            return (ktop1, ktop2, msthr), None
        
        # Scan from k=kx-4 down to k=2
        layers = jnp.arange(2, kx-3, dtype=jnp.int32)[::-1]  # Reverse: top to bottom
        (ktop1, ktop2, msthr), _ = jax.lax.scan(check_layer, (ktop1, ktop2, msthr), layers)
        
        # Apply surface pressure threshold
        psa_ok = psa > param.psmin
        ktop1_valid = ktop1 < kx-1
        
        # Determine itop and qdif based on conditions
        # Case 1: Both conditional and convective instability
        case1 = psa_ok & ktop1_valid & (ktop2 < kx-1)
        qdif1 = jnp.maximum(qa[:, :, kx-1] - qthr0, (mse_sfc - msthr) / alhc)
        
        # Case 2: Conditional instability + RH threshold
        case2 = psa_ok & ktop1_valid & (ktop2 >= kx-1) & lqthr
        qdif2 = qa[:, :, kx-1] - qthr0
        
        # Apply cases
        itop = jnp.where(case1 | case2, ktop1, itop)
        qdif = jnp.where(case1, qdif1, jnp.where(case2, qdif2, qdif))
        
        return itop, qdif
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_fluxes(
        self,
        param: Param,
        psa: jax.Array,
        se: jax.Array,
        qa: jax.Array,
        qsat: jax.Array,
        itop: jax.Array,
        qdif: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Compute convective fluxes (corrected version)."""
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        alhc = self.constants.alhc

        # param-dependt parameters
        # Mass flux scaling: fm0 = p0 * dhs[kx-1] / (g * trcnv * 3600)
        fm0 = self.constants.p0 * self.dhs[kx-1] / (self.constants.grav * param.trcnv * 3600.0)
        # Pressure threshold scaling
        rdps = 2.0 / (1.0 - param.psmin)
        # Entrainment profile (up to sigma = 0.5)
        entr = jnp.maximum(0.0, self.fsg - 0.5) ** 2  # [kx]
        sentr = jnp.sum(entr[1:kx-1])  # Sum for k=1 to kx-2 (layers 2 to nl1 in Fortran)
        #sentr = jnp.maximum(sentr, 1e-10)  # Avoid division by zero
        entr = entr * (param.entmax / sentr)
        
        dfse = jnp.zeros((ix, il, kx))
        dfqa = jnp.zeros((ix, il, kx))
        cbmf = jnp.zeros((ix, il))
        precnv = jnp.zeros((ix, il))
        
        active = itop < self.nlp
        
        # Boundary layer (cloud base): k = kx-1
        k = kx - 1
        k1 = k - 1
        
        # Compute intermediate values (no masking needed - they're cheap)
        qmax = jnp.maximum(1.01 * qa[:, :, k], qsat[:, :, k])
        sb = se[:, :, k1] + self.wvi[k1] * (se[:, :, k] - se[:, :, k1])
        qb = qa[:, :, k1] + self.wvi[k1] * (qa[:, :, k] - qa[:, :, k1])
        qb = jnp.minimum(qb, qa[:, :, k])

        fpsa = psa * jnp.minimum(1.0, (psa - param.psmin) * rdps)

        # Mask only fmass (the key variable that gets carried forward)
        fmass = jnp.where(active, fm0 * fpsa * jnp.minimum(self.fqmax, qdif / (qmax - qb)), 0.0)
        cbmf = fmass

        # Compute fluxes from (already masked) fmass
        # No additional masking needed - if fmass=0, all fluxes=0
        fus = fmass * se[:, :, k]
        fuq = fmass * qmax
        fds = fmass * sb
        fdq = fmass * qb

        # Already implicitly masked through fmass
        dfse = dfse.at[:, :, k].set(fds - fus)
        dfqa = dfqa.at[:, :, k].set(fdq - fuq)

        # Intermediate layers: k = kx-2 down to itop+1 (EXCLUDE top layer)
        def process_layer(carry, k):
            fmass, fus, fuq, fds, fdq, dfse, dfqa = carry
            k1 = k - 1
            
            layer_active = active & (k > itop)
            
            # Initial fluxes (update dfse/dfqa with current flux balance)
            dfse = dfse.at[:, :, k].set(jnp.where(layer_active, fus - fds, dfse[:, :, k]))
            dfqa = dfqa.at[:, :, k].set(jnp.where(layer_active, fuq - fdq, dfqa[:, :, k]))
            
            # Entrainment (only add to fmass where active)
            enmass = entr[k] * psa * cbmf
            fmass = jnp.where(layer_active, fmass + enmass, fmass)
            fus = jnp.where(layer_active, fus + enmass * se[:, :, k], fus)
            fuq = jnp.where(layer_active, fuq + enmass * qa[:, :, k], fuq)
            
            # Downward fluxes at upper boundary (compute unconditionally)
            sb = se[:, :, k1] + self.wvi[k1] * (se[:, :, k] - se[:, :, k1])
            qb = qa[:, :, k1] + self.wvi[k1] * (qa[:, :, k] - qa[:, :, k1])
            
            # Update downward fluxes (only where active)
            fds = jnp.where(layer_active, fmass * sb, fds)
            fdq = jnp.where(layer_active, fmass * qb, fdq)
            
            # Add contribution from upper boundary
            dfse = dfse.at[:, :, k].add(jnp.where(layer_active, fds - fus, 0.0))
            dfqa = dfqa.at[:, :, k].add(jnp.where(layer_active, fdq - fuq, 0.0))
            
            # Secondary moisture flux
            delq = param.rhil * qsat[:, :, k] - qa[:, :, k]
            fsq_active = layer_active & (delq > 0.0)
            fsq = jnp.where(fsq_active, param.smf * cbmf * delq, 0.0)
            
            dfqa = dfqa.at[:, :, k].add(fsq)
            dfqa = dfqa.at[:, :, kx-1].add(-fsq)  # Remove from surface layer
            
            return (fmass, fus, fuq, fds, fdq, dfse, dfqa), None

        layers = jnp.arange(3, kx-1, dtype=jnp.int32)[::-1]
        (fmass, fus, fuq, fds, fdq, dfse, dfqa), _ = jax.lax.scan(
            process_layer,
            (fmass, fus, fuq, fds, fdq, dfse, dfqa),
            layers
        )
        
        # Top layer (includes entrainment + detrainment + precipitation)
        # Use advanced indexing to directly access the top layer for each grid point
        i_idx = jnp.arange(ix)[:, None]
        j_idx = jnp.arange(il)[None, :]

        # Safe indexing: clip itop+1 to valid range (result ignored when not active)
        itop_below = jnp.minimum(itop + 1, kx - 1)

        # Interpolation weight at top boundary
        wvi_top = self.wvi[itop]  # [ix, il]

        # Saturation humidity at top boundary (interpolated)
        qsat_top = qsat[i_idx, j_idx, itop]
        qsat_below = qsat[i_idx, j_idx, itop_below]  # Safe indexing
        qsatb = qsat_top + wvi_top * (qsat_below - qsat_top)

        # Convective precipitation
        precnv_top = jnp.maximum(fuq - fmass * qsatb, 0.0)
        precnv = jnp.where(active, precnv_top, precnv)

        # Net fluxes at top layer
        dfse_top = fus - fds + alhc * precnv
        dfqa_top = fuq - fdq - precnv

        # Update only for active convection points
        dfse = dfse.at[i_idx, j_idx, itop].set(jnp.where(active, dfse_top, dfse[i_idx, j_idx, itop]))
        dfqa = dfqa.at[i_idx, j_idx, itop].set(jnp.where(active, dfqa_top, dfqa[i_idx, j_idx, itop]))
        
        return precnv, dfse, dfqa
