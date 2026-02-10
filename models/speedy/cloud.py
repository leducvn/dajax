#!/usr/bin/env python3
"""
Cloud diagnostics for radiation in SPEEDY.

Diagnoses:
- Cloud-top level (icltop)
- Total cloud cover (cloudc)
- Stratiform cloud cover (clstr)
- Equivalent specific humidity of clouds (qcloud)

Cloud cover is computed from:
1. A term proportional to sqrt(precipitation rate)
2. A quadratic function of max relative humidity where q > qacl

Based on SPEEDY shortwave_radiation.f90:clouds()
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, NamedTuple

from .state import Config, Param
from .constants import Constants
from .static import StaticFields

class Cloud:
    """
    Cloud diagnostics for radiation.
    
    Diagnoses cloud cover and cloud-top level from:
    - Relative humidity profile
    - Precipitation (convective + large-scale)
    - Dry static energy gradient (for stratiform clouds)
    
    Based on SPEEDY shortwave_radiation.f90:clouds()
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        static_fields: StaticFields,
    ):
        """
        Initialize cloud diagnostics.
        
        Args:
            config: Model configuration
            constants: Physical constants
            static_fields: Static boundary conditions
        """
        self.config = config
        self.constants = constants
        self.static_fields = static_fields
        
        # Setup constants
        self._setup()
    
    def _setup(self):
        """
        Initialize cloud constants.
        """
        kx = self.config.kx
        
        # Layer indices
        self.nl1 = kx - 2   # Second-to-bottom layer
        self.nlp = kx   # No-cloud marker
        
        # Cloud cover factor for reducing stratiform when convective clouds exist
        self.clfact = 1.2

        # Get static fields
        self.fmask_l = self.static_fields.get_field('fmask_l')
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        qa: jax.Array,
        rh: jax.Array,
        precnv: jax.Array,
        precls: jax.Array,
        iptop: jax.Array,
        gse: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute cloud diagnostics for radiation.
        Note that we use 0-based Python index for iptop
        Args:
            qa: Specific humidity [ix, il, kx] (g/kg)
            rh: Relative humidity [ix, il, kx] (0-1)
            precnv: Convective precipitation [ix, il] (g/m²/s)
            precls: Large-scale precipitation [ix, il] (g/m²/s)
            iptop: Precipitation top level [ix, il] (from convection/condensation)
            gse: Vertical gradient of dry static energy [ix, il]
            fmask: Land fraction mask [ix, il]
            
        Returns:
            icltop, cloudc, clstr
        """
        kx = self.config.kx
        fmask = self.fmask_l

        # RH threshold scaling
        rrcl = 1.0 / (param.rhcl2 - param.rhcl1)
        # Stratiform cloud gradient scaling
        rgse = 1.0 / (param.gse_s1 - param.gse_s0)
        
        # ====================================================================
        # 1. Cloud cover from relative humidity
        # ====================================================================
        # Find maximum RH above boundary layer where q > qacl
        # Cloud cover = f(RH - rhcl1), with cloud-top at level of max RH
        
        # Initialize from layer nl1 (second-to-bottom)
        rh_nl1 = rh[:, :, kx-2]
        drh_init = jnp.maximum(rh_nl1 - param.rhcl1, 0.0)
        
        # Check both RH threshold and humidity threshold for consistency
        init_mask = (rh_nl1 > param.rhcl1)
        cloudc_init = jnp.where(init_mask, drh_init, 0.0)
        icltop_init = jnp.where(init_mask, self.nl1, self.nlp)
        
        # Scan through tropospheric layers k=2 to kx-3 (Fortran k=3 to kx-2)
        # Update cloud cover and top where drh > current and qa > qacl
        def update_cloud(carry, k):
            cloudc, icltop = carry
            
            drh = rh[:, :, k] - param.rhcl1
            
            # Update where this layer has higher RH excess and sufficient humidity
            update_mask = (drh > cloudc) & (qa[:, :, k] > param.qacl)
            
            cloudc_new = jnp.where(update_mask, drh, cloudc)
            icltop_new = jnp.where(update_mask, k, icltop)
            
            return (cloudc_new, icltop_new), None
        
        # Layers k=2 to kx-3 (in Fortran terms k=3 to kx-2)
        layers = jnp.arange(2, kx - 2)
        (cloudc, icltop), _ = jax.lax.scan(update_cloud, (cloudc_init, icltop_init), layers)
        
        # ====================================================================
        # 2. Add precipitation contribution to cloud cover
        # ====================================================================
        # pr1 = min(pmaxcl, 86.4 * (precnv + precls))
        # 86.4 converts g/m²/s to mm/day
        pr1 = jnp.minimum(param.pmaxcl, 86.4 * (precnv + precls))
        
        # cloudc = min(1, wpcl*sqrt(pr1) + min(1, cloudc*rrcl)²)
        rh_term = jnp.minimum(1.0, cloudc * rrcl) ** 2
        precip_term = param.wpcl * jnp.sqrt(pr1)
        cloudc = jnp.minimum(1.0, precip_term + rh_term)
        
        # Cloud top is minimum of precipitation top and RH-based top
        icltop = jnp.minimum(iptop, icltop)
        
        # ====================================================================
        # 3. Equivalent specific humidity of clouds
        # ====================================================================
        qcloud = qa[:, :, kx-2]
        
        # ====================================================================
        # 4. Stratiform clouds at top of PBL
        # ====================================================================
        # fstab = clamp((gse - gse_s0) / (gse_s1 - gse_s0), 0, 1)
        fstab = jnp.clip((gse - param.gse_s0) * rgse, 0.0, 1.0)
        
        # Stratocumulus over sea: depends on stability and existing cloud cover
        clstr_sea = fstab * jnp.maximum(param.clsmax - self.clfact * cloudc, 0.0)
        
        # Stratocumulus over land: min(clstr, clsminl) * RH at surface
        clstr_land = jnp.maximum(clstr_sea, param.clsminl) * rh[:, :, kx-1]
        
        # Blend based on land fraction
        clstr = clstr_sea + fmask * (clstr_land - clstr_sea)
        
        return icltop, cloudc, clstr, qcloud
