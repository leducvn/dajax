#!/usr/bin/env python3
"""
Vertical diffusion parameterization for SPEEDY.

Computes tendencies due to three processes:
1. Shallow convection - Redistribution of moisture and dry static energy
   between the lowest two layers where there is conditional instability
2. Slow diffusion of moisture in stable conditions
3. Fast redistribution of dry static energy where the lapse rate is close to
   dry adiabatic limit

Note: Momentum tendencies (ut_vdi, vt_vdi) are always zero in SPEEDY.
Surface momentum flux is added separately in the physics driver.

Based on SPEEDY vertical_diffusion.f90

Index Convention:
    Python uses 0-based indexing, Fortran uses 1-based for layers.
    
    Layer indices (0-based Python):
        k = 0: top level
        k = kx-2 = 6: second-to-bottom level (nl1 in Fortran = kx-1 = 7)
        k = kx-1 = 7: bottom level (surface)
    
    Half-level (sigh) indexing in Fortran:
        sigh is 0-indexed in Fortran: sigh(0:kx)
        sigh(k) = hsg(k+1) where hsg is 1-indexed
        
    In Python, we use hsg directly:
        hsg[k] corresponds to Fortran sigh(k) for the same numerical k
        For Fortran layer k (1-indexed), the bottom interface is sigh(k)
        For Python layer k (0-indexed), the bottom interface is hsg[k+1]
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .state import Config, Param
from .constants import Constants
from .vertical import VerticalGrid

class VerticalDiffusion:
    """
    Vertical diffusion and shallow convection scheme.
    
    Computes tendencies for temperature and humidity due to:
    - Shallow convection in conditionally unstable PBL
    - Moisture diffusion in stable conditions
    - Super-adiabatic damping
    
    Based on SPEEDY vertical_diffusion.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
    ):
        """
        Initialize vertical diffusion scheme.
        
        Args:
            config: Model configuration
            constants: Physical constants
            vertical_grid: Vertical grid structure
        """
        self.config = config
        self.constants = constants
        self.vertical_grid = vertical_grid
        
        self._setup()
    
    def _setup(self):
        """Initialize vertical diffusion constants."""
        kx = self.config.kx
        
        # Level indices (0-based Python)
        # Fortran nl1 = kx - 1 = 7 (1-indexed second-to-bottom)
        # Python k_nl1 = kx - 2 = 6 (0-indexed second-to-bottom)
        self.k_nl1 = kx - 2  # Second-to-bottom layer
        self.k_sfc = kx - 1  # Bottom (surface) layer
        
        dhs = self.vertical_grid.dhs
        hsg = self.vertical_grid.hsg  # Half sigma levels [kx+1], 0-indexed
        fsg = self.vertical_grid.fsg
        
        cp = self.constants.cp
        
        # Relaxation times (hours)
        trshc = 6.0   # Shallow convection
        
        # Reduction factor for shallow convection in deep convection areas
        self.redshc = 0.5
        
        # Maximum gradient thresholds
        self.rhgrad = 0.5  # Max d_RH/d_sigma
        self.segrad = 0.1  # Min d_DSE/d_phi
        
        # Flux coefficients
        # Fortran: cshc = dhs(kx)/3600.0
        # Python: dhs[kx-1] is the bottom layer thickness
        cshc = dhs[self.k_sfc] / 3600.0
        
        self.fshcq = cshc / trshc
        self.fshcse = cshc / (trshc * cp)
        
        # Inverse layer thicknesses
        self.rsig = 1.0 / jnp.array(dhs)  # [kx]
        
        # Factor for redistributing flux to layers below
        self.rsig1 = 1.0 / (1.0 - jnp.array(hsg[1:kx+1]))  # [kx]
        
        # Store grid parameters
        self.hsg = jnp.array(hsg)  # [kx+1]
        self.fsg = jnp.array(fsg)  # [kx]
        self.dhs = jnp.array(dhs)  # [kx]
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        se: jax.Array,
        rh: jax.Array,
        qa: jax.Array,
        qsat: jax.Array,
        phig: jax.Array,
        icnv: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute vertical diffusion tendencies.
        
        Args:
            se: Dry static energy [ix, il, kx] (J/kg)
            rh: Relative humidity [ix, il, kx] (0-1)
            qa: Specific humidity [ix, il, kx] (g/kg)
            qsat: Saturation specific humidity [ix, il, kx] (g/kg)
            phig: Geopotential [ix, il, kx] (m²/s²)
            icnv: Deep convection depth [ix, il] (number of layers)
            
        Returns:
            Tuple of:
            - ut_vdi: U-wind tendency [ix, il, kx] (always zero)
            - vt_vdi: V-wind tendency [ix, il, kx] (always zero)
            - tt_vdi: Temperature tendency [ix, il, kx] (K/s)
            - qt_vdi: Humidity tendency [ix, il, kx] (g/kg/s)
        """
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        
        # Initialize tendencies
        ut_vdi = jnp.zeros((ix, il, kx))
        vt_vdi = jnp.zeros((ix, il, kx))
        tt_vdi = jnp.zeros((ix, il, kx))
        qt_vdi = jnp.zeros((ix, il, kx))
        
        # ====================================================================
        # 1. Shallow convection between bottom (k_sfc) and second-to-bottom (k_nl1)
        # ====================================================================
        tt_vdi, qt_vdi = self._shallow_convection(param, se, rh, qa, qsat, icnv, tt_vdi, qt_vdi)
        
        # ====================================================================
        # 2. Vertical diffusion of moisture above PBL
        # ====================================================================
        qt_vdi = self._moisture_diffusion(param, rh, qsat, qt_vdi)
        
        # ====================================================================
        # 3. Damping of super-adiabatic lapse rate
        # ====================================================================
        tt_vdi = self._superadiabatic_damping(param, se, phig, tt_vdi)
        
        return ut_vdi, vt_vdi, tt_vdi, qt_vdi
    
    def _shallow_convection(
        self,
        param: Param,
        se: jax.Array,
        rh: jax.Array,
        qa: jax.Array,
        qsat: jax.Array,
        icnv: jax.Array,
        tt_vdi: jax.Array,
        qt_vdi: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Shallow convection between bottom layer (k_sfc) and layer above (k_nl1).
        
        Redistributes moisture and dry static energy when:
        - Conditional instability exists (dmse >= 0)
        - Or stable with sufficient humidity gradient
        
        Based on Fortran lines 78-106
        """
        k_nl1 = self.k_nl1  # Second-to-bottom (index kx-2 = 6)
        k_sfc = self.k_sfc  # Bottom/surface (index kx-1 = 7)
        kx = self.config.kx
        
        alhc = self.constants.alhc
        # Relaxation times (hours)
        trvdi = param.trvdi  # Moisture diffusion (default 24.0)
        nl1 = kx - 1  # = 7, Fortran's nl1
        cvdi = (self.hsg[nl1] - self.hsg[1]) / ((nl1 - 1) * 3600.0)
        fvdiq = cvdi / trvdi
        
        # Moist static energy difference: bottom - second-to-bottom
        # Fortran: dmse = se(kx) - se(nl1) + alhc*(qa(kx) - qsat(nl1))
        dmse = (se[:, :, k_sfc] - se[:, :, k_nl1] + 
                alhc * (qa[:, :, k_sfc] - qsat[:, :, k_nl1]))
        
        # Relative humidity gradient: bottom - second-to-bottom
        drh = rh[:, :, k_sfc] - rh[:, :, k_nl1]
        
        # Threshold for moisture diffusion in stable conditions
        # Fortran: drh0 = rhgrad*(fsg(kx) - fsg(nl1))
        drh0 = self.rhgrad * (self.fsg[k_sfc] - self.fsg[k_nl1])
        
        # Coefficient for stable moisture diffusion
        fvdiq2 = fvdiq * self.hsg[nl1]
        
        # Reduction factor where deep convection is active
        fcnv = jnp.where(icnv > 0, self.redshc, 1.0)
        
        # Conditionally unstable (dmse >= 0)
        unstable = dmse >= 0.0
        
        # DSE flux for unstable case
        fluxse = fcnv * self.fshcse * dmse
        fluxse = jnp.where(unstable, fluxse, 0.0)
        
        # Fortran: ttenvd(nl1) = fluxse*rsig(nl1), ttenvd(kx) = -fluxse*rsig(kx)
        tt_vdi = tt_vdi.at[:, :, k_nl1].add(fluxse * self.rsig[k_nl1])
        tt_vdi = tt_vdi.at[:, :, k_sfc].add(-fluxse * self.rsig[k_sfc])
        
        # Moisture flux for unstable case with drh >= 0
        # Fortran: fluxq = fcnv*fshcq*qsat(kx)*drh
        fluxq_unstable = fcnv * self.fshcq * qsat[:, :, k_sfc] * drh
        fluxq_unstable = jnp.where(unstable & (drh >= 0.0), fluxq_unstable, 0.0)
        
        # Moisture flux for stable case with drh > drh0
        # Fortran: fluxq = fvdiq2*qsat(nl1)*drh
        fluxq_stable = fvdiq2 * qsat[:, :, k_nl1] * drh
        stable_diffuse = (~unstable) & (drh > drh0)
        fluxq_stable = jnp.where(stable_diffuse, fluxq_stable, 0.0)
        
        # Total moisture flux
        fluxq = fluxq_unstable + fluxq_stable
        
        qt_vdi = qt_vdi.at[:, :, k_nl1].add(fluxq * self.rsig[k_nl1])
        qt_vdi = qt_vdi.at[:, :, k_sfc].add(-fluxq * self.rsig[k_sfc])
        
        return tt_vdi, qt_vdi
    
    def _moisture_diffusion(
        self,
        param: Param,
        rh: jax.Array,
        qsat: jax.Array,
        qt_vdi: jax.Array,
    ) -> jax.Array:
        """
        Vertical diffusion of moisture above PBL.
        
        Active at levels where sigh > 0.5 and humidity gradient exceeds threshold.
        
        """
        kx = self.config.kx
        # Relaxation times (hours)
        trvdi = param.trvdi  # Moisture diffusion (default 24.0)
        nl1 = kx - 1  # = 7, Fortran's nl1
        cvdi = (self.hsg[nl1] - self.hsg[1]) / ((nl1 - 1) * 3600.0)
        fvdiq = cvdi / trvdi
        
        def process_layer(qt_vdi, k):
            active = self.hsg[k+1] > 0.5
            
            drh0 = self.rhgrad * (self.fsg[k+1] - self.fsg[k])
            fvdiq2 = fvdiq * self.hsg[k+1]
            
            drh = rh[:, :, k+1] - rh[:, :, k]
            
            # Only diffuse if drh >= drh0
            do_diffuse = active & (drh >= drh0)
            
            fluxq = fvdiq2 * qsat[:, :, k] * drh
            fluxq = jnp.where(do_diffuse, fluxq, 0.0)
            
            qt_vdi = qt_vdi.at[:, :, k].add(fluxq * self.rsig[k])
            qt_vdi = qt_vdi.at[:, :, k+1].add(-fluxq * self.rsig[k+1])
            
            return qt_vdi, None
        
        layers = jnp.arange(2, kx - 2)
        qt_vdi, _ = jax.lax.scan(process_layer, qt_vdi, layers)
        
        return qt_vdi
    
    def _superadiabatic_damping(
        self,
        param: Param,
        se: jax.Array,
        phig: jax.Array,
        tt_vdi: jax.Array,
    ) -> jax.Array:
        """
        Damping of super-adiabatic lapse rate.
        
        When DSE at level k is less than threshold, redistribute energy
        from layers below.
        
        Fortran: do k = 1, nl1 (1-indexed, k=1 to kx-1=7)
        Python:  k = 0 to kx-2 (0-indexed, k=0 to 6)
        """
        kx = self.config.kx
        k_nl1 = self.k_nl1  # = kx - 2 = 6
        # Relaxation times (hours)
        trvds = param.trvds  # Super-adiabatic (default 6.0)
        nl1 = kx - 1  # = 7, Fortran's nl1
        cvdi = (self.hsg[nl1] - self.hsg[1]) / ((nl1 - 1) * 3600.0)
        cp = self.constants.cp
        fvdise = cvdi / (trvds * cp)

        def process_layer(tt_vdi, k):
            # Reference DSE at level k (threshold for stability)
            se0 = se[:, :, k+1] + self.segrad * (phig[:, :, k] - phig[:, :, k+1])
            
            # Super-adiabatic condition
            superadiabatic = se[:, :, k] < se0
            
            fluxse = fvdise * (se0 - se[:, :, k])
            fluxse = jnp.where(superadiabatic, fluxse, 0.0)
            
            # Add flux to level k
            tt_vdi = tt_vdi.at[:, :, k].add(fluxse * self.rsig[k])
            
            # Subtract from all levels below (k+1 to kx-1)
            layer_mask = jnp.arange(kx) > k  # [kx] boolean mask for layers below k
            delta = fluxse[:, :, jnp.newaxis] * self.rsig1[k] * layer_mask[jnp.newaxis, jnp.newaxis, :]
            tt_vdi = tt_vdi - delta
            
            return tt_vdi, None
        
        layers = jnp.arange(0, k_nl1 + 1)
        tt_vdi, _ = jax.lax.scan(process_layer, tt_vdi, layers)
        
        return tt_vdi
