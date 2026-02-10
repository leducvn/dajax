#!/usr/bin/env python3
"""
Shortwave radiation parameterization for SPEEDY.

Computes:
- Ozone absorption in stratosphere
- Shortwave transmissivities (visible + near-IR bands)
- Downward flux with cloud reflection
- Surface reflection and upward flux
- Initializes longwave transmissivities

Returns CachedState for use by longwave and surface fluxes.

Based on SPEEDY shortwave_radiation.f90

FULLY VECTORIZED VERSION - no explicit Python for loops.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .state import Config, Param, ForcingState, CachedState
from .constants import Constants
from .vertical import VerticalGrid

class Shortwave:
    """
    Shortwave radiation scheme (fully vectorized).
    
    Computes absorption of solar radiation through:
    - Two spectral bands (visible + near-IR)
    - Ozone absorption in stratosphere
    - Water vapor and cloud absorption in troposphere
    - Cloud reflection
    - Surface reflection based on albedo
    
    Also initializes transmissivities for longwave radiation.
    
    Based on SPEEDY shortwave_radiation.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
    ):
        """
        Initialize shortwave radiation scheme.
        
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
        """Initialize shortwave radiation constants."""
        self.dhs = jnp.array(self.vertical_grid.dhs)
        self.fsg = jnp.array(self.vertical_grid.fsg)
        
        # Spectral band fractions
        self.fband2 = 0.05  # Near-IR fraction
        self.fband1 = 1.0 - self.fband2  # Visible fraction
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        psa: jax.Array,
        qa: jax.Array,
        icltop: jax.Array,
        cloudc: jax.Array,
        clstr: jax.Array,
        qcloud: jax.Array,
        forcing: ForcingState,
    ) -> CachedState:
        """
        Compute shortwave radiation (fully vectorized).
        
        Args:
            psa: Normalized surface pressure [ix, il] (p/p0)
            qa: Specific humidity [ix, il, kx] (g/kg)
            icltop: Cloud-top level [ix, il] (0-based index)
            cloudc: Total cloud cover [ix, il] (0-1)
            clstr: Stratiform cloud cover [ix, il] (0-1)
            qcloud: Cloud equivalent humidity [ix, il] (g/kg)
            forcing: ForcingState containing albsfc and solar fields
            
        Returns:
            CachedState containing:
            - tau2: [ix, il, kx, 4] LW transmissivity
            - stratc: [ix, il, 2] Stratospheric correction
            - ssrd: [ix, il] Surface downward SW (W/m²)
            - dfabs_sw: [ix, il, kx] SW absorbed in each layer (W/m²)
            - tsr: [ix, il] TOA net SW (W/m²)
            - ssr: [ix, il] Surface net SW (W/m²)
        """
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        nl1 = kx - 1
        # Stratospheric correction factor for longwave
        eps1 = param.epslw / (self.dhs[0] + self.dhs[1])
        
        # Extract fields from forcing
        albsfc = forcing.albsfc
        solar = forcing.solar
        
        # ====================================================================
        # 1. Initialize shortwave transmissivities
        # ====================================================================
        tau2_sw = jnp.zeros((ix, il, kx, 4))
        
        # Cloud reflectance - VECTORIZED
        k_levels = jnp.arange(kx)[jnp.newaxis, jnp.newaxis, :]  # [1, 1, kx]
        is_cltop = (icltop[:, :, jnp.newaxis] == k_levels)  # [ix, il, kx]
        cloud_refl = jnp.where(is_cltop, param.albcl * cloudc[:, :, jnp.newaxis], 0.0)
        
        # Stratiform cloud reflectance at surface layer
        cloud_refl = cloud_refl.at[:, :, kx-1].set(param.albcls * clstr)
        tau2_sw = tau2_sw.at[:, :, :, 2].set(cloud_refl)
        
        # ====================================================================
        # 2. Shortwave transmissivity (visible band) - VECTORIZED
        # ====================================================================
        psaz = psa * solar.zenit
        acloud_sw = cloudc * jnp.minimum(param.abscl1 * qcloud, param.abscl2)
        
        # Layer 0 (top stratosphere)
        tau2_sw = tau2_sw.at[:, :, 0, 0].set(jnp.exp(-psaz * self.dhs[0] * param.absdry))
        
        # Layers 1 to kx-2: VECTORIZED
        k_array = jnp.arange(1, nl1)  # [1, 2, ..., kx-2]
        abs1_array = param.absdry + param.absaer * self.fsg[k_array]**2  # [kx-2]
        
        # Water vapor absorption
        abs_wv = param.abswv1 * qa[:, :, 1:nl1]  # [ix, il, kx-2]
        
        # Cloud absorption: active below cloud top
        in_cloud = k_array[jnp.newaxis, jnp.newaxis, :] >= icltop[:, :, jnp.newaxis]  # [ix, il, kx-2]
        acloud_contrib = jnp.where(in_cloud, acloud_sw[:, :, jnp.newaxis], 0.0)  # [ix, il, kx-2]
        
        abs_total = abs1_array[jnp.newaxis, jnp.newaxis, :] + abs_wv + acloud_contrib  # [ix, il, kx-2]
        
        # Layer thicknesses
        dhs_mid = self.dhs[1:nl1]  # [kx-2]
        tau_mid = jnp.exp(-psaz[:, :, jnp.newaxis] * dhs_mid[jnp.newaxis, jnp.newaxis, :] * abs_total)
        tau2_sw = tau2_sw.at[:, :, 1:nl1, 0].set(tau_mid)
        
        # Bottom layer (kx-1)
        abs1_bot = param.absdry + param.absaer * self.fsg[kx-1]**2
        tau2_sw = tau2_sw.at[:, :, kx-1, 0].set(
            jnp.exp(-psaz * self.dhs[kx-1] * (abs1_bot + param.abswv1 * qa[:, :, kx-1]))
        )
        
        # ====================================================================
        # 3. Shortwave transmissivity (near-IR band) - VECTORIZED
        # ====================================================================
        dhs_nir = self.dhs[1:kx]  # [kx-1]
        qa_nir = qa[:, :, 1:kx]  # [ix, il, kx-1]
        
        tau_nir = jnp.exp(-psaz[:, :, jnp.newaxis] * dhs_nir[jnp.newaxis, jnp.newaxis, :] * param.abswv2 * qa_nir)
        tau2_sw = tau2_sw.at[:, :, 1:kx, 1].set(tau_nir)
        
        # ====================================================================
        # 4. Downward shortwave flux - use scan for sequential processing
        # ====================================================================
        flux = jnp.zeros((ix, il, 2))
        flux = flux.at[:, :, 0].set(solar.fsol * self.fband1)
        flux = flux.at[:, :, 1].set(solar.fsol * self.fband2)
        
        ftop = solar.fsol.copy()
        dfabs = jnp.zeros((ix, il, kx))
        
        # Layer 0: ozone absorption
        dfabs = dfabs.at[:, :, 0].set(flux[:, :, 0])
        flux = flux.at[:, :, 0].set(tau2_sw[:, :, 0, 0] * (flux[:, :, 0] - solar.ozupp * psa))
        dfabs = dfabs.at[:, :, 0].add(-flux[:, :, 0])
        
        # Layer 1: ozone absorption
        dfabs = dfabs.at[:, :, 1].set(flux[:, :, 0])
        flux = flux.at[:, :, 0].set(tau2_sw[:, :, 1, 0] * (flux[:, :, 0] - solar.ozone * psa))
        dfabs = dfabs.at[:, :, 1].add(-flux[:, :, 0])
        
        # Troposphere (k=2 to kx-1): sequential processing with scan
        def downward_visible_step(carry, k):
            flux_vis, dfabs, tau2_sw = carry
            
            tau0 = tau2_sw[:, :, k, 0]
            tau2 = tau2_sw[:, :, k, 2]
            
            # Cloud reflection
            reflected = flux_vis * tau2
            flux_vis = flux_vis - reflected
            
            # Absorption
            dfabs = dfabs.at[:, :, k].set(flux_vis)
            flux_vis = tau0 * flux_vis
            dfabs = dfabs.at[:, :, k].add(-flux_vis)
            
            return (flux_vis, dfabs, tau2_sw), reflected
        
        (flux_vis_final, dfabs, _), reflected_stack = jax.lax.scan(
            downward_visible_step,
            (flux[:, :, 0], dfabs, tau2_sw),
            jnp.arange(2, kx)
        )
        flux = flux.at[:, :, 0].set(flux_vis_final)
        
        # Store reflected flux for upward pass
        tau2_sw = tau2_sw.at[:, :, 2:kx, 2].set(jnp.moveaxis(reflected_stack, 0, -1))
        
        # Near-IR absorption - VECTORIZED with scan
        def downward_nir_step(carry, k):
            flux_nir, dfabs = carry
            tau1 = tau2_sw[:, :, k, 1]
            dfabs = dfabs.at[:, :, k].add(flux_nir)
            flux_nir = tau1 * flux_nir
            dfabs = dfabs.at[:, :, k].add(-flux_nir)
            return (flux_nir, dfabs), None
        
        (flux_nir_final, dfabs), _ = jax.lax.scan(
            downward_nir_step,
            (flux[:, :, 1], dfabs),
            jnp.arange(1, kx)
        )
        flux = flux.at[:, :, 1].set(flux_nir_final)
        
        # ====================================================================
        # 5. Surface reflection and upward flux
        # ====================================================================
        fsfcd = flux[:, :, 0] + flux[:, :, 1]  # Downward SW at surface
        flux = flux.at[:, :, 0].set(flux[:, :, 0] * albsfc)
        fsfc = fsfcd - flux[:, :, 0]  # Net SW at surface
        
        # Upward absorption - scan from bottom to top
        def upward_step(carry, k):
            flux_vis, dfabs = carry
            tau0 = tau2_sw[:, :, k, 0]
            tau2 = tau2_sw[:, :, k, 2]
            
            dfabs = dfabs.at[:, :, k].add(flux_vis)
            flux_vis = tau0 * flux_vis
            dfabs = dfabs.at[:, :, k].add(-flux_vis)
            flux_vis = flux_vis + tau2
            
            return (flux_vis, dfabs), None
        
        (flux_vis_up, dfabs), _ = jax.lax.scan(
            upward_step,
            (flux[:, :, 0], dfabs),
            jnp.arange(kx-1, -1, -1)
        )
        
        # Net TOA flux
        ftop = ftop - flux_vis_up  # tsr = net downward SW at TOA
        
        # ====================================================================
        # 6. Initialize longwave transmissivities
        # ====================================================================
        tau2_lw = self._compute_lw_transmissivity(param, psa, qa, icltop, cloudc)
        
        # Stratospheric correction for longwave
        stratc = jnp.zeros((ix, il, 2))
        stratc = stratc.at[:, :, 0].set(solar.stratz * psa)
        stratc = stratc.at[:, :, 1].set(eps1 * psa)
        
        # ====================================================================
        # 7. Create CachedState
        # ====================================================================
        cached = CachedState(
            tau2=tau2_lw,
            stratc=stratc,
            ssrd=fsfcd,  # Surface downward SW
            dfabs_sw=dfabs, # SW absorbed flux (for temp tendency)
            tsr=ftop,    # TOA net SW
            ssr=fsfc,    # Surface net SW
        )
        
        return cached
    
    def _compute_lw_transmissivity(
        self,
        param: Param,
        psa: jax.Array,
        qa: jax.Array,
        icltop: jax.Array,
        cloudc: jax.Array,
    ) -> jax.Array:
        """
        Compute longwave transmissivities (VECTORIZED).
        
        Args:
            psa: Normalized surface pressure [ix, il]
            qa: Specific humidity [ix, il, kx]
            icltop: Cloud top level [ix, il]
            cloudc: Total cloud cover [ix, il]
            
        Returns:
            tau2: Transmissivity [ix, il, kx, 4]
                  Bands: window, CO2, H2O weak, H2O strong
        """
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        nl1 = kx - 1
        
        tau2 = jnp.zeros((ix, il, kx, 4))
        acloud = cloudc * param.ablcl2
        
        # ====================================================================
        # Layer 0 (top stratosphere)
        # ====================================================================
        deltap_0 = psa * self.dhs[0]
        tau2 = tau2.at[:, :, 0, 0].set(jnp.exp(-deltap_0 * param.ablwin))
        tau2 = tau2.at[:, :, 0, 1].set(jnp.exp(-deltap_0 * param.ablco2))
        tau2 = tau2.at[:, :, 0, 2].set(1.0)
        tau2 = tau2.at[:, :, 0, 3].set(1.0)
        
        # ====================================================================
        # Layer 1 (lower stratosphere) - cloud-free
        # ====================================================================
        deltap_1 = psa * self.dhs[1]
        tau2 = tau2.at[:, :, 1, 0].set(jnp.exp(-deltap_1 * param.ablwin))
        tau2 = tau2.at[:, :, 1, 1].set(jnp.exp(-deltap_1 * param.ablco2))
        tau2 = tau2.at[:, :, 1, 2].set(jnp.exp(-deltap_1 * param.ablwv1 * qa[:, :, 1]))
        tau2 = tau2.at[:, :, 1, 3].set(jnp.exp(-deltap_1 * param.ablwv2 * qa[:, :, 1]))
        
        # ====================================================================
        # Layer kx-1 (PBL) - cloud-free
        # ====================================================================
        deltap_bot = psa * self.dhs[kx-1]
        tau2 = tau2.at[:, :, kx-1, 0].set(jnp.exp(-deltap_bot * param.ablwin))
        tau2 = tau2.at[:, :, kx-1, 1].set(jnp.exp(-deltap_bot * param.ablco2))
        tau2 = tau2.at[:, :, kx-1, 2].set(jnp.exp(-deltap_bot * param.ablwv1 * qa[:, :, kx-1]))
        tau2 = tau2.at[:, :, kx-1, 3].set(jnp.exp(-deltap_bot * param.ablwv2 * qa[:, :, kx-1]))
        
        # ====================================================================
        # Cloudy layers (k=2 to kx-2) - VECTORIZED
        # ====================================================================
        k_cloudy = jnp.arange(2, nl1)  # [2, 3, ..., kx-2]
        deltap_cloudy = psa[:, :, jnp.newaxis] * self.dhs[k_cloudy][jnp.newaxis, jnp.newaxis, :]  # [ix, il, nl1-2]
        
        # Cloud absorption: ablcl2 below cloud top, ablcl1 at/above
        above_cloud = k_cloudy[jnp.newaxis, jnp.newaxis, :] < icltop[:, :, jnp.newaxis]  # [ix, il, nl1-2]
        acloud1 = jnp.where(above_cloud, acloud[:, :, jnp.newaxis], param.ablcl1 * cloudc[:, :, jnp.newaxis])
        
        # Window band
        tau_win = jnp.exp(-deltap_cloudy * (param.ablwin + acloud1))
        tau2 = tau2.at[:, :, 2:nl1, 0].set(tau_win)
        
        # CO2 band
        tau_co2 = jnp.exp(-deltap_cloudy * param.ablco2)
        tau2 = tau2.at[:, :, 2:nl1, 1].set(tau_co2)
        
        # H2O bands
        qa_cloudy = qa[:, :, 2:nl1]  # [ix, il, nl1-2]
        h2o_weak = jnp.maximum(param.ablwv1 * qa_cloudy, acloud[:, :, jnp.newaxis])
        h2o_strong = jnp.maximum(param.ablwv2 * qa_cloudy, acloud[:, :, jnp.newaxis])
        
        tau2 = tau2.at[:, :, 2:nl1, 2].set(jnp.exp(-deltap_cloudy * h2o_weak))
        tau2 = tau2.at[:, :, 2:nl1, 3].set(jnp.exp(-deltap_cloudy * h2o_strong))
        
        return tau2
