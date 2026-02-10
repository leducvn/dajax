#!/usr/bin/env python3
"""
Longwave radiation parameterization for SPEEDY.

Computes absorption and emission of longwave radiation in 4 spectral bands:
1. Window band (8-12 μm)
2. CO2 band (15 μm)
3. H2O weak band
4. H2O strong band

Uses transmissivities (tau2) and stratospheric correction (stratc) from
CachedState computed by shortwave radiation.

IMPORTANT: downward() and upward() must be called separately because
surface fluxes are computed between them:
1. downward(ta, tau2) → slrd, dfabs, lw_state
2. get_surface_fluxes(..., slrd, ...) → ts, slru
3. upward(ta, ts, tau2, stratc, slrd, slru, dfabs, lw_state) → slr, olr, dfabs

Based on SPEEDY longwave_radiation.f90

FULLY VECTORIZED VERSION - no explicit Python for loops.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple

from .state import Config, Param
from .constants import Constants
from .vertical import VerticalGrid

class Longwave:
    """
    Longwave radiation scheme (fully vectorized).
    
    Computes absorption and emission of longwave radiation using:
    - 4 spectral bands with temperature-dependent fractions
    - Layer transmissivities from CachedState (shortwave)
    - Surface emission and reflection
    - Stratospheric correction for polar night
    
    Based on SPEEDY longwave_radiation.f90
    """
    
    NBAND = 4
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
    ):
        """
        Initialize longwave radiation scheme.
        
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
        """Initialize longwave radiation constants (VECTORIZED)."""
        self.dhs = jnp.array(self.vertical_grid.dhs)
        self.wvi = jnp.array(self.vertical_grid.wvi)
        
        # Band fractions lookup table
        self.fband = self._compute_fband()
        
    def _compute_fband(self) -> jax.Array:
        """
        Compute energy fractions in longwave bands (VECTORIZED).
        
        Based on SPEEDY longwave_radiation.f90:radset()
        
        Returns:
            fband: [301, 4] Band fractions for T = 100-400 K
        """
        fband = np.zeros((301, 4))
        
        # VECTORIZED: T = 200-320 K
        jtemp = np.arange(200, 321)
        idx = jtemp - 100
        
        fband[idx, 1] = (0.148 - 3.0e-6 * (jtemp - 247)**2)
        fband[idx, 2] = (0.356 - 5.2e-6 * (jtemp - 282)**2)
        fband[idx, 3] = (0.314 + 1.0e-5 * (jtemp - 315)**2)
        fband[idx, 0] = 1.0 - (fband[idx, 1] + fband[idx, 2] + fband[idx, 3])
        
        # VECTORIZED: Extend to T < 200 K
        fband[:100, :] = fband[100, :]  # Copy from T=200K
        
        # VECTORIZED: Extend to T > 320 K
        fband[221:, :] = fband[220, :]  # Copy from T=320K
        
        return jnp.array(fband)
    
    def _lookup_fband(self, temp: jax.Array) -> jax.Array:
        """Look up band fractions for given temperatures."""
        idx = jnp.clip(jnp.round(temp).astype(jnp.int32) - 100, 0, 300)
        return self.fband[idx]
    
    @partial(jax.jit, static_argnames=['self'])
    def downward(
        self,
        param: Param,
        ta: jax.Array,
        tau2: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute downward longwave radiation flux (VECTORIZED).
        
        Based on SPEEDY longwave_radiation.f90:get_downward_longwave_rad_fluxes()
        
        MUST be called before get_surface_fluxes(), which uses slrd.
        
        Args:
            ta: Absolute temperature [ix, il, kx] (K)
            tau2: Layer transmissivity [ix, il, kx, 4] (from CachedState)
            
        Returns:
            Tuple of:
            - slrd: Downward LW flux at surface [ix, il] (W/m²)
            - dfabs: LW flux absorbed in each layer [ix, il, kx] (W/m²)
            - st4a: Blackbody emission [ix, il, kx, 2] (for upward pass)
            - flux: Band fluxes [ix, il, 4] (for upward pass)
        """
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        eps1 = 1.0 - param.epslw
        
        # Blackbody emission
        st4a = self._compute_blackbody_emission(ta)
        
        # Initialize
        dfabs = jnp.zeros((ix, il, kx))
        flux = jnp.zeros((ix, il, self.NBAND))
        
        # ====================================================================
        # Stratosphere (k=0, bands 0,1 only) - VECTORIZED over bands
        # ====================================================================
        k = 0
        emis_strat = 1.0 - tau2[:, :, k, :2]  # [ix, il, 2]
        fband_k0 = eps1 * self._lookup_fband(ta[:, :, k])[:, :, :2]  # [ix, il, 2]
        brad_strat = fband_k0 * (st4a[:, :, k, 0:1] + emis_strat * st4a[:, :, k, 1:2])
        flux_strat = emis_strat * brad_strat
        flux = flux.at[:, :, :2].set(flux_strat)
        dfabs = dfabs.at[:, :, k].add(-jnp.sum(flux_strat, axis=2))
        
        # ====================================================================
        # Troposphere (k=1 to kx-1, all bands) - scan
        # ====================================================================
        def process_layer_down(carry, k):
            flux, dfabs = carry
            
            emis = 1.0 - tau2[:, :, k, :]  # [ix, il, 4]
            fband_k = eps1 * self._lookup_fband(ta[:, :, k])  # [ix, il, 4]
            brad = fband_k * (st4a[:, :, k, 0:1] + emis * st4a[:, :, k, 1:2])  # [ix, il, 4]
            
            # Absorption of incoming flux
            dfabs_k = jnp.sum(flux, axis=2)  # [ix, il]
            
            # Transmission and emission
            flux_new = tau2[:, :, k, :] * flux + emis * brad  # [ix, il, 4]
            
            # Outgoing flux subtracted
            dfabs_k = dfabs_k - jnp.sum(flux_new, axis=2)
            dfabs = dfabs.at[:, :, k].add(dfabs_k)
            
            return (flux_new, dfabs), None
        
        layers = jnp.arange(1, kx)
        (flux, dfabs), _ = jax.lax.scan(process_layer_down, (flux, dfabs), layers)
        
        # Surface downward flux
        slrd = param.emisfc * jnp.sum(flux, axis=2)
        
        # Correction for "black" band
        corlw = param.epslw * param.emisfc * st4a[:, :, kx-1, 0]
        dfabs = dfabs.at[:, :, kx-1].add(-corlw)
        slrd = slrd + corlw
        
        return slrd, dfabs, st4a, flux
    
    @partial(jax.jit, static_argnames=['self'])
    def upward(
        self,
        param: Param,
        ta: jax.Array,
        ts: jax.Array,
        tau2: jax.Array,
        stratc: jax.Array,
        slrd: jax.Array,
        slru: jax.Array,
        dfabs: jax.Array,
        st4a: jax.Array,
        flux: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute upward longwave radiation flux (VECTORIZED).
        
        Based on SPEEDY longwave_radiation.f90:get_upward_longwave_rad_fluxes()
        
        MUST be called after get_surface_fluxes(), which provides ts and slru.
        
        Args:
            ta: Absolute temperature [ix, il, kx] (K)
            ts: Surface/skin temperature [ix, il] (K) - from surface fluxes
            tau2: Layer transmissivity [ix, il, kx, 4]
            stratc: Stratospheric correction [ix, il, 2]
            slrd: Downward LW flux at surface [ix, il] (from downward pass)
            slru: Surface upward LW emission [ix, il] (from surface fluxes)
            dfabs: LW flux absorbed (from downward pass) [ix, il, kx]
            st4a: Blackbody emission [ix, il, kx, 2] (from downward pass)
            flux: Band fluxes [ix, il, 4] (from downward pass)
            
        Returns:
            Tuple of:
            - slr: Net upward LW flux at surface [ix, il] (W/m²)
            - olr: Outgoing LW flux at TOA [ix, il] (W/m²)
            - dfabs: Updated absorbed flux [ix, il, kx] (W/m²)
        """
        kx = self.config.kx
        eps1 = 1.0 - param.epslw

        # ====================================================================
        # Net upward LW at surface
        # ====================================================================
        slr = slru - slrd  # [ix, il]
        
        # ====================================================================
        # Surface emission and reflection - VECTORIZED over bands
        # ====================================================================
        refsfc = 1.0 - param.emisfc
        fband_s = eps1 * self._lookup_fband(ts)  # [ix, il, 4]
        flux = fband_s * slru[:, :, jnp.newaxis] + refsfc * flux  # [ix, il, 4]
        
        # ====================================================================
        # Troposphere (upward: k = kx-1 to 1) - scan
        # ====================================================================
        # Correction for "black" band at surface
        dfabs = dfabs.at[:, :, kx-1].add(param.epslw * slru)
        
        def process_layer_up(carry, k):
            flux, dfabs = carry
            
            emis = 1.0 - tau2[:, :, k, :]  # [ix, il, 4]
            fband_k = eps1 * self._lookup_fband(ta[:, :, k])  # [ix, il, 4]
            # Note: minus sign on gradient for upward
            brad = fband_k * (st4a[:, :, k, 0:1] - emis * st4a[:, :, k, 1:2])  # [ix, il, 4]
            
            # Absorption of incoming flux
            dfabs_k = jnp.sum(flux, axis=2)  # [ix, il]
            
            # Transmission and emission
            flux_new = tau2[:, :, k, :] * flux + emis * brad  # [ix, il, 4]
            
            # Outgoing flux subtracted
            dfabs_k = dfabs_k - jnp.sum(flux_new, axis=2)
            dfabs = dfabs.at[:, :, k].add(dfabs_k)
            
            return (flux_new, dfabs), None
        
        # Process from bottom to top (k = kx-1 down to k=1)
        layers = jnp.arange(1, kx)[::-1]
        (flux, dfabs), _ = jax.lax.scan(process_layer_up, (flux, dfabs), layers)
        
        # ====================================================================
        # Stratosphere (k=0, bands 0,1 only) - VECTORIZED
        # ====================================================================
        k = 0
        emis_strat = 1.0 - tau2[:, :, k, :2]  # [ix, il, 2]
        fband_k0 = eps1 * self._lookup_fband(ta[:, :, k])[:, :, :2]  # [ix, il, 2]
        brad_strat = fband_k0 * (st4a[:, :, k, 0:1] - emis_strat * st4a[:, :, k, 1:2])
        
        dfabs = dfabs.at[:, :, k].add(jnp.sum(flux[:, :, :2], axis=2))
        flux_strat = tau2[:, :, k, :2] * flux[:, :, :2] + emis_strat * brad_strat
        flux = flux.at[:, :, :2].set(flux_strat)
        dfabs = dfabs.at[:, :, k].add(-jnp.sum(flux_strat, axis=2))
        
        # ====================================================================
        # Stratospheric correction for polar night cooling
        # ====================================================================
        corlw1 = self.dhs[0] * stratc[:, :, 1] * st4a[:, :, 0, 0] + stratc[:, :, 0]
        corlw2 = self.dhs[1] * stratc[:, :, 1] * st4a[:, :, 1, 0]
        
        dfabs = dfabs.at[:, :, 0].add(-corlw1)
        dfabs = dfabs.at[:, :, 1].add(-corlw2)
        
        # Outgoing longwave radiation
        olr = corlw1 + corlw2 + jnp.sum(flux, axis=2)
        
        return slr, olr, dfabs
    
    def _compute_blackbody_emission(self, ta: jax.Array) -> jax.Array:
        """
        Compute blackbody emission from atmospheric levels (VECTORIZED).
        
        Based on SPEEDY get_downward_longwave_rad_fluxes lines 31-66
        
        Args:
            ta: Absolute temperature [ix, il, kx] (K)
            
        Returns:
            st4a: Blackbody emission [ix, il, kx, 2]
                  st4a[:,:,k,0] = σT⁴ (emission)
                  st4a[:,:,k,1] = 4σT³ΔT (gradient correction)
        """
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        nl1 = kx - 1
        sbc = self.constants.sbc
        
        st4a = jnp.zeros((ix, il, kx, 2))
        
        # ====================================================================
        # Temperature at layer boundaries - VECTORIZED
        # ====================================================================
        t_diff = ta[:, :, 1:kx] - ta[:, :, :nl1]  # [ix, il, kx-1]
        t_boundary = ta[:, :, :nl1] + self.wvi[:nl1][jnp.newaxis, jnp.newaxis, :] * t_diff  # [ix, il, kx-1]
        
        # Pad t_boundary to kx (last value not used but needed for shape)
        t_boundary = jnp.concatenate([t_boundary, ta[:, :, kx-1:kx]], axis=2)  # [ix, il, kx]
        
        # ====================================================================
        # Mean temperature in stratospheric layers (k=0, k=1)
        # ====================================================================
        st4a_k0_mean = 0.75 * ta[:, :, 0] + 0.25 * t_boundary[:, :, 0]
        st4a_k1_mean = 0.50 * ta[:, :, 1] + 0.25 * (t_boundary[:, :, 0] + t_boundary[:, :, 1])
        
        st4a = st4a.at[:, :, 0, 0].set(sbc * st4a_k0_mean**4)
        st4a = st4a.at[:, :, 0, 1].set(0.0)
        st4a = st4a.at[:, :, 1, 0].set(sbc * st4a_k1_mean**4)
        st4a = st4a.at[:, :, 1, 1].set(0.0)
        
        # ====================================================================
        # Temperature gradient in tropospheric layers (k=2 to kx-2) - VECTORIZED
        # ====================================================================
        anis = 1.0
        
        grad_tropo = 0.5 * anis * jnp.maximum(
            t_boundary[:, :, 2:nl1] - t_boundary[:, :, 1:nl1-1], 0.0
        )  # [ix, il, nl1-2]
        
        st3a_tropo = sbc * ta[:, :, 2:nl1]**3  # [ix, il, nl1-2]
        
        st4a = st4a.at[:, :, 2:nl1, 0].set(st3a_tropo * ta[:, :, 2:nl1])
        st4a = st4a.at[:, :, 2:nl1, 1].set(4.0 * st3a_tropo * grad_tropo)
        
        # ====================================================================
        # Bottom layer (k = kx-1)
        # ====================================================================
        grad_bot = anis * jnp.maximum(ta[:, :, kx-1] - t_boundary[:, :, nl1-1], 0.0)
        st3a_bot = sbc * ta[:, :, kx-1]**3
        
        st4a = st4a.at[:, :, kx-1, 0].set(st3a_bot * ta[:, :, kx-1])
        st4a = st4a.at[:, :, kx-1, 1].set(4.0 * st3a_bot * grad_bot)
        
        return st4a
