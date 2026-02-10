#!/usr/bin/env python3
"""
Main physics integration module for SPEEDY-JAX.

Orchestrates all physical parameterizations:
1. Convection (deep convection)
2. Large-scale condensation
3. Cloud diagnostics
4. Shortwave radiation (every nstrad steps)
5. Longwave radiation (every step)
6. Surface fluxes
7. Vertical diffusion and shallow convection
8. SPPT (Stochastically Perturbed Parameterization Tendencies)

Based on SPEEDY physics.f90

Data Flow:
    
    Convection → precnv, iptop, cbmf, tt_cnv, qt_cnv
         ↓
    Condensation → precls, iptop (updated), tt_lsc, qt_lsc
         ↓
    Clouds → icltop, cloudc, clstr, qcloud
         ↓
    [if compute_radiation]
    Shortwave → CachedState (tau2, stratc, ssrd, dfabs_sw, tsr, ssr)
         ↓
    Longwave.downward → slrd, dfabs_lw, st4a, flux
         ↓
    Surface Fluxes → ustr, vstr, shf, evap, slru, hfluxn, ts, tskin, ...
         ↓
    Longwave.upward → slr, olr, dfabs_lw (updated)
         ↓
    Vertical Diffusion → ut_vdi, vt_vdi, tt_vdi, qt_vdi
         ↓
    Add surface flux tendencies to bottom layer
         ↓
    [if sppt_on]
    Apply SPPT perturbations
         ↓
    Return tendencies and diagnostics
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, NamedTuple, Optional

from .util import Utility
from .state import Config, Param, TimeInfo, ForcingState, CachedState, DiagState
from .constants import Constants
from .vertical import VerticalGrid
from .transformer import Transformer
from .static import StaticFields

# Physics modules
from .convection import Convection
from .condensation import Condensation
from .cloud import Cloud
from .shortwave import Shortwave
from .longwave import Longwave
from .flux import SurfaceFluxes
from .diffusion import VerticalDiffusion

class Physics:
    """
    Main physics driver.
    
    Computes physical parameterization tendencies and updates
    surface models. Called every dynamics timestep.
    
    Based on SPEEDY physics.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
        transformer: Transformer,
        static_fields: StaticFields,
    ):
        """
        Initialize physics driver.
        
        Args:
            config: Model configuration
            constants: Physical constants
            vertical_grid: Vertical grid structure
            transformer: Spectral transformer
            static_fields: Static boundary conditions
        """
        self.config = config
        self.constants = constants
        self.vertical_grid = vertical_grid
        self.transformer = transformer
        self.static_fields = static_fields
        
        # Initialize physics modules
        self._init_modules()
        
        # Setup conversion factors
        self._setup()
    
    def _init_modules(self):
        """Initialize all physics sub-modules."""
        # Precipitation
        self.convection = Convection(self.config, self.constants, self.vertical_grid)
        self.condensation = Condensation(self.config, self.constants, self.vertical_grid)
        
        # Clouds and radiation
        self.cloud = Cloud(self.config, self.constants, self.static_fields)
        self.shortwave = Shortwave(self.config, self.constants, self.vertical_grid)
        self.longwave = Longwave(self.config, self.constants, self.vertical_grid)
        
        # Surface fluxes
        self.surface_fluxes = SurfaceFluxes(self.config, self.constants, self.vertical_grid, self.transformer, self.static_fields)
        
        # Boundary layer
        self.vertical_diffusion = VerticalDiffusion(self.config, self.constants, self.vertical_grid)
    
    def _setup(self):
        """Setup conversion factors for tendencies."""
        self.fsg = jnp.array(self.vertical_grid.fsg)
        self.grdsig = jnp.array(self.vertical_grid.grdsig)
        self.grdscp = jnp.array(self.vertical_grid.grdscp)
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        ug: jax.Array,
        vg: jax.Array,
        tg: jax.Array,
        qg: jax.Array,
        phig: jax.Array,
        psg: jax.Array,
        forcing: ForcingState,
        cached: CachedState,
        time: TimeInfo
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, CachedState, DiagState]:
        """
        Compute all physical parameterization tendencies.
        
        Based on SPEEDY physics.f90:get_physical_tendencies()
        
        Args:
            ug: U-wind [ix, il, kx] (m/s)
            vg: V-wind [ix, il, kx] (m/s)
            tg: Temperature [ix, il, kx] (K)
            qg: Specific humidity [ix, il, kx] (g/kg)
            phig: Geopotential [ix, il, kx] (m²/s²)
            psg: Surface pressure [ix, il] (normalized, p/p0)
            forcing: ForcingState with land, sea, solar, albsfc
            cached: CachedState with tau2, stratc, ssrd, dfabs_sw
            time: TimeInfo
            
        Returns:
            Tuple of:
            - Tendencies with ut, vt, tt, qt
            - CachedState (updated if compute_radiation)
            - PhysicsDiagnostics
        """
        ix, il = self.config.ix, self.config.il
        kx = self.config.kx
        
        # Unpack forcing state
        land = forcing.land
        sea = forcing.sea
        
        # ====================================================================
        # Compute thermodynamic variables
        # ====================================================================
        rps = 1.0 / psg  # Inverse surface pressure
        
        # Ensure non-negative humidity
        qg = jnp.maximum(qg, 0.0)
        
        # Dry static energy
        se = self.constants.cp * tg + phig
        
        # Relative humidity and saturation humidity
        fsg = self.fsg
        rh, qsat = jax.vmap(
            lambda tg_k, fsg_k, qg_k: Utility.spec_hum_to_rel_hum(tg_k, psg, fsg_k, qg_k),
            in_axes=(2, 0, 2),  # Map over k dimension: tg[...,k], fsg[k], qg[...,k]
            out_axes=2          # Output along k dimension: rh[...,k], qsat[...,k]
        )(tg, fsg, qg)
        #max_idx = jnp.unravel_index(jnp.argmax(qsat[:,:,kx-1]), qsat[:,:,kx-1].shape)
        #jax.debug.print("step={}, max_qsat={}, at i={}, j={}, T={}", 
                        #time.step, qsat[max_idx[0], max_idx[1], kx-1],
                        #max_idx[0], max_idx[1], 
                        #tg[max_idx[0], max_idx[1], kx-1])
        
        # ====================================================================
        # Initialize tendencies
        # ====================================================================
        ut = jnp.zeros((ix, il, kx))
        vt = jnp.zeros((ix, il, kx))
        tt = jnp.zeros((ix, il, kx))
        qt = jnp.zeros((ix, il, kx))
        
        # ====================================================================
        # 1. Deep Convection
        # ====================================================================
        iptop, precnv, tt_cnv, qt_cnv = self.convection(param, psg, se, qg, qsat)
        
        # Convert fluxes to tendencies (scale by inverse pressure)
        # tt_cnv and qt_cnv from convection are already scaled correctly
        # but need rps factor
        tt_cnv = tt_cnv.at[:, :, 1:].set(
            tt_cnv[:, :, 1:] * rps[:, :, jnp.newaxis] * self.grdscp[jnp.newaxis, jnp.newaxis, 1:]
        )
        qt_cnv = qt_cnv.at[:, :, 1:].set(
            qt_cnv[:, :, 1:] * rps[:, :, jnp.newaxis] * self.grdsig[jnp.newaxis, jnp.newaxis, 1:]
        )
        # Convection depth indicator for vertical diffusion
        # IMPORTANT: Compute icnv BEFORE condensation modifies iptop
        icnv = kx - 1 - iptop
        
        # ====================================================================
        # 2. Large-scale Condensation
        # ====================================================================
        iptop, precls, tt_lsc, qt_lsc = self.condensation(param, psg, qg, qsat, iptop)
        
        # Add precipitation tendencies
        tt = tt + tt_cnv + tt_lsc
        qt = qt + qt_cnv + qt_lsc
        #jax.debug.print("tt_cnv: step={}, min={}, max={}", time.step, jnp.min(tt_cnv), jnp.max(tt_cnv))
        #jax.debug.print("tt_lsc: step={}, min={}, max={}", time.step, jnp.min(tt_lsc), jnp.max(tt_lsc))
        
        # ====================================================================
        # 3. Clouds and Radiation
        # ====================================================================
        # Shortwave radiation (every nstrad steps)
        def compute_shortwave():
            # Dry static energy gradient for stratiform clouds
            gse = (se[:, :, kx-2] - se[:, :, kx-1]) / (phig[:, :, kx-2] - phig[:, :, kx-1])
            # Compute cloud diagnostics
            icltop, cloudc, clstr, qcloud = self.cloud(param, qg, rh, precnv, precls, iptop, gse)
            # Compute new shortwave radiation
            new_cached = self.shortwave(param, psg, qg, icltop, cloudc, clstr, qcloud, forcing)
            return new_cached
        
        def use_cached():
            # Return existing cached state
            return cached
        
        compute_radiation = (time.step % self.config.nstrad) == 0
        cached = jax.lax.cond(compute_radiation, compute_shortwave, use_cached)
        
        # ====================================================================
        # 4. Longwave Downward
        # ====================================================================
        slrd, dfabs_lw, st4a, flux = self.longwave.downward(param, tg, cached.tau2)
        
        # ====================================================================
        # 5. Surface Fluxes
        # ====================================================================
        (ustr, vstr, shf, evap, slru, hfluxn, 
         ts, tskin, u0, v0, t0, t1, q1, denvvs) = self.surface_fluxes(
            param, psg, ug, vg, tg, qg, rh, phig,
            land.stl_am, land.soilw_am, sea.sst_am,
            cached.ssrd, slrd, forcing.alb_l, forcing.alb_s, forcing.snowc
        )
        
        # For anomaly coupling: recompute sea fluxes with ocean model SST
        if self.config.sea_coupling_flag > 0:
            shf_s, evap_s, slru_s, hfluxn_s = self.surface_fluxes.recompute_sea_fluxes(
                param, psg, sea.ssti_om, cached.ssrd, slrd, 
                forcing.alb_s, t1[:,:,1], q1[:,:,1], denvvs[:,:,1]
            )
            shf = shf.at[:, :, 1].set(shf_s)
            evap = evap.at[:, :, 1].set(evap_s)
            slru = slru.at[:, :, 1].set(slru_s)
            hfluxn = hfluxn.at[:, :, 1].set(hfluxn_s)
        
        # ====================================================================
        # 6. Longwave Upward
        # ====================================================================
        slr, olr, dfabs_lw = self.longwave.upward(
            param, tg, ts, cached.tau2, cached.stratc,
            slrd, slru[:, :, 2], dfabs_lw, st4a, flux
        )
        
        # Shortwave temperature tendency (use cached.dfabs_sw)
        tt_rsw = cached.dfabs_sw * rps[:, :, jnp.newaxis] * self.grdscp[jnp.newaxis, jnp.newaxis, :]
        # Longwave temperature tendency
        tt_rlw = dfabs_lw * rps[:, :, jnp.newaxis] * self.grdscp[jnp.newaxis, jnp.newaxis, :]
        #jax.debug.print("tt_rsw: step={}, min={}, max={}", time.step, jnp.min(tt_rsw), jnp.max(tt_rsw))
        #jax.debug.print("tt_rlw: step={}, min={}, max={}", time.step, jnp.min(tt_rlw), jnp.max(tt_rlw))

        # Add radiation tendencies
        tt = tt + tt_rsw + tt_rlw
        
        # ====================================================================
        # 7. Vertical Diffusion and Shallow Convection
        # ====================================================================
        ut_vdi, vt_vdi, tt_vdi, qt_vdi = self.vertical_diffusion(param, se, rh, qg, qsat, phig, icnv)
        
        # Add surface flux tendencies to bottom layer
        grdsig_kx = self.grdsig[kx-1]
        grdscp_kx = self.grdscp[kx-1]
        
        ut_pbl = ut_vdi.at[:, :, kx-1].add(ustr[:, :, 2] * rps * grdsig_kx)
        vt_pbl = vt_vdi.at[:, :, kx-1].add(vstr[:, :, 2] * rps * grdsig_kx)
        tt_pbl = tt_vdi.at[:, :, kx-1].add(shf[:, :, 2] * rps * grdscp_kx)
        qt_pbl = qt_vdi.at[:, :, kx-1].add(evap[:, :, 2] * rps * grdsig_kx)
        #jax.debug.print("tt_pbl: step={}, min={}, max={}", time.step, jnp.min(tt_pbl), jnp.max(tt_pbl))
        
        # Add PBL tendencies
        ut = ut + ut_pbl
        vt = vt + vt_pbl
        tt = tt + tt_pbl
        qt = qt + qt_pbl
        
        # ====================================================================
        # 8. Create outputs
        # ====================================================================
        
        # Create diagnostics (subset for land/sea models)
        diagnostics = DiagState(
            hfluxn=hfluxn,                    # [ix, il, 2]
            shf=shf,                          # [ix, il, 3] land, sea only
            evap=evap,                        # [ix, il, 3]
            ustr=ustr,                        # [ix, il, 3]
            vstr=vstr,                        # [ix, il, 3]
            slru=slru,                        # [ix, il, 3]
            ssrd=cached.ssrd,                 # [ix, il]
            slrd=slrd,                        # [ix, il]
            slr=slr,                          # [ix, il]
            olr=olr,                          # [ix, il]
            precnv=precnv,                    # [ix, il]
            precls=precls,                    # [ix, il]
            ts=ts,                            # [ix, il]
            tskin=tskin,                      # [ix, il]
            u0=u0,                            # [ix, il]
            v0=v0,                            # [ix, il]
            t0=t0,                            # [ix, il]
        )
        
        return ut, vt, tt, qt, cached, diagnostics

def create_physics(
    config: Config,
    param: Optional[Param] = None,
    constants: Optional[Constants] = None,
    vertical: Optional[VerticalGrid] = None,
    transformer=None,
    static_fields=None,
) -> Physics:
    """
    Factory function to create Physics instance.
    
    Args:
        config: Model configuration (required)
        param: Tunable parameters (optional, uses defaults)
        constants: Physical constants (optional, uses defaults)
        vertical: Vertical grid (optional, creates from config)
        transformer: Spectral transformer (required for full physics)
        static_fields: Static fields (required for full physics)
        
    Returns:
        Physics instance
    """
    if param is None:
        param = Param()
    
    if constants is None:
        constants = Constants()
    
    if vertical is None:
        vertical = VerticalGrid(config.kx, constants)
    
    return Physics(
        config=config,
        param=param,
        constants=constants,
        vertical=vertical,
        transformer=transformer,
        static_fields=static_fields,
    )
