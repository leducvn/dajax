#!/usr/bin/env python3
"""
Core atmospheric dynamics for SPEEDY model.

Implements:
- Geopotential calculation (hydrostatic equation)
- Tendency computation (primitive equations)
- Horizontal diffusion (∇⁴ hyperdiffusion)

Based on SPEEDY:
- geopotential.f90
- tendencies.f90
- horizontal_diffusion.f90
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple

from .state import Config, Param, SpectralState, ForcingState, CachedState, DiagState
from .util import TimeInfo
from .constants import Constants
from .vertical import VerticalGrid
from .transformer import Transformer
from .static import StaticFields
from .implicit import ImplicitSolver
from .physics import Physics

class Dynamics:
    """
    Complete dynamical core including:
    - Primitive equations (advection, pressure gradient, Coriolis)
    - Horizontal diffusion (∇⁴ hyperdiffusion for numerical stability)
    
    Provides:
    - Geopotential calculation (hydrostatic balance)
    - Tendency computation (primitive equations + diffusion + physics)
    """
    
    def __init__(self, config: Config, constants: Constants, vertical_grid: VerticalGrid, 
                 transformer: Transformer, static_fields: StaticFields):
        """
        Initialize dynamics with optional diffusion and simple physics.
        
        Args:
            config: Config instance from speedy_jax
            constants: Constants instance from speedy_jax
            vertical_grid: VerticalGrid instance from speedy_jax
            transformer: Transformer instance
            static_fields: StaticFields instance
            
        """
        self.config = config
        self.constants = constants
        self.vertical_grid = vertical_grid
        self.transformer = transformer
        self.static_fields = static_fields
        
        # Setup diffusion coefficients
        self._setup_diffusion()
        self._setup_orographic_corrections()
        
        # Working arrays
        self.hsg = jnp.array(self.vertical_grid.hsg)
        self.fsg = jnp.array(self.vertical_grid.fsg)
        self.fsgr = jnp.array(self.vertical_grid.fsgr)
        self.dhs = jnp.array(self.vertical_grid.dhs)
        self.dhsr = jnp.array(self.vertical_grid.dhsr)
        self.tref = jnp.array(self.vertical_grid.tref)
        self.tref2 = jnp.array(self.vertical_grid.tref2)
        self.tref3 = jnp.array(self.vertical_grid.tref3)
        self.xgeop1 = jnp.array(self.vertical_grid.xgeop1)
        self.xgeop2 = jnp.array(self.vertical_grid.xgeop2)
    
    # ========================================================================
    # Main Public Interface
    # ========================================================================

    @partial(jax.jit, static_argnames=['self', 'implicit_solver', 'physics'])
    def compute_tendencies(
        self,
        implicit_solver: ImplicitSolver,
        physics: Physics,
        param: Param,
        state_dyn: SpectralState,
        state_phy: SpectralState,
        forcing: ForcingState,
        cached: CachedState,
        time: TimeInfo
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, CachedState, DiagState]:
        """
        Compute complete tendencies including:
        1. Dynamical tendencies (primitive equations)
        2. Physics parameterizations
        3. Horizontal diffusion
        
        Args:
            implicit_solver: ImplicitSolver instance
            physics: Physics instance for parameterizations
            state_dyn: Spectral state at dynamics time level (j2)
            state_phy: Spectral state at physics time level (j1)
            forcing: Forcing state (albedo, solar, surface conditions)
            cached: Cached radiation fields
            time: Time information
            
        Returns:
            Tuple of (vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis)
        """
        # 1. Core dynamics + physics (primitive equations + parameterizations)
        vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis = self._compute_dynamical_tendencies(
            implicit_solver, physics, param, state_dyn, state_phy, forcing, cached, time
        )
        
        # 2. Add horizontal diffusion (critical for numerical stability)
        vordt, divdt, tdt, qdt = self._add_horizontal_diffusion(
            implicit_solver, vordt, divdt, tdt, qdt, state_phy, forcing
        )
        
        # 3. Apply truncation to tendencies if needed
        if self.config.ix == self.config.iy * 4:
            vordt = self.transformer.apply_truncation_3d(vordt)
            divdt = self.transformer.apply_truncation_3d(divdt)
            tdt = self.transformer.apply_truncation_3d(tdt)
            qdt = self.transformer.apply_truncation_3d(qdt)
            psdt = self.transformer.apply_truncation(psdt)
        
        return vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis
    
    # ========================================================================
    # Horizontal Diffusion Setup and Application
    # ========================================================================
    
    def _setup_diffusion(self):
        """
        Setup horizontal diffusion coefficients (VECTORIZED).
        Based on SPEEDY horizontal_diffusion.f90:36-82
        
        Uses ∇⁴ (4th order) hyperdiffusion for:
        - Vorticity and temperature: thd timescale
        - Divergence: thdd timescale
        - Stratospheric extra diffusion: thds timescale
        """
        mx, nx = self.config.mx, self.config.nx
        trunc = self.config.trunc
        
        # Diffusion timescales (hours) - from dynamical_constants module
        thd = self.config.thd    # Vorticity and temperature
        thdd = self.config.thdd   # Divergence
        thds = self.config.thds   # Stratospheric extra diffusion
        tdrs = self.config.tdrs   # Stratospheric drag timescale (hours)
        npowhd = self.config.npowhd # Power of Laplacian (∇⁴ = 4th order)

        # Convert to 1/seconds
        hdiff = 1.0 / (thd * 3600.0)
        hdifd = 1.0 / (thdd * 3600.0)
        hdifs = 1.0 / (thds * 3600.0)
        
        # Normalization factor for Laplacian
        rlap = 1.0 / float(trunc * (trunc + 1))
        
        # VECTORIZED: Create 2D arrays of wavenumbers
        # k ranges from 0 to mx-1 (m index)
        # j ranges from 0 to nx-1 (n index)
        k_grid = np.arange(mx)[:, jnp.newaxis]  # [mx, 1]
        j_grid = np.arange(nx)[jnp.newaxis, :]  # [1, nx]
        
        # Total wavenumber: l = m + n
        twn = (k_grid + j_grid).astype(float)  # [mx, nx]
        
        # Normalized Laplacian: l(l+1) / [T(T+1)]
        elap = (twn * (twn + 1.0)) * rlap
        
        # Hyperdiffusion: [l(l+1) / T(T+1)]^4
        elapn = elap ** npowhd
        
        # Explicit damping coefficients (all vectorized!)
        dmp = hdiff * elapn    # Vorticity and temperature
        dmpd = hdifd * elapn   # Divergence
        dmps = hdifs * elap    # Stratospheric (note: elap, not elapn)

        # Store both explicit and implicit coefficients
        self.dmp = jnp.array(dmp)
        self.dmpd = jnp.array(dmpd)
        self.dmps = jnp.array(dmps)
        self.sdrag = 1.0 / (tdrs * 3600.0)
    
    def _setup_orographic_corrections(self):
        """
        Setup orographic correction terms for temperature and humidity - FULLY VECTORIZED.
        
        Based on SPEEDY:
        - horizontal_diffusion.f90:69-81 (vertical components tcorv, qcorv)
        - forcing.f90:73-99 (horizontal components tcorh, qcorh)
        
        Orographic corrections account for:
        - Temperature: Effect of topography on surface temperature lapse rate
        - Humidity: Effect of topography on surface humidity
        
        """
        kx = self.config.kx
        rgas = self.constants.rgas
        grav = self.constants.grav
        gamma = self.constants.gamma
        hscale = self.constants.hscale
        hshum = self.constants.hshum
        fsg = self.vertical_grid.fsg
        
        # ========================================================================
        # Vertical Components (function of sigma level)
        # ========================================================================
        
        # Temperature correction exponent
        rgam = rgas * gamma / (1000.0 * grav)
        
        # Humidity correction exponent
        qexp = hscale / hshum
        
        # VECTORIZED: compute all levels at once
        k_range = np.arange(kx)
        
        # tcorv[k] = fsg[k]^rgam for k >= 1, else 0
        tcorv = np.where(k_range >= 1, fsg ** rgam, 0.0)
        
        # qcorv[k] = fsg[k]^qexp for k >= 2, else 0
        qcorv = np.where(k_range >= 2, fsg ** qexp, 0.0)
        
        self.tcorv = jnp.array(tcorv)
        self.qcorv = jnp.array(qcorv)
        
        # ========================================================================
        # Horizontal Components (function of surface geopotential)
        # ========================================================================
        
        # Temperature correction: tcorh = gamlat * phis
        # Reference lapse rate (constant for all latitudes in SPEEDY)
        gamlat = gamma / (1000.0 * grav)
        
        # Multiply by constant gamlat (broadcast automatically)
        phis_grid = self.static_fields.get_field('phis0')
        tcorh_grid = gamlat * phis_grid  # [ix, il]
        
        # Transform back to spectral space
        tcorh = self.transformer.grid_to_spec(tcorh_grid)  # [mx, nx]
        self.tcorh = jax.lax.cond(
            self.config.ix == self.config.iy * 4,
            lambda x: self.transformer.apply_truncation(tcorh),
            lambda x: x,
            tcorh
        )

    @partial(jax.jit, static_argnames=['self', 'implicit_solver'])
    def _add_horizontal_diffusion(
        self,
        implicit_solver: ImplicitSolver,
        vordt: jax.Array,      # [mx, nx, kx]
        divdt: jax.Array,      # [mx, nx, kx]
        tdt: jax.Array,        # [mx, nx, kx]
        qdt: jax.Array,        # [mx, nx, kx]
        state_phy: SpectralState,
        forcing: ForcingState
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Add horizontal diffusion tendencies using BACKWARD IMPLICIT scheme - FULLY VECTORIZED.
        
        Based on SPEEDY:
        - horizontal_diffusion.f90:86-105 (diffusion formula)
        - time_stepping.f90:62-85 (application order and orographic corrections)
        
        CRITICAL: Uses backward implicit formula for unconditional stability:
            fdt_out = (fdt_in - dmp*field) * dmp1
        
        where dmp1 = 1/(1 + dmp*dt)
        
        Application order (from time_stepping.f90):
        1. Primary diffusion (∇⁴) of vorticity, divergence, temperature
        - Temperature uses orographic correction: t + tcorh*tcorv
        2. Stratospheric zonal wind damping (m=0 modes only, top level)
        3. Stratospheric extra diffusion (∇²) of vorticity, divergence, temperature
        
        Args:
            implicit_solver: ImplicitSolver instance
            vordt, divdt, tdt, qdt: Current tendencies [mx, nx, kx]
            state_phy: Spectral state at physics time level
            forcing: ForcingState containing qcorh for humidity correction
            
        Returns:
            Updated tendencies with diffusion added
        """
        # Unpack spectral fields from state_phy
        vor_spec = state_phy.vor
        div_spec = state_phy.div
        t_spec = state_phy.t
        q_spec = state_phy.q

        # ========================================================================
        # Step 1: Primary Diffusion (∇⁴ hyperdiffusion)
        # ========================================================================
        
        dmp1 = implicit_solver.dmp1
        dmp1d = implicit_solver.dmp1d
        dmp1s = implicit_solver.dmp1s

        # Vorticity diffusion: vordt = (vordt - dmp*vor)*dmp1
        vordt = (vordt - self.dmp[:, :, jnp.newaxis] * vor_spec) * dmp1[:, :, jnp.newaxis]
        
        # Divergence diffusion: divdt = (divdt - dmpd*div)*dmp1d
        divdt = (divdt - self.dmpd[:, :, jnp.newaxis] * div_spec) * dmp1d[:, :, jnp.newaxis]
        
        # Temperature diffusion with orographic correction
        # Apply orographic correction: t_corrected = t + tcorh*tcorv
        t_corrected = t_spec + self.tcorh[:, :, jnp.newaxis] * self.tcorv[jnp.newaxis, jnp.newaxis, :]
        
        # Apply diffusion to corrected temperature
        tdt = (tdt - self.dmp[:, :, jnp.newaxis] * t_corrected) * dmp1[:, :, jnp.newaxis]
        
        # ========================================================================
        # Step 2: Stratospheric Zonal Wind Damping
        # ========================================================================
        
        # Apply damping only to m=0 modes (zonal mean) at top level (k=0)
        vordt = vordt.at[0, :, 0].add(-self.sdrag * vor_spec[0, :, 0])
        divdt = divdt.at[0, :, 0].add(-self.sdrag * div_spec[0, :, 0])
        
        # ========================================================================
        # Step 3: Stratospheric Extra Diffusion (∇² instead of ∇⁴)
        # ========================================================================
        
        # Apply ∇² diffusion using dmps, dmp1s
        # Note: dmps uses elap (not elapn), so this is ∇² not ∇⁴
        # Vorticity: vordt = (vordt - dmps*vor)*dmp1s
        #vordt = (vordt - self.dmps[:, :, jnp.newaxis] * vor_spec) * dmp1s[:, :, jnp.newaxis]
        # Divergence: divdt = (divdt - dmps*div)*dmp1s
        #divdt = (divdt - self.dmps[:, :, jnp.newaxis] * div_spec) * dmp1s[:, :, jnp.newaxis]
        # Temperature: tdt = (tdt - dmps*t_corrected)*dmp1s
        # Use the same orographic-corrected temperature
        #tdt = (tdt - self.dmps[:, :, jnp.newaxis] * t_corrected) * dmp1s[:, :, jnp.newaxis]
        # Stratospheric extra diffusion - ONLY apply to top level (k=0)
        vordt = vordt.at[:, :, 0].set((vordt[:, :, 0] - self.dmps * vor_spec[:, :, 0]) * dmp1s)
        divdt = divdt.at[:, :, 0].set((divdt[:, :, 0] - self.dmps * div_spec[:, :, 0]) * dmp1s)
        tdt = tdt.at[:, :, 0].set((tdt[:, :, 0] - self.dmps * t_corrected[:, :, 0]) * dmp1s)
        # ========================================================================
        # Step 4: Tracer Diffusion
        # ========================================================================
        
        # Get qcorh from forcing (computed in forcing.py)
        qcorh = forcing.qcorh
        # Apply orographic correction to qv
        q_corrected = q_spec + qcorh[:, :, jnp.newaxis] * self.qcorv[jnp.newaxis, jnp.newaxis, :]
        # Apply diffusion with divergence coefficients
        qdt = (qdt - self.dmpd[:, :, jnp.newaxis] * q_corrected) * dmp1d[:, :, jnp.newaxis]
        
        return vordt, divdt, tdt, qdt
    
    # ========================================================================
    # Tendency Computation (Main Method)
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self', 'implicit_solver', 'physics'])
    def _compute_dynamical_tendencies(
        self,
        implicit_solver: ImplicitSolver,
        physics: Physics,
        param: Param,
        state_dyn: SpectralState,
        state_phy: SpectralState,
        forcing: ForcingState,
        cached: CachedState,
        time: TimeInfo
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, CachedState, DiagState]:
        """
        Compute dynamical tendencies from primitive equations and physics.
        Based on SPEEDY tendencies.f90
        
        Args:
            implicit_solver: ImplicitSolver instance
            physics: Physics instance for parameterizations
            state_dyn: Spectral state at dynamics time level (j2)
            state_phy: Spectral state at physics time level (j1)
            forcing: Forcing state
            cached: Cached radiation fields
            time: Time information
            
        Returns:
            Tuple of (vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis)
        """
        # Unpack dynamics time level spectral state
        vor_spec = state_dyn.vor
        div_spec = state_dyn.div
        t_spec = state_dyn.t
        q_spec = state_dyn.q
        ps_spec = state_dyn.ps
        
        # Unpack physics time level spectral state
        vor_spec_phy = state_phy.vor
        div_spec_phy = state_phy.div
        t_spec_phy = state_phy.t
        q_spec_phy = state_phy.q
        ps_spec_phy = state_phy.ps

        # 1. Transform to grid space
        vorg, divg, tg, qg, ug, vg, tgg = self._transform_to_grid(vor_spec, div_spec, t_spec, q_spec)
        
        # 2. Surface pressure tendency
        psdt, umean, vmean, dmean, px, py = self._compute_surface_pressure_tendency(ps_spec, ug, vg, divg)
        
        # 3. Vertical velocity (sigma-dot)
        sigdt, sigm, puv = self._compute_vertical_velocity(ug, vg, divg, umean, vmean, dmean, px, py)
        
        # 4. Grid-point tendencies
        utend, vtend, ttend, qtend = self._compute_grid_tendencies(ug, vg, tg, qg, tgg, vorg, divg, px, py, sigdt, sigm, puv, dmean)
        
        # 5. Add physical tendencies
        utend_phy, vtend_phy, ttend_phy, qtend_phy, new_cached, diagnosis = self._add_physical_tendencies(
            physics, param, state_phy, forcing, cached, time
        )
        # Optional: Apply SPPT perturbations to physics tendencies
        # (Would need sppt_pattern and mu from configuration)
        # if self.config.sppt_on:
        #     utend_phy, vtend_phy, ttend_phy, qtend_phy = apply_sppt(
        #         utend_phy, vtend_phy, ttend_phy, qtend_phy, sppt_pattern, mu
        #     )
        # Add physics tendencies to dynamical tendencies
        utend = utend + utend_phy
        vtend = vtend + vtend_phy
        ttend = ttend + ttend_phy
        qtend = qtend + qtend_phy
        
        # 6. Transform to spectral
        vordt, divdt, tdt, qdt = self._transform_tendencies_to_spectral(utend, vtend, ttend, qtend, ug, vg, tgg, qg)
        #jax.debug.print("After tendencies_to_spectral: tdt[0,2,:]: {x}", x=tdt[0,2,:])
        
        # 7. Add spectral corrections
        alph = self.config.alph  # Semi-implicit parameter
        if alph < 0.5:
            # Explicit scheme: just add spectral corrections
            divdt, tdt, psdt = self._add_spectral_corrections(divdt, tdt, psdt, div_spec, t_spec, ps_spec)
        else:
            # Semi-implicit scheme: add spectral corrections then apply implicit solver
            divdt, tdt, psdt = self._add_spectral_corrections(divdt, tdt, psdt, div_spec_phy, t_spec_phy, ps_spec_phy)
            #jax.debug.print("Before implicit: tdt[0,2,:]: {x}", x=tdt[0,2,:])
            
            # Apply implicit correction for gravity waves
            divdt, tdt, psdt = implicit_solver.apply(divdt, tdt, psdt)
            #jax.debug.print("After implicit: tdt[0,2,:]: {x}", x=tdt[0,2,:])

        return vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis
    
    @partial(jax.jit, static_argnames=['self', 'physics'])
    def _add_physical_tendencies(
        self,
        physics: Physics,
        param: Param,
        state: SpectralState,
        forcing: ForcingState,
        cached: CachedState,
        time: TimeInfo
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, CachedState, DiagState]:
        """
        Add physical parameterization tendencies.
        
        Based on SPEEDY get_physical_tendencies() in tendencies.f90
        
        Takes spectral state variables and modifies grid-space tendencies.
        This is where physics modules (convection, radiation, surface fluxes, etc.) 
        would add their contributions to the tendencies.
        
        Args:
            physics: Physics instance for parameterizations
            state_phy: Spectral state at physics time level (j1)
            forcing: Forcing state
            cached: Cached radiation fields
            time: Time information
            
        Returns:
            Updated (vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis)
        """
        # Unpack physics time level spectral state
        vor_spec = state.vor
        div_spec = state.div
        t_spec = state.t
        q_spec = state.q
        ps_spec = state.ps
        phi_spec = self._geopotential(t_spec)  # [mx, nx, kx]

        # Convert vorticity/divergence to U/V winds (vectorized over levels)
        u_spec, v_spec = self.transformer.vor_div_3d_to_uv(vor_spec, div_spec)
        # Transform to grid with wind weighting
        ug = self.transformer.spec_3d_to_grid(u_spec, kcos = True)
        vg = self.transformer.spec_3d_to_grid(v_spec, kcos = True)
        tg = self.transformer.spec_3d_to_grid(t_spec)
        qg = self.transformer.spec_3d_to_grid(q_spec)
        phig = self.transformer.spec_3d_to_grid(phi_spec)
        psg = self.transformer.spec_to_grid(ps_spec)
        psg = jnp.exp(psg)
        
        utend, vtend, ttend, qtend, new_cached, diagnosis = physics(
            param, ug, vg, tg, qg, phig, psg, forcing, cached, time
        )
        
        return utend, vtend, ttend, qtend, new_cached, diagnosis

    # ========================================================================
    # Helper Method 1: Transform to Grid Space
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _transform_to_grid(
        self,
        vor_spec: jax.Array,  # [mx, nx, kx]
        div_spec: jax.Array,  # [mx, nx, kx]
        t_spec: jax.Array,    # [mx, nx, kx]
        q_spec: jax.Array     # [mx, nx, kx]
    ):
        """
        Transform spectral fields to grid point space.
        
        Args:
            vor_spec: Vorticity [mx, nx, kx]
            div_spec: Divergence [mx, nx, kx]
            t_spec: Temperature [mx, nx, kx]
            q_spec: Humidity [mx, nx, kx]
        
        Returns:
            vorg: Vorticity + planetary vorticity [ix, il, kx]
            divg: Divergence [ix, il, kx]
            tg: Temperature [ix, il, kx]
            qg: Humidity [ix, il, kx]
            ug: Zonal wind [ix, il, kx]
            vg: Meridional wind [ix, il, kx]
            tgg: Temperature deviation from reference [ix, il, kx]
        """
        # Transform all levels at once using vmap
        vorg = self.transformer.spec_3d_to_grid(vor_spec)  # [ix, il, kx]
        divg = self.transformer.spec_3d_to_grid(div_spec)
        tg = self.transformer.spec_3d_to_grid(t_spec)
        
        # Convert vorticity/divergence to U/V winds (vectorized over levels)
        u_spec, v_spec = self.transformer.vor_div_3d_to_uv(vor_spec, div_spec)
        
        # Transform to grid with wind weighting
        ug = self.transformer.spec_3d_to_grid(u_spec, kcos = True)
        vg = self.transformer.spec_3d_to_grid(v_spec, kcos = True)
        
        # Add planetary vorticity (Coriolis parameter)
        coriol = self.transformer.coriol  # [il]
        vorg = vorg + coriol[jnp.newaxis, :, jnp.newaxis]  # Broadcasting
        
        # Subtract reference temperature
        tgg = tg - self.tref[jnp.newaxis, jnp.newaxis, :]

        # Transform qv
        qg = self.transformer.spec_3d_to_grid(q_spec)

        return vorg, divg, tg, qg, ug, vg, tgg
    
    # ========================================================================
    # Helper Method 2: Surface Pressure Tendency
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_surface_pressure_tendency(
        self,
        ps_spec: jax.Array,
        ug: jax.Array,
        vg: jax.Array,
        divg: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute surface pressure tendency and related fields.
        
        Args:
            ps_spec: Log surface pressure [mx, nx]
            ug, vg: Winds [ix, il, kx]
            divg: Divergence [ix, il, kx]
            
        Returns:
            psdt: Surface pressure tendency [mx, nx]
            umean: Vertical mean U [ix, il]
            vmean: Vertical mean V [ix, il]
            dmean: Vertical mean divergence [ix, il]
            px: âˆ‚ps/âˆ‚Î» [ix, il]
            py: âˆ‚ps/âˆ‚Î¸ [ix, il]
        """
        dhs = self.dhs  # [kx]
        
        # Vertical mean: sum over k with weights dhs (VECTORIZED with einsum)
        umean = jnp.einsum('ijk,k->ij', ug, dhs)  # [ix, il]
        vmean = jnp.einsum('ijk,k->ij', vg, dhs)
        dmean = jnp.einsum('ijk,k->ij', divg, dhs)
        
        # Surface pressure gradient
        px_spec = self.transformer.grad_lon(ps_spec)
        py_spec = self.transformer.grad_lat(ps_spec)
        px = self.transformer.spec_to_grid(px_spec, kcos = True)  # [ix, il]
        py = self.transformer.spec_to_grid(py_spec, kcos = True)
        
        # Surface pressure tendency: psdt = -umean*px - vmean*py
        psdt_grid = -umean * px - vmean * py
        psdt = self.transformer.grid_to_spec(psdt_grid)
        psdt = psdt.at[0, 0].set(0.0 + 0.0j)  # Zero mean
        
        return psdt, umean, vmean, dmean, px, py
    
    # ========================================================================
    # Helper Method 3: Vertical Velocity
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_vertical_velocity(
        self,
        ug: jax.Array,
        vg: jax.Array,
        divg: jax.Array,
        umean: jax.Array,
        vmean: jax.Array,
        dmean: jax.Array,
        px: jax.Array,
        py: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute vertical velocity in sigma coordinates.
        
        Args:
            ug, vg: Winds [ix, il, kx]
            divg: Divergence [ix, il, kx]
            umean, vmean, dmean: Vertical means [ix, il]
            px, py: Pressure gradients [ix, il]
            
        Returns:
            sigdt: ÏƒÌ‡ for continuity [ix, il, kx+1]
            sigm: ÏƒÌ‡ for pressure advection [ix, il, kx+1]
            puv: Pressure gradient work term [ix, il, kx]
        """
        ix, il, kx = ug.shape
        dhs = self.dhs  # [kx]
        
        # Deviation from mean (for temperature equation)
        puv = (ug - umean[:, :, jnp.newaxis]) * px[:, :, jnp.newaxis] + \
              (vg - vmean[:, :, jnp.newaxis]) * py[:, :, jnp.newaxis]  # [ix, il, kx]
        
        # Vertical velocity at interfaces using scan
        def scan_sigma(carry, k):
            sigdt_k = carry
            sigdt_kp1 = sigdt_k - dhs[k] * (puv[:, :, k] + divg[:, :, k] - dmean)
            return sigdt_kp1, sigdt_kp1
        init_sigdt = jnp.zeros((ix, il))
        _, sigdt_stack = jax.lax.scan(scan_sigma, init_sigdt, jnp.arange(kx))
        sigdt = jnp.concatenate([init_sigdt[..., jnp.newaxis], jnp.transpose(sigdt_stack, (1, 2, 0))], axis=2)  # [ix, il, kx+1]
        
        # For temperature equation
        def scan_sigm(carry, k):
            sigm_k = carry
            sigm_kp1 = sigm_k - dhs[k] * puv[:, :, k]
            return sigm_kp1, sigm_kp1
        init_sigm = jnp.zeros((ix, il))
        _, sigm_stack = jax.lax.scan(scan_sigm, init_sigm, jnp.arange(kx))
        sigm = jnp.concatenate([init_sigm[..., jnp.newaxis], jnp.transpose(sigm_stack, (1, 2, 0))], axis=2)  # [ix, il, kx+1]
        
        return sigdt, sigm, puv
    
    # ========================================================================
    # Helper Method 4: Grid-Point Tendencies
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_grid_tendencies(
        self,
        ug: jax.Array,
        vg: jax.Array,
        tg: jax.Array,
        qg: jax.Array,
        tgg: jax.Array,
        vorg: jax.Array,
        divg: jax.Array,
        px: jax.Array,
        py: jax.Array,
        sigdt: jax.Array,
        sigm: jax.Array,
        puv: jax.Array,
        dmean: jax.Array  
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Compute wind and temperature tendencies in grid space.
        
        This is the core primitive equation dynamics!
        
        Returns:
            utend: Zonal wind tendency [ix, il, kx]
            vtend: Meridional wind tendency [ix, il, kx]
            ttend: Temperature tendency [ix, il, kx]
            qtend: Humidity tendency [ix, il, kx]
        """
        ix, il, kx = ug.shape
        rgas = self.constants.rgas
        akap = self.constants.akap
        dhsr = self.dhsr  # [kx]
        fsgr = self.fsgr
        tref = self.tref
        tref3 = self.tref3
        
        # ====================================================================
        # Wind Tendencies
        # ====================================================================
        
        # Vertical advection of u 
        temp_u = jnp.zeros((ix, il, kx + 1))
        temp_u = temp_u.at[:, :, 1:kx].set(
            sigdt[:, :, 1:kx] * (ug[:, :, 1:] - ug[:, :, :-1])
        )
        
        # Zonal wind tendency (vectorized over k)
        utend = vg * vorg - tgg * rgas * px[:, :, jnp.newaxis]
        utend = utend - (temp_u[:, :, 1:] + temp_u[:, :, :-1]) * dhsr[jnp.newaxis, jnp.newaxis, :]
        
        # Vertical advection of v - VECTORIZED (no scan needed)
        temp_v = jnp.zeros((ix, il, kx + 1))
        temp_v = temp_v.at[:, :, 1:kx].set(
            sigdt[:, :, 1:kx] * (vg[:, :, 1:] - vg[:, :, :-1])
        )
        
        # Meridional wind tendency (vectorized over k)
        vtend = -ug * vorg - tgg * rgas * py[:, :, jnp.newaxis]
        vtend = vtend - (temp_v[:, :, 1:] + temp_v[:, :, :-1]) * dhsr[jnp.newaxis, jnp.newaxis, :]
        
        # ====================================================================
        # Temperature Tendency
        # ====================================================================
        
        # Vertical advection of temperature - VECTORIZED (no scan needed)
        temp_t = jnp.zeros((ix, il, kx + 1))
        temp_t = temp_t.at[:, :, 1:kx].set(
            sigdt[:, :, 1:kx] * (tgg[:, :, 1:] - tgg[:, :, :-1]) +
            sigm[:, :, 1:kx] * (tref[1:kx] - tref[:kx-1])[jnp.newaxis, jnp.newaxis, :]
        )
        
        # Temperature tendency (vectorized)
        ttend = tgg * divg
        ttend = ttend - (temp_t[:, :, 1:] + temp_t[:, :, :-1]) * dhsr[jnp.newaxis, jnp.newaxis, :]
        ttend = ttend + fsgr[jnp.newaxis, jnp.newaxis, :] * tgg * (sigdt[:, :, 1:] + sigdt[:, :, :-1])
        ttend = ttend + tref3[jnp.newaxis, jnp.newaxis, :] * (sigm[:, :, 1:] + sigm[:, :, :-1])
        ttend = ttend + akap * (tg * puv - tgg * dmean[:, :, jnp.newaxis])
        
        # ====================================================================
        # Tracer Tendencies
        # ====================================================================

        # Vertical advection of qv
        temp_q = jnp.zeros((ix, il, kx + 1))
        temp_q = temp_q.at[:, :, 1:kx].set(
            sigdt[:, :, 1:kx] * (qg[:, :, 1:] - qg[:, :, :-1])
        )
        # Zero out levels 2 and 3 (indices 1 and 2 in 0-based) as in Fortran
        temp_q = temp_q.at[:, :, 1:3].set(0.0)
        # Tracer tendency: horizontal advection + vertical advection
        qtend = qg * divg - (temp_q[:, :, 1:] + temp_q[:, :, :-1]) * dhsr[jnp.newaxis, jnp.newaxis, :]
        
        return utend, vtend, ttend, qtend
    
    # ========================================================================
    # Helper Method 5: Transform Tendencies to Spectral
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _transform_tendencies_to_spectral(
        self,
        utend: jax.Array,
        vtend: jax.Array,
        ttend: jax.Array,
        qtend: jax.Array,
        ug: jax.Array,
        vg: jax.Array,
        tgg: jax.Array,
        qg: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Transform grid-point tendencies to spectral space.
        
        Args:
            utend, vtend: Wind tendencies [ix, il, kx]
            ttend: Temperature tendency [ix, il, kx]
            qtend: Humidity tendencies [ix, il, kx]
            ug, vg: Winds (for kinetic energy) [ix, il, kx]
            tgg: Temperature deviation (for advection) [ix, il, kx]
            qg: Humidity (for advection) [ix, il, kx]
            
        Returns:
            vordt: Vorticity tendency [mx, nx, kx]
            divdt: Divergence tendency [mx, nx, kx]
            tdt: Temperature tendency [mx, nx, kx]
            qdt: Humidity tendencies [mx, nx, kx]
        """
        # Transform wind tendencies to spectral (vectorized)
        vordt, divdt = self.transformer.grid_3d_to_vor_div(utend, vtend, kcos=True)

        # Add kinetic energy term to divergence
        ke_3d = 0.5 * (ug**2 + vg**2)  # [ix, il, kx]
        ke_spec_3d = self.transformer.grid_3d_to_spec(ke_3d)  # [mx, nx, kx]
        divdt = divdt - self.transformer.laplacian_3d(ke_spec_3d)
        
        # Temperature tendency: transform and add advection
        ttend_spec = self.transformer.grid_3d_to_spec(ttend)
        
        # Add temperature advection: -div(u*T', v*T')
        uT = -ug * tgg  # [ix, il, kx]
        vT = -vg * tgg  # [ix, il, kx]
        _, temp_advection = self.transformer.grid_3d_to_vor_div(uT, vT, kcos=True)
        tdt = ttend_spec + temp_advection
        
        # Qv tendencies: transform and add advection
        # Transform grid tendency to spectral
        qtend_spec = self.transformer.grid_3d_to_spec(qtend)
        
        # Add advection: -div(u*tr, v*tr)
        uQ = -ug * qg  # [ix, il, kx]
        vQ = -vg * qg  # [ix, il, kx]
        _, q_advection = self.transformer.grid_3d_to_vor_div(uQ, vQ, kcos=True)
        qdt = qtend_spec + q_advection

        return vordt, divdt, tdt, qdt
    
    # ========================================================================
    # Helper Method 6: Add Spectral Corrections
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _add_spectral_corrections(
        self,
        divdt: jax.Array,
        tdt: jax.Array,
        psdt: jax.Array,
        div_spec: jax.Array,
        t_spec: jax.Array,
        ps_spec: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Add remaining terms computed in spectral space.
        
        Includes:
        - Geopotential calculation
        - Vertical velocity in spectral space
        - Reference atmosphere contributions
        - Pressure gradient force on divergence
        
        Returns:
            Updated (divdt, tdt, psdt)
        """
        mx, nx, kx = self.config.mx, self.config.nx, self.config.kx
        rgas = self.constants.rgas
        dhs = self.dhs
        dhsr = self.dhsr
        tref = self.tref
        tref2 = self.tref2
        tref3 = self.tref3
        
        # Vertical mean divergence (spectral)
        dmeanc = jnp.einsum('mnk,k->mn', div_spec, dhs)  # [mx, nx]
        
        # Update surface pressure tendency
        psdt = psdt - dmeanc
        psdt = psdt.at[0, 0].set(0.0 + 0.0j)
        
        # Vertical velocity in spectral space using scan
        def scan_sigdtc(carry, k):
            sigdtc_k = carry
            sigdtc_kp1 = sigdtc_k - dhs[k] * (div_spec[:, :, k] - dmeanc)
            return sigdtc_kp1, sigdtc_kp1
        init_sigdtc = jnp.zeros((mx, nx), dtype=jnp.complex64)
        _, sigdtc_stack = jax.lax.scan(scan_sigdtc, init_sigdtc, jnp.arange(kx))
        sigdtc = jnp.concatenate([init_sigdtc[..., jnp.newaxis], jnp.transpose(sigdtc_stack, (1, 2, 0))], axis=2)  # [mx, nx, kx+1]
        sigdtc = sigdtc.at[:, :, kx].set(0.0 + 0.0j)

        # Add reference atmosphere terms to temperature tendency
        temp_spec = jnp.zeros((mx, nx, kx + 1), dtype=jnp.complex64)
        temp_spec = temp_spec.at[:, :, 1:kx].set(
            sigdtc[:, :, 1:kx] * (tref[1:kx] - tref[:kx-1])[jnp.newaxis, jnp.newaxis, :]
        )
        
        # Add to temperature tendency (vectorized)
        tdt_correction = (
            -(temp_spec[:, :, 1:] + temp_spec[:, :, :-1]) * dhsr[jnp.newaxis, jnp.newaxis, :] +
            tref3[jnp.newaxis, jnp.newaxis, :] * (sigdtc[:, :, 1:] + sigdtc[:, :, :-1]) -
            tref2[jnp.newaxis, jnp.newaxis, :] * dmeanc[:, :, jnp.newaxis]
        )
        tdt = tdt + tdt_correction
        
        # Compute geopotential
        phi_spec = self._geopotential(t_spec)
        
        # Add pressure gradient to divergence tendency (vectorized over k)
        phi_plus_ps = phi_spec + rgas * tref[jnp.newaxis, jnp.newaxis, :] * ps_spec[:, :, jnp.newaxis]  # [mx, nx, kx]
        divdt = divdt - self.transformer.laplacian_3d(phi_plus_ps)
        
        return divdt, tdt, psdt

    # ========================================================================
    # Geopotential Calculation
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def _geopotential(self, t_spec):
        """
        Compute spectral geopotential from spectral temperature and surface geopotential.
        Based on SPEEDY geopotential.f90
        
        Integrates hydrostatic equation: dÎ¦/d(ln p) = -R*T
        
        Args:
            t_spec: Temperature [mx, nx, kx] (complex)
            
        Returns:
            phi_spec: Geopotential [mx, nx, kx] (complex)
        """
        kx = self.config.kx
        xgeop1 = self.xgeop1
        xgeop2 = self.xgeop2
        hsg = self.hsg
        fsg = self.fsg
        phis_spec = self.static_fields.get_field('phis_spec')
        
        # 1. Bottom layer (integration over half a layer)
        phi_bottom = phis_spec + xgeop1[kx-1] * t_spec[:, :, kx-1]
        
        # 2. Other layers using scan (bottom-up integration)
        def scan_geopotential(phi_below, k):
            """Compute geopotential at level k given phi at k+1"""
            phi_k = phi_below + xgeop2[k+1] * t_spec[:, :, k+1] + xgeop1[k] * t_spec[:, :, k]
            return phi_k, phi_k
        
        # Scan from kx-2 down to 0
        k_indices = jnp.arange(kx-2, -1, -1)
        _, phi_stack = jax.lax.scan(scan_geopotential, phi_bottom, k_indices)
        
        # phi_stack has shape (kx-1, mx, nx) - transpose to (mx, nx, kx-1)
        phi_stack = jnp.transpose(phi_stack, (1, 2, 0))  # [mx, nx, kx-1]
        
        # phi_stack is in order [phi[kx-2], phi[kx-3], ..., phi[1], phi[0]]
        # Reverse to get [phi[0], phi[1], ..., phi[kx-2]]
        phi_upper = phi_stack[:, :, ::-1]  # [mx, nx, kx-1]
        
        # Concatenate: [phi[0], ..., phi[kx-2]] + [phi[kx-1]]
        phi = jnp.concatenate([phi_upper, phi_bottom[..., jnp.newaxis]], axis=2)  # [mx, nx, kx]
        
        # 3. Lapse-rate correction in the free troposphere (VECTORIZED)
        # (only for m=0 modes to reduce computational cost)
        k_array = jnp.arange(1, kx-1)  # [1, 2, ..., kx-2]
        
        # Compute correction factors for all levels at once
        corf = xgeop1[k_array] * 0.5 * jnp.log(hsg[k_array+1]/fsg[k_array]) / \
            jnp.log(fsg[k_array+1]/fsg[k_array-1])
        
        # Compute temperature differences for all levels
        # Note: advanced indexing gives shape (len(k_array), nx)
        t_diff = t_spec[0, :, k_array+1] - t_spec[0, :, k_array-1]  # [len(k_array), nx]
        
        # Apply correction (vectorized over levels)
        # corf: (6,), t_diff: (6, 32) -> correction: (6, 32)
        correction = corf[:, jnp.newaxis] * t_diff  # [len(k_array), nx]
        
        # Add to phi (vectorized update)
        phi = phi.at[0, :, k_array].add(correction)
        
        return phi
    