#!/usr/bin/env python3
"""
Time stepping schemes for SPEEDY model.

Based on SPEEDY time_stepping.f90 module.
Implements Leapfrog scheme with Robert-Asselin-Williams filter.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional

from .util import TimeInfo
from .state import State, SpectralState, ForcingState, CachedState, DiagState, Param, Config
from .dynamics import Dynamics
from .implicit import ImplicitSolver
from .forcing import Forcing
from .physics import Physics

class TimeStepper:
    """
    Time stepping for SPEEDY using Leapfrog + Robert-Asselin-Williams filter.
    
    Based on SPEEDY time_stepping.f90.
    
    Implements:
    - Forward Euler for initialization
    - Leapfrog for main integration
    - Robert-Asselin-Williams filter for time stability
    """
    
    def __init__(self, config: Config, dynamics: Dynamics):
        """
        Initialize time stepper.
        
        Args:
            dynamics: Dynamics instance
        """
        self.config = config
        self.dynamics = dynamics

        # Create 3 implicit solvers
        constants = self.dynamics.constants
        vertical_grid = self.dynamics.vertical_grid
        dmp = self.dynamics.dmp
        dmpd = self.dynamics.dmpd
        dmps = self.dynamics.dmps
        self.implicit_solver_half = ImplicitSolver(config, constants, vertical_grid, 0.5*config.dt, dmp, dmpd, dmps)
        self.implicit_solver_full = ImplicitSolver(config, constants, vertical_grid, config.dt, dmp, dmpd, dmps)
        self.implicit_solver_double = ImplicitSolver(config, constants, vertical_grid, 2.0*config.dt, dmp, dmpd, dmps)

        # Initialize physics
        transformer = self.dynamics.transformer
        static_fields = self.dynamics.static_fields
        self.physics = Physics(
            config=config,
            constants=constants,
            vertical_grid=vertical_grid,
            transformer=transformer,
            static_fields=static_fields
        )
        # Initialize forcing
        self.forcing = Forcing(
            config=config,
            constants=constants,
            transformer=transformer,
            static_fields=static_fields
        )

    def _create_initial_cached_state(self) -> CachedState:
        """
        Create initial (dummy) CachedState for first timestep.
        
        These will be overwritten on the first radiation calculation.
        """
        ix, il, kx = self.config.ix, self.config.il, self.config.kx
        
        return CachedState(
            tau2=jnp.zeros((ix, il, kx, 4)),      # LW transmissivity
            stratc=jnp.zeros((ix, il, 2)),        # Stratospheric correction
            ssrd=jnp.zeros((ix, il)),             # Surface downward SW
            dfabs_sw=jnp.zeros((ix, il, kx)),     # SW absorbed flux
            tsr=jnp.zeros((ix, il)),              # TOA net SW
            ssr=jnp.zeros((ix, il)),              # Surface net SW
        )
    
    def _create_initial_diag_state(self) -> DiagState:
        """
        Create initial (dummy) DiagState for first timestep.
        
        These will be computed on the first physics call.
        """
        ix, il = self.config.ix, self.config.il
        
        return DiagState(
            hfluxn=jnp.zeros((ix, il, 2)),    # Net heat flux (land, sea)
            shf=jnp.zeros((ix, il, 3)),       # Sensible heat flux
            evap=jnp.zeros((ix, il, 3)),      # Evaporation
            ustr=jnp.zeros((ix, il, 3)),      # U-stress
            vstr=jnp.zeros((ix, il, 3)),      # V-stress
            slru=jnp.zeros((ix, il, 3)),      # Surface upward LW
            ssrd=jnp.zeros((ix, il)),         # Surface downward SW
            slrd=jnp.zeros((ix, il)),         # Surface downward LW
            slr=jnp.zeros((ix, il)),          # Surface net LW
            olr=jnp.zeros((ix, il)),          # Outgoing LW
            precnv=jnp.zeros((ix, il)),       # Convective precipitation
            precls=jnp.zeros((ix, il)),       # Large-scale precipitation
            ts=jnp.zeros((ix, il)),           # Surface temperature
            tskin=jnp.zeros((ix, il)),        # Skin temperature
            u0=jnp.zeros((ix, il)),           # Near-surface u-wind
            v0=jnp.zeros((ix, il)),           # Near-surface v-wind
            t0=jnp.zeros((ix, il)),           # Near-surface temperature
        )

    def initialize_from_rest(self, param: Param, time: TimeInfo) -> 'State':
        """
        Create initial state from reference atmosphere at rest.
        
        Based on SPEEDY prognostics.f90:initialize_from_rest_state()
        
        Initializes:
        1. Vorticity and divergence to zero
        2. Temperature with:
        - Stratosphere (k=0,1): T = 216K
        - Troposphere (k>=2): T decreases with height following lapse rate
        3. Surface pressure consistent with hydrostatic balance
        4. Specific humidity in troposphere
        
        Returns:
            State with both time levels initialized identically
        """
        mx, nx, kx = self.config.mx, self.config.nx, self.config.kx
        
        # Get necessary constants
        grav = self.dynamics.constants.grav
        rgas = self.dynamics.constants.rgas
        gamma = self.dynamics.constants.gamma
        hscale = self.dynamics.constants.hscale
        hshum = self.dynamics.constants.hshum
        refrh1 = self.dynamics.constants.refrh1
        
        # Get vertical coordinate and surface geopotential
        fsg = jnp.array(self.dynamics.vertical_grid.fsg)  # Full sigma levels [kx]
        phis0 = self.dynamics.static_fields.get_field('phis0')  # Surface geopotential grid [ix, il]
        phis = self.dynamics.static_fields.get_field('phis_spec')    # Surface geopotential spectral [mx, nx]
        
        # ========================================================================
        # 1. Initialize vorticity, divergence, humidity to zero
        # ========================================================================
        vor = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
        div = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
        q = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
        
        # ========================================================================
        # 2. Initialize temperature (VECTORIZED)
        # ========================================================================
        
        # Constants for temperature initialization
        tref = 288.0   # Surface reference temperature (K)
        ttop = 216.0   # Stratospheric temperature (K)
        gam1 = gamma / (1000.0 * grav)
        rgam = rgas * gam1
        rgamr = 1.0 / rgam
        
        # Surface temperature (spectral): surfs = tref - gam1*phis
        surfs = - gam1 * phis  # [mx, nx]
        surfs = surfs.at[0, 0].set(surfs[0,0] + jnp.sqrt(2.0) * tref)
        
        # VECTORIZED: Compute all tropospheric levels (k=2:kx) at once
        t_tropo = surfs[:, :, jnp.newaxis] * (fsg[jnp.newaxis, jnp.newaxis, 2:] ** rgam)
        
        # Initialize temperature array
        t = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
        
        # Set stratosphere (k=0,1): only m=0, n=0 non-zero
        t = t.at[0, 0, 0].set(jnp.sqrt(2.0) * ttop + 0.0j)
        t = t.at[0, 0, 1].set(jnp.sqrt(2.0) * ttop + 0.0j)
        
        # Set troposphere (k=2:kx)
        t = t.at[:, :, 2:].set(t_tropo)
        
        # ========================================================================
        # 3. Initialize log surface pressure
        # ========================================================================
        
        # p_ref = 1013 hPa at z = 0
        rlog0 = jnp.log(1.013)  # Reference pressure
        gam2 = gam1 / tref
        
        # Compute in grid space: ps = log(1.013) + (1/rgam)*log(1 - gam2*phis0)
        surfg = rlog0 + rgamr * jnp.log(1.0 - gam2 * phis0)  # [ix, il]
        
        # Transform to spectral space
        ps = self.dynamics.transformer.grid_to_spec(surfg)  # [mx, nx]
        
        # Apply truncation if needed
        if self.config.ix == self.config.iy * 4:
            ps = self.dynamics.transformer.apply_truncation(ps)
        
        # ========================================================================
        # 4. Initialize specific humidity (VECTORIZED)
        # ========================================================================
        
        # Reference humidity: Qref = RHref * Qsat(288K, 1013hPa)
        esref = 17.0
        qref = refrh1 * 0.622 * esref  # Specific humidity at surface (g/kg)
        qexp = hscale / hshum
        
        # Surface specific humidity (grid space): q = qref * exp(qexp * log_ps)
        surfg_q = qref * jnp.exp(qexp * surfg)  # [ix, il]
        
        # Transform to spectral space
        surfs_q = self.dynamics.transformer.grid_to_spec(surfg_q)  # [mx, nx]
        
        # Apply truncation if needed
        if self.config.ix == self.config.iy * 4:
            surfs_q = self.dynamics.transformer.apply_truncation(surfs_q)
        
        # VECTORIZED: Compute all tropospheric levels (k=2:kx) at once
        q_tropo = surfs_q[:, :, jnp.newaxis] * (fsg[jnp.newaxis, jnp.newaxis, 2:] ** qexp)
        
        # Initialize humidity array
        q = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
        q = q.at[:, :, 2:].set(q_tropo)
        
        # ========================================================================
        # 5. Create spectral state
        # ========================================================================
        spec_state = SpectralState(vor=vor, div=div, t=t, q=q, ps=ps)
        
        # ========================================================================
        # 6. Initialize time
        # ========================================================================
        #default_time = TimeInfo.create(year=1982, month=1, day=1, hour=0, minute=0)
        
        # ========================================================================
        # 7. Initialize forcing state
        # ========================================================================
        forcing_state = self.forcing.initialize(param, time)
        
        # ========================================================================
        # 8. Create initial dummy cached and diagnostic states
        # ========================================================================
        cached_state = self._create_initial_cached_state()
        diag_state = self._create_initial_diag_state()
        
        # Both filt and curr start with the same state
        return State(
            filt=spec_state,
            curr=spec_state,
            forcing=forcing_state,
            cached=cached_state,
            diag=diag_state,
            time=time
        )

    @partial(jax.jit, static_argnames=['self'])
    def forward(self, state: State, param: Param) -> 'State':
        """
        Perform one Leapfrog time step with Robert-Asselin-Williams filter.
        
        Based on SPEEDY time_stepping.f90 lines 26-167.
        
        Implements:
            F_new = F(1) + dt * [T_dyn(F(2)) + T_phy(F(1))]
            F(1) = F(j1) + wil*eps*(F(1) - 2*F(j1) + F_new)
            F(2) = F_new - (1-wil)*eps*(F(1) - 2*F(j1) + F_new)
        
        For regular leapfrog: j1=2 (use curr), j2=2 (use curr), eps=ROB
        
        Args:
            state: Current State(filt, curr)
            param: Model parameters
            
        Returns:
            Updated State(filt_new, curr_new)
        """
        
        # Regular leapfrog parameters (after initialization)
        eps = self.config.rob
        wil = self.config.wil
        dt = self.implicit_solver_double.dt
        
        # ====================================================================
        # 1. Compute tendencies
        # ====================================================================
        
        new_forcing = self.forcing.forward(param, state.forcing, state.diag, state.time)

        # Compute tendencies with physics
        vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis = self.dynamics.compute_tendencies(
            implicit_solver=self.implicit_solver_double,
            physics=self.physics,
            param=param,
            state_dyn=state.curr,
            state_phy=state.filt,
            forcing=new_forcing,
            cached=state.cached,
            time=state.time
        )
        
        # ====================================================================
        # 2. Time integration with Robert-Asselin-Williams filter
        # ====================================================================
        
        # Leapfrog formula: F_new = F(1) + dt*dF/dt
        # Always use state.filt (F(1) in Fortran) as base
        vor_new = state.filt.vor + dt * vordt
        div_new = state.filt.div + dt * divdt
        t_new = state.filt.t + dt * tdt
        q_new = state.filt.q + dt * qdt
        ps_new = state.filt.ps + dt * psdt
        
        # Robert-Asselin-Williams filter
        # F(1) = F(j1) + wil*eps*(F(1) - 2*F(j1) + F_new)
        vor_filt = state.curr.vor + wil*eps*(state.filt.vor - 2*state.curr.vor + vor_new)
        div_filt = state.curr.div + wil*eps*(state.filt.div - 2*state.curr.div + div_new)
        t_filt = state.curr.t + wil*eps*(state.filt.t - 2*state.curr.t + t_new)
        q_filt = state.curr.q + wil*eps*(state.filt.q - 2*state.curr.q + q_new)
        ps_filt = state.curr.ps + wil*eps*(state.filt.ps - 2*state.curr.ps + ps_new)
        
        # F(2) = F_new - (1-wil)*eps*(F(1) - 2*F(j1) + F_new)
        vor_curr = vor_new - (1-wil)*eps*(vor_filt - 2*state.curr.vor + vor_new)
        div_curr = div_new - (1-wil)*eps*(div_filt - 2*state.curr.div + div_new)
        t_curr = t_new - (1-wil)*eps*(t_filt - 2*state.curr.t + t_new)
        q_curr = q_new - (1-wil)*eps*(q_filt - 2*state.curr.q + q_new)
        ps_curr = ps_new - (1-wil)*eps*(ps_filt - 2*state.curr.ps + ps_new)
        
        # Return new state
        new_filt = SpectralState(vor=vor_filt, div=div_filt, t=t_filt, q=q_filt, ps=ps_filt)
        new_curr = SpectralState(vor=vor_curr, div=div_curr, t=t_curr, q=q_curr, ps=ps_curr)
        # Advance time
        new_time = state.time.forward(dt=0.5*dt)
        
        return State(
            filt=new_filt,
            curr=new_curr,
            forcing=new_forcing,
            cached=new_cached,
            diag=diagnosis,
            time=new_time
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def _forward_euler(self, state: State, param: Param) -> 'State':
        """
        Forward Euler step (for initialization).
        
        Based on time_stepping.f90 with j1=1, j2=1, eps=0.
        
        Args:
            state: Current state
            param: Model parameters  
            dt: Time step (may be different from self.config.dt for half-step)
            
        Returns:
            Updated state
        """
        dt = self.implicit_solver_half.dt

        # Forward Euler: j1=1, j2=1, eps=0
        # Both dynamics and physics use state.filt
        vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis = self.dynamics.compute_tendencies(
            implicit_solver=self.implicit_solver_half,
            physics=self.physics,
            param=param,
            state_dyn=state.filt,
            state_phy=state.filt,
            forcing=state.forcing,
            cached=state.cached,
            time=state.time
        )
        
        # Forward Euler: F_new = F + dt*dF/dt (no filter, eps=0)
        vor_new = state.filt.vor + dt * vordt
        div_new = state.filt.div + dt * divdt
        t_new = state.filt.t + dt * tdt
        q_new = state.filt.q + dt * qdt
        ps_new = state.filt.ps + dt * psdt
        
        # For forward Euler: both filt and curr get the same new value
        # (Actually curr gets new, filt stays old per Fortran line 163-166 with eps=0)
        new_state = SpectralState(vor=vor_new, div=div_new, t=t_new, q=q_new, ps=ps_new)
        
        # Update: filt keeps old, curr gets new (matches Fortran with eps=0)
        return State(
            filt=state.filt,
            curr=new_state,
            forcing=state.forcing,
            cached=new_cached,
            diag=diagnosis,
            time=state.time
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def _initial_leapfrog(self, state: State, param: Param) -> 'State':
        """
        Initial leapfrog step (for initialization).
        
        Based on time_stepping.f90 with j1=1, j2=2, eps=0.
        
        Args:
            state: Current state
            param: Model parameters
            
        Returns:
            Updated state
        """
        dt = self.implicit_solver_full.dt
        
        # Initial leapfrog: j1=1, j2=2, eps=0
        # Dynamics uses state.curr, physics uses state.filt
        vordt, divdt, tdt, qdt, psdt, new_cached, diagnosis = self.dynamics.compute_tendencies(
            implicit_solver=self.implicit_solver_full,
            physics=self.physics,
            param=param,
            state_dyn=state.curr,
            state_phy=state.filt,
            forcing=state.forcing,
            cached=state.cached,
            time=state.time
        )
        
        # Leapfrog: F_new = F(1) + dt*dF/dt (eps=0, no filter)
        vor_new = state.filt.vor + dt * vordt
        div_new = state.filt.div + dt * divdt
        t_new = state.filt.t + dt * tdt
        q_new = state.filt.q + dt * qdt
        ps_new = state.filt.ps + dt * psdt
        
        new_curr = SpectralState(vor=vor_new, div=div_new, t=t_new, q=q_new, ps=ps_new)

        # filt stays as curr (state.filt = old state.curr), curr gets new
        return State(
            filt=state.filt,
            curr=new_curr,
            forcing=state.forcing,
            cached=new_cached,
            diag=diagnosis,
            time=state.time
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def first_step(self, state0: State, param: Param) -> 'State':
        """
        Perform initialization steps to start leapfrog integration.
        
        Based on SPEEDY time_stepping.f90 first_step() lines 12-24:
        1. Forward Euler with dt/2
        2. Initial leapfrog with dt
        
        Args:
            state0: Initial state
            param: Model parameters
            
        Returns:
            State ready for regular leapfrog stepping
        """
        
        # Step 1: Forward Euler with dt/2
        state1 = self._forward_euler(state0, param)
        #jax.debug.print("After first step 1: t[0,2,:]: {x}", x=state1.curr.t[0,2,:])
        
        # Step 2: Initial leapfrog with dt
        state2 = self._initial_leapfrog(state1, param)
        #jax.debug.print("After first step 2: t[0,2,:]: {x}", x=state2.curr.t[0,2,:])
        
        return state2
    
    @partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
    def integrate(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple['State', Optional[SpectralState]]:
        """
        Integrate the model forward in time.
        
        This function automatically performs the initialization steps (first_step)
        before running the main leapfrog integration.
        
        Args:
            state0: Initial state (raw, not yet initialized for leapfrog)
            param: Model parameters
            nstep: Number of total time steps (including first_step, must be >= 1)
            save_freq: If provided, save states every save_freq steps.
                    Must divide nstep evenly.
            
        Returns:
            final_state: State after nstep steps
            trajectory: If save_freq provided, SpectralState trajectory from state0.curr
                    to final_state.curr (length = nstep/save_freq + 1)
        """
        # Step 1: Perform initialization (first_step counts as step 1)
        state_init = self.first_step(state0, param)
        
        # Define the forward step function
        def step_fn(state, _):
            state_new = self.forward(state, param)
            return state_new, state_new
        
        if save_freq is None:
            # Only return final state, no trajectory
            if nstep == 1: return state_init, None
            final_state, _ = jax.lax.scan(step_fn, state_init, jnp.arange(nstep - 1))
            return final_state, None
        
        # With trajectory saving
        n_saves = nstep // save_freq  # Number of save points after state0
        
        # First chunk: first_step + (save_freq - 1) forwards = save_freq steps total
        first_chunk_forwards = save_freq - 1
        if first_chunk_forwards > 0:
            state_first_save, _ = jax.lax.scan(step_fn, state_init, jnp.arange(first_chunk_forwards))
        else:
            # save_freq == 1, so first save is right after first_step
            state_first_save = state_init
        
        # Remaining chunks: each does save_freq forwards
        n_remaining = n_saves - 1
        
        if n_remaining == 0:
            # Only one save point after state0
            final_state = state_first_save
            trajectory = jax.tree_util.tree_map(lambda x0, x1: jnp.stack([x0, x1], axis=0), state0.curr, state_first_save.curr)
        else:
            # Multiple save points
            def inner_loop(state, _):
                next_state, _ = jax.lax.scan(step_fn, state, jnp.arange(save_freq))
                return next_state, next_state.curr
            
            final_state, later_saves = jax.lax.scan(inner_loop, state_first_save, jnp.arange(n_remaining))
            
            # Build trajectory: [state0.curr, state_first_save.curr, later_saves...]
            trajectory = jax.tree_util.tree_map(
                lambda x0, x1, xrest: jnp.concatenate([x0[None], x1[None], xrest], axis=0),
                state0.curr, state_first_save.curr, later_saves
            )
        
        return final_state, trajectory
    
    @partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
    def integrate_old(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple['State', Optional['State']]:
        """
        Integrate the model forward in time.
        
        Follows the Protocol interface from dajax.models.base.
        
        Args:
            state0: Initial state (should be output of first_step)
            param: Model parameters
            nstep: Number of time steps to integrate
            save_freq: If provided, save states every save_freq steps
            
        Returns:
            final_state: State after nstep steps
            trajectory: If save_freq provided, State array of saved states
        """
        if nstep <= 0: return state0, None
        
        # Note: first_step cannot be jitted due to object reassignment
        # So we assume state0 is already initialized via first_step
        
        def step_fn(state, _):
            state_new = self.forward(state, param)
            return state_new, state_new
        
        if save_freq is None:
            # Only return final state
            final_state, _ = jax.lax.scan(step_fn, state0, jnp.arange(nstep))
            return final_state, None
        else:
            # Inner loop: integrate for save_freq steps without saving
            def inner_loop(state, _):
                final_state, _ = jax.lax.scan(step_fn, state, jnp.arange(save_freq))
                return final_state, final_state.curr  # Save only curr (State2, State3, ...)
            
            # Outer loop: save states every save_freq steps
            n_saves = nstep // save_freq
            final_state, trajectory = jax.lax.scan(inner_loop, state0, jnp.arange(n_saves))
            
            # Add initial state to trajectory
            trajectory_with_init = jax.tree_util.tree_map(
                lambda x0, x: jnp.concatenate([x0[None, ...], x], axis=0),
                state0.curr, trajectory
            )
            
            return final_state, trajectory_with_init
