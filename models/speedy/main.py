#!/usr/bin/env python3
"""
SPEEDY (Simplified Parameterizations, primitivE-Equation DYnamics) model in JAX
Converted from Fortran to JAX following the pattern of the shqg model.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Tuple, Dict, Optional

# Import core modules
from .util import TimeInfo
from .state import State, Param, Config, SpectralState, PhyState
from .constants import Constants
from .vertical import VerticalGrid
from .legendre import LegendreTransform
from .transformer import Transformer
from .static import StaticFields
from .dynamics import Dynamics
from .integration import TimeStepper
from .diagnostics import Diagnostics

class SPEEDY:
    """
    Main SPEEDY atmospheric model class.
    
    Coordinates the modular components:
    - Transformer: Spectral transforms and operators
    - ImplicitSolver: Semi-implicit scheme for gravity waves
    - Dynamics: Geopotential and tendency computations
    
    Handles:
    - Initialization
    - Time stepping (leapfrog + Robert-Williams filter)
    - Integration
    """
    
    def __init__(self, config: Config, data_dir: Optional[str] = None):
        """
        Initialize SPEEDY model.
        
        Args:
            config: Model configuration
            data_dir: Path to data directory (for StaticFields)
        """
        self.config = config
        self.constants = Constants()
        
        # Setup vertical grid
        self.vertical_grid = VerticalGrid(config.kx, self.constants)
        
        # Initialize spectral transform
        self.legendre = LegendreTransform(config)
        self.transformer = Transformer(config, self.constants, self.legendre)
        
        # Initialize static/boundary fields
        self.static_fields = StaticFields(config, self.constants, self.transformer, data_dir)

        # Initialize modular components
        self.dynamics = Dynamics(config, self.constants, self.vertical_grid, self.transformer, self.static_fields)
        
        # Initialize time stepper
        self.time_stepper = TimeStepper(config=config, dynamics=self.dynamics)
        
        print("SPEEDY model initialized successfully!")
        print(f"  Resolution: T{config.trunc}, {config.ix}x{config.il} grid, {config.kx} levels")
        print(f"  Time step: {config.dt}s ({config.dt/60:.1f} min)")
        print(f"  Semi-implicit: α = {config.alph}")
    
    @staticmethod
    def default_param():
        """Get default parameters."""
        return Param()
    
    def random_param(self, key: jax.Array, base, noise_scale: float):
        """Random parameters (currently same as default)."""
        return Param()
    
    # ========================================================================
    # Initial Conditions
    # ========================================================================
    
    def default_state(self, param: Param, time: TimeInfo) -> State:
        """
        Create default initial state from reference atmosphere at rest.
        
        Delegates to TimeStepper.initialize_from_rest()
        
        Returns:
            State with both time levels initialized identically
        """
        return self.time_stepper.initialize_from_rest(param, time)
    
    def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float = 0.01) -> State:
        """Create random state perturbation around base state.
        Perturbs the prognostic spectral variables (vor, div, t, q, ps).
        Perturbations are applied in physical space for physical consistency,
        then converted back to spectral space.
        Args:
            key: JAX random key
            param: Model parameters
            base: Base state to perturb
            noise_scale: Relative perturbation scale
        Returns:
            Perturbed State
        """
        keys = jax.random.split(key, 5)

        # Convert base state to physical space
        phy_base = self.mod2phy(base.curr, param)
        
        # Compute perturbation scales based on typical field magnitudes
        # or use standard deviations of the fields themselves
        u_scale = noise_scale * jnp.maximum(jnp.std(phy_base.u), 1.0)  # m/s
        v_scale = noise_scale * jnp.maximum(jnp.std(phy_base.v), 1.0)  # m/s
        t_scale = noise_scale * jnp.maximum(jnp.std(phy_base.t), 1.0)  # K
        q_scale = noise_scale * jnp.maximum(jnp.std(phy_base.q), 0.1)  # g/kg
        ps_scale = noise_scale * jnp.maximum(jnp.std(phy_base.ps), 0.001) # log(p)
        
        # Generate smooth perturbations (spectrally truncated noise)
        def smooth_noise(k, shape, scale):
            """Generate spatially correlated noise."""
            noise = scale * jax.random.normal(k, shape)
            # Optional: apply spectral smoothing for large-scale perturbations
            return noise
        du = smooth_noise(keys[0], phy_base.u.shape, u_scale)
        dv = smooth_noise(keys[1], phy_base.v.shape, v_scale)
        dt = smooth_noise(keys[2], phy_base.t.shape, t_scale)
        dq = smooth_noise(keys[3], phy_base.q.shape, q_scale)
        dps = smooth_noise(keys[4], phy_base.ps.shape, ps_scale)
        
        # Ensure humidity stays positive
        q_pert = jnp.maximum(phy_base.q + dq, 0.0)
        dq = q_pert - phy_base.q
        
        # Convert perturbations to spectral space
        phy_increment = PhyState(u=du, v=dv, t=dt, q=dq, ps=dps)
        return self.phy2mod(phy_increment, base, param)
    # ========================================================================
    # Time Stepping
    # ========================================================================
    
    @partial(jax.jit, static_argnames=['self'])
    def forward(self, state: State, param: Param):
        """
        Perform one time step.
        
        Delegates to TimeStepper.forward()
        
        Args:
            state: Current State(filt, curr)
            param: Model parameters
            
        Returns:
            Updated State(filt, curr)
        """
        return self.time_stepper.forward(state, param)
    
    @partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
    def integrate(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None):
        """
        Integrate forward in time.
        
        Delegates to TimeStepper.integrate()
        
        Args:
            state0: Initial state (should be output of first_step)
            param: Model parameters
            nstep: Number of time steps
            save_freq: Save frequency (None = no trajectory)
            
        Returns:
            final_state: State after nstep steps
            trajectory: If save_freq provided, SpectralState array of curr states 
                       (State1, State2, State3, ...) where State_i = state_i.curr
        """
        return self.time_stepper.integrate(state0, param, nstep, save_freq)
    
    @partial(jax.jit, static_argnames=['self'])
    def _mod2phy(self, state: SpectralState, param: Param) -> PhyState:
        """
        Convert a single spectral state to physical space.
        
        Args:
            state: Single SpectralState in spectral space
            param: Model parameters
            
        Returns:
            PhyState with grid-point fields
        """
        # Convert vorticity/divergence to U/V winds
        u_spec, v_spec = self.transformer.vor_div_3d_to_uv(state.vor, state.div)
        
        # Transform to grid space
        u = self.transformer.spec_3d_to_grid(u_spec, kcos=True).transpose((2,1,0))
        v = self.transformer.spec_3d_to_grid(v_spec, kcos=True).transpose((2,1,0))
        t = self.transformer.spec_3d_to_grid(state.t).transpose((2,1,0))
        q = self.transformer.spec_3d_to_grid(state.q).transpose((2,1,0))
        ps = self.transformer.spec_to_grid(state.ps).transpose((1,0))
        
        return PhyState(u=u, v=v, t=t, q=q, ps=ps)

    @partial(jax.jit, static_argnames=['self'])
    def mod2phy(self, state, param: Param) -> PhyState:
        """Convert model state to physical space.
        
        Args:
            state: Can be either:
                - State (full model state) - extracts curr
                - SpectralState (prognostic variables only)
                - Batched version of either (trajectory/ensemble)
            param: Model parameters
        
        Returns:
            PhyState with grid-point fields
        """
        # Extract SpectralState if given full State
        if hasattr(state, 'curr'): spectral = state.curr
        else: spectral = state  # Already SpectralState
        
        # Get expected shape for a single state
        expected_shape = self.state_info['vor']
        actual_shape = spectral.vor.shape
        
        # Check if this is a single state or batched
        if len(actual_shape) == len(expected_shape):
            return self._mod2phy(spectral, param)
        else:
            return jax.vmap(self._mod2phy, in_axes=(0, None))(spectral, param)

    @partial(jax.jit, static_argnames=['self'])
    def _phy2mod(self, phy_increment: PhyState, ref_state: State, param: Param) -> State:
        """Convert physical space increment back to model space.
        Args:
            phy_increment: Analysis increment in physical space (δu, δv, δt, δq, δps)
            ref_state: Background state to add increment to
            param: Model parameters
        Returns:
            Analysis state = ref_state + converted increment
        """
        # Convert from [lev,lat,lon] to [lon,lat,lev]
        du = phy_increment.u.transpose((2,1,0))
        dv = phy_increment.v.transpose((2,1,0))
        dt = phy_increment.t.transpose((2,1,0))
        dq = phy_increment.q.transpose((2,1,0))
        dps = phy_increment.ps.transpose((1,0))
        
        # Convert (δu, δv) → (δvor, δdiv) in spectral space
        dvor_spec, ddiv_spec = self.transformer.grid_3d_to_vor_div(du, dv, kcos=True)
        
        # Convert (δt, δq, δps) → spectral
        dt_spec = self.transformer.grid_3d_to_spec(dt)
        dq_spec = self.transformer.grid_3d_to_spec(dq)
        dps_spec = self.transformer.grid_to_spec(dps)
        
        # Apply truncation to ensure consistency
        dvor_spec = self.transformer.apply_truncation_3d(dvor_spec)
        ddiv_spec = self.transformer.apply_truncation_3d(ddiv_spec)
        dt_spec = self.transformer.apply_truncation_3d(dt_spec)
        dq_spec = self.transformer.apply_truncation_3d(dq_spec)
        dps_spec = self.transformer.apply_truncation(dps_spec)
        # Localization at k = 0, 1 for humidity
        dq_spec = dq_spec.at[:,:,0:2].set(0.0)
        
        # Enforce positivity
        #q_spec = ref_state.curr.q + dq_spec
        #q_grid = self.transformer.spec_3d_to_grid(q_spec)
        #q_grid = jnp.maximum(q_grid, 0.0)
        #q_spec = self.transformer.grid_3d_to_spec(q_grid)

        # Add increment to reference state's curr
        ref_curr = ref_state.curr
        new_curr = SpectralState(
            vor=ref_curr.vor + dvor_spec,
            div=ref_curr.div + ddiv_spec,
            t=ref_curr.t + dt_spec,
            #q=q_spec,
            q=ref_curr.q + dq_spec,
            ps=ref_curr.ps + dps_spec,
        )
        
        # For leapfrog: set filt = curr to avoid filter shock after DA
        # (Alternative: could also update filt with same increment)
        return State(
            filt=new_curr,           # Reset filter state
            curr=new_curr,           # Updated current state
            forcing=ref_state.forcing,
            cached=ref_state.cached,
            diag=ref_state.diag,
            time=ref_state.time,
        )

    @partial(jax.jit, static_argnames=['self'])
    def phy2mod(self, phy_increment: PhyState, ref_state: State, param: Param) -> State:
        """Convert physical space increment to model space (handles ensembles).
        
        Args:
            phy_increment: Analysis increment (single or ensemble)
            ref_state: Background state (single or ensemble)
            param: Model parameters
        
        Returns:
            Analysis state
        """
        # Check if ensemble by comparing shapes
        expected_ndim = 3  # [ix, il, kx] for u, v, t, q
        actual_ndim = phy_increment.u.ndim
        
        if actual_ndim == expected_ndim:
            # Single state
            return self._phy2mod(phy_increment, ref_state, param)
        else:
            # Ensemble: vmap over member dimension (axis 0)
            return jax.vmap(self._phy2mod, in_axes=(0, 0, None))(phy_increment, ref_state, param)

    @property
    def state_info(self) -> Dict[str, Tuple[int, ...]]:
        """Return shape information about the state space."""
        mx, nx, kx = self.config.mx, self.config.nx, self.config.kx
        
        return {
            'vor': (mx, nx, kx),
            'div': (mx, nx, kx),
            't': (mx, nx, kx),
            'q': (mx, nx, kx),
            'ps': (mx, nx),
        }
    
    @property
    def grid_info(self) -> Dict[str, any]:
        """Return shape information about the physical space."""
        return {
            'nlon': self.config.ix,
            'nlat': self.config.il,
            'nlev': self.config.kx,
            'lat': self.transformer.lat,
            'lon': self.transformer.lon,
            'landsea_mask': self.static_fields.get_field('fmask')
        }
    
# ============================================================================
# Testing and Validation
# ============================================================================

def main():
    """Test SPEEDY model with NetCDF output."""
    import sys
    from timeit import default_timer as timer
    
    print(f"JAX version: {jax.__version__}")
    print()
    
    # ========================================================================
    # Parameters
    # ========================================================================
    trunc = 30              # Spectral truncation (T30)
    kx = 8                  # Number of vertical levels
    dt = 2400.0             # Time step (seconds) = 40 minutes
    
    # Integration parameters
    ndays = 360              # Number of days to integrate
    save_hours = 6          # Save every 6 hours
    
    # Derived parameters
    steps_per_day = int(86400 / dt)
    nstep = ndays * steps_per_day
    save_freq = int(save_hours * 3600 / dt)
    
    print("="*70)
    print("SPEEDY Model Test")
    print("="*70)
    print(f"  Resolution: T{trunc}, {kx} levels")
    print(f"  Time step: {dt}s ({dt/60:.1f} min)")
    print(f"  Integration: {ndays} days ({nstep} steps)")
    print(f"  Save frequency: {save_hours} hours ({save_freq} steps)")
    print()
    
    # ========================================================================
    # Create model
    # ========================================================================
    print("Initializing model...")
    config = Config.create(trunc=trunc, dt=dt, kx=kx)
    param = SPEEDY.default_param()
    model = SPEEDY(config=config, data_dir=None)
    print()
    
    # ========================================================================
    # Initialize state
    # ========================================================================
    print("Creating initial state...")
    key = jax.random.PRNGKey(0)
    time0 = TimeInfo.create(year=2000, month=1, day=1)
    state0 = model.default_state(param, time0)
    state0 = model.random_state(key, param, state0, noise_scale=1.)
    print(f"  Initial time: {time0.year}/{time0.month:02d}/{time0.day:02d}")
    print(f"  State shape (vor): {state0.curr.vor.shape}")
    print()
    
    # ========================================================================
    # Warmup (JIT compilation)
    # ========================================================================
    print("Warmup (JIT compilation)...")
    warmup_start = timer()
    # Run a short integration for compilation
    _, _ = model.integrate(state0, param, save_freq, save_freq)
    jax.block_until_ready(_)
    warmup_time = timer() - warmup_start
    print(f"  Warmup completed in {warmup_time:.2f}s")
    print()
    
    # ========================================================================
    # Main integration
    # ========================================================================
    print(f"Integrating for {ndays} days...")
    integration_start = timer()
    final_state, trajectory = model.integrate(state0, param, nstep, save_freq)
    #state0 = model.random_state(key, param, final_state, noise_scale=1.)
    #final_state, trajectory = model.integrate(state0, param, nstep, save_freq)
    jax.block_until_ready(trajectory)
    integration_time = timer() - integration_start
    
    print(f"  Integration completed in {integration_time:.2f}s")
    print(f"  Time per step: {integration_time/nstep*1000:.2f}ms")
    print(f"  Time per day: {integration_time/ndays:.2f}s")
    print(f"  Trajectory shape (vor): {trajectory.vor.shape}")
    print()
    
    # ========================================================================
    # Convert to physical space
    # ========================================================================
    print("Converting to physical space...")
    convert_start = timer()
    phy_trajectory = model.mod2phy(trajectory, param)
    phy_trajectory = phy_trajectory._replace(ps=1000.*jnp.exp(phy_trajectory.ps))
    jax.block_until_ready(phy_trajectory)
    convert_time = timer() - convert_start
    print(f"  Conversion completed in {convert_time:.2f}s")
    print(f"  Physical shape (u): {phy_trajectory.u.shape}")
    print()
    
    # ========================================================================
    # Output to NetCDF
    # ========================================================================
    print("Writing output...")
    try:
        from dajax.utils.inout import write_trajectory, write_state
        
        # Time coordinate in hours
        n_times = trajectory.vor.shape[0]
        time_hours = jnp.arange(n_times) * save_hours
        
        # Coordinates
        lon = np.array(model.transformer.lon)
        lat = np.array(model.transformer.lat_full)
        lev = np.arange(kx)
        
        coords = {
            'time': np.array(time_hours),
            'lev': lev,
            'lat': lat,
            'lon': lon
        }
        
        dims = {
            'u': ('time', 'lev', 'lat', 'lon'),
            'v': ('time', 'lev', 'lat', 'lon'),
            't': ('time', 'lev', 'lat', 'lon'),
            'q': ('time', 'lev', 'lat', 'lon'),
            'ps': ('time', 'lat', 'lon')
        }
        
        # Write trajectory
        output_file = 'true.nc'
        write_trajectory(
            filename=output_file,
            state=phy_trajectory,
            time=time_hours,
            coords=coords,
            dims=dims
        )
        print(f"  Trajectory written to {output_file}")
        
        # Write static fields
        fmask_l = model.static_fields.get_field('fmask_l')
        fmask_s = model.static_fields.get_field('fmask_s')
        phis0 = model.static_fields.get_field('phis0')
        alb0 = model.static_fields.get_field('alb0')
        
        StaticState = NamedTuple('StaticState', [
            ('land_fraction', jax.Array),
            ('sea_fraction', jax.Array),
            ('orography', jax.Array),
            ('albedo', jax.Array)
        ])
        static_state = StaticState(
            land_fraction=fmask_l,
            sea_fraction=fmask_s,
            orography=phis0 / model.constants.grav,  # Convert to meters
            albedo=alb0
        )
        
        static_coords = {'lat': lat, 'lon': lon}
        static_dims = {
            'land_fraction': ('lon', 'lat'),
            'sea_fraction': ('lon', 'lat'),
            'orography': ('lon', 'lat'),
            'albedo': ('lon', 'lat')
        }
        
        static_file = 'static.nc'
        write_state(
            filename=static_file,
            state=static_state,
            coords=static_coords,
            dims=static_dims
        )
        print(f"  Static fields written to {static_file}")
        
    except ImportError as e:
        print(f"  Warning: Could not write NetCDF output: {e}")
        print(f"  (dajax.utils.inout not available)")
    except Exception as e:
        print(f"  Warning: Error writing output: {e}")
    
    print()
    
    # ========================================================================
    # Print statistics
    # ========================================================================
    print("="*70)
    print("Output Statistics")
    print("="*70)
    print(f"  U wind:  min={float(jnp.min(phy_trajectory.u)):8.2f}, "
          f"max={float(jnp.max(phy_trajectory.u)):8.2f}, "
          f"mean={float(jnp.mean(phy_trajectory.u)):8.2f} m/s")
    print(f"  V wind:  min={float(jnp.min(phy_trajectory.v)):8.2f}, "
          f"max={float(jnp.max(phy_trajectory.v)):8.2f}, "
          f"mean={float(jnp.mean(phy_trajectory.v)):8.2f} m/s")
    print(f"  Temp:    min={float(jnp.min(phy_trajectory.t)):8.2f}, "
          f"max={float(jnp.max(phy_trajectory.t)):8.2f}, "
          f"mean={float(jnp.mean(phy_trajectory.t)):8.2f} K")
    print(f"  Humidity:min={float(jnp.min(phy_trajectory.q)):8.2e}, "
          f"max={float(jnp.max(phy_trajectory.q)):8.2e}, "
          f"mean={float(jnp.mean(phy_trajectory.q)):8.2e} g/kg")
    print(f"  Ps:      min={float(jnp.min(phy_trajectory.ps)):8.4f}, "
          f"max={float(jnp.max(phy_trajectory.ps)):8.4f}, "
          f"mean={float(jnp.mean(phy_trajectory.ps)):8.4f} hPa")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("Test Summary")
    print("="*70)
    print(f"  ✓ Model initialization: SUCCESS")
    print(f"  ✓ Integration ({ndays} days): SUCCESS")
    print(f"  ✓ Performance: {integration_time/ndays:.2f}s per simulated day")
    print(f"  ✓ Throughput: {ndays*86400/integration_time:.1f}x real-time")
    print("="*70)
    
    return final_state, trajectory

if __name__ == "__main__":
    main()
