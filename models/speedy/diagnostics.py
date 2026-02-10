#!/usr/bin/env python3
"""
Diagnostics for SPEEDY model.

Based on SPEEDY diagnostics.f90.
Computes:
- Eddy kinetic energy (rotational and divergent components)
- Global-mean temperature
- Checks for numerical instability
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Optional

from .state import SpectralState, Config
from .transformer import Transformer

class DiagnosticValues(NamedTuple):
    """
    Diagnostic values for one timestep.
    
    All arrays have shape [kx] for vertical profiles.
    """
    reke: jax.Array  # Rotational eddy kinetic energy [kx]
    deke: jax.Array  # Divergent eddy kinetic energy [kx]
    temp: jax.Array  # Global-mean temperature [kx]
    step: int        # Timestep number

class Diagnostics:
    """
    Compute and check model diagnostics.
    
    Based on SPEEDY diagnostics.f90:check_diagnostics()
    
    Computes:
    1. Eddy kinetic energy from vorticity (rotational component)
    2. Eddy kinetic energy from divergence (divergent component)
    3. Global-mean temperature
    
    Checks for numerical stability:
    - Kinetic energy < 500 m²/s²
    - Temperature in range [180, 320] K
    """
    
    def __init__(self, config: Config, transformer: Transformer):
        """
        Initialize diagnostics.
        
        Args:
            config: Model configuration
            transformer: Transformer for spectral operations
        """
        self.config = config
        self.transformer = transformer
        
        # Thresholds for stability checks (from diagnostics.f90:61-62)
        self.ke_max = 500.0      # Maximum kinetic energy (m²/s²)
        self.temp_min = 180.0    # Minimum temperature (K)
        self.temp_max = 320.0    # Maximum temperature (K)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute(self, spec_state: SpectralState, step: int) -> DiagnosticValues:
        """
        Compute diagnostic values from spectral state.
        
        Based on diagnostics.f90:check_diagnostics() lines 29-50.
        
        Args:
            spec_state: SpectralState with vor, div, t fields
        
        Returns:
            DiagnosticValues with reke, deke, temp profiles
        """
        vor = spec_state.vor  # [mx, nx, kx]
        div = spec_state.div  # [mx, nx, kx]
        t = spec_state.t      # [mx, nx, kx]
        
        # Initialize diagnostic arrays
        kx = self.config.kx
        reke = jnp.zeros(kx)  # Rotational kinetic energy
        deke = jnp.zeros(kx)  # Divergent kinetic energy
        temp = jnp.zeros(kx)  # Global-mean temperature
        
        # ====================================================================
        # Compute diagnostics for each vertical level
        # ====================================================================
        
        def compute_level(k):
            """Compute diagnostics for one vertical level."""
            
            # 1. Global-mean temperature (m=0, n=0 mode)
            # Factor sqrt(0.5) comes from normalization (Fortran line 33)
            temp_k = jnp.sqrt(0.5) * jnp.real(t[0, 0, k])
            
            # 2. Rotational eddy kinetic energy
            # KE = -∑(ψ * conj(ζ)) where ψ = ∇⁻²ζ (streamfunction)
            # Fortran lines 35-41
            psi = self.transformer.inverse_laplacian(vor[:, :, k])  # Streamfunction
            
            # Sum over eddy modes (m>=2, all n) - exclude m=0 (zonal mean)
            # real(psi * conj(vor)) = real part of complex product
            reke_k = -jnp.sum(jnp.real(psi[1:, :] * jnp.conj(vor[1:, :, k])))
            
            # 3. Divergent eddy kinetic energy
            # KE = -∑(χ * conj(δ)) where χ = ∇⁻²δ (velocity potential)
            # Fortran lines 43-49
            chi = self.transformer.inverse_laplacian(div[:, :, k])  # Velocity potential
            
            # Sum over eddy modes (m>=1, all n)
            deke_k = -jnp.sum(jnp.real(chi[1:, :] * jnp.conj(div[1:, :, k])))
            
            return reke_k, deke_k, temp_k
        
        # VECTORIZED: Compute all levels at once using vmap
        reke, deke, temp = jax.vmap(compute_level)(jnp.arange(kx))
        
        return DiagnosticValues(reke=reke, deke=deke, temp=temp, step=step)
    
    def check_stability(self, diag: DiagnosticValues) -> tuple[bool, Optional[str]]:
        """
        Check if diagnostic values are within acceptable ranges.
        
        Based on diagnostics.f90:check_diagnostics() lines 60-70.
        
        Args:
            diag: DiagnosticValues to check
        
        Returns:
            (is_stable, error_message)
            - is_stable: True if all values are within range
            - error_message: None if stable, otherwise description of problem
        """
        # Check each level
        for k in range(self.config.kx):
            # Check rotational kinetic energy
            if diag.reke[k] > self.ke_max:
                return False, f"Rotational KE too large at level {k}: {diag.reke[k]:.2f} > {self.ke_max}"
            
            # Check divergent kinetic energy
            if diag.deke[k] > self.ke_max:
                return False, f"Divergent KE too large at level {k}: {diag.deke[k]:.2f} > {self.ke_max}"
            
            # Check temperature range
            if diag.temp[k] < self.temp_min:
                return False, f"Temperature too low at level {k}: {diag.temp[k]:.2f} < {self.temp_min}"
            
            if diag.temp[k] > self.temp_max:
                return False, f"Temperature too high at level {k}: {diag.temp[k]:.2f} > {self.temp_max}"
        
        return True, None
    
    def format_diagnostics(self, diag: DiagnosticValues) -> str:
        """
        Format diagnostics for printing.
        
        Matches Fortran output format (diagnostics.f90:72-74).
        
        Args:
            diag: DiagnosticValues to format
        
        Returns:
            Formatted string
        """
        lines = []
        
        # Format kinetic energy values
        reke_str = ' '.join([f'{val:8.2f}' for val in diag.reke])
        deke_str = ' '.join([f'{val:8.2f}' for val in diag.deke])
        temp_str = ' '.join([f'{val:8.2f}' for val in diag.temp])
        
        lines.append(f' step = {diag.step:6d} reke = {reke_str}')
        lines.append(f'              deke = {deke_str}')
        lines.append(f'              temp = {temp_str}')
        
        return '\n'.join(lines)
    
    def print_diagnostics(self, diag: DiagnosticValues):
        """Print diagnostics to console."""
        print(self.format_diagnostics(diag))
    
    def check_and_print(self, spec_state: SpectralState, step: int, print_freq: Optional[int] = None) -> bool:
        """
        Compute diagnostics, optionally print, and check stability.
        
        This is the main method to use during integration.
        
        Args:
            spec_state: SpectralState to diagnose
            step: Current timestep number
            print_freq: Print every N steps (None = never print)
        
        Returns:
            True if stable, False if unstable
        """
        # Compute diagnostics
        diag = self.compute(spec_state, step)
        
        # Print if requested
        if print_freq is not None and step % print_freq == 0:
            self.print_diagnostics(diag)
        
        # Check stability
        is_stable, error_msg = self.check_stability(diag)
        
        if not is_stable:
            # Print diagnostics before raising error
            print("\n" + "="*70)
            print("ERROR: Model variables out of accepted range")
            print("="*70)
            self.print_diagnostics(diag)
            print("="*70)
            print(f"Reason: {error_msg}")
            print("="*70)
        
        return is_stable

# ============================================================================
# Helper Functions for Integration
# ============================================================================

def create_diagnostic_callback(diagnostics: Diagnostics, print_freq: int = 1):
    """
    Create a callback function for use during integration.
    
    Args:
        diagnostics: Diagnostics instance
        print_freq: Print diagnostics every N steps
    
    Returns:
        Callback function that can be called with (state, step)
    """
    def callback(state, step):
        """Check diagnostics and raise error if unstable."""
        # Use current state for diagnostics
        is_stable = diagnostics.check_and_print(state.curr, step, print_freq)
        
        if not is_stable:
            raise RuntimeError(f"Model became unstable at step {step}")
        
        return is_stable
    
    return callback

# ============================================================================
# Testing
# ============================================================================

def test_diagnostics():
    """Test diagnostics computation."""
    import sys
    sys.path.insert(0, '/mnt/project')
    
    from speedy_jax import create_config, SpectralState
    from transformer import Transformer
    from legendre import LegendreTransform
    from constants import Constants
    
    # Create minimal setup
    config = create_config(trunc=30, dt=2400.0, kx=8)
    constants = Constants()
    legendre = LegendreTransform(config)
    transformer = Transformer(config, constants, legendre)
    
    # Create diagnostics
    diag_module = Diagnostics(config, transformer)
    
    # Create test state
    mx, nx, kx = config.mx, config.nx, config.kx
    
    # Initialize with reasonable values
    vor = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
    div = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
    t = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
    q = jnp.zeros((mx, nx, kx), dtype=jnp.complex64)
    ps = jnp.zeros((mx, nx), dtype=jnp.complex64)
    
    # Set global-mean temperature to 250K (reasonable)
    # t[0,0,k] = T / sqrt(0.5) for spectral coefficient
    for k in range(kx):
        t = t.at[0, 0, k].set(250.0 / jnp.sqrt(0.5) + 0.0j)
    
    # Add small vorticity perturbation
    vor = vor.at[2, 1, 4].set(1.0e-5 + 0.0j)
    
    spec_state = SpectralState(vor=vor, div=div, t=t, q=q, ps=ps)
    
    # Compute diagnostics
    diag_values = diag_module.compute(spec_state)
    
    print("="*70)
    print("Diagnostics Test")
    print("="*70)
    print(f"\nRotational KE: {diag_values.reke}")
    print(f"Divergent KE:  {diag_values.deke}")
    print(f"Temperature:   {diag_values.temp}")
    
    # Check stability
    is_stable, msg = diag_module.check_stability(diag_values)
    print(f"\nStability check: {'PASS' if is_stable else 'FAIL'}")
    if msg:
        print(f"Message: {msg}")
    
    # Test formatting
    print("\n" + "="*70)
    print("Formatted output:")
    print("="*70)
    diag_values = diag_values._replace(step=100)
    print(diag_module.format_diagnostics(diag_values))
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_diagnostics()
