#!/usr/bin/env python3
"""
Spectral transforms and differential operators for SPEEDY model.

Combines spherical geometry computations with spectral operators.
Uses pyshtools-based Legendre transforms.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple

from .state import Config
from .constants import Constants
from .legendre import LegendreTransform

class Transformer:
    """
    Spectral transforms and differential operators on the sphere.
    
    Combines:
    - Spherical geometry (Gaussian grid, Coriolis, etc.)
    - Spectral transforms (grid ↔ spectral)
    - Differential operators (gradients, Laplacian, etc.)
    - Wind conversions (vor/div ↔ U/V)
    """
    
    def __init__(self, config: Config, constants: Constants, legendre: LegendreTransform):
        """
        Initialize transformer.
        
        Args:
            config: Config instance from speedy_jax
            constants: Constants instance
            legendre: LegendreTransform instance
        """
        self.config = config
        self.constants = constants
        self.legendre = legendre
        
        # ====================================================================
        # Spherical Geometry
        # ====================================================================
        
        # Use Gaussian latitudes and weights from the transform
        # (More accurate than manual computation - from pyshtools)
        self.lat = legendre.lat       # [iy] - North Pole to Equator
        
        # Longitude grid (equally spaced)
        self.lon = jnp.linspace(0, 360, config.ix, endpoint=False)
        
        # Geometric factors (half grid - Northern hemisphere only)
        self.sia_half = jnp.sin(self.lat * jnp.pi / 180.0)  # sin(lat)
        self.coa_half = jnp.cos(self.lat * jnp.pi / 180.0)  # cos(lat)
        
        # For full grid (South Pole to North Pole)
        self.lat_full = jnp.concatenate([-self.lat, self.lat[::-1]])
        self.cosg = jnp.cos(self.lat_full * jnp.pi / 180.0)
        self.cosgr = 1./self.cosg
        self.cosgr2 = 1./self.cosg**2
        
        # Coriolis parameter: f = 2Ω sin(φ)
        self.coriol = 2.0 * constants.omega * jnp.sin(self.lat_full * jnp.pi / 180.0)
        
        # ====================================================================
        # Spectral Operators
        # ====================================================================
        
        # Compute spectral coefficients
        self.el2, self.elm2, self.el4 = self._setup_spectral_coefficients()
        self.gradx, self.gradym, self.gradyp = self._setup_gradient_coefficients()
        self.uvdx, self.uvdym, self.uvdyp = self._setup_uv_coefficients()
        self.vddym, self.vddyp = self._setup_vd_coefficients()
        self.trfilt = self._setup_truncation_filter()
    
    # ========================================================================
    # Convenience transformation methods 
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,2))
    def spec_to_grid(self, spec: jax.Array, kcos: bool = False) -> jax.Array:
        """
        Transform from complex spectral representation to grid.
        
        Input: spec [mx, nx] (complex)
               kcos: 0 for regular fields, 1 for winds (differ from SPEEDY with 1 for regular, 2 winds)
        Output: grid [ix, il] (real)
        """
        grid = self.legendre.spec_to_grid(spec)
        if kcos: grid = grid * self.cosgr[jnp.newaxis, :]
        return grid
    
    @partial(jax.jit, static_argnums=(0,2))
    def spec_3d_to_grid(self, spec_3d: jax.Array, kcos: bool = False) -> jax.Array:
        """
        Vectorized transform for 3D spectral fields.
        
        Input: spec_3d [mx, nx, kx] (complex)
        Output: grid_3d [ix, il, kx] (real)
        
        Uses vmap to transform all levels at once - much faster than looping!
        """
        # Transpose to put level axis first for vmap
        spec_transposed = jnp.transpose(spec_3d, (2, 0, 1))  # [kx, mx, nx]
        
        # Map transform over levels
        grid_transposed = jax.vmap(self.spec_to_grid, in_axes=(0, None))(spec_transposed, kcos)  # [kx, ix, il]
        
        # Transpose back to [ix, il, kx]
        return jnp.transpose(grid_transposed, (1, 2, 0))
    
    @partial(jax.jit, static_argnums=(0,))
    def grid_to_spec(self, grid: jax.Array) -> jax.Array:
        """
        Transform from grid to complex spectral representation.
        
        Input: grid [ix, il] (real)
        Output: spec_complex [mx, nx] (complex)
        """
        spec = self.legendre.grid_to_spec(grid)
        return spec
    
    @partial(jax.jit, static_argnums=(0,))
    def grid_3d_to_spec(self, grid_3d: jax.Array) -> jax.Array:
        """
        Vectorized transform for 3D grid fields.
        
        Input: grid_3d [ix, il, kx] (real)
        Output: spec_3d [mx, nx, kx] (complex)
        
        Uses vmap to transform all levels at once - much faster than looping!
        """
        # Transpose to put level axis first for vmap
        grid_transposed = jnp.transpose(grid_3d, (2, 0, 1))  # [kx, ix, il]
        
        # Map transform over levels
        spec_transposed = jax.vmap(self.grid_to_spec)(grid_transposed)  # [kx, mx, nx]
        
        # Transpose back to [mx, nx, kx]
        return jnp.transpose(spec_transposed, (1, 2, 0))

    # ========================================================================
    # Spectral Coefficient Setup
    # ========================================================================
    
    def _setup_spectral_coefficients(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Setup L^2 and L^4 operators - VECTORIZED"""
        mx, nx = self.config.mx, self.config.nx
        rearth = self.constants.rearth
        
        # Create mesh grid for all (m, n) pairs
        m_grid, n_grid = np.meshgrid(np.arange(mx), np.arange(nx), indexing='ij')
        l_grid = m_grid + n_grid
        
        # el2[m,n] = l(l+1) / r²
        el2 = l_grid * (l_grid + 1) / (rearth ** 2)
        el4 = el2 ** 2
        
        # Inverse L² (avoid division by zero)
        elm2 = np.zeros_like(el2)
        mask = l_grid > 0
        elm2[mask] = 1.0 / el2[mask]
        
        return jnp.array(el2), jnp.array(elm2), jnp.array(el4)
    
    def _setup_gradient_coefficients(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Setup gradient operator coefficients - CORRECTED to match Fortran spectral.f90
        
        Fortran lines 67,73,77:
        - gradx(m) = m1/rearth where m1 = m-1 (Fortran)
        - gradym(m,n) = (el1 - 1) * epsi(m,n) / rearth
        - gradyp(m,n) = (el1 + 2) * epsi(m,n+1) / rearth
        where el1 = m+n-2 (Fortran) = m+n (Python)
        """
        mx, nx = self.config.mx, self.config.nx
        rearth = self.constants.rearth
        epsi = self.legendre.epsi
        
        # Gradient in x (longitude): gradx[m] = m / r (Python 0-indexed)
        gradx = np.arange(mx, dtype=float) / rearth
        
        # Create mesh grid for gradient in y (latitude)
        m_grid, n_grid = np.meshgrid(np.arange(mx), np.arange(nx), indexing='ij')
        l_grid = m_grid + n_grid  # Total wavenumber at each (m,n)
        # gradym: (l - 1) * eps(m,l) / rearth (for n>0 only)
        # Multiplies the (m, n-1) term in latitude gradient
        gradym = np.where(n_grid > 0, (l_grid - 1) * epsi[:-1,:-1] / rearth, 0.0)
        # gradyp: (l + 2) * eps(m,l+1) / rearth
        # Multiplies the (m, n+1) term in latitude gradient
        gradyp = (l_grid + 2) * epsi[:-1,1:] / rearth

        return jnp.array(gradx), jnp.array(gradym), jnp.array(gradyp)
    
    def _setup_uv_coefficients(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Setup coefficients for UV spectral conversion - CORRECTED to match Fortran spectral.f90
        
        Fortran lines 66-78:
        - For n=1: uvdx(m,1) = -rearth / (m1+1) where m1=m-1
        - For n>1: uvdx(m,n) = -rearth * m1 / (el1*(el1+1))
        - uvdym(m,n) = -rearth * epsi(m,n) / el1  
        - uvdyp(m,n) = -rearth * epsi(m,n+1) / (el1+1)
        """
        mx, nx = self.config.mx, self.config.nx
        rearth = self.constants.rearth
        epsi = self.legendre.epsi
        
        # Create mesh grid
        m_grid, n_grid = np.meshgrid(np.arange(mx), np.arange(nx), indexing='ij')
        l_grid = m_grid + n_grid  # Total wavenumber
        
        # uvdx: coefficient for d/dlambda term
        # Special case for n=0 (Fortran n=1): -rearth/(m+1)
        # For n>0: -rearth * m / (l*(l+1))
        uvdx = np.zeros((mx, nx))
        uvdx[:, 0] = -rearth / (np.arange(mx, dtype=float) + 1)
        mask = n_grid > 0
        uvdx[mask] = -rearth * m_grid[mask] / (l_grid[mask] * (l_grid[mask] + 1))

        # uvdym: -rearth * eps(m,l) / l (for n>0 only)
        uvdym = np.zeros((mx, nx))
        mask = n_grid > 0
        uvdym[mask] = -rearth * epsi[:-1,:-1][mask] / l_grid[mask]
        
        # uvdyp: -rearth * eps(m,l+1) / (l+1)
        uvdyp = -rearth * epsi[:-1,1:] / (l_grid + 1)
        
        return jnp.array(uvdx), jnp.array(uvdym), jnp.array(uvdyp)
    
    def _setup_vd_coefficients(self) -> Tuple[jax.Array, jax.Array]:
        """
        Setup coefficients for vorticity/divergence conversion - CORRECTED to match Fortran spectral.f90
        
        Fortran lines 75,79:
        - vddym(m,n) = (el1 + 1) * epsi(m,n) / rearth
        - vddyp(m,n) = el1 * epsi(m,n+1) / rearth
        """
        mx, nx = self.config.mx, self.config.nx
        rearth = self.constants.rearth
        epsi = self.legendre.epsi
        
        # Create mesh grid
        m_grid, n_grid = np.meshgrid(np.arange(mx), np.arange(nx), indexing='ij')
        l_grid = m_grid + n_grid
        
        # vddym: (l + 1) * eps(m,l) / rearth (for n>0 only)
        vddym = np.zeros((mx, nx))
        mask = n_grid > 0
        vddym[mask] = (l_grid[mask] + 1) * epsi[:-1,:-1][mask] / rearth
        
        # vddyp: l * eps(m,l+1) / rearth
        vddyp = l_grid * epsi[:-1,1:] / rearth
        
        return jnp.array(vddym), jnp.array(vddyp)
    
    def _setup_truncation_filter(self) -> jax.Array:
        """Setup triangular truncation filter - VECTORIZED"""
        mx, nx = self.config.mx, self.config.nx
        
        # Create mesh grid
        m_grid, n_grid = np.meshgrid(np.arange(mx), np.arange(nx), indexing='ij')
        l_grid = m_grid + n_grid
        
        # Triangular truncation: keep if l <= trunc
        trfilt = np.where(l_grid <= self.config.trunc, 1.0, 0.0)
        
        return jnp.array(trfilt)
    
    # ========================================================================
    # Differential Operators
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def laplacian(self, spec_input: jax.Array) -> jax.Array:
        """Apply Laplacian operator: ∇²"""
        return -spec_input * self.el2
    
    @partial(jax.jit, static_argnums=(0,))
    def laplacian_3d(self, spec_3d: jax.Array) -> jax.Array:
        """
        Apply Laplacian to 3D spectral field: ∇²
        
        Input: spec_3d [mx, nx, kx] (complex)
        Output: laplacian_3d [mx, nx, kx] (complex)
        
        Uses broadcasting - more efficient than vmap since el2 is [mx, nx].
        """
        # el2 is [mx, nx], spec_3d is [mx, nx, kx]
        # Broadcasting handles the level dimension automatically
        return -spec_3d * self.el2[:, :, jnp.newaxis]

    @partial(jax.jit, static_argnums=(0,))
    def inverse_laplacian(self, spec_input: jax.Array) -> jax.Array:
        """Apply inverse Laplacian: ∇⁻²"""
        return -spec_input * self.elm2
    
    @partial(jax.jit, static_argnums=(0,))
    def grad_lon(self, spec_input: jax.Array) -> jax.Array:
        """
        Gradient in longitude direction: ∂/∂λ
        For complex spectral coefficients: multiply by im
        """
        # Broadcast gradx over n dimension
        # gradx[m] → gradx[m, newaxis] broadcasts to [mx, nx]
        return 1j * self.gradx[:, jnp.newaxis] * spec_input
    
    @partial(jax.jit, static_argnums=(0,))
    def grad_lat(self, spec_input: jax.Array) -> jax.Array:
        """
        Gradient in latitude direction: ∂/∂θ
        Uses epsilon recursion with gradym and gradyp coefficients
        
        CORRECTED: Signs now match Fortran spectral.f90 line 141
        Formula: ∂f/∂θ = -gradym[m,n]·f[m,n-1] + gradyp[m,n]·f[m,n+1]
        """
        result = jnp.zeros_like(spec_input)
        
        # Term from n-1: -gradym[:, 1:] * spec_input[:, :-1] (NEGATIVE sign)
        result = result.at[:, 1:].add(-self.gradym[:, 1:] * spec_input[:, :-1])
        
        # Term from n+1: +gradyp[:, :-1] * spec_input[:, 1:] (POSITIVE sign)
        result = result.at[:, :-1].add(self.gradyp[:, :-1] * spec_input[:, 1:])
        
        return result
    
    # ========================================================================
    # Wind Conversions
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def vor_div_to_uv(self, vor_spec: jax.Array, div_spec: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Convert vorticity and divergence to U and V winds in spectral space.
        
        ψ = ∇⁻²ζ (streamfunction from vorticity)
        χ = ∇⁻²δ (velocity potential from divergence)
        
        CRITICAL: The uvdx, uvdym, uvdyp coefficients ALREADY include the 
        inverse Laplacian factor elm2! So we work with vorm/divm directly,
        NOT with psi/chi!
        
        Fortran uvspec (spectral.f90:180-195):
        zc = uvdx*divm*i  [uvdx has elm2 built in!]
        zp = uvdx*vorm*i  [uvdx has elm2 built in!]
        """
        # CRITICAL: Use vor_spec and div_spec directly!
        # The uvdx coefficients already include elm2 (inverse Laplacian)
        
        # Compute longitude gradient terms
        zc = 1j * self.uvdx * div_spec  # NOT chi! uvdx has elm2
        zp = 1j * self.uvdx * vor_spec  # NOT psi! uvdx has elm2
        
        u_spec = jnp.zeros_like(vor_spec)
        v_spec = jnp.zeros_like(vor_spec)
        
        # Term from n-1: 
        u_spec = u_spec.at[:, 1:].add(self.uvdym[:, 1:] * vor_spec[:, :-1])
        v_spec = v_spec.at[:, 1:].add(-self.uvdym[:, 1:] * div_spec[:, :-1])
        
        # Term from n+1:
        u_spec = u_spec.at[:, :-1].add(zc[:,:-1]-self.uvdyp[:, :-1] * vor_spec[:, 1:])
        v_spec = v_spec.at[:, :-1].add(zp[:,:-1]+self.uvdyp[:, :-1] * div_spec[:, 1:])
        
        return u_spec, v_spec
    
    @partial(jax.jit, static_argnums=(0,))
    def vor_div_3d_to_uv(self, vor_spec_3d: jax.Array, div_spec_3d: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Vectorized conversion for 3D spectral fields.
        
        Input: vor_spec_3d [mx, nx, kx], div_spec_3d [mx, nx, kx] (complex)
        Output: u_spec_3d [mx, nx, kx], v_spec_3d [mx, nx, kx] (complex)
        
        Uses vmap to transform all levels at once - much faster than looping!
        """
        # Transpose to put level axis first for vmap
        vor_transposed = jnp.transpose(vor_spec_3d, (2, 0, 1))  # [kx, mx, nx]
        div_transposed = jnp.transpose(div_spec_3d, (2, 0, 1))  # [kx, mx, nx]
        
        # Map transform over levels
        # in_axes=(0, 0) means vectorize over axis 0 of both arguments
        u_transposed, v_transposed = jax.vmap(self.vor_div_to_uv, in_axes=(0, 0))(vor_transposed, div_transposed)  # [kx, mx, nx] each
        
        # Transpose back to [mx, nx, kx]
        u_spec_3d = jnp.transpose(u_transposed, (1, 2, 0))
        v_spec_3d = jnp.transpose(v_transposed, (1, 2, 0))
        
        return u_spec_3d, v_spec_3d

    @partial(jax.jit, static_argnums=(0, 3))
    def grid_to_vor_div(self, ug: jax.Array, vg: jax.Array, kcos: bool = False) -> Tuple[jax.Array, jax.Array]:
        """
        Convert grid U,V to spectral vorticity/divergence (vdspec).
        Based on Fortran spectral.f90 vds() lines 146-171
        
        Has special boundary conditions for n=1 and n=nx!
        """
        # Apply cosine weighting
        if kcos:
            ug_weighted = ug * self.cosgr[jnp.newaxis, :]
            vg_weighted = vg * self.cosgr[jnp.newaxis, :]
        else:
            ug_weighted = ug * self.cosgr2[jnp.newaxis, :]
            vg_weighted = vg * self.cosgr2[jnp.newaxis, :]
        
        # Transform to spectral
        u_spec = self.grid_to_spec(ug_weighted)
        v_spec = self.grid_to_spec(vg_weighted)
        
        # Compute longitude gradient terms
        zp = 1j * self.gradx[:, jnp.newaxis] * u_spec  # gradx*u*i
        zc = 1j * self.gradx[:, jnp.newaxis] * v_spec  # gradx*v*i
        
        vor_spec = jnp.zeros_like(u_spec)
        div_spec = jnp.zeros_like(u_spec)
        
        # Term from n-1: 
        vor_spec = vor_spec.at[:, 1:].add(self.vddym[:, 1:] * u_spec[:, :-1])
        div_spec = div_spec.at[:, 1:].add(-self.vddym[:, 1:] * v_spec[:, :-1])
        
        # Term from n+1:
        vor_spec = vor_spec.at[:, :-1].add(zc[:,:-1]-self.vddyp[:, :-1] * u_spec[:, 1:])
        div_spec = div_spec.at[:, :-1].add(zp[:,:-1]+self.vddyp[:, :-1] * v_spec[:, 1:])
        
        return vor_spec, div_spec
    
    @partial(jax.jit, static_argnums=(0, 3))
    def grid_3d_to_vor_div(self, ug_3d: jax.Array, vg_3d: jax.Array, kcos: bool = False) -> Tuple[jax.Array, jax.Array]:
        """
        Vectorized conversion for 3D grid fields.
        
        Input: ug_3d [ix, il, kx], vg_3d [ix, il, kx] (real)
        Output: vor_spec_3d [mx, nx, kx], div_spec_3d [mx, nx, kx] (complex)
        
        Uses vmap to transform all levels at once - much faster than looping!
        """
        # Transpose to put level axis first for vmap
        ug_transposed = jnp.transpose(ug_3d, (2, 0, 1))  # [kx, ix, il]
        vg_transposed = jnp.transpose(vg_3d, (2, 0, 1))  # [kx, ix, il]
        
        # Map transform over levels
        # in_axes=(0, 0, None) means vectorize over axis 0 of first two args, broadcast kcos
        vor_transposed, div_transposed = jax.vmap(self.grid_to_vor_div, in_axes=(0, 0, None))(ug_transposed, vg_transposed, kcos)  # [kx, mx, nx] each
        
        # Transpose back to [mx, nx, kx]
        vor_spec_3d = jnp.transpose(vor_transposed, (1, 2, 0))
        div_spec_3d = jnp.transpose(div_transposed, (1, 2, 0))
        
        return vor_spec_3d, div_spec_3d

    @partial(jax.jit, static_argnums=(0,))
    def apply_truncation(self, spec: jax.Array) -> jax.Array:
        """
        Apply triangular truncation filter.
        Based on Fortran spectral.f90 trunct() lines 229-233
        
        Uses the trfilt mask computed in _setup_truncation_filter()
        """
        return spec * self.trfilt
    
    @partial(jax.jit, static_argnums=(0,))
    def apply_truncation_3d(self, spec: jax.Array) -> jax.Array:
        """
        Apply triangular truncation filter.
        Based on Fortran spectral.f90 trunct() lines 229-233
        
        Uses the trfilt mask computed in _setup_truncation_filter()
        """
        return spec * self.trfilt[:, :, jnp.newaxis]
    
    @partial(jax.jit, static_argnums=(0,))
    def spectral_truncation(self, fg: jax.Array) -> jax.Array:
        """
        Compute a spectrally-filtered grid-point field.
        Applies triangular truncation by zeroing modes with l > trunc.
        
        From Fortran boundaries.f90 spectral_truncation()
        
        Args:
            fg: Original grid-point field [ix, il]
            
        Returns:
            Filtered grid-point field [ix, il]
        """
        # Convert to spectral space
        fsp = self.grid_to_spec(fg)
        
        # Apply triangular truncation filter (zeros out modes with l > trunc)
        fsp_filtered = fsp * self.trfilt
        
        # Convert back to grid space with kcos=1
        fg_filtered = self.spec_to_grid(fsp_filtered)
        
        return fg_filtered
