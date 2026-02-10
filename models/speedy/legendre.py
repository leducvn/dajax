#!/usr/bin/env python3
"""
VECTORIZED Legendre Transform - Based on legendre.f90
"""

import jax
import jax.numpy as jnp
import numpy as np
import pyshtools as pysh
from typing import Tuple
from functools import partial

from .state import Config

class LegendreTransform:
    """
    Fast vectorized Legendre transform.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Compute epsilon (needed for some spectral operations)
        self.epsi = self._compute_epsilon()
        
        # Use pyshtools for Gaussian grid and Legendre polynomials
        cpol, wt, self.lat = self._setup_legendre_polynomials()
        #cpol, wt, self.lat = self._setup_legendre_polynomials2()
        
        # Split into even/odd for hemispheric symmetry
        self.cpol_even = cpol[:, ::2, :]
        self.cpol_odd = cpol[:, 1::2, :]
        
        # Weighted polynomials for grid_to_spec
        cpol_weighted = cpol * wt[jnp.newaxis, jnp.newaxis, :]
        self.cpol_weighted_even = cpol_weighted[:, ::2, :]
        self.cpol_weighted_odd = cpol_weighted[:, 1::2, :]

    def _compute_epsilon(self):
        """
        Compute epsilon coefficients for Legendre recursion.
        
        Formula: epsi[m,n] = sqrt((l^2 - m^2) / (4*l^2 - 1))
        where l = m + n is the total wavenumber.
        
        Returns:
            epsi: Epsilon coefficients [mx+1, nx+1]
        """
        mx, nx = self.config.mx, self.config.nx
        
        m_grid, n_grid = np.meshgrid(np.arange(mx + 1), np.arange(nx + 1), indexing='ij')
        
        emm2 = m_grid.astype(float) ** 2
        ell = (m_grid + n_grid).astype(float)
        ell2 = ell ** 2
        
        denom = 4.0 * ell2 - 1.0
        denom = np.where(denom == 0, 1.0, denom)
        
        epsi = np.sqrt((ell2 - emm2) / denom)
        
        epsi[:, nx] = 0.0
        epsi[0, 0] = 0.0
        
        return epsi

    def _setup_legendre_polynomials(self):
        """
        Compute Legendre polynomials using pyshtools.
        
        Returns:
            cpol: Legendre polynomials [mx, nx, iy]
            wt: Gaussian weights [iy]
            lat: latitude
        """
        mx, nx, iy = self.config.mx, self.config.nx, self.config.iy
        trunc = self.config.trunc
        
        # Truncation array
        nsh2 = np.array([min(mx, trunc - n + 2) for n in range(nx)])
        
        # Gaussian quadrature from pyshtools
        # Full sphere then take hemisphere (SPEEDY method)
        cost_full, wt_full = pysh.expand.SHGLQ(2 * iy - 1)
        cost = cost_full[:iy] # North pole to equator
        wt = wt_full[:iy]

        # Convert to latitude in degrees (for reference)
        lat = np.arcsin(cost) * 180.0 / np.pi
        
        # Storage for polynomials [mx, nx, iy] - simplified from [2*mx, nx, iy]
        cpol = np.zeros((mx, nx, iy))
        
        for j in range(iy):
            # CRITICAL: csphase=-1 to exclude Condon-Shortley phase!
            p, _ = pysh.legendre.PlmBar_d1(trunc+1, cost[j], cnorm=0, csphase=-1)
            
            for n in range(nx):
                for m in range(nsh2[n]):
                    l = m + n
                    if l <= trunc+1:
                        idx = (l * (l + 1)) // 2 + m
                        # CRITICAL: Apply normalization conversion
                        # SPEEDY = PlmBar(csphase=-1) / sqrt(2 * (2 - delta_m0))
                        if m == 0: 
                            norm_factor = np.sqrt(2.0)
                        else: 
                            norm_factor = 2.0 * ((-1) ** m)
                        cpol[m, n, j] = p[idx] / norm_factor
        
        return jnp.array(cpol), jnp.array(wt), np.array(lat)

    # ========================================================================
    # Main Transform Functions
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def spec_to_grid(self, spec: jax.Array) -> jax.Array:
        """
        Spectral to grid - FULLY VECTORIZED
        
        Args:
            spec: Complex spectral coefficients [mx, nx]
        
        Returns:
            grid: Real grid values [ix, il]
        """
        fourier = self._legendre_inv(spec)   # [mx, il] complex
        grid = self._fourier_inv(fourier)    # [ix, il] real
        return grid
    
    @partial(jax.jit, static_argnums=(0,))
    def grid_to_spec(self, grid: jax.Array) -> jax.Array:
        """
        Grid to spectral - FULLY VECTORIZED
        
        Args:
            grid: Real grid values [ix, il]
        
        Returns:
            spec: Complex spectral coefficients [mx, nx]
        """
        fourier = self._fourier_dir(grid)    # [mx, il] complex
        spec = self._legendre_dir(fourier)   # [mx, nx] complex
        return spec
    
    # ========================================================================
    # Legendre Transforms (Vectorized with einsum)
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _legendre_inv(self, spec: jax.Array) -> jax.Array:
        """
        Inverse Legendre transform: spectral → Fourier coefficients.
        
        Uses hemispheric symmetry with even/odd parity decomposition.
        einsum handles complex * real multiplication correctly.
        
        Args:
            spec: Complex spectral coefficients [mx, nx]
        
        Returns:
            fourier: Complex Fourier coefficients [mx, il]
        """
        
        # Split even/odd modes
        spec_even = spec[:, ::2]   # [mx, nx_even]
        spec_odd = spec[:, 1::2]   # [mx, nx_odd]
        
        # Compute even and odd contributions
        # einsum handles complex * real correctly
        even = jnp.einsum('mn,mnj->mj', spec_even, self.cpol_even)  # [mx, iy]
        odd = jnp.einsum('mn,mnj->mj', spec_odd, self.cpol_odd)     # [mx, iy]
        
        # Reconstruct both hemispheres
        # Northern hemisphere (j = 0 to iy-1): even - odd
        north = even - odd  # [mx, iy]
        
        # Southern hemisphere (j = iy to il-1): even + odd (reversed)
        south = jnp.flip(even + odd, axis=-1)  # [mx, iy]
        
        # Concatenate: [mx, il]
        fourier = jnp.concatenate([north, south], axis=-1)
        
        return fourier

    @partial(jax.jit, static_argnums=(0,))
    def _legendre_dir(self, fourier: jax.Array) -> jax.Array:
        """
        Direct Legendre transform: Fourier coefficients → spectral.
        
        Uses Gaussian quadrature with hemispheric symmetry.
        
        Args:
            fourier: Complex Fourier coefficients [mx, il]
        
        Returns:
            spec: Complex spectral coefficients [mx, nx]
        """
        iy = self.config.iy
        nx = self.config.nx
        trunc = self.config.trunc
        
        # Split hemispheres
        north = fourier[:, :iy]                      # [mx, iy]
        south = jnp.flip(fourier[:, iy:], axis=-1)   # [mx, iy]
        
        # Parity decomposition (weights already in cpol_weighted)
        even = north + south   # Symmetric combination
        odd = south - north    # Antisymmetric combination
        
        # Number of modes for truncation
        n_even_modes = (trunc // 2) + 1      # Even modes: 0, 2, 4, ..., trunc
        n_odd_modes = (trunc + 1) // 2       # Odd modes: 1, 3, 5, ..., trunc-1
        
        # Slice cpol arrays to only include modes up to trunc
        cpol_even_trunc = self.cpol_weighted_even[:, :n_even_modes, :]
        cpol_odd_trunc = self.cpol_weighted_odd[:, :n_odd_modes, :]
        
        # Project onto spectral modes
        spec_even = jnp.einsum('mj,mnj->mn', even, cpol_even_trunc)  # [mx, n_even]
        spec_odd = jnp.einsum('mj,mnj->mn', odd, cpol_odd_trunc)     # [mx, n_odd]
        
        # Interleave even and odd coefficients into output
        spec = jnp.zeros((self.config.mx, nx), dtype=fourier.dtype)
        
        # Even modes at indices 0, 2, 4, ...
        even_indices = jnp.arange(0, 2 * n_even_modes, 2)
        spec = spec.at[:, even_indices].set(spec_even)
        
        # Odd modes at indices 1, 3, 5, ...
        odd_indices = jnp.arange(1, 2 * n_odd_modes, 2)
        spec = spec.at[:, odd_indices].set(spec_odd)
        
        return spec
    
    # ========================================================================
    # Fourier Transforms (using rfft/irfft - much simpler!)
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _fourier_inv(self, fourier: jax.Array) -> jax.Array:
        """
        Inverse Fourier transform: Fourier coefficients → grid.
        
        Uses irfft which automatically handles conjugate symmetry for real output.
        Much simpler than manual real/imag packing!
        
        Args:
            fourier: Complex Fourier coefficients [mx, il]
        
        Returns:
            grid: Real grid values [ix, il]
        """
        mx = self.config.mx
        ix = self.config.ix
        
        # irfft expects [n//2+1, ...] for output size n
        full_mx = ix // 2 + 1
        
        # Pad with zeros for truncated high-frequency modes if needed
        if mx < full_mx:
            padded = jnp.zeros((full_mx, fourier.shape[1]), dtype=fourier.dtype)
            padded = padded.at[:mx, :].set(fourier)
        else:
            padded = fourier[:full_mx, :]
        
        # irfft: [ix//2+1, il] complex → [ix, il] real
        # Multiply by ix to match Fortran scaling convention
        grid = jnp.fft.irfft(padded, n=ix, axis=0) * ix
        
        return grid
    
    @partial(jax.jit, static_argnums=(0,))
    def _fourier_dir(self, grid: jax.Array) -> jax.Array:
        """
        Direct Fourier transform: grid → Fourier coefficients.
        
        Uses rfft which produces complex output from real input.
        Much simpler than manual real/imag packing!
        
        Args:
            grid: Real grid values [ix, il]
        
        Returns:
            fourier: Complex Fourier coefficients [mx, il]
        """
        mx = self.config.mx
        ix = self.config.ix
        
        # rfft: [ix, il] real → [ix//2+1, il] complex
        # Divide by ix to match Fortran scaling convention
        full_fourier = jnp.fft.rfft(grid, axis=0) / ix
        
        # Truncate to mx modes (spectral truncation)
        fourier = full_fourier[:mx, :]
        
        return fourier
    
    def _setup_legendre_polynomials2(self):
        """
        Compute Legendre polynomials using the original Fortran.
        
        Returns:
            cpol: Legendre polynomials [mx, nx, iy]
            wt: Gaussian weights [iy]
            lat: latitude
        """
        wt, lat, self.zeros = self._setup_gaussian_grid()

        cpol = self._precompute_legendre_matrices()

        return cpol, wt, lat
    
    def _setup_gaussian_grid(self) -> Tuple[jax.Array, jax.Array]:
        """
        Compute Gaussian latitudes and weights using SPEEDY's method.
        
        Args:
            iy: Number of latitudes in one hemisphere
            
        Returns:
            sia_half: sin(latitude) for one hemisphere [iy] 
            weights: Gaussian quadrature weights [iy]
            mu: Gaussian latitudes (sin values) [iy]
        """
        iy = self.config.iy
        n = 2 * iy  # Total number of latitudes
        
        # Arrays for one hemisphere
        zeros = np.zeros(iy)
        weights = np.zeros(iy)
        z1 = 2.0
        # Compute Gaussian latitudes and weights using Newton's method
        for i in range(iy):
            # Initial approximation for latitude
            z = np.cos(np.pi * (i + 1 - 0.25) / (n + 0.5))
            # Store the Gaussian latitude (this is sin(latitude))
            zeros[i] = z # Fortran uses this which is only approximate
            
            # Newton's method iteration to find root of Legendre polynomial
            while np.abs(z - z1) > np.finfo(jnp.float64).eps:
                p1 = 1.0
                p2 = 0.0
                
                # Compute Legendre polynomial using recurrence relation
                for j in range(1, n + 1):
                    p3 = p2
                    p2 = p1
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
                
                # Compute derivative of Legendre polynomial
                pp = n * (z * p1 - p2) / (z**2 - 1.0)
                
                # Newton's method update
                z1 = z
                z = z1 - p1 / pp
            #zeros[i] = z
            # Compute Gaussian weight
            weights[i] = 2.0 / ((1.0 - z**2) * pp**2)
        
        # Convert to latitudes
        lat_rad = np.arcsin(zeros)
        lat_deg = lat_rad * 180.0 / np.pi
        
        return jnp.array(weights), jnp.array(lat_deg), jnp.array(zeros)

    def _precompute_legendre_matrices(self):
        """
        Precompute Legendre polynomial matrices for ALL latitudes.
        """
        #print("  Precomputing Legendre matrices...", end="", flush=True)
        ix = self.config.ix
        iy = self.config.iy
        mx = self.config.mx
        nx = self.config.nx
        trunc = self.config.trunc
        
        # Compute nsh2 for triangular truncation
        # We have modified the truncation here to accomodate the use of ein later
        nsh2 = np.zeros(nx, dtype=int)
        for n in range(nx):
            for m in range(mx):
                if m + n <= trunc + 1 or ix != 4*iy:
                    nsh2[n] += 1

        # First compute epsilon and repsi (needed for recursion)
        epsi = np.zeros((mx + 1, nx + 1))
        repsi = np.zeros((mx + 1, nx + 1))
        for m in range(mx + 1):
            for n in range(nx + 1):
                emm2 = float(m) ** 2
                ell2 = float(n + m) ** 2
                if n == nx or (n == 0 and m == 0):
                    epsi[m, n] = 0.0
                else:
                    epsi[m, n] = np.sqrt((ell2 - emm2) / (4.0 * ell2 - 1.0))
                    if n != 0: repsi[m, n] = 1.0 / epsi[m, n]
        #repsi = np.where(epsi > 1.e-6, 1.0 / epsi, 0.0)
        
        # Storage for polynomials
        cpol = np.zeros((mx, nx, iy))
        # Compute for each latitude using three-term recursion
        for j in range(iy):
            poly = self._get_legendre_poly(j, epsi, repsi)
            for n in range(nx):
                # Truncation
                for m in range(nsh2[n]): 
                    cpol[m, n, j] = poly[m, n]
        return jnp.array(cpol)
    
    def _get_legendre_poly(self, j: int, epsi: np.ndarray, repsi: np.ndarray) -> np.ndarray:
        """
        Compute Legendre polynomials at latitude j.
        """
        mx = self.config.mx
        nx = self.config.nx
        
        x = float(self.zeros[j])  # sin(latitude)
        y = np.sqrt(1.0 - x**2)  # cos(latitude)
        
        small = 1.0e-30
        
        # Compute consq array
        consq = np.zeros(mx)
        for m in range(mx):
            consq[m] = np.sqrt(0.5 * (2.0 * float(m + 1) + 1.0) / float(m + 1))
        
        # Initialize alp array (mx+1 to include m=mx)
        alp = np.zeros((mx + 1, nx))
        
        # Start recursion with N=1 (M=L) diagonal
        alp[0, 0] = np.sqrt(0.5)
        for m in range(1, mx + 1):
            alp[m, 0] = consq[m - 1] * y * alp[m - 1, 0]
        
        # Continue with other elements
        for m in range(mx + 1):
            alp[m, 1] = (x * alp[m, 0]) * repsi[m, 1]
        
        # Recursion for n >= 2
        for n in range(2, nx):
            for m in range(mx + 1):
                alp[m, n] = (x * alp[m, n - 1] - epsi[m, n - 1] * alp[m, n - 2]) * repsi[m, n]
        
        # Zero small values
        alp = np.where(np.abs(alp) <= small, 0.0, alp)
        
        # Pick off required polynomials
        poly = alp[:mx, :nx]
        
        return poly
    
if __name__ == "__main__":
    print("This is the vectorized transform module.")
    print("Import it with: from legendre import LegendreTransform")
