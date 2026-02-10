#!/usr/bin/env python3
"""
Semi-implicit solver for SPEEDY model.

Handles gravity wave terms implicitly to allow larger time steps.
Based on SPEEDY implicit.f90
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple

from .state import Config
from .constants import Constants
from .vertical import VerticalGrid

class ImplicitSolver:
    """
    Semi-implicit solver for gravity wave terms.
    
    Solves the coupled system:
    (I - α²Δt²L) * [div_new, t_new, ps_new] = RHS
    
    where L includes vertical coupling through gravity waves.
    Built once during initialization, then applied at each timestep.
    """
    
    def __init__(self, config: Config, constants: Constants, vertical_grid: VerticalGrid, 
                 dt: float, dmp: jax.Array, dmpd: jax.Array, dmps: jax.Array):
        """
        Build implicit solver matrices.
        
        Args:
            config: Config instance 
            constants: Constants instance 
            vertical_grid: VerticalGrid instance 
        """
        self.config = config
        self.constants = constants
        self.vertical_grid = vertical_grid
        self.dt = dt

        # Compute implicit diffusion coefficients for backward implicit scheme 
        self.dmp1 = 1.0 / (1.0 + dmp * dt)
        self.dmp1d = 1.0 / (1.0 + dmpd * dt)
        self.dmp1s = 1.0 / (1.0 + dmps * dt)

        # Build all matrices
        self._setup_implicit_matrices()
    
    def _setup_implicit_matrices(self):
        """
        Build matrices for semi-implicit time stepping - FULLY VECTORIZED!
        
        Creates matrices for solving:
        (I - α²Δt²L) * div_new = RHS
        
        where L includes vertical coupling through gravity waves.
        """
        mx, nx, kx = self.config.mx, self.config.nx, self.config.kx
        dt = self.dt
        alph = self.config.alph
        rearth = self.constants.rearth
        rgas = self.constants.rgas
        akap = self.constants.akap
        
        # Get reference temperature and sigma coordinates
        tref = self.vertical_grid.tref
        tref1 = self.vertical_grid.tref1
        dhs = self.vertical_grid.dhs
        fsg = self.vertical_grid.fsg
        hsg = self.vertical_grid.hsg
        
        # Time stepping parameter
        xi = dt * alph  # For leapfrog with semi-implicit
        xxi = xi / (rearth * rearth)
        
        # Precompute dhsx = xi * dhs (used in surface pressure tendency)
        dhsx = xi * dhs
        
        # ====================================================================
        # Build coupling matrices (following SPEEDY implicit.f90)
        # FULLY VECTORIZED - NO LOOPS!
        # ====================================================================
        
        # YA: temperature increment due to divergence
        # T(k) = T_ex(k) + YA(k,k') * div(k')
        ya = -akap * np.outer(tref, dhs)  # [kx, kx]
        
        # XA: temperature increment due to log(ps) tendency
        xa = np.zeros((kx, kx))
        
        # Upper diagonal: xa[k, k-1] for k=1..kx-1
        k_range = np.arange(1, kx)
        xa[k_range, k_range-1] = 0.5 * (
            akap * tref[k_range] / fsg[k_range] - 
            (tref[k_range] - tref[k_range-1]) / dhs[k_range]
        )
        
        # Main diagonal: xa[k, k] for k=0..kx-2
        k_range = np.arange(kx-1)
        xa[k_range, k_range] = 0.5 * (
            akap * tref[k_range] / fsg[k_range] - 
            (tref[k_range+1] - tref[k_range]) / dhs[k_range]
        )
        
        # XB: log(ps) increment due to divergence
        dsum = np.cumsum(dhs)  # Cumulative sum
        
        # Create indices for lower triangle
        k_idx = np.arange(kx-1)[:, np.newaxis]  # [kx-1, 1]
        k1_idx = np.arange(kx)[np.newaxis, :]   # [1, kx]
        
        xb = np.zeros((kx, kx))
        xb[k_idx, k1_idx] = dhs[k1_idx] * dsum[k_idx]
        
        # Subtract dhs[k1] where k1 <= k (lower triangle including diagonal)
        mask = k1_idx <= k_idx
        xb[:kx-1, :] = np.where(mask, xb[:kx-1, :] - dhs[np.newaxis, :], xb[:kx-1, :])
        
        # XC: total temperature increment 
        xc = ya + np.dot(xa, xb)
        
        # XD: geopotential increment due to temperature
        xd = np.zeros((kx, kx))
        
        # Upper triangle: xd[k, k1] for k1 > k
        k_grid, k1_grid = np.meshgrid(np.arange(kx), np.arange(kx), indexing='ij')
        upper_mask = k1_grid > k_grid
        xd[upper_mask] = rgas * np.log(hsg[k1_grid[upper_mask]+1] / hsg[k1_grid[upper_mask]])
        
        # Diagonal: xd[k, k]
        xd[np.arange(kx), np.arange(kx)] = rgas * np.log(hsg[np.arange(kx)+1] / fsg)
        
        # XE: geopotential increment due to divergence 
        xe = np.dot(xd, xc)
        
        # Store matrices for use in implicit solver
        self.tref1 = jnp.array(tref1)
        self.dhsx = jnp.array(dhsx)
        #self.xa = jnp.array(xa)
        #self.xb = jnp.array(xb)
        self.xc = jnp.array(xc) * xi  # Pre-multiply by xi
        self.xd = jnp.array(xd)
        #self.xe = jnp.array(xe)
        
        # ====================================================================
        # Build implicit operator matrices for each total wavenumber l
        # FULLY VECTORIZED - NO LOOPS!
        # ====================================================================
        
        # For l=0 (planetary scale), special treatment
        # For l>0, build and invert (I - α²Δt²L) matrix
        
        # This is detected by Opus 5.4
        # xf and xj actually use 0-based index like Python and not 1-based index
        # This can be verified later by seeing the code when xj(:,:,l) is applied
        max_l = mx + nx - 1
        
        # VECTORIZED: compute all l values at once
        l_values = np.arange(max_l)  # [0, 1, 2, ..., max_l-1]
        
        # Horizontal wavenumber squared for all l
        #xxx = (l_values + 1) * (l_values + 2) / (rearth * rearth)  # [max_l] WRONG
        xxx = l_values * (l_values + 1) / (rearth * rearth)  # [max_l]
        
        # Build XF matrix for all l at once using broadcasting
        # Reshape for broadcasting: [kx, kx, 1] and [1, 1, max_l]
        tref_dhs = np.outer(rgas * tref, dhs)[:, :, np.newaxis]  # [kx, kx, 1]
        xe_3d = xe[:, :, np.newaxis]  # [kx, kx, 1]
        xxx_3d = xxx[np.newaxis, np.newaxis, :]  # [1, 1, max_l]
        
        xf = xi * xi * xxx_3d * (tref_dhs - xe_3d)  # [kx, kx, max_l]
        
        # Add identity to diagonal
        xf[np.arange(kx), np.arange(kx), :] += 1.0
        
        # Invert to get XJ - VECTORIZED over l
        xj = np.zeros((kx, kx, max_l))
        
        # Invert all matrices at once
        # Transpose to [max_l, kx, kx] for NumPy's batch inversion
        xf_transposed = np.transpose(xf, (2, 0, 1))  # [max_l, kx, kx]
        xj_transposed = np.linalg.inv(xf_transposed)  # Batch invert all at once
        xj = np.transpose(xj_transposed, (1, 2, 0))   # Back to [kx, kx, max_l]
        
        #self.xf = jnp.array(xf)
        self.xj = jnp.array(xj)
        
        # VECTORIZED: elz = dt * alph * l(l+1) / r² for all (m,n)
        m_grid, n_grid = np.meshgrid(np.arange(mx), np.arange(nx), indexing='ij')
        l_grid = m_grid + n_grid
        elz = l_grid * (l_grid + 1) * xxi
        
        self.elz = jnp.array(elz)
    
    @partial(jax.jit, static_argnames=['self'])
    def apply(self, divdt: jax.Array, tdt: jax.Array, psdt: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Apply implicit corrections to tendencies for gravity wave terms.
        Based on SPEEDY implicit.f90: implicit_terms()
        
        Uses vectorized operations via einsum for efficiency.
        
        Args:
            divdt: Divergence tendency [mx, nx, kx] (complex)
            tdt: Temperature tendency [mx, nx, kx] (complex)
            psdt: Log surface pressure tendency [mx, nx] (complex)
            
        Returns:
            Updated (divdt, tdt, psdt) with implicit corrections
        """
        mx, nx = self.config.mx, self.config.nx
        
        # Step 1: Compute geopotential tendency from temperature
        ye = jnp.einsum('kl,mnl->mnk', self.xd, tdt)
        
        # Add surface pressure contribution: ye[m,n,k] += tref1[k] * psdt[m,n]
        ye = ye + self.tref1[jnp.newaxis, jnp.newaxis, :] * psdt[:, :, jnp.newaxis]
        
        # Step 2: Add gravity wave term to divergence tendency
        yf = divdt + self.elz[:, :, jnp.newaxis] * ye
        
        # Step 3: Apply implicit solver (matrix-vector product for each m,n)
        # Create total wavenumber array
        m_indices = jnp.arange(mx)[:, jnp.newaxis]  # [mx, 1]
        n_indices = jnp.arange(nx)[jnp.newaxis, :]  # [1, nx]
        l_indices = m_indices + n_indices  # [mx, nx] - total wavenumber for each (m,n)
        
        # For each (m,n), apply the corresponding xj[:,:,l] matrix
        # This is a batched matrix-vector product
        # yf is [mx, nx, kx], we want to apply xj[:,:,l[m,n]] to yf[m,n,:]
        # Gather the appropriate xj matrices for each (m,n)
        # xj_gathered[m,n,k,k1] = xj[k,k1,l[m,n]]
        xj_gathered = self.xj[:, :, l_indices]  # [kx, kx, mx, nx]
        xj_gathered = jnp.transpose(xj_gathered, (2, 3, 0, 1))  # [mx, nx, kx, kx]
        
        # Matrix-vector product: divdt_new[m,n,k] = sum_k1 xj_gathered[m,n,k,k1] * yf[m,n,k1]
        divdt_new = jnp.einsum('mnkl,mnl->mnk', xj_gathered, yf)
        
        # Handle l=0 special case: zero out the (0,0,0) mode
        divdt_new = divdt_new.at[0, 0, :].set(0.0)
        
        # Step 4: Update surface pressure tendency
        psdt_new = psdt - jnp.einsum('mnk,k->mn', divdt_new, self.dhsx)
        
        # Step 5: Update temperature tendency
        tdt_new = tdt + jnp.einsum('kl,mnl->mnk', self.xc, divdt_new)
        
        return divdt_new, tdt_new, psdt_new
