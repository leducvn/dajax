#!/usr/bin/env python3
"""
Utility functions for SPEEDY model.

Implements:
- Humidity conversions (specific ↔ relative humidity)
- Boundary condition utilities (field checking and filling)
- Interpolation

Based on SPEEDY Fortran modules:
- humidity.f90
- boundaries.f90
- interpolation.f90
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, NamedTuple

class TimeInfo(NamedTuple):
    """Time information for SPEEDY model physics."""
    step: int          # Timestep counter (purely for tracking)
    year: int          # Year (e.g., 1982)
    month: int         # Month 1-12
    day: int           # Day 1-31
    hour: int          # Hour 0-23
    minute: int        # Minute 0-59
    # Derived quantities for physics (computed from date)
    tyear: float       # Fraction of year elapsed (0-1)
    tmonth: float      # Fraction of month elapsed (0-1)
    imont1: int        # Current month (same as month, kept for Fortran compatibility)
    
    @staticmethod
    def create(year: int = 1982, month: int = 1, day: int = 1,
               hour: int = 0, minute: int = 0, step: int = 0) -> 'TimeInfo':
        """
        Create TimeInfo with automatic computation of derived quantities.
        
        Based on SPEEDY date.f90:initialize_date()
        """
        # 365-day calendar
        days_in_month = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        cumulative_days = jnp.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        
        # Compute derived quantities
        imont1 = month
        tmonth = (day - 0.5) / days_in_month[month - 1]
        tyear = (cumulative_days[month - 1] + day - 0.5) / 365.0
        
        return TimeInfo(
            step=step,
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            tyear=float(tyear),
            tmonth=float(tmonth),
            imont1=imont1
        )
    
    def forward(self, dt: float) -> 'TimeInfo':
        """
        Advance time by one timestep.
        
        Based on SPEEDY date.f90:newdate()
        
        Args:
            dt: Time step in seconds
        
        Returns:
            New TimeInfo with updated datetime
        """
        # Calendar constants
        days_in_month = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        cumulative_days = jnp.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        
        # Compute minute increment
        minutes_per_step = int(dt/60)
        
        # Update step counter
        new_step = self.step + 1
        
        # Update minutes and handle overflow
        new_minute = self.minute + minutes_per_step
        carry_hour = new_minute // 60  # How many hours to carry
        new_minute = new_minute % 60
        
        # Update hours and handle overflow
        new_hour = self.hour + carry_hour
        carry_day = new_hour // 24  # How many days to carry
        new_hour = new_hour % 24
        
        # Update days
        new_day = self.day + carry_day
        new_month = self.month
        new_year = self.year
        
        # Handle day overflow - need to check month-specific day limits
        # Check if it's a leap year
        is_leap_year = (new_year % 4 == 0)
        
        # Get the maximum days in current month
        # If February in leap year, use 29, else use days_in_month
        max_days = jax.lax.cond(
            jnp.logical_and(is_leap_year, new_month == 2),
            lambda: 29,
            lambda: days_in_month[new_month - 1]
        )
        
        # Check if day overflows
        day_overflow = new_day > max_days
        
        # Update day and month if overflow
        new_day = jax.lax.cond(day_overflow, lambda: 1, lambda: new_day)
        
        new_month = jax.lax.cond(day_overflow, lambda: new_month + 1, lambda: new_month)
        
        # Handle month overflow
        month_overflow = new_month > 12
        
        new_month = jax.lax.cond(month_overflow, lambda: 1, lambda: new_month)
        
        new_year = jax.lax.cond(month_overflow, lambda: new_year + 1, lambda: new_year)
        
        # Recompute derived quantities
        imont1 = new_month
        
        # Get days in new month for tmonth calculation
        is_leap_year_new = (new_year % 4 == 0)
        days_in_new_month = jax.lax.cond(
            jnp.logical_and(is_leap_year_new, new_month == 2),
            lambda: 29,
            lambda: days_in_month[new_month - 1]
        )
        
        tmonth = (new_day - 0.5) / days_in_new_month
        
        # For tyear calculation
        # If leap year, use 366 days, else 365
        # Also need to adjust cumulative days if after February in leap year
        days_in_year = jax.lax.cond(is_leap_year_new, lambda: 366, lambda: 365)
        
        # Adjustment for leap year: add 1 day if month > 2
        leap_adjustment = jax.lax.cond(jnp.logical_and(is_leap_year_new, new_month > 2), lambda: 1, lambda: 0)
        tyear = (cumulative_days[new_month - 1] + leap_adjustment + new_day - 0.5) / days_in_year
        
        return TimeInfo(
            step=new_step,
            year=new_year,
            month=new_month,
            day=new_day,
            hour=new_hour,
            minute=new_minute,
            tyear=tyear,
            tmonth=tmonth,
            imont1=imont1
        )

class Utility:
    """
    Utility class providing common functions for SPEEDY model.
    
    All methods are static - no initialization required.
    """
    
    # ========================================================================
    # Humidity Conversions
    # ========================================================================
    
    @staticmethod
    @jax.jit
    def get_qsat(ta: jax.Array, ps: jax.Array, sig: float) -> jax.Array:
        """
        Compute saturation specific humidity.
        Based on SPEEDY humidity.f90:get_qsat
        
        Uses Tetens formula for saturation vapor pressure:
        - For T ≥ 273.16 K (water): e_sat = e0 * exp(c1*(T-T0)/(T-t1))
        - For T < 273.16 K (ice):   e_sat = e0 * exp(c2*(T-T0)/(T-t2))
        
        Args:
            ta: Absolute temperature [ix, il] (K)
            ps: Normalized pressure (p/1000 hPa) [ix, il]
            sig: Sigma level (if sig ≤ 0, use ps[0,0] = const)
            
        Returns:
            qsat: Saturation specific humidity [ix, il] (g/kg)
        """
        # Tetens formula constants
        e0 = 6.108e-3  # Reference vapor pressure (hPa)
        c1 = 17.269    # Water constant
        c2 = 21.875    # Ice constant
        t0 = 273.16    # Triple point (K)
        t1 = 35.86     # Water offset (K)
        t2 = 7.66      # Ice offset (K)
        
        # Compute saturation vapor pressure using Tetens formula
        # Vectorized: use jnp.where for conditional
        e_sat = jnp.where(
            ta >= t0,
            e0 * jnp.exp(c1 * (ta - t0) / (ta - t1)),  # Water
            e0 * jnp.exp(c2 * (ta - t0) / (ta - t2))   # Ice
        )
        
        # Convert to specific humidity: q = 622 * e / (p - 0.378*e)
        def qsat1():
            # Use constant pressure
            qsat = 622.0 * e_sat / (ps[0, 0] - 0.378 * e_sat)
            return qsat
        def qsat2():
            # Use sigma level
            qsat = 622.0 * e_sat / (sig * ps - 0.378 * e_sat)
            return qsat
        negative = sig <= 0.0
        qsat = jax.lax.cond(negative, qsat1, qsat2)
        
        return qsat
    
    @staticmethod
    @jax.jit
    def spec_hum_to_rel_hum(ta: jax.Array, ps: jax.Array, sig: float, qa: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Convert specific humidity to relative humidity.
        Based on SPEEDY humidity.f90:spec_hum_to_rel_hum
        
        Args:
            ta: Absolute temperature [ix, il] (K)
            ps: Normalized pressure (p/1000 hPa) [ix, il]
            sig: Sigma level
            qa: Specific humidity [ix, il] (g/kg)
            
        Returns:
            Tuple of (rh, qsat):
                rh: Relative humidity [ix, il] (dimensionless, 0-1)
                qsat: Saturation specific humidity [ix, il] (g/kg)
        """
        qsat =  Utility.get_qsat(ta, ps, sig)
        rh = qa / qsat
        return rh, qsat
    
    @staticmethod
    @jax.jit
    def rel_hum_to_spec_hum(ta: jax.Array, ps: jax.Array, sig: float, rh: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Convert relative humidity to specific humidity.
        Based on SPEEDY humidity.f90:rel_hum_to_spec_hum
        
        Args:
            ta: Absolute temperature [ix, il] (K)
            ps: Normalized pressure (p/1000 hPa) [ix, il]
            sig: Sigma level
            rh: Relative humidity [ix, il] (dimensionless, 0-1)
            
        Returns:
            Tuple of (qa, qsat):
                qa: Specific humidity [ix, il] (g/kg)
                qsat: Saturation specific humidity [ix, il] (g/kg)
        """
        qsat =  Utility.get_qsat(ta, ps, sig)
        qa = rh * qsat
        return qa, qsat
    
    # ========================================================================
    # Boundary Condition Utilities
    # ========================================================================
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1, ))
    def forchk(fmask: jax.Array, fset: float, field: jax.Array) -> jax.Array:
        """
        Check consistency of surface fields with land-sea mask and set undefined
        values to a constant (to avoid over/underflow).
        Based on SPEEDY boundaries.f90:forchk
        
        VECTORIZED VERSION - fully JIT-compatible
        
        Args:
            fmask: Fractional land-sea mask [ix, il]
            fset: Replacement for undefined values
            field: Input field [ix, il, nf]
            
        Returns:
            field: Output field [ix, il, nf] with corrected values
        """
        # Create masks (broadcast over nf dimension)
        # fmask is [ix, il], need to broadcast to [ix, il, nf]
        land_mask = fmask[:, :, jnp.newaxis] > 0.0  # [ix, il, 1] -> broadcasts to [ix, il, nf]
        
        # Apply correction: where land_mask is False (ocean), set to fset
        # Where land_mask is True, keep original value (even if out of range - matches Fortran)
        field_out = jnp.where(land_mask, field, fset)
        
        # Optional: count faults for each field (for diagnostics)
        # Check for out-of-range values on land
        #out_of_range = (field < fmin) | (field > fmax)  # [ix, il, nf]
        # nfaults = jnp.sum(land_mask & out_of_range, axis=(0, 1))  # [nf]
        
        return field_out
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def fillsf(sf: jax.Array, fmis: float) -> jax.Array:
        """
        Replace missing values in surface fields.
        Based on SPEEDY boundaries.f90:fillsf
        
        VECTORIZED VERSION - uses jax.lax.scan for latitude iteration
        
        Note: It is assumed that non-missing values exist near the Equator.
        
        Args:
            sf: Field to replace missing values in [ix, il]
            fmis: Missing value threshold (values < fmis are considered missing)
            
        Returns:
            sf: Field with missing values filled [ix, il]
        """
        ix, il = sf.shape
        
        # Process each hemisphere separately
        def process_hemisphere(sf_in, hemisphere):
            """Process one hemisphere using scan."""
            if hemisphere == 0:
                # Southern hemisphere: equator to south pole
                j_start = il // 2
                j_indices = jnp.arange(j_start, -1, -1)  # [j_start, j_start-1, ..., 0]
            else:
                # Northern hemisphere: equator to north pole
                j_start = il // 2 + 1
                j_indices = jnp.arange(j_start, il)  # [j_start, j_start+1, ..., il-1]
            
            def scan_latitude(sf_state, j):
                """Process one latitude band."""
                # Get current latitude values
                sf_lat = sf_state[j, :]  # [ix]
                
                # Identify missing values
                is_missing = sf_lat < fmis  # [ix]
                
                # Count missing values
                nmis = jnp.sum(is_missing)
                
                # Compute mean of non-missing values
                # Set missing to 0 for sum, then divide by count of non-missing
                sf_nonmissing = jnp.where(is_missing, 0.0, sf_lat)
                n_valid = ix - nmis
                fmean = jnp.sum(sf_nonmissing) / jnp.maximum(n_valid, 1.0)  # avoid division by zero
                
                # First pass: replace missing with mean
                sf_lat_filled = jnp.where(is_missing, fmean, sf_lat)
                
                # Second pass: average with neighbors (with periodic boundaries)
                # Create extended array with periodic boundaries
                sf_extended = jnp.concatenate([
                    sf_lat_filled[-1:],     # sf[ix-1] wraps to position 0
                    sf_lat_filled,          # sf[0:ix]
                    sf_lat_filled[:1]       # sf[0] wraps to position ix+1
                ])  # [ix+2]
                
                # Compute neighbor average: 0.5 * (sf[i-1] + sf[i+1])
                # For each i in [0, ix), use sf_extended[i] and sf_extended[i+2]
                neighbor_avg = 0.5 * (sf_extended[:-2] + sf_extended[2:])  # [ix]
                
                # Apply neighbor averaging only where values were missing
                sf_lat_final = jnp.where(is_missing, neighbor_avg, sf_lat_filled)
                
                # Update the state
                sf_state = sf_state.at[j, :].set(sf_lat_final)
                
                return sf_state, None
            
            # Scan over latitudes in this hemisphere
            sf_out, _ = jax.lax.scan(scan_latitude, sf_in, j_indices)
            
            return sf_out
        
        # Process southern hemisphere
        sf = process_hemisphere(sf, hemisphere=0)
        
        # Process northern hemisphere
        sf = process_hemisphere(sf, hemisphere=1)
        
        return sf

    # ========================================================================
    # Interpolation Utilities
    # ========================================================================
    
    @staticmethod
    @jax.jit
    def forint(for12: jax.Array, imon: int, tmonth: float) -> jax.Array:
        """
        Linear interpolation of monthly-mean forcing fields.
        Based on SPEEDY interpolation.f90:forint
        
        Interpolates between current month and adjacent month based on
        fractional month position (tmonth).
        
        Args:
            for12: Monthly forcing fields [ix*il, 12] or [ix, il, 12]
            imon: Current month (1-12, Fortran-style indexing)
            tmonth: Fractional position within month (0.0-1.0)
                    0.0 = start of month, 0.5 = mid-month, 1.0 = end
            
        Returns:
            for1: Interpolated field [ix*il] or [ix, il]
        """
        # Determine adjacent month and weight based on tmonth
        # If tmonth <= 0.5: interpolate with previous month
        # If tmonth > 0.5: interpolate with next month
        
        # Convert to 0-based indexing for Python
        imon_idx = imon - 1
        
        # Vectorized month selection and weight computation
        is_first_half = tmonth <= 0.5
        
        # Adjacent month (with wraparound)
        imon2_prev = (imon_idx - 1) % 12  # Previous month
        imon2_next = (imon_idx + 1) % 12  # Next month
        imon2 = jnp.where(is_first_half, imon2_prev, imon2_next)
        
        # Weight for adjacent month
        wmon = jnp.where(is_first_half, 0.5 - tmonth, tmonth - 0.5)
        
        # Linear interpolation (fully vectorized)
        for1 = for12[..., imon_idx] + wmon * (for12[..., imon2] - for12[..., imon_idx])
        
        return for1

    @staticmethod
    @jax.jit
    def forin5(for12: jax.Array, imon: int, tmonth: float) -> jax.Array:
        """
        Nonlinear, mean-conserving interpolation of monthly-mean forcing fields.
        Based on SPEEDY interpolation.f90:forin5
        
        Uses 5-point stencil (2 months before, current, 2 months after) with
        optimized weights that conserve the monthly mean.
        
        Args:
            for12: Monthly forcing fields [ix*il, 12] or [ix, il, 12]
            imon: Current month (1-12, Fortran-style indexing)
            tmonth: Fractional position within month (0.0-1.0)
            
        Returns:
            for1: Interpolated field [ix*il] or [ix, il]
        """
        # Convert to 0-based indexing
        imon_idx = imon - 1
        
        # Compute indices for 5-point stencil (with wraparound)
        im2 = (imon_idx - 2) % 12  # 2 months before
        im1 = (imon_idx - 1) % 12  # 1 month before
        ip1 = (imon_idx + 1) % 12  # 1 month after
        ip2 = (imon_idx + 2) % 12  # 2 months after
        
        # Compute weights for mean-conserving interpolation
        c0 = 1.0 / 12.0
        t0 = c0 * tmonth
        t1 = c0 * (1.0 - tmonth)
        t2 = 0.25 * tmonth * (1.0 - tmonth)
        
        wm2 = -t1 + t2
        wm1 = -c0 + 8.0 * t1 - 6.0 * t2
        w0  = 7.0 * c0 + 10.0 * t2
        wp1 = -c0 + 8.0 * t0 - 6.0 * t2
        wp2 = -t0 + t2
        
        # 5-point weighted interpolation (fully vectorized)
        for1 = (wm2 * for12[..., im2] + 
                wm1 * for12[..., im1] + 
                w0  * for12[..., imon_idx] + 
                wp1 * for12[..., ip1] + 
                wp2 * for12[..., ip2])
        
        return for1