#!/usr/bin/env python3
"""
Surface flux parameterization for SPEEDY.

Computes surface fluxes of momentum, heat, and moisture over land and sea.
Includes skin temperature adjustment from energy balance.

Split into separate methods for land and sea to support:
1. Full computation (land + sea + combine)
2. Sea-only recomputation for anomaly coupling

Based on SPEEDY surface_fluxes.f90
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .util import Utility
from .state import Config, Param
from .constants import Constants
from .vertical import VerticalGrid
from .transformer import Transformer
from .static import StaticFields

class SurfaceFluxes:
    """
    Surface flux parameterization.
    
    Computes:
    - Wind stress (momentum flux)
    - Sensible heat flux
    - Evaporation (latent heat flux)
    - Surface longwave emission
    - Net heat flux into surface
    - Skin temperature from energy balance
    
    Based on SPEEDY surface_fluxes.f90
    """
    
    def __init__(
        self,
        config: Config,
        constants: Constants,
        vertical_grid: VerticalGrid,
        transformer: Transformer,  # For latitude (coa)
        static_fields: StaticFields,  # For phis0
    ):
        """
        Initialize surface flux scheme.
        
        Args:
            config: Model configuration
            constants: Physical constants
            vertical_grid: Vertical grid structure
            transformer: Transformer for lat/lon info
            static_fields: Static fields (phis0)
        """
        self.config = config
        self.constants = constants
        self.vertical_grid = vertical_grid
        self.transformer = transformer
        self.static_fields = static_fields
        
        self._setup()
    
    def _setup(self):
        """Initialize surface flux constants."""
        # Get surface geopotential for orographic drag
        self.phis0 = self.static_fields.get_field('phis0')  # [ix, il]
        
        # Get cosine of latitude for daily cycle correction
        # transformer.cosg is [il]
        self.coa = self.transformer.cosg
        
        # Vertical interpolation weight for near-surface extrapolation
        self.wvi_kx = jnp.array(self.vertical_grid.wvi[self.config.kx - 1])
        
        # Sigma at lowest level
        self.sigl_kx = self.vertical_grid.sigl[self.config.kx - 1]

        # Get static fields
        self.fmask_l = self.static_fields.get_field('fmask_l')
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self,
        param: Param,
        psa: jax.Array,
        ua: jax.Array,
        va: jax.Array,
        ta: jax.Array,
        qa: jax.Array,
        rh: jax.Array,
        phig: jax.Array,
        stl_am: jax.Array,
        soilw_am: jax.Array,
        sst_am: jax.Array,
        ssrd: jax.Array,
        slrd: jax.Array,
        alb_l: jax.Array,
        alb_s: jax.Array,
        snowc: jax.Array,
    ) -> Tuple[jax.Array, ...]:
        """
        Compute surface fluxes for land and sea.
        
        Args:
            psa: Normalized surface pressure [ix, il]
            ua: U-wind [ix, il, kx]
            va: V-wind [ix, il, kx]
            ta: Temperature [ix, il, kx]
            qa: Specific humidity [ix, il, kx] (g/kg)
            phig: Geopotential [ix, il, kx]
            stl_am: Land surface temperature [ix, il] (K)
            soilw_am: Soil water availability [ix, il] (0-1)
            sst_am: Sea surface temperature [ix, il] (K)
            ssrd: Surface downward SW [ix, il] (W/m²)
            slrd: Surface downward LW [ix, il] (W/m²)
            alb_l: Land albedo [ix, il]
            alb_s: Sea albedo [ix, il]
            snowc: Snow cover fraction [ix, il]
            
        Returns:
            Tuple of:
            - ustr: U-stress [ix, il, 3] (land, sea, combined)
            - vstr: V-stress [ix, il, 3]
            - shf: Sensible heat flux [ix, il, 3] (W/m²)
            - evap: Evaporation [ix, il, 3] (g/m²/s)
            - slru: Surface upward LW [ix, il, 3] (W/m²)
            - hfluxn: Net heat flux into surface [ix, il, 2] (land, sea)
            - ts: Surface temperature [ix, il] (K)
            - tskin: Skin temperature [ix, il] (K)
            - u0: Near-surface u-wind [ix, il]
            - v0: Near-surface v-wind [ix, il]
            - t0: Near-surface temperature [ix, il]
            - t1: Extrapolated temperature [ix, il, 2] (land, sea)
            - q1: Extrapolated humidity [ix, il, 2] (land, sea)
            - denvvs: Density * wind speed [ix, il, 3]
        """
        phis0 = self.phis0
        fmask_l = self.fmask_l
        
        # 1. Extrapolate to near-surface
        u0, v0, t1, t2, q1, denvvs_base = self._extrapolate_to_surface(
            param, psa, ua, va, ta, qa, rh, phig, phis0, fmask_l
        )
        
        # 2. Compute land fluxes
        (ustr_l, vstr_l, shf_l, evap_l, slru_l, hfluxn_l, 
         tskin, denvvs_land) = self._compute_land_fluxes(
            param, psa, ua, va, ta, stl_am, soilw_am, ssrd, slrd,
            alb_l, snowc, t1[:, :, 0], t2[:, :, 0], q1[:, :, 0], denvvs_base
        )
        
        # 3. Compute sea fluxes  
        (ustr_s, vstr_s, shf_s, evap_s, slru_s, hfluxn_s,
         denvvs_sea) = self._compute_sea_fluxes(
            param, psa, ua, va, sst_am, ssrd, slrd, alb_s,
            t1[:, :, 1], t2[:, :, 1], q1[:, :, 1], denvvs_base
        )
        
        # 4. Combine land and sea fluxes
        ustr, vstr, shf, evap, slru, ts, tskin_combined, t0 = self._combine_fluxes(
            fmask_l, stl_am, sst_am,
            ustr_l, vstr_l, shf_l, evap_l, slru_l, tskin,
            ustr_s, vstr_s, shf_s, evap_s, slru_s,
            t1
        )
        
        # Stack hfluxn
        hfluxn = jnp.stack([hfluxn_l, hfluxn_s], axis=-1)  # [ix, il, 2]
        
        # Stack denvvs
        denvvs = jnp.stack([denvvs_land, denvvs_sea, denvvs_base], axis=-1)  # [ix, il, 3]
        
        return (ustr, vstr, shf, evap, slru, hfluxn, 
                ts, tskin_combined, u0, v0, t0, t1, q1, denvvs)
    
    @partial(jax.jit, static_argnames=['self'])
    def recompute_sea_fluxes(
        self,
        param: Param,
        psa: jax.Array,
        ssti_om: jax.Array,
        ssrd: jax.Array,
        slrd: jax.Array,
        alb_s: jax.Array,
        t1_sea: jax.Array,
        q1_sea: jax.Array,
        denvvs_sea: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Recompute sea surface fluxes with different SST.
        
        Used for anomaly coupling when sea_coupling_flag > 0.
        Reuses cached t1, q1, denvvs from first call.
        
        Args:
            psa: Normalized surface pressure [ix, il]
            ssti_om: Ocean model SST [ix, il] (K)
            ssrd: Surface downward SW [ix, il] (W/m²)
            slrd: Surface downward LW [ix, il] (W/m²)
            alb_s: Sea albedo [ix, il]
            t1_sea: Extrapolated sea temperature [ix, il] (from first call)
            q1_sea: Extrapolated sea humidity [ix, il] (from first call)
            denvvs_sea: Density*wind for sea [ix, il] (from first call)
            
        Returns:
            Tuple of:
            - shf_s: Sea sensible heat flux [ix, il] (W/m²)
            - evap_s: Sea evaporation [ix, il] (g/m²/s)
            - slru_s: Sea upward LW [ix, il] (W/m²)
            - hfluxn_s: Net heat flux into sea [ix, il] (W/m²)
        """
        cp = self.constants.cp
        alhc = self.constants.alhc
        chs = param.chs
        # Surface emissivity * Stefan-Boltzmann constant
        esbc = param.emisfc * self.constants.sbc
        
        # Sensible heat flux
        shf_s = chs * cp * denvvs_sea * (ssti_om - t1_sea)
        
        # Evaporation
        qsat_sea = Utility.get_qsat(ssti_om, psa, 1.0)
        evap_s = chs * denvvs_sea * (qsat_sea - q1_sea)
        
        # Surface upward LW
        slru_s = esbc * ssti_om**4
        
        # Net heat flux into sea surface
        #hfluxn_s = ssrd * (1.0 - alb_s) + slrd - slru_s + shf_s + alhc * evap_s
        hfluxn_s = ssrd * (1.0 - alb_s) + slrd - slru_s - shf_s - alhc * evap_s
        
        return shf_s, evap_s, slru_s, hfluxn_s
    
    def _extrapolate_to_surface(
        self,
        param: Param,
        psa: jax.Array,
        ua: jax.Array,
        va: jax.Array,
        ta: jax.Array,
        qa: jax.Array,
        rh: jax.Array,
        phig: jax.Array,
        phis0: jax.Array,
        fmask_l: jax.Array
    ) -> Tuple[jax.Array, ...]:
        """
        Extrapolate atmospheric variables to near-surface.
        
        Returns:
            u0, v0: Near-surface wind [ix, il]
            t0_land, t0_sea: Near-surface temp for land/sea [ix, il]
            t1: Extrapolated temp [ix, il, 2] (land, sea)
            t2: Dry adiabatic temp [ix, il, 2]
            q1: Extrapolated humidity [ix, il, 2]
            denvvs_base: Base density*wind [ix, il]
        """
        kx = self.config.kx
        nl1 = kx - 1
        
        p0 = self.constants.p0
        rgas = self.constants.rgas
        cp = self.constants.cp
        rcp = 1.0 / cp
        
        fwind0 = param.fwind0
        ftemp0 = param.ftemp0
        fhum0 = param.fhum0
        vgust = param.vgust
        
        gtemp0 = 1.0 - ftemp0
        ghum0 = 1.0 - fhum0

        # Wind components
        u0 = fwind0 * ua[:, :, kx-1]
        v0 = fwind0 * va[:, :, kx-1]
        
        # Temperature extrapolation
        dt1 = self.wvi_kx * (ta[:, :, kx-1] - ta[:, :, nl1-1])
        
        # Extrapolated using actual lapse rate
        t1_land_raw = ta[:, :, kx-1] + dt1
        t1_sea_raw = t1_land_raw - phis0 * dt1 / (rgas * 288.0 * self.sigl_kx)
        
        # Extrapolated using dry adiabatic lapse rate
        t2_sea = ta[:, :, kx-1] + rcp * phig[:, :, kx-1]
        t2_land = t2_sea - rcp * phis0
        
        # Use extrapolated if dT/dz < 0, else use lowest level
        use_extrap = ta[:, :, kx-1] > ta[:, :, nl1-1]
        t1_land = jnp.where(use_extrap, ftemp0 * t1_land_raw + gtemp0 * t2_land, ta[:, :, kx-1])
        t1_sea = jnp.where(use_extrap, ftemp0 * t1_sea_raw + gtemp0 * t2_sea, ta[:, :, kx-1])
        
        # Stack for output
        t1 = jnp.stack([t1_land, t1_sea], axis=-1)  # [ix, il, 2]
        t2 = jnp.stack([t2_land, t2_sea], axis=-1)
        
        # Humidity extrapolation
        # For simplicity, using fhum0=0 (constant specific humidity)
        #q1_land = qa[:, :, kx-1]
        #q1_sea = qa[:, :, kx-1]
        # Convert RH to specific humidity at surface temperature
        q1_land, _ = Utility.rel_hum_to_spec_hum(t1_land, psa, 1.0, rh[:, :, kx-1])
        q1_sea, _ = Utility.rel_hum_to_spec_hum(t1_sea, psa, 1.0, rh[:, :, kx-1])
        q1_land = fhum0 * q1_land + ghum0 * qa[:, :, kx-1]
        q1_sea = fhum0 * q1_sea + ghum0 * qa[:, :, kx-1]
        q1 = jnp.stack([q1_land, q1_sea], axis=-1)
        
        # Base density * wind speed (including gustiness)
        wind_speed = jnp.sqrt(u0**2 + v0**2 + vgust**2)
        # Use average of land and sea temps for base density
        #t0_avg = 0.5 * (t1_land + t1_sea)
        #denvvs_base = (p0 * psa / (rgas * t0_avg)) * wind_speed
        t0 = t1_sea + fmask_l * (t1_land - t1_sea)
        denvvs_base = (p0 * psa / (rgas * t0)) * wind_speed
        
        return u0, v0, t1, t2, q1, denvvs_base
    
    def _compute_land_fluxes(
        self,
        param: Param,
        psa: jax.Array,
        ua: jax.Array,
        va: jax.Array,
        ta: jax.Array,
        stl_am: jax.Array,
        soilw_am: jax.Array,
        ssrd: jax.Array,
        slrd: jax.Array,
        alb_l: jax.Array,
        snowc: jax.Array,
        t1_land: jax.Array,
        t2_land: jax.Array,
        q1_land: jax.Array,
        denvvs_base: jax.Array,
    ) -> Tuple[jax.Array, ...]:
        """
        Compute land surface fluxes.
        
        Returns:
            ustr_l, vstr_l: Wind stress [ix, il]
            shf_l: Sensible heat flux [ix, il]
            evap_l: Evaporation [ix, il]
            slru_l: Upward LW [ix, il]
            hfluxn_l: Net heat flux into land [ix, il]
            tskin: Skin temperature [ix, il]
            denvvs_land: Density*wind with stability [ix, il]
        """
        kx = self.config.kx
        cp = self.constants.cp
        alhc = self.constants.alhc
        
        cdl = param.cdl
        chl = param.chl
        dtheta = param.dtheta
        fstab = param.fstab
        ctday = param.ctday
        clambda = param.clambda
        clambsn = param.clambsn

        # Compute orographic enhancement factor for land drag
        rhdrag = 1.0 / (self.constants.grav * param.hdrag)
        forog = 1.0 + rhdrag * (1.0 - jnp.exp(-jnp.maximum(self.phis0, 0.0) * rhdrag))
        # Surface emissivity * Stefan-Boltzmann constant
        esbc = param.emisfc * self.constants.sbc
        
        # Effective skin temperature with daily cycle correction
        tskin = stl_am + ctday * jnp.sqrt(self.coa)[jnp.newaxis, :] * ssrd * (1.0 - alb_l) * psa
        
        # Stability correction
        rdth = fstab / dtheta
        astab = 0.5  # Asymmetric stability
        
        dth_land = jnp.where(
            tskin > t2_land,
            jnp.minimum(dtheta, tskin - t2_land),
            jnp.maximum(-dtheta, astab * (tskin - t2_land))
        )
        denvvs_land = denvvs_base * (1.0 + dth_land * rdth)
        
        # Wind stress
        cdldv = cdl * denvvs_base * forog
        ustr_l = -cdldv * ua[:, :, kx-1]
        vstr_l = -cdldv * va[:, :, kx-1]
        
        # Sensible heat flux
        chlcp = chl * cp
        shf_l = chlcp * denvvs_land * (tskin - t1_land)
        
        # Evaporation
        qsat_skin = Utility.get_qsat(tskin, psa, 1.0)
        evap_l = chl * denvvs_land * jnp.maximum(0.0, soilw_am * qsat_skin - q1_land)
        
        # Surface LW emission and initial heat flux
        tsk3 = tskin**3
        dslr = 4.0 * esbc * tsk3
        slru_l = esbc * tsk3 * tskin
        hfluxn_l = ssrd * (1.0 - alb_l) + slrd - (slru_l + shf_l + alhc * evap_l)
        
        # Energy balance: redefine skin temperature
        clamb = clambda + snowc * (clambsn - clambda)
        hfluxn_l = hfluxn_l - clamb * (tskin - stl_am)
        
        # Derivative of qsat w.r.t. temperature
        dtskin_test = tskin + 1.0
        qsat_plus = Utility.get_qsat(dtskin_test, psa, 1.0)
        dqsat = jnp.where(evap_l > 0.0, soilw_am * (qsat_plus - qsat_skin), 0.0)
        
        # Skin temperature adjustment
        denom = clamb + dslr + chl * denvvs_land * (cp + alhc * dqsat)
        dtskin = hfluxn_l / denom
        tskin = tskin + dtskin
        
        # Adjust fluxes
        shf_l = shf_l + chlcp * denvvs_land * dtskin
        evap_l = evap_l + chl * denvvs_land * dqsat * dtskin
        slru_l = slru_l + dslr * dtskin
        hfluxn_l = clamb * (tskin - stl_am)
        
        return ustr_l, vstr_l, shf_l, evap_l, slru_l, hfluxn_l, tskin, denvvs_land
    
    def _compute_sea_fluxes(
        self,
        param: Param,
        psa: jax.Array,
        ua: jax.Array,
        va: jax.Array,
        tsea: jax.Array,
        ssrd: jax.Array,
        slrd: jax.Array,
        alb_s: jax.Array,
        t1_sea: jax.Array,
        t2_sea: jax.Array,
        q1_sea: jax.Array,
        denvvs_base: jax.Array,
    ) -> Tuple[jax.Array, ...]:
        """
        Compute sea surface fluxes.
        
        Returns:
            ustr_s, vstr_s: Wind stress [ix, il]
            shf_s: Sensible heat flux [ix, il]
            evap_s: Evaporation [ix, il]
            slru_s: Upward LW [ix, il]
            hfluxn_s: Net heat flux into sea [ix, il]
            denvvs_sea: Density*wind with stability [ix, il]
        """
        kx = self.config.kx
        cp = self.constants.cp
        alhc = self.constants.alhc
        # Surface emissivity * Stefan-Boltzmann constant
        esbc = param.emisfc * self.constants.sbc
        
        cds = param.cds
        chs = param.chs
        dtheta = param.dtheta
        fstab = param.fstab
        
        # Stability correction
        rdth = fstab / dtheta
        astab = 0.5
        
        dth_sea = jnp.where(
            tsea > t2_sea,
            jnp.minimum(dtheta, tsea - t2_sea),
            jnp.maximum(-dtheta, astab * (tsea - t2_sea))
        )
        denvvs_sea = denvvs_base * (1.0 + dth_sea * rdth)
        
        # Wind stress
        cdsdv = cds * denvvs_sea
        ustr_s = -cdsdv * ua[:, :, kx-1]
        vstr_s = -cdsdv * va[:, :, kx-1]
        
        # Sensible heat flux
        shf_s = chs * cp * denvvs_sea * (tsea - t1_sea)
        
        # Evaporation
        qsat_sea = Utility.get_qsat(tsea, psa, 1.0)
        evap_s = chs * denvvs_sea * (qsat_sea - q1_sea)
        
        # Surface LW emission
        slru_s = esbc * tsea**4
        
        # Net heat flux into sea surface
        #hfluxn_s = ssrd * (1.0 - alb_s) + slrd - slru_s + shf_s + alhc * evap_s
        hfluxn_s = ssrd * (1.0 - alb_s) + slrd - slru_s - shf_s - alhc * evap_s
        
        return ustr_s, vstr_s, shf_s, evap_s, slru_s, hfluxn_s, denvvs_sea
    
    def _combine_fluxes(
        self,
        fmask_l: jax.Array,
        stl_am: jax.Array,
        sst_am: jax.Array,
        ustr_l: jax.Array,
        vstr_l: jax.Array,
        shf_l: jax.Array,
        evap_l: jax.Array,
        slru_l: jax.Array,
        tskin_l: jax.Array,
        ustr_s: jax.Array,
        vstr_s: jax.Array,
        shf_s: jax.Array,
        evap_s: jax.Array,
        slru_s: jax.Array,
        t1: jax.Array,
    ) -> Tuple[jax.Array, ...]:
        """
        Combine land and sea fluxes using land fraction.
        
        Returns:
            ustr, vstr, shf, evap, slru: [ix, il, 3]
            ts: Surface temperature [ix, il]
            tskin: Skin temperature [ix, il]
            t0: Near-surface temperature [ix, il]
        """
        # Weighted average
        ustr = jnp.stack([ustr_l, ustr_s, ustr_s + fmask_l * (ustr_l - ustr_s)], axis=-1)
        vstr = jnp.stack([vstr_l, vstr_s, vstr_s + fmask_l * (vstr_l - vstr_s)], axis=-1)
        shf = jnp.stack([shf_l, shf_s, shf_s + fmask_l * (shf_l - shf_s)], axis=-1)
        evap = jnp.stack([evap_l, evap_s, evap_s + fmask_l * (evap_l - evap_s)], axis=-1)
        slru = jnp.stack([slru_l, slru_s, slru_s + fmask_l * (slru_l - slru_s)], axis=-1)
        
        # Surface temperature
        ts = sst_am + fmask_l * (stl_am - sst_am)
        tskin = sst_am + fmask_l * (tskin_l - sst_am)
        
        # Near-surface temperature
        t0 = t1[:, :, 1] + fmask_l * (t1[:, :, 0] - t1[:, :, 1])
        
        return ustr, vstr, shf, evap, slru, ts, tskin, t0
