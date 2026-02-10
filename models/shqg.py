#!/usr/bin/env python3
import pathlib, jax, os
import jax.numpy as jnp
import numpy as np  # needed for pyshtools interaction
import xarray as xr
import pyshtools as pysh
from functools import partial
from typing import NamedTuple, List, Tuple, Dict, Optional, Union
from dajax.models.base import add_operators, Model

@add_operators
class State(NamedTuple):
   q: jax.Array  # [nlev,2,T+1,T+1]
@add_operators
class PhyState(NamedTuple):
	u: jax.Array # [nlev,nlat,nlon]
	v: jax.Array

@add_operators
class ObsState(NamedTuple):
	u: jax.Array # [nlev,nlat,nlon]
	v: jax.Array
	u2: jax.Array
	v2: jax.Array
	wind: jax.Array

@add_operators
class Param(NamedTuple):
	Hscale: Union[float, jax.Array]    # Scale height
	Hekman: Union[float, jax.Array]    # Ekman layer height
	Wekman: Union[float, jax.Array]    # Ekman coefficient for land
	Tekman: Union[float, jax.Array]    # Ekman timescale (3 days)
	Tselective: Union[float, jax.Array]  # Selective damping timescale
	Tthermal: Union[float, jax.Array]   # Thermal damping timescale (25 days)
	
class Config(NamedTuple):
	T: int
	Tgrid: int
	nlev: int
	dt: float
	Re: float = 6371000.
	Omega: float = 7.292e-05
	Rrossby: List[float] = [700000., 450000.]
	padding: bool = False
	const_file: str = 'data_t63.zarr'

class Transformer:
	def __init__(self, config: Config):
		self.config = config
		self.lat, self.lon = self._setup_grid()
		self.PLM, self.PPLM, self.ALM, self.PW = self._setup_legendre()
		self.laplacian = self._setup_laplacian()

	def _setup_grid(self) -> Tuple[jax.Array, jax.Array]:
		lat, _ = pysh.expand.GLQGridCoord(self.config.Tgrid)
		lat = lat[::-1]
		lon = np.mod(np.linspace(0, 360, 2*self.config.Tgrid+2, endpoint=False)-180, 360)
		return jnp.array(lat), jnp.array(lon)

	def _setup_legendre(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
		PLM = np.zeros((self.config.Tgrid+1, self.config.T+1, self.config.T+1))
		PPLM = np.zeros((self.config.Tgrid+1, self.config.T+1, self.config.T+1))
		ALM = np.zeros((self.config.Tgrid+1, self.config.T+1, self.config.T+1))
		PW = np.zeros((self.config.Tgrid+1, self.config.T+1, self.config.T+1))
		cost, w = pysh.expand.SHGLQ(self.config.Tgrid)
		cosl = np.cos(np.array(self.lat)*np.pi/180)
		for i in range(self.config.Tgrid+1):
			p, a = pysh.legendre.PlmBar_d1(self.config.T, cost[i], cnorm=1, csphase=1)
			for l in range(self.config.T+1):
				for m in range(l+1):
					ind = (l*(l+1))//2 + m
					PLM[i,m,l] = p[ind]
					PPLM[i,m,l] = p[ind]/(cosl[i]*self.config.Re)
					ALM[i,m,l] = a[ind]*cosl[i]/self.config.Re
					PW[i,m,l] = 0.5*p[ind]*w[i]
		return (jnp.array(PLM), jnp.array(PPLM), jnp.array(ALM), jnp.array(PW))

	def _setup_laplacian(self) -> jax.Array:
		l_range = jnp.arange(self.config.T + 1)
		return -l_range*(l_range+1)/self.config.Re**2
		
	@partial(jax.jit, static_argnums=(0,))
	def spectra2grid_generic(self, spec_x: jax.Array, plm: jax.Array) -> jax.Array:
		leg_x = jnp.einsum('...jm,imj->...im', spec_x, plm)
		return self.apply_fft(leg_x)
		
	@partial(jax.jit, static_argnums=(0,))
	def spectra2grid(self, spec_x: jax.Array) -> jax.Array:
		return self.spectra2grid_generic(spec_x, self.PLM)
	
	@partial(jax.jit, static_argnums=(0,))
	def spectra2grid_gradtheta(self, spec_x: jax.Array) -> jax.Array:
		return self.spectra2grid_generic(spec_x, self.ALM)
	
	@partial(jax.jit, static_argnums=(0,))
	def spectra2grid_gradphi(self, spec_x: jax.Array) -> jax.Array:
		range_vals = jnp.arange(self.config.T+1)
		spec_dx_real = -range_vals * spec_x[..., 1, :, :]
		spec_dx_real = jnp.expand_dims(spec_dx_real, axis=-3)
		spec_dx_imag = range_vals * spec_x[..., 0, :, :]
		spec_dx_imag = jnp.expand_dims(spec_dx_imag, axis=-3)
		spec_dx = jnp.concatenate([spec_dx_real, spec_dx_imag], axis=-3)
		return self.spectra2grid_generic(spec_dx, self.PPLM)

	@partial(jax.jit, static_argnums=(0,))
	def grid2spectra(self, x: jax.Array) -> jax.Array:
		leg_x = self.apply_ifft(x)
		return jnp.einsum('...im,iml->...lm', leg_x, self.PW)

	@partial(jax.jit, static_argnums=(0,))
	def apply_fft(self, leg_x: jax.Array) -> jax.Array:
		leg_x = leg_x[..., 0, :, :] + 1j * leg_x[..., 1, :, :]
		padded = jnp.pad(leg_x, 
			[(0,0)]*(leg_x.ndim-1) + [(0, self.config.Tgrid+1-self.config.T)],
			mode='constant', constant_values=0)
		return jnp.fft.hfft(padded, axis=-1)

	@partial(jax.jit, static_argnums=(0,))
	def apply_ifft(self, x: jax.Array) -> jax.Array:
		leg_x = jnp.fft.ihfft(x, axis=-1)[..., :self.config.T+1]
		return jnp.stack([leg_x.real, leg_x.imag], axis=-3)
	
	@partial(jax.jit, static_argnums=(0,))
	def grid_to_vorticity_spec(self, u: jax.Array, v: jax.Array) -> jax.Array:
		"""
		Compute spectral vorticity from grid-point winds.
		Based on empirical testing, the correct formula for SHQG is:
		ζ = gradtheta(u_spec) + (tan φ / a) * u - (1/(a cos φ)) * dv/dλ
		Args:
			u: [nlev, nlat, nlon]
			v: [nlev, nlat, nlon]
		Returns:
			zeta_spec: [nlev, 2, T+1, T+1]
		"""
		nlon = 2 * (self.config.Tgrid + 1)
		lat_rad = self.lat * jnp.pi / 180.0
		cos_lat = jnp.cos(lat_rad)
		tan_lat = jnp.tan(lat_rad)
		a = self.config.Re
		
		# Term 1: gradtheta(grid2spectra(u))
		u_spec = self.grid2spectra(u)
		term1 = self.spectra2grid_gradtheta(u_spec)
		
		# Term 2: (tan φ / a) * u
		term2 = (tan_lat[None, :, None] / a) * u
		
		# Term 3: -(1/(a cos φ)) * dv/dλ
		v_fft = jnp.fft.rfft(v, axis=-1)
		m_fft = jnp.arange(v_fft.shape[-1])
		dv_dlambda = jnp.fft.irfft(1j * m_fft * v_fft, n=nlon, axis=-1)
		term3 = -(1 / (a * cos_lat[None, :, None])) * dv_dlambda
		
		# Combine
		zeta_grid = term1 + term2 + term3
		zeta_spec = self.grid2spectra(zeta_grid)
		return zeta_spec

class StaticFields:
	def __init__(self, transformer: Transformer):
		self.transformer = transformer
		self.config = transformer.config
		self.fields: Dict[str, jax.Array] = {}
		self._read_fields()
	
	def set_field(self, name: str, data: jax.Array) -> None:
		self.fields[name] = data
	
	def get_field(self, name: str) -> jax.Array:
		return self.fields[name]
	
	def _read_fields(self) -> None:
		dajax_root = pathlib.Path(__file__).parent.parent
		file_path = dajax_root/'data'/'shqg'/pathlib.Path(self.config.const_file).name
		if not file_path.exists(): raise FileNotFoundError(f"Constant file not found at {file_path}.")
		ds = xr.open_zarr(file_path).load()
		lon = np.array(self.transformer.lon)
		lat = np.array(self.transformer.lat)
		if self.config.padding:
			# Add padding to latitude dimension
			def _add_padding(x: np.ndarray) -> np.ndarray:
				shape = list(x.shape)
				shape[-2] += 2
				augmented_x = np.zeros(shape)
				mean = np.mean(x[...,0,:], axis=-1)
				mean = np.repeat(mean[...,np.newaxis], shape[-1], axis=-1)
				augmented_x[...,0,:] = mean
				augmented_x[...,1:-1,:] = x
				mean = np.mean(x[...,-1,:], axis=-1)
				mean = np.repeat(mean[...,np.newaxis], shape[-1], axis=-1)
				augmented_x[...,-1,:] = mean
				return augmented_x
			# Apply padding to dataset
			augmented_lat = np.array([-90] + list(ds.lat.to_numpy()) + [90])
			ds = xr.apply_ufunc(_add_padding, ds, input_core_dims=(('lat', 'lon'),),
				output_core_dims=(('augmented_lat', 'lon'),), dask='parallelized'
				).assign_coords(augmented_lat=augmented_lat,).rename(dict(augmented_lat='lat',))
		for var in ('orography', 'land_sea_mask', 'forcing'):
			if var == 'forcing':
				# Scale up for interpolation (a severe error when zarr updated their methods)
				scale_factor = 1e6
				interpolated = (ds[var]*scale_factor).interp(lon=lon, lat=lat, method='cubic')/scale_factor
			else: interpolated = ds[var].interp(lon=lon, lat=lat, method='cubic')
			self.set_field(var, jnp.array(interpolated.values)) 
		#print("Forcing min/max:", jnp.min(self.fields['forcing']), jnp.max(self.fields['forcing']))
		#print("Orography min/max:", jnp.min(self.fields['orography']), jnp.max(self.fields['orography']))
		#self.fields['orography'] = jnp.zeros_like(self.fields['orography'])
		#self.fields['land_sea_mask'] = jnp.zeros_like(self.fields['land_sea_mask'])


class Poisson:
	def __init__(self, param: Param, transformer: Transformer, staticFields: StaticFields):
		self.Hscale = param.Hscale
		self.transformer = transformer
		self.staticFields = staticFields
		self.config = transformer.config
		self.spec_qp = self._setup_qplanet(self.Hscale)
		self.coupling, self.coupling_matrix = self._setup_coupling()

	def _setup_qplanet(self, Hscale: jax.Array) -> jax.Array:
		qp = jnp.zeros((2, self.config.T+1, self.config.T+1))
		qp = qp.at[0,1,0].set(2*self.config.Omega/jnp.sqrt(3))
		qp = self.transformer.spectra2grid(qp)
		qp = jnp.repeat(jnp.expand_dims(qp, axis=0), self.config.nlev, axis=0)
		hs = self.staticFields.fields['orography']
		qp = qp.at[-1].multiply(1 + hs/Hscale)
		return self.transformer.grid2spectra(qp)

	def _setup_coupling(self) -> jax.Array:
		coupling_matrix = jnp.zeros((self.config.nlev, self.config.nlev))
		for z, Rrossby in enumerate(self.config.Rrossby):
			coupling = 1/Rrossby**2
			coupling_block = coupling*jnp.array([[1,-1], [-1,1]])
			coupling_matrix = coupling_matrix.at[z:z+2, z:z+2].add(-coupling_block)
		pass
		
		# Build inverse coupling for totalq_to_psi: (∇² + S)⁻¹
		coupling_inv = jnp.zeros((self.config.nlev, self.config.nlev, self.config.T+1))
		# special case for l = 0
		coupling_inv = coupling_inv.at[1:,1:,0].set(jnp.linalg.inv(coupling_matrix[1:,1:]))
		# general case for l > 0
		spectrum = self.transformer.laplacian
		for l in range(1, self.config.T+1):
			matrix = spectrum[l]*jnp.eye(self.config.nlev) + coupling_matrix
			coupling_inv = coupling_inv.at[..., l].set(jnp.linalg.inv(matrix))
		return coupling_inv, coupling_matrix

	@partial(jax.jit, static_argnums=(0,))
	def q_to_totalq(self, spec_q: jax.Array, Hscale: jax.Array) -> jax.Array:
		spec_qp = jnp.where(jnp.array_equal(Hscale, self.Hscale), self.spec_qp, self._setup_qplanet(Hscale))
		return spec_q + spec_qp

	@partial(jax.jit, static_argnums=(0,))
	def totalq_to_psi(self, spec_total_q: jax.Array) -> jax.Array:
		return jnp.einsum('ijl,...jklm->...iklm', self.coupling, spec_total_q)

	@partial(jax.jit, static_argnums=(0,))
	def psi_to_zeta(self, spec_psi: jax.Array) -> jax.Array:
		spec_zeta = jnp.einsum('...lm,l->...lm', spec_psi[...,-1,:,:,:], self.transformer.laplacian)
		return self.transformer.spectra2grid(spec_zeta)
	
	@partial(jax.jit, static_argnums=(0,))
	def psi_to_totalq(self, spec_psi: jax.Array) -> jax.Array:
		"""Forward coupling: totalq = (∇² + S) @ psi"""
		# ∇²ψ term - laplacian depends on l, which is axis -2
		laplacian_term = spec_psi * self.transformer.laplacian[None, None, :, None]
		# S @ ψ term (layer coupling)
		stretching_term = jnp.einsum('ij,...jklm->...iklm', self.coupling_matrix, spec_psi)
		return laplacian_term + stretching_term

class Dissipation:
	def __init__(self, param: Param, transformer: Transformer, staticFields: StaticFields):
		self.Hekman = param.Hekman
		self.Wekman = param.Wekman
		self.Tekman = param.Tekman
		self.transformer = transformer
		self.staticFields = staticFields
		self.config = transformer.config
		self.mu, self.dmu_dphi, self.dmu_dtheta = self._setup_mu(self.Hekman, self.Wekman, self.Tekman)
		self.spectrum = self._setup_spectrum()
		self.coupling = self._setup_coupling()

	def _setup_mu(self, Hekman: jax.Array, Wekman: jax.Array, Tekman: jax.Array) -> jax.Array:
		hs = self.staticFields.fields['orography']
		ls = self.staticFields.fields['land_sea_mask']
		mu = (1 + (1-Wekman)*ls + Wekman*(1-jnp.exp(-jnp.maximum(0,hs/Hekman))))/Tekman
		spec_mu = self.transformer.grid2spectra(mu)
		dmu_dphi = self.transformer.spectra2grid_gradphi(spec_mu)
		dmu_dtheta = self.transformer.spectra2grid_gradtheta(spec_mu)
		return (mu, dmu_dphi, dmu_dtheta)

	def _setup_spectrum(self) -> jax.Array:
		T = self.config.T
		Re = self.config.Re
		spectrum = (Re**2*self.transformer.laplacian/(T*(T+1)))**4
		return spectrum
	
	def _setup_coupling(self) -> jax.Array:
		coupling_matrix = jnp.zeros((self.config.nlev, self.config.nlev))
		for z, Rrossby in enumerate(self.config.Rrossby):
			coupling = 1/Rrossby**2
			coupling_block = coupling*jnp.array([[1,-1], [-1,1]])
			coupling_matrix = coupling_matrix.at[z:z+2, z:z+2].add(-coupling_block)
		return coupling_matrix

	@partial(jax.jit, static_argnums=(0,))
	def ekman(self, zeta: jax.Array, dpsi_dtheta: jax.Array, dpsi_dphi: jax.Array, 
				Hekman: jax.Array, Wekman: jax.Array, Tekman: jax.Array) -> jax.Array:
		same_hekman = jnp.array_equal(Hekman, self.Hekman)
		same_wekman = jnp.array_equal(Wekman, self.Wekman)
		same_tekman = jnp.array_equal(Tekman, self.Tekman)
		use_precomputed = jnp.logical_and(jnp.logical_and(same_hekman, same_wekman), same_tekman)
		def true_fn(_): return self.mu, self.dmu_dphi, self.dmu_dtheta
		def false_fn(_): return self._setup_mu(Hekman, Wekman, Tekman)
		mu, dmu_dphi, dmu_dtheta = jax.lax.cond(use_precomputed, true_fn, false_fn, None)	
		num_levels = dpsi_dtheta.shape[-3]
		ekman_1 = zeta*mu
		ekman_2 = (dmu_dtheta*dpsi_dtheta[...,-1,:,:] + dmu_dphi*dpsi_dphi[...,-1,:,:])
		ekman = jnp.expand_dims(ekman_1+ekman_2, -3)
		rank = len(ekman.shape)
		paddings = [(0,0) for _ in range(rank)]
		paddings[-3] = (num_levels-1,0)
		return jnp.pad(ekman, paddings, mode='constant', constant_values=0)

	@partial(jax.jit, static_argnums=(0,))
	def selective(self, spec_totalq: jax.Array, Tselective: jax.Array) -> jax.Array:
		return jnp.einsum('...lm,l->...lm', spec_totalq, self.spectrum/Tselective)
	
	@partial(jax.jit, static_argnums=(0,))
	def thermal(self, spec_psi: jax.Array, Tthermal: jax.Array) -> jax.Array:
		return jnp.einsum('...jklm,ij->...iklm', spec_psi, self.coupling/Tthermal)

class SHQG(Model[State, PhyState, Param, Config]):
	def __init__(self, config: Config, param: Param):
		self.config = config
		self.transformer = Transformer(config)
		self.staticFields = StaticFields(self.transformer)
		self.poisson = Poisson(param, self.transformer, self.staticFields)
		self.dissipation = Dissipation(param, self.transformer, self.staticFields)
		
		# JIT compile core operations
		self._rhs = jax.jit(self._make_rhs())
		self._forward = jax.jit(self._make_forward())
		self.spectra2grid = jax.jit(self.transformer.spectra2grid)
		self.spectra2grid_gradtheta = jax.jit(self.transformer.spectra2grid_gradtheta)
		self.spectra2grid_gradphi = jax.jit(self.transformer.spectra2grid_gradphi)
		self.grid2spectra = jax.jit(self.transformer.grid2spectra)
		self.q_to_totalq = jax.jit(self.poisson.q_to_totalq)
		self.totalq_to_psi = jax.jit(self.poisson.totalq_to_psi)
		self.psi_to_zeta = jax.jit(self.poisson.psi_to_zeta)
		self.ekman = jax.jit(self.dissipation.ekman)
		self.selective = jax.jit(self.dissipation.selective)
		self.thermal = jax.jit(self.dissipation.thermal)
	
	@staticmethod
	def default_param() -> Param:
		return Param(Hscale=jnp.array(9000.), Hekman=jnp.array(1000.), Wekman=jnp.array(0.5), 
			Tekman=jnp.array(259200.), Tselective=jnp.array(8640.), Tthermal=jnp.array(2160000.))
	def create_default_param(self) -> None:
		return None
	
	def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
		"""Create random parameters around base parameters."""
		keys = jax.random.split(key, 6)
		return Param(Hscale = base.Hscale + noise_scale*jax.random.normal(keys[0], shape=(1,)),
						Hekman = base.Hekman + noise_scale*jax.random.normal(keys[1], shape=(1,)),
						Wekman = base.Wekman + noise_scale*jax.random.normal(keys[2], shape=(1,)),
						Tekman = base.Tekman + noise_scale*jax.random.normal(keys[3], shape=(1,)),
						Tselective = base.Tselective + noise_scale*jax.random.normal(keys[4], shape=(1,)),
						Tthermal = base.Tthermal + noise_scale*jax.random.normal(keys[5], shape=(1,)))
	
	def default_state(self, param: Param) -> State:
		dajax_root = pathlib.Path(__file__).parent.parent
		file_path = dajax_root/'data'/'shqg'/pathlib.Path(f'test_t{self.config.T}_t{self.config.Tgrid}.zarr')
		if not file_path.exists(): raise FileNotFoundError(f"Initial file not found at {file_path}.")
		ds = xr.open_zarr(file_path).load()
		q = jnp.array(ds.isel(ensemble=0).spec_q.to_numpy())
		return State(q=q)
	
	def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float = 1.e-6) -> State:
		q = base.q + noise_scale*jax.random.uniform(key, base.q.shape)
		return State(q=q)
	
	def _make_rhs(self):
		def rhs(q: jax.Array, param: Param) -> jax.Array:
			totalq = self.q_to_totalq(q, param.Hscale)
			psi = self.totalq_to_psi(totalq)
			zeta = self.psi_to_zeta(psi)
			dq_dtheta = self.spectra2grid_gradtheta(q)
			dq_dphi = self.spectra2grid_gradphi(q)
			dpsi_dtheta = self.spectra2grid_gradtheta(psi)
			dpsi_dphi = self.spectra2grid_gradphi(psi)
			jacobian = dq_dphi*dpsi_dtheta - dq_dtheta*dpsi_dphi
			forcing = self.staticFields.fields['forcing']
			dissipation_ekman = self.ekman(zeta, dpsi_dtheta, dpsi_dphi, param.Hekman, param.Wekman, param.Tekman)
			dq = self.grid2spectra(jacobian+forcing-dissipation_ekman)
			dissipation_selective = self.selective(totalq, param.Tselective)
			dissipation_thermal = self.thermal(psi, param.Tthermal)
			return dq - dissipation_selective - dissipation_thermal
		return rhs
	
	def _make_forward(self):
		def forward(state: State, param: Param) -> State:
			q = state.q
			k1 = self._rhs(q, param)
			k2 = self._rhs(q + 0.5*self.config.dt*k1, param)
			k3 = self._rhs(q + 0.5*self.config.dt*k2, param)
			k4 = self._rhs(q + self.config.dt*k3, param)
			qnew = q + (self.config.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
			return State(qnew)
		return forward
	
	def forward(self, state: State, param: Param) -> State:
		return self._forward(state, param)
	
	@partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
	def integrate(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple[State,State]:
		def step(state, _):
			state_new = self._forward(state, param)
			return state_new, state_new
		
		if save_freq is None:
			# Only return final state
			final_state, _ = jax.lax.scan(step, state0, jnp.arange(nstep))
			return final_state, None
		else:
			# Inner loop: integrate for save_freq steps
			def inner_loop(state, _):
				final_state, _ = jax.lax.scan(step, state, jnp.arange(save_freq))
				return final_state, final_state
			# Outer loop: save states every save_freq steps
			final_state, trajectory = jax.lax.scan(inner_loop, state0, jnp.arange(nstep//save_freq))
			#return State(q=jnp.concatenate([state0.q[None],q])), ObsState(dpsi_dtheta,dpsi_dphi)
			trajectory_with_init = jax.tree_util.tree_map(
				lambda x0, x: jnp.concatenate([x0[None, ...], x], axis=0) if x is not None else None,
				state0, trajectory)
			return final_state, trajectory_with_init
	
	@partial(jax.jit, static_argnames=['self', ])
	def _mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform a single time slice."""
		totalq = self.q_to_totalq(state.q, param.Hscale)
		psi = self.totalq_to_psi(totalq)
		dpsi_dtheta = self.spectra2grid_gradtheta(psi)
		dpsi_dphi = self.spectra2grid_gradphi(psi)
		#psi = self.spectra2grid(psi)
		return PhyState(u=dpsi_dtheta, v=dpsi_dphi)
	
	@partial(jax.jit, static_argnames=['self', ])
	def mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform trajectory by vmapping over time slices."""
		# Automatically detects whether input is single state
		expected_shape = self.state_info['q']
		actual_shape = state.q.shape
		if len(actual_shape) == len(expected_shape): return self._mod2phy(state, param)
		return jax.vmap(self._mod2phy, in_axes=(0,None))(state, param)

	@partial(jax.jit, static_argnames=['self'])
	def _phy2mod_abs(self, phystate: PhyState, ref_state: State, param: Param) -> State:
		"""Convert physical state (u, v) back to PV state q."""
		u, v = phystate.u, phystate.v
		
		# Step 1: Compute spectral vorticity from (u, v)
		zeta_spec = self.transformer.grid_to_vorticity_spec(u, v)
		
		# Step 2: Get streamfunction via inverse Laplacian
		# ζ = ∇²ψ, so ψ = ∇⁻²ζ
		# laplacian[l] = -l(l+1)/Re², applied along axis -2 (l axis)
		laplacian = self.transformer.laplacian  # [T+1]
		laplacian_safe = jnp.where(laplacian == 0, 1.0, laplacian)
		# Apply along axis -2 (l), not axis -1 (m)
		psi = zeta_spec / laplacian_safe[None, None, :, None]
		# Zero out l=0 mode
		psi = psi.at[:, :, 0, :].set(0.0)
		
		# Step 3: Get totalq = (∇² + S) @ psi
		totalq = self.poisson.psi_to_totalq(psi)
		
		# Step 4: Recover l=0 mode from reference state
		ref_totalq = self.poisson.q_to_totalq(ref_state.q, param.Hscale)
		totalq = totalq.at[:, :, 0, :].set(ref_totalq[:, :, 0, :])
		
		# Step 5: Subtract planetary PV
		q = totalq - self.poisson.spec_qp
		
		return State(q=q)
	
	@partial(jax.jit, static_argnames=['self'])
	def _phy2mod(self, phy_increment: PhyState, ref_state: State, param: Param) -> State:
		"""Convert physical space increment back to model space.
		q_analysis = q_background + L⁻¹(δu, δv)
		where L⁻¹ is the linear inverse operator (u,v) → q
		Args:
			phy_increment: Analysis increment in physical space (δu, δv)
			ref_state: Background state to add increment to
			param: Model parameters
		Returns:
			Analysis state = ref_state + converted increment
		"""
		du, dv = phy_increment.u, phy_increment.v
		
		# δζ from (δu, δv)
		dzeta_spec = self.transformer.grid_to_vorticity_spec(du, dv)
		
		# δψ = ∇⁻² δζ
		laplacian = self.transformer.laplacian
		laplacian_safe = jnp.where(laplacian == 0, 1.0, laplacian)
		dpsi = dzeta_spec / laplacian_safe[None, None, :, None]
		dpsi = dpsi.at[:, :, 0, :].set(0.0)  # l=0 increment is zero
		
		# δq = (∇² + S) @ δψ
		dq = self.poisson.psi_to_totalq(dpsi)
		
		# Analysis = background + increment
		return State(q=ref_state.q + dq)
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2mod(self, phystate: PhyState, ref_state: State, param: Param) -> State:
		"""
		Convert physical state back to model state.
		Handles both single state and trajectory (vmapped over time).
		"""
		# Check if input is trajectory or single state
		expected_shape = (self.config.nlev, len(self.transformer.lat), len(self.transformer.lon))
		actual_shape = phystate.u.shape
		
		if len(actual_shape) == len(expected_shape):
			# Single state
			return self._phy2mod(phystate, ref_state, param)
		else:
			# Trajectory: vmap over time axis
			return jax.vmap(self._phy2mod, in_axes=(0, 0, None))(phystate, ref_state, param)

	@property
	def state_info(self) -> dict[str, Tuple[int, ...]]:
		return {'q': (self.config.nlev,2,self.config.T+1,self.config.T+1)}
	@property
	def grid_info(self) -> dict[str, Union[int, jax.Array]]:
		return { 'nlon': len(self.transformer.lon), 'nlat': len(self.transformer.lat), 'nlev': self.config.nlev, 
					'lon': self.transformer.lon, 'lat': self.transformer.lat,
					'landsea_mask': self.staticFields.fields['land_sea_mask']}

def main():
	print(f"JAX version: {jax.__version__}")

	# Parameters
	T = 21
	Tgrid = 31
	nlev = 3
	dt = 3600.
	save_dt = 6*3600.
	tstart = 365*24*3600.
	tend = 1*365*24*3600.
	nstep = tend//dt
	save_freq = save_dt//dt
	config = Config(T=T, Tgrid=Tgrid, nlev=nlev, dt=dt)
	param = Param(Hscale=9000., Hekman=1000., Wekman=0.5, Tekman=259200., Tselective=8640., Tthermal=2160000.)
	model = SHQG(config=config, param=param)
	
	# Initialize state
	key = jax.random.PRNGKey(0)
	q0 = model.default_state(param)
	q0 = model.random_state(key, param, q0, noise_scale=1.e-9)
	print(model.state_info, q0.q.shape)
	
	# test_phy2mod_inverse
	#state = model.default_state(param)
	#state,trajectory = model.integrate(state, param, nstep, save_freq)
	#phystate = model.mod2phy(state, param)
	#state_recovered = model.phy2mod(phystate, state, param)
	#error = jnp.max(jnp.abs(state.q - state_recovered.q))
	#print(f"Round-trip error: {error:.2e}")
	#q_mag = jnp.max(jnp.abs(state.q))
	#print(f"q magnitude: {q_mag:.2e}")
	#print(f"Relative error: {error/q_mag:.2e}")
	#state_f = model.default_state(param)
	# Create a small perturbation (simulating DA increment)
	#key = jax.random.PRNGKey(42)
	#dq = 0.01 * state_f.q * jax.random.normal(key, state_f.q.shape)
	#state_a = State(q=state_f.q + dq)
	# Forward: get (u,v) for both states
	#phy_f = model.mod2phy(state_f, param)
	#phy_a = model.mod2phy(state_a, param)
	# Now convert back and check increment
	#state_a_recovered = model.phy2mod(phy_a-phy_f, state_f, param)
	#dq_recovered = state_a_recovered.q - state_f.q
	# Compare increments
	#dq_error = jnp.max(jnp.abs(dq - dq_recovered))
	#dq_mag = jnp.max(jnp.abs(dq))
	#print(f"Increment magnitude: {dq_mag:.2e}")
	#print(f"Increment error: {dq_error:.2e}")
	#print(f"Relative error in increment: {dq_error/dq_mag:.2%}")
	#sys.exit(0)

	# Warmup
	_ = model.integrate(q0, param, save_freq, 1)
	#_ = model.integrate(q0, param, nstep, save_freq)
	# Time integration
	from timeit import default_timer as timer
	s = timer()
	_,trajectory = model.integrate(q0, param, nstep, save_freq)
	jax.block_until_ready(trajectory)
	e = timer()
	print('JAX execution time:', e-s)
	trajectory = model.mod2phy(trajectory, param)
	#for i in range(200): print(i, trajectory.u[i,0].mean(), trajectory.u[i,0].std())
	
	# Output
	from dajax.utils.inout import write_trajectory, write_state
	time = jnp.arange(nstep//save_freq+1)*save_dt/3600.
	lon = np.array(model.transformer.lon)
	lat = np.array(model.transformer.lat)
	invalid = lon >= 180
	lon[invalid] -= 360
	coords = {'time': time, 'lev': jnp.arange(nlev), 'lat': lat, 'lon': lon}
	dims = { 'u': ('time', 'lev', 'lat', 'lon'),
				'v': ('time', 'lev', 'lat', 'lon')}
	write_trajectory(filename='true.nc', state=trajectory, time=time, coords=coords, dims=dims)
	
	hs = np.array(model.staticFields.fields['orography'])
	ls = np.array(model.staticFields.fields['land_sea_mask'])
	state = NamedTuple('StaticState', [('orography', jax.Array), ('land_sea_mask', jax.Array)])(
							orography=hs, land_sea_mask=ls)
	coords={'lat': lat, 'lon': lon}
	dims={'orography': ('lat', 'lon'),
			'land_sea_mask': ('lat', 'lon')}
	write_state(filename='static.nc', state=state, coords=coords, dims=dims)
	print('Orography:', np.min(hs), np.max(hs))

if __name__ == "__main__":
	main()
