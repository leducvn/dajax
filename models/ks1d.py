#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Optional, Tuple, Union
from dajax.models.base import add_operators, Model

@add_operators
class State(NamedTuple):
	u: jax.Array  # [nx]

class PhyState(NamedTuple):
	u: jax.Array  # [nx]

@add_operators
class Param(NamedTuple):
	alpha: Union[float, jax.Array]
	gamma: Union[float, jax.Array]

class Config(NamedTuple):
	nx: int
	Lx: float  # Domain length (typically 32π)
	dt: float

class ETDCoeffs(NamedTuple):
	eLdt: jax.Array
	eLdt2: jax.Array
	q: jax.Array
	f1: jax.Array
	f2: jax.Array
	f3: jax.Array

class KS1D(Model[State, PhyState, Param, Config]):
	"""Kuramoto-Sivashinsky model implementation using ETDRK4 timestepping."""
	
	def __init__(self, config: Config, param: Param):
		self.config = config
		self.alpha = param.alpha
		self.gamma = param.gamma
		self._setup_operators()
		self._coeffs = self._etdrk4_coeffs(self.alpha, self.gamma)
		# JIT compile functions
		self._nonlinear = jax.jit(self._make_nonlinear())
		self._forward = jax.jit(self._make_forward())
	
	@staticmethod
	def default_param() -> Param:
		return Param(alpha=jnp.array(1.0), gamma=jnp.array(1.0))
	def create_default_param(self) -> None:
		return None
	
	def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
		keys = jax.random.split(key, 2)
		return Param(alpha = base.alpha + noise_scale*jax.random.normal(keys[0], shape=(1,)),
						gamma = base.gamma + noise_scale*jax.random.normal(keys[1], shape=(1,)))
	
	def default_state(self, param: Param) -> State:
		u = jnp.zeros(self.config.nx)
		u = u.at[0].set(0.01)
		return State(u=u)
	
	def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float = 0.1) -> State:
		u = base.u + noise_scale*jax.random.normal(key, base.u.shape)
		return State(u=u)

	def _setup_operators(self):
		"""Setup wavenumbers and operators for spectral computation"""
		nx = self.config.nx
		Lx = self.config.Lx
		# Wavenumbers
		k = jnp.concatenate([jnp.arange(nx//2), jnp.array([0.]), jnp.arange(-nx//2+1,0)])
		self.k = k*(2*jnp.pi/Lx)
		# Operator for nonlinear term
		self.g = -0.5j*self.k

	def _etdrk4_coeffs(self, alpha: jax.Array, gamma: jax.Array) -> ETDCoeffs:
		"""Compute ETDRK4 coefficients."""
		dt = self.config.dt
		# Linear operator L = αk² - γk⁴
		L = alpha*self.k**2 - gamma*self.k**4
		Ldt = L*dt
		Ldt2 = 0.5*Ldt
		eLdt = jnp.exp(Ldt)
		eLdt2 = jnp.exp(Ldt2)
		# Contour integral points
		nroot = 128
		jroot = jnp.exp(jnp.pi*1j*(jnp.arange(1,nroot/2+1)-0.5)/(nroot/2))
		# Compute coefficients using Cauchy integral
		Ldt2 = Ldt2.reshape(-1, 1)
		Ldt = Ldt.reshape(-1, 1)
		z = Ldt2 + jroot[None,:]
		q = jnp.mean((jnp.exp(z)-1)/z, axis=1).real
		z = Ldt + jroot[None,:]
		f1 = jnp.mean((jnp.exp(z)*(z**2-3*z+4)-(z+4))/z**3, axis=1).real
		f2 = jnp.mean((jnp.exp(z)*(z-2)+(z+2))/z**3, axis=1).real
		f3 = jnp.mean((jnp.exp(z)*(4-z)-(z**2+3*z+4))/z**3, axis=1).real
		return ETDCoeffs(eLdt, eLdt2, dt*q, dt*f1, dt*f2, dt*f3)
	
	def _make_nonlinear(self):
		def nonlinear(uhat: jax.Array) -> jax.Array:
			u = jnp.fft.ifft(uhat).real
			return self.g*jnp.fft.fft(u**2)
		return nonlinear
	
	def _make_forward(self):
		"""Create forward time-stepping function using ETDRK4 scheme"""
		def forward(state: State, param: Param) -> State:
			use_precomputed = jnp.logical_and(self.alpha == param.alpha, self.gamma == param.gamma)
			def true_fn(_): return self._coeffs
			def false_fn(_): return self._etdrk4_coeffs(param.alpha, param.gamma)
			coeffs = jax.lax.cond(use_precomputed, true_fn, false_fn, None)
			# ETDRK4 stages
			uhat = jnp.fft.fft(state.u)
			N0 = self._nonlinear(uhat)
			ahat = coeffs.eLdt2*uhat + 0.5*coeffs.q*N0
			Na = self._nonlinear(ahat)
			bhat = coeffs.eLdt2*uhat + 0.5*coeffs.q*Na
			Nb = self._nonlinear(bhat)
			chat = coeffs.eLdt2*ahat + 0.5*coeffs.q*(2*Nb-N0)
			Nc = self._nonlinear(chat)
			uhat = coeffs.eLdt*uhat + coeffs.f1*N0 + 2*coeffs.f2*(Na+Nb) + coeffs.f3*Nc
			return State(u=jnp.fft.ifft(uhat).real)
		return forward
	
	def forward(self, state: State, param: Param) -> State:
		return self._forward(state, param)
	
	@partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
	def integrate(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple[State, State]:
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
			return final_state, trajectory
	
	@partial(jax.jit, static_argnames=['self'])
	def _mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform a single state."""
		return PhyState(u=state.u)

	@partial(jax.jit, static_argnames=['self'])
	def mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform state(s) to physical space."""
		expected_shape = self.state_info['u']
		actual_shape = state.u.shape
		if len(actual_shape) == len(expected_shape): return self._mod2phy(state, param)
		return jax.vmap(self._mod2phy, in_axes=(0,None))(state, param)
	
	@property
	def state_info(self) -> dict[str, Tuple[int, ...]]:
		return {'u': (self.config.nx,)}
	
	@property
	def grid_info(self) -> dict[str, Union[int, jax.Array]]:
		return {'nx': self.config.nx, 'Lx': self.config.Lx}
	
def main():
	# Parameters
	nx = 256
	dt = 0.25
	Lx = 32.*jnp.pi
	tstart = 50.
	tend = 150.
	nstep = int(tend/dt)
	config = Config(nx=nx, Lx=Lx, dt=dt)
	param = Param(alpha=1, gamma=1)
	model = KS1D(config=config, param=param)
	
	# Initialize state
	key = jax.random.PRNGKey(0)
	u0 = model.default_state(param)
	u0 = model.random_state(key, param, u0, noise_scale=0.1)
	
	# Warmup
	_ = model.integrate(u0, param, nstep, 1)
	# Time integration
	from timeit import default_timer as timer
	s = timer()
	_,trajectory = model.integrate(u0, param, nstep, 1)
	jax.block_until_ready(trajectory)
	e = timer()
	print('JAX execution time:', e-s)
	
	# Output
	from dajax.utils.inout import write_trajectory
	time = jnp.arange(nstep)*dt
	istart = int(tstart/dt)
	coords = {'time': time[istart:], 'x': jnp.arange(nx)}
	dims = {'u': ('time', 'x')}
	write_trajectory( filename='true.nc', state=State(u=trajectory.u[istart:,:]), 
							time=time[istart:], coords=coords, dims=dims)

if __name__ == "__main__":
	main()
