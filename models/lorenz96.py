#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Optional, Tuple, Union
from dajax.models.base import add_operators, Model

@add_operators
class State(NamedTuple):
   u: jax.Array  # [nx]
@add_operators
class PhyState(NamedTuple):
   u: jax.Array  # [nx]

@add_operators
class ObsState(NamedTuple):
	u: jax.Array
	u2: jax.Array

@add_operators
class Param(NamedTuple):
   F: Union[float, jax.Array]

class Config(NamedTuple):
	nx: int
	dt: float

class ETDCoeffs(NamedTuple):
	eLdt: float
	eLdt2: float
	q: float
	f1: float
	f2: float
	f3: float

class Lorenz96(Model[State, PhyState, Param, Config]):
	"""Lorenz96 model implementation using ETDRK4 timestepping."""

	def __init__(self, config: Config):
		self.config = config
		# Compute coefficients once during initialization
		self._coeffs = self._etdrk4_coeffs(config.dt)
		# JIT compile nonlinear function once during initialization
		self._nonlinear = jax.jit(self._make_nonlinear())
		self._forward = jax.jit(self._make_forward())
	
	@staticmethod
	def _etdrk4_coeffs(dt: float) -> ETDCoeffs:
		Ldt = -dt
		Ldt2 = 0.5*Ldt
		eLdt = jnp.exp(Ldt)
		eLdt2 = jnp.exp(Ldt2)
		# Contour integral points
		nroot = 128
		jroot = jnp.exp(jnp.pi*1j*(jnp.arange(1,nroot/2+1)-0.5)/(nroot/2))
		# Compute coefficients using Cauchy integral
		z = Ldt2 + jroot
		q = jnp.mean((jnp.exp(z)-1)/z).real
		z = Ldt + jroot
		f1 = jnp.mean((jnp.exp(z)*(z**2-3*z+4)-(z+4))/z**3).real
		f2 = jnp.mean((jnp.exp(z)*(z-2)+(z+2))/z**3).real
		f3 = jnp.mean((jnp.exp(z)*(4-z)-(z**2+3*z+4))/z**3).real
		return ETDCoeffs(eLdt, eLdt2, dt*q, dt*f1, dt*f2, dt*f3)
	
	@staticmethod
	def default_param() -> Param:
		return Param(F=jnp.array(8.0))
	def create_default_param(self) -> None:
		return None
	
	def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
		return Param(F=base.F+noise_scale*jax.random.normal(key, shape=(1,)))

	def default_state(self, param: Param) -> State:
		u = jnp.zeros(self.config.nx)
		u = u.at[0].set(0.01)
		return State(u=u)
	
	def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float = 0.1) -> State:
		u = base.u + noise_scale*jax.random.normal(key, base.u.shape)
		return State(u=u)
	
	def _make_nonlinear(self):
		def nonlinear(u: jax.Array, F: jax.Array) -> jax.Array:
			return jnp.roll(u,1)*(jnp.roll(u,-1)-jnp.roll(u,2)) + F
		return nonlinear
	
	def _make_forward(self):
		def forward(state: State, param: Param) -> State:
			u = state.u
			N0 = self._nonlinear(u, param.F)
			ahat = self._coeffs.eLdt2*u + 0.5*self._coeffs.q*N0
			Na = self._nonlinear(ahat, param.F)
			bhat = self._coeffs.eLdt2*u + 0.5*self._coeffs.q*Na
			Nb = self._nonlinear(bhat, param.F)
			chat = self._coeffs.eLdt2*ahat + 0.5*self._coeffs.q*(2*Nb-N0)
			Nc = self._nonlinear(chat, param.F)
			unew = self._coeffs.eLdt*u + self._coeffs.f1*N0 + 2*self._coeffs.f2*(Na+Nb) + self._coeffs.f3*Nc
			return State(unew)
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
			#return State(u=jnp.concatenate([state0.u[None],trajectory.u])), trajectory
			trajectory_with_init = jax.tree_util.tree_map(
				lambda x0, x: jnp.concatenate([x0[None, ...], x], axis=0) if x is not None else None,
				state0, trajectory)
			return final_state, trajectory_with_init
	
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
		result = jax.vmap(self._mod2phy, in_axes=(0,None))(state, param)
		if isinstance(result, PhyState): return result
		return PhyState(*result)

	@partial(jax.jit, static_argnames=['self'])
	def _phy2mod(self, phy_increment: PhyState, ref_state: State, param: Param) -> State:
		"""Convert physical space increment back to model space."""
		return State(u=ref_state.u+phy_increment.u)
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2mod(self, phystate: PhyState, ref_state: State, param: Param) -> State:
		"""
		Convert physical state back to model state.
		Handles both single state and trajectory (vmapped over time).
		"""
		# Check if input is trajectory or single state
		expected_shape = (self.config.nx, )
		actual_shape = phystate.u.shape
		
		if len(actual_shape) == len(expected_shape):
			# Single state
			return self._phy2mod(phystate, ref_state, param)
		else:
			# Trajectory: vmap over time axis
			return jax.vmap(self._phy2mod, in_axes=(0, 0, None))(phystate, ref_state, param)
	
	@property
	def state_info(self) -> dict[str, Tuple[int, ...]]:
		return {'u': (self.config.nx,)}
	@property
	def grid_info(self) -> dict[str, Union[int, jax.Array]]:
		return {'nx': self.config.nx}

def main():
	# Parameters
	nx = 1024
	dt = 0.05
	F = 8.0
	tstart = 50.
	tend = 150.
	nstep = int(tend/dt)
	config = Config(nx=nx, dt=dt)
	param = Param(F=F)
	model = Lorenz96(config=config)
	
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
