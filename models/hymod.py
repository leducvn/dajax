#!/usr/bin/env python3
import pathlib, datetime, re
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Optional, Tuple, Union
from datetime import datetime
from dajax.models.base import add_operators, Model
from dajax.models.mixin import ForcedModelMixin

@add_operators
class State(NamedTuple):
	qloss: jax.Array    # Soil moisture deficit
	qslow: jax.Array    # Slow tank state
	qquick: jax.Array   # Quick tank states [nq]
	qout: jax.Array     # outflow
	rain: jax.Array     # Precipitation time series
	pet: jax.Array      # Evaporation time series 
	t: jax.Array        # Current time index

class PhyState(NamedTuple):
	qloss: jax.Array    # Soil moisture deficit
	qslow: jax.Array    # Slow tank state
	qquick: jax.Array   # Quick tank states [nq]
	qout: jax.Array     # outflow
   
@add_operators
class Param(NamedTuple):
	cmax: Union[float, jax.Array]   # Maximum storage capacity
	bexp: Union[float, jax.Array]   # Degree of spatial variability of soil moisture capacity
	alpha: Union[float, jax.Array]  # Quick/slow split
	ks: Union[float, jax.Array]     # Slow tank rate parameter
	kq: Union[float, jax.Array]     # Quick tank rate parameter

class Config(NamedTuple):
	nq: int             # Number of quick flow reservoirs
	initFlow: bool      # Whether to initialize with non-zero flow

class Forcing(NamedTuple):
	date: jax.Array  # Corresponding dates
	rain: jax.Array      # Precipitation
	pet: jax.Array      # Evaporation
	qobs: jax.Array   # Observation

class ForcingConfig(NamedTuple):
	forcing_file: str
	start_date: Optional[int] = None
	end_date: Optional[int] = None

class HyModMixin(ForcedModelMixin[Forcing, ForcingConfig]):
	"""Specific implementation of forcing functionality for HyMod."""

	def load_forcing(self) -> Forcing:
		"""Load HyMod-specific forcing data."""
		dajax_root = pathlib.Path(__file__).parent.parent
		file_path = dajax_root/'data'/'hymod'/pathlib.Path(self.forcing_config.forcing_file)
		# Read file
		dates = []
		rain = []
		pet = []
		qobs = []
		with open(file_path, 'r') as f:
			for line in f:
				if not line.strip(): continue
				# Split and clean line
				fields = re.split(r'\s+', line.strip())
				date = int(fields[0])*10000 + int(fields[1])*100 + int(fields[2])
				dates.append(date)
				rain.append(float(fields[5]))
				pet.append(float(fields[8]))
				qobs.append(float(fields[11]))
		# Convert to arrays
		date = jnp.array(dates)
		rain = jnp.array(rain)
		pet = jnp.array(pet)
		qobs = jnp.array(qobs)
		forcing = Forcing(date=date, rain=rain, pet=pet, qobs=qobs)
		return forcing

	def preprocess_forcing(self, forcing: Forcing) -> Forcing:
		"""Preprocess HyMod forcing data."""
		if self.forcing_config.start_date is not None and self.forcing_config.end_date is not None:
			mask = (forcing.date >= self.forcing_config.start_date) & (forcing.date <= self.forcing_config.end_date)
			forcing = Forcing(date=forcing.date[mask], rain=forcing.rain[mask], pet=forcing.pet[mask], qobs=forcing.qobs[mask])
		return forcing

	def _hargreaves(self, dates, tmin: jax.Array, tmax: jax.Array) -> jax.Array:
		"""Compute potential evapotranspiration using Hargreaves method"""
		Gsc = 367.0
		doy = dates  # Assuming dates are already in day of year format
		# Average temperature
		tavg = 0.5*(tmin+tmax)
		# Compute extraterrestrial radiation
		b = 2*jnp.pi*(doy/365)
		Rav = 1.00011 + 0.034221*jnp.cos(b) + 0.00128*jnp.sin(b) + 0.000719*jnp.cos(2*b) + 0.000077*jnp.sin(2*b)
		Ho = ((Gsc*Rav)*86400)/1e6
		# Compute ETo
		return 0.0023*Ho*(tmax-tmin)**0.5*(tavg+17.8)

class HyMod(Model[State, PhyState, Param, Config], HyModMixin):
	"""HyMod hydrological model implementation"""
	def __init__(self, config: Config, forcing_config: ForcingConfig):
		self.config = config
		self.forcing_config = forcing_config
		# JIT compile core functions
		self._excess = jax.jit(self._make_excess())
		self._linear_reservoir = jax.jit(self._make_linear_reservoir())
		self._simulate = jax.jit(self._make_simulate())
		self._forward = jax.jit(self._make_forward())
	
	@staticmethod
	def default_param() -> Param:
		return Param(cmax=jnp.array(50.), bexp=jnp.array(1.), alpha=jnp.array(0.6), 
			ks=jnp.array(0.25), kq=jnp.array(0.8))
	def create_default_param(self) -> None:
		return None
	
	def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
		keys = jax.random.split(key, 5)
		# noise_scale should depend on parameters here
		return Param(cmax = base.cmax + noise_scale*jax.random.normal(keys[0], shape=(1,)),
						bexp = base.bexp + noise_scale*jax.random.normal(keys[1], shape=(1,)),
						alpha = base.alpha + noise_scale*jax.random.normal(keys[2], shape=(1,)),
						ks = base.ks + noise_scale*jax.random.normal(keys[3], shape=(1,)),
						kq = base.kq + noise_scale*jax.random.normal(keys[4], shape=(1,)),)
	
	def default_state(self, param: Param) -> State:
		"""Create initial state including forcing time series"""
		forcing = self.load_forcing()
		forcing = self.preprocess_forcing(forcing)
		if self.config.initFlow: qslow = 2.3503/(param.ks*22.5)
		else: qslow = 0.0
		return State(qloss=jnp.array([0.0]), qslow=jnp.array([qslow]), qquick=jnp.zeros(self.config.nq), qout=jnp.array([0.0]), 
						rain=forcing.rain, pet=forcing.pet, t=jnp.array(0))

	def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float = 0.01) -> State:
		keys = jax.random.split(key, 3)
		return State(
			qloss=base.qloss*(1+noise_scale*jax.random.normal(keys[0],base.qloss.shape)),
			qslow=base.qslow*(1+noise_scale*jax.random.normal(keys[1],base.qslow.shape)),
			qquick=base.qquick*(1+noise_scale*jax.random.normal(keys[2],base.qquick.shape)),
			qout=base.qout, rain=base.rain, pet=base.pet, t=base.t)
	
	def _make_excess(self):
		def excess(qloss: float, cmax: float, bexp: float, Pval: float, PETval: float) -> Tuple[float, float, float]:
			# Calculate capacity
			ct_prev = cmax * (1-jnp.power(abs(1-((bexp+1)*qloss/cmax)),1/(bexp+1)))
			# Calculate Effective rainfall 1
			ER1 = jnp.maximum(Pval-cmax+ct_prev, 0.0)
			Pval = Pval - ER1
			# Calculate new state
			dummy = jnp.minimum((ct_prev+Pval)/cmax, 1)
			qloss_new = (cmax/(bexp+1)) * (1-jnp.power(abs(1-dummy),bexp+1))
			# Calculate Effective rainfall 2
			ER2 = jnp.maximum(Pval-(qloss_new-qloss), 0)
			# Calculate evaporation
			evap = (1-(((cmax/(bexp+1))-qloss_new)/(cmax/(bexp+1)))) * PETval
			qloss_new = jnp.maximum(qloss_new-evap, 0)
			return ER1, ER2, qloss_new
		return excess
	
	def _make_linear_reservoir(self):
		def linear_reservoir(q: float, inflow: float, R: float) -> Tuple[float, float]:
			qnew = (1-R)*q + (1-R)*inflow
			outflow = (R/(1-R))*qnew
			return qnew, outflow
		return linear_reservoir
	
	def _make_simulate(self):
		def simulate(qstate: PhyState, param: Param, p: float, e: float) -> PhyState:
			# Extract current states
			qloss = qstate.qloss[0]
			qslow = qstate.qslow[0]
			qquick = qstate.qquick
			# Calculate excess precipitation
			ER1, ER2, qloss_new = self._excess(qloss, param.cmax, param.bexp, p, e)
			# Total effective rainfall
			ET = ER1 + ER2
			# Partition between quick and slow
			UQ = param.alpha*ET
			US = (1-param.alpha)*ET
			# Route slow flow
			qslow_new, QS = self._linear_reservoir(qslow, US, param.ks)
			# Route quick flow through series of reservoirs
			inflow = UQ
			def quick_step(i, carry):
				qquick, inflow = carry
				qnew, outflow = self._linear_reservoir(qquick[i], inflow, param.kq)
				qquick = qquick.at[i].set(qnew)
				return qquick, outflow
			qquick_new, outflow = jax.lax.fori_loop(0, self.config.nq, quick_step, (qquick,inflow))
			return PhyState(qloss=jnp.array([qloss_new]), qslow=jnp.array([qslow_new]), qquick=qquick_new, qout=jnp.array([QS+outflow]))
		return simulate
	
	def _make_forward(self):
		def forward(state: State, param: Param) -> State:
			# Get current forcing values
			rain = state.rain[state.t]
			pet = state.pet[state.t]
			# Update dynamic state
			phystate = PhyState(qloss=state.qloss, qslow=state.qslow, qquick=state.qquick, qout=state.qout)
			phystate = self._simulate(phystate, param, rain, pet)
			# Return new state with incremented time
			return State(qloss=phystate.qloss, qslow=phystate.qslow, qquick=phystate.qquick, qout=phystate.qout, 
							rain=state.rain, pet=state.pet, t=state.t+1)
		return forward

	def forward(self, state: State, param: Param) -> State:
		return self._forward(state, param)
	
	@partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
	def integrate(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple[State,State]:
		def step(state, _):
			next_state = self.forward(state, param)
			# Create trajectory state with p,e at the current time index
			traj_state = State(qloss=next_state.qloss, qslow=next_state.qslow, qquick=next_state.qquick, qout=next_state.qout, 
									rain=jnp.array([state.rain[state.t]]), pet=jnp.array([state.pet[state.t]]), t=state.t)
			return next_state, traj_state
			
		if save_freq is None:
			final_state, _ = jax.lax.scan(step, state0, jnp.arange(nstep))
			return final_state, None
		else:
			# Inner loop: integrate for save_freq steps
			def inner_loop(state, _):
				final_state, trajectory = jax.lax.scan(step, state, jnp.arange(save_freq))
				return final_state, trajectory
			# Outer loop: save states every save_freq steps
			final_state, trajectory = jax.lax.scan(inner_loop, state0, jnp.arange(nstep//save_freq))
			return final_state, trajectory

	@partial(jax.jit, static_argnames=['self'])
	def _mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform a single state."""
		return PhyState(qloss=state.qloss, qslow=state.qslow, qquick=state.qquick, qout=state.qout)

	@partial(jax.jit, static_argnames=['self'])
	def mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform state(s) to physical space."""
		expected_shape = self.state_info['qquick']
		actual_shape = state.qquick.shape
		if len(actual_shape) == len(expected_shape): return self._mod2phy(state, param)
		return jax.vmap(self._mod2phy, in_axes=(0,None))(state, param)
	
	@property
	def state_info(self) -> dict[str, Tuple[int, ...]]:
		return {'qloss': (1,), 'qslow': (1,), 'qquick': (self.config.nq,), 'qout': (1,)}
	
	@property
	def grid_info(self) -> dict[str, Union[int, jax.Array]]:
		return {'nq': self.config.nq}
	
def main():
	# Parameters
	nq = 5
	initFlow = True
	start_date=20000101
	end_date=20101231
	forcing_file='input_leafriver.txt'
	config = Config(nq=nq, initFlow=initFlow)
	forcing_config = ForcingConfig(forcing_file=forcing_file, start_date=start_date, end_date=end_date)
	param = Param(cmax=50.,bexp=1., alpha=0.6, ks=0.25, kq=0.8)
	model = HyMod(config=config, forcing_config=forcing_config)
	
	# Initialize state
	key = jax.random.PRNGKey(0)
	state0 = model.default_state(param)
	state0 = model.random_state(key, param, state0, noise_scale=0.01)
	nstep = len(state0.rain)
	
	# Warmup
	_ = model.integrate(state0, param, nstep, 1)
	# Time integration
	from timeit import default_timer as timer
	s = timer()
	_,trajectory = model.integrate(state0, param, nstep, 1)
	jax.block_until_ready(trajectory)
	e = timer()
	print('JAX execution time:', e-s)
	
	# Output
	trajectory = model.mod2phy(trajectory, param)
	from dajax.utils.inout import write_trajectory
	time = jnp.arange(nstep)
	coords = {'time': time, 't': jnp.arange(1), 'x': jnp.arange(1), 'y': jnp.arange(nq)}
	dims = {'qloss': ('time', 't', 'x'), 'qslow': ('time', 't', 'x'), 'qquick': ('time', 't', 'y'), 'qout': ('time', 't', 'x'), }
	write_trajectory( filename='true.nc', state=trajectory, time=time, coords=coords, dims=dims)

if __name__ == "__main__":
	main()

