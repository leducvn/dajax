#!/usr/bin/env python3
import os, pathlib, jax, onnx
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from functools import partial
from typing import NamedTuple, Optional, Tuple, Union
from dajax.models.base import add_operators, Model

@add_operators
class State(NamedTuple):
	pmsl: jax.Array    # Surface pressure
	u10m: jax.Array    # 10m u-wind
	v10m: jax.Array    # 10m v-wind
	t2m: jax.Array     # 2m temperature
	h: jax.Array       # Geopotential height
	qv: jax.Array      # Specific humidity
	t: jax.Array       # Temperature
	u: jax.Array       # U-wind
	v: jax.Array       # V-wind

@add_operators
class PhyState(NamedTuple):
	pmsl: jax.Array    # Surface pressure
	u10m: jax.Array    # 10m u-wind
	v10m: jax.Array    # 10m v-wind
	t2m: jax.Array     # 2m temperature
	h: jax.Array       # Geopotential height
	qv: jax.Array      # Specific humidity
	t: jax.Array       # Temperature
	u: jax.Array       # U-wind
	v: jax.Array       # V-wind

@add_operators
class Param(NamedTuple):
	dummy: int

class Config(NamedTuple):
	model_path: str  # Path to ONNX model file
	data_path: str  # Path to input files
	nlon: int = 1440
	nlat: int = 721
	nlev: int = 13
	nuvar: int = 5
	nsvar: int = 4
	device: str = 'cuda' # Device to run inference on

class Pangu(Model[State, PhyState, Param, Config]):
	def __init__(self, config: Config):
		self.config = config
		self._initialize_onnx_sessions(config.model_path)

	def _initialize_onnx_sessions(self, model_path: str):
		# Initialize ONNX runtime session
		#model = onnx.load(model_path)
		#print(model.graph.input)  # Input specs
		#print(model.graph.output)  # Output specs
		#print(model.graph.node)
		options = ort.SessionOptions()
		options.enable_cpu_mem_arena = False
		options.enable_mem_pattern = False
		options.enable_mem_reuse = False
		options.intra_op_num_threads = 1
		#options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
		cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}
		providers = [('CUDAExecutionProvider', cuda_provider_options)] if self.config.device == 'cuda' else ['CPUExecutionProvider']
		self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

	@staticmethod
	def default_param() -> Param:
		return Param(dummy=0)
	def create_default_param(self) -> None:
		return None

	def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
		"""Parameters are deterministic for this model"""
		return base

	def default_state(self, param: Param) -> State:
		upper = np.load(os.path.join(self.config.data_path, 'input_upper.npy')).astype(np.float32)
		surface = np.load(os.path.join(self.config.data_path, 'input_surface.npy')).astype(np.float32)
		return State(pmsl=surface[0], u10m=surface[1], v10m=surface[2], t2m=surface[3], 
         h=upper[0], qv=upper[1], t=upper[2], u=upper[3], v=upper[4])

	def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float) -> State:
		"""Initialize with random perturbations around base state"""
		keys = jax.random.split(key, 9)
		return State(pmsl = base.pmsl + noise_scale*jax.random.normal(keys[0], base.pmsl.shape),
			u10m = base.u10m + noise_scale*jax.random.normal(keys[1], base.u10m.shape),
			v10m = base.v10m + noise_scale*jax.random.normal(keys[2], base.v10m.shape),
			t2m = base.t2m + noise_scale*jax.random.normal(keys[3], base.t2m.shape),
			h = base.h + noise_scale*jax.random.normal(keys[4], base.h.shape),
			qv = base.qv + noise_scale*jax.random.normal(keys[5], base.qv.shape),
			t = base.t + noise_scale*jax.random.normal(keys[6], base.t.shape),
			u = base.u + noise_scale*jax.random.normal(keys[7], base.u.shape),
			v = base.v + noise_scale*jax.random.normal(keys[8], base.v.shape))

	#@partial(jax.jit, static_argnames=['self'])
	def forward(self, state: State, param: Param) -> State:
		"""Single forward step using ONNX model"""
		#def onnx_step(inputs):
			#upper, surface = inputs
			#upper = np.array(state.upper,dtype=np.float32)
			#surface = np.array(state.surface,dtype=np.float32)
			#upper, surface = self.session.run(None, {'input': upper,'input_surface': surface})
			#return upper, surface
		#out_shapes = (jax.ShapeDtypeStruct(shape=state.upper.shape, dtype=jnp.float32),
						#jax.ShapeDtypeStruct(shape=state.surface.shape, dtype=jnp.float32))
		#upper, surface = jax.pure_callback(onnx_step, out_shapes, (state.upper,state.surface))
		upper = np.stack([state.h, state.qv, state.t, state.u, state.v]).astype(np.float32)
		surface = np.stack([state.pmsl, state.u10m, state.v10m, state.t2m]).astype(np.float32)
		upper, surface = self.session.run(None, {'input': upper,'input_surface': surface})
		return State(pmsl=surface[0], u10m=surface[1], v10m=surface[2], t2m=surface[3],
         h=upper[0], qv=upper[1], t=upper[2], u=upper[3], v=upper[4])

	#@partial(jax.jit, static_argnames=['self', 'nstep', 'save_freq'])
	def integrate(self, state0: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple[State,State]:
		state = state0
		if save_freq is None:
			for _ in range(nstep): state = self.forward(state,param)
			return state, None
		else:
			states = []
			for step in range(nstep):
				state = self.forward(state,param)
				if step%save_freq == 0: states.append(state)
			trajectory = jax.tree_map(lambda *xs: jnp.stack(xs), *states)
			return state, trajectory

	@partial(jax.jit, static_argnames=['self'])
	def _mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform a single state."""
		return PhyState(**state._asdict())

	@partial(jax.jit, static_argnames=['self'])
	def mod2phy(self, state: State, param: Param) -> PhyState:
		"""Transform state(s) to physical space."""
		expected_shape = self.state_info['h']
		actual_shape = state.h.shape
		if len(actual_shape) == len(expected_shape): return self._mod2phy(state, param)
		return jax.vmap(self._mod2phy, in_axes=(0,None))(state, param)

	@property
	def state_info(self) -> dict[str, Tuple[int, ...]]:
		return { 'pmsl': (self.config.nlat, self.config.nlon),
					'u10m': (self.config.nlat, self.config.nlon),
					'v10m': (self.config.nlat, self.config.nlon),
					't2m': (self.config.nlat, self.config.nlon),
					'h': (self.config.nlev, self.config.nlat, self.config.nlon),
					'qv': (self.config.nlev, self.config.nlat, self.config.nlon),
					't': (self.config.nlev, self.config.nlat, self.config.nlon),
					'u': (self.config.nlev, self.config.nlat, self.config.nlon),
					'v': (self.config.nlev, self.config.nlat, self.config.nlon)}

	@property
	def grid_info(self) -> dict[str, Union[int, jax.Array]]:
		return { 'nlev': self.config.nlev,
					'nlat': self.config.nlat,
					'nlon': self.config.nlon}
	
def main():
	# Parameters
	dt = 24*3600.
	tend = 24*3600.
	nstep = int(tend/dt)
	dajax_root = pathlib.Path(__file__).parent.parent
	model_path = dajax_root/'data'/'pangu'/'pangu_weather_24.onnx'
	data_path = dajax_root/'data'/'pangu'
	config = Config(model_path=str(model_path), data_path=str(data_path))
	param = Pangu.default_param()
	model = Pangu(config=config)
	
	# Initialize state
	key = jax.random.PRNGKey(0)
	state0 = model.default_state(param)
	
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
	from dajax.utils.inout import write_trajectory
	time = jnp.arange(nstep)*dt
	lev = jnp.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
	lat = jnp.arange(90, -90-0.25, -0.25)
	lon = jnp.arange(0, 360, 0.25)
	coords = {'time': time, 'lev': lev, 'lat': lat, 'lon': lon}
	dims = {'pmsl': ('time','lat','lon'), 'u10m': ('time','lat','lon'),
				'v10m': ('time','lat','lon'), 't2m': ('time','lat','lon'),
				'h': ('time','lev','lat','lon'), 'qv': ('time','lev','lat','lon'),
				't': ('time','lev','lat','lon'), 'u': ('time','lev','lat','lon'), 'v': ('time','lev','lat','lon')}
	write_trajectory( filename='true.nc', state=trajectory, time=time, coords=coords, dims=dims)

if __name__ == "__main__":
	main()
