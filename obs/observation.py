import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Union, Dict
from functools import partial
from dajax.obs.likelihood import Distribution, ObsParam, Likelihood
from dajax.obs.obsoperator import ObsMap, ObsOperator
from dajax.obs.obslocation import ObsMask, ObsLocation
from dajax.obs.obstime import ObsTimeMask, ObsTime
from dajax.models.base import PhyState, ObsState

class ObsSetting(NamedTuple):
   distribution: Distribution 
   mapper: ObsMap
   mask: ObsMask
   timemask: Optional[ObsTimeMask] = None
   biasmin: Optional[Union[float, jax.Array]] = None
   biasmax: Optional[Union[float, jax.Array]] = None
   threshold: Optional[Union[float, jax.Array]] = None

class Observation:
	def __init__(self, settings: Dict[str, ObsSetting], obsstate_class: type):
		self.likelihood = Likelihood({k: v.distribution for k,v in settings.items()})
		self.operator = ObsOperator({k: v.mapper for k,v in settings.items()}, obsstate_class)
		self.location = ObsLocation({k: v.mask for k,v in settings.items()})
		self.time = ObsTime({k: v.timemask for k,v in settings.items()})
		self.biasmin = ObsParam({k: v.biasmin for k,v in settings.items() if v.biasmin is not None})
		self.biasmax = ObsParam({k: v.biasmax for k,v in settings.items() if v.biasmax is not None})
		self.threshold = ObsParam({k: v.threshold for k,v in settings.items() if v.threshold is not None})
		self.obsstate_class = obsstate_class

	@staticmethod
	@jax.jit
	def combine_states(state1: ObsState, state2: ObsState) -> ObsState:
		"""Combine observation states handling None fields."""
		template = type(state1)(**{field: field for field in state1._fields})
		def combine(field_name):
			x = getattr(state1,field_name)
			y = getattr(state2,field_name)
			if x is None and y is None: return None
			if x is None: return y
			if y is None: return x
			return jnp.concatenate([x, y], axis=-1)
		return jax.tree_util.tree_map(combine, template)
	
	@partial(jax.jit, static_argnames=['self'])
	def double_states(self, state: ObsState) -> ObsState:
		"""Double fields that have biasmin attribute, keep others intact."""
		def double(field, var):
			if var in self.biasmin.params:
				return jnp.concatenate([field, field], axis=-1) if field is not None else None
			return field
		return self.obsstate_class(**{var: double(field, var) for var, field in state._asdict().items()})

	@partial(jax.jit, static_argnames=['self'])
	def Hforward(self, phystate: PhyState) -> ObsState:
			obs = self.operator.phy2obs(phystate)
			obs = self.location.select(obs)
			obs = self.time.select(obs)
			return obs

	@partial(jax.jit, static_argnames=['self'])
	def sample(self, key: jax.Array, state: ObsState) -> tuple[jax.Array, ObsState]:
		return self.likelihood.sample(key, state)

	@partial(jax.jit, static_argnames=['self'])
	def scale(self, state: ObsState) -> ObsState:
		return self.likelihood.scale(state)
	
	@partial(jax.jit, static_argnames=['self'])
	def e2isample(self, key: jax.Array, state: ObsState) -> tuple[jax.Array, ObsState, ObsState]:
		return self.likelihood.e2isample(key, state, self.biasmin, self.biasmax)

	@partial(jax.jit, static_argnames=['self'])
	def e2iscale(self, state: ObsState) -> tuple[ObsState, ObsState]:
		return self.likelihood.e2iscale(state, self.biasmin, self.biasmax)

	@partial(jax.jit, static_argnames=['self'])
	def isample(self, key: jax.Array, state: ObsState) -> tuple[jax.Array, ObsState, ObsState]:
		return self.likelihood.isample(key, state, self.threshold)
	
	@partial(jax.jit, static_argnames=['self'])
	def get_valid_mask(self, *obs_states: ObsState) -> ObsState:
		"""Compute validity mask where ALL input states have non-NaN values.
		All inputs should have the same shape (use ensemble mean for ensemble states).
		Returns: ObsState with boolean mask arrays (True = valid for ALL inputs)
		"""
		def compute_field_mask(field_name):
			valid_arrays = []
			for obs in obs_states:
				arr = getattr(obs, field_name)
				if arr is None: continue
				valid_arrays.append(~jnp.isnan(arr))
			if not valid_arrays: return None
			return jnp.all(jnp.stack(valid_arrays), axis=0)
		
		first_obs = obs_states[0]
		return self.obsstate_class(**{f: compute_field_mask(f) for f in first_obs._fields})
	
	@partial(jax.jit, static_argnames=['self'])
	def filter_valid_obs(self, obs: ObsState, mask: ObsState) -> ObsState:
		"""Filter observations keeping only valid (masked) points.
		This does not work because JIT expects fixed shapes for all arrays.
		Args:
			obs: Observation state to filter (can be single or ensemble)
			mask: Boolean mask from get_valid_mask (True = keep)
		Returns:
			Filtered ObsState with reduced observation count
		"""
		def filter_field(arr, m):
			if arr is None or m is None: return None
			# m has shape (n_obs,), arr has shape (..., n_obs)
			# Use boolean indexing on last dimension
			return arr[..., m]
		
		return self.obsstate_class(**{f: filter_field(getattr(obs, f), getattr(mask, f)) for f in obs._fields})
	
	@partial(jax.jit, static_argnames=['self'])
	def mask_invalid_obs(self, obs: ObsState, valid: ObsState, fill_value: float = jnp.nan) -> ObsState:
		"""Mask invalid observations by setting them to 0.
		Args:
			obs: Observation state (can have NaN)
			valid: Boolean mask (True = valid)
		Returns:
			ObsState with invalid values set to 0
		"""
		def mask_field(arr, v):
			if arr is None or v is None: return None
			# v has shape (n_obs,), arr has shape (..., n_obs)
			return jnp.where(v, arr, fill_value)
		return self.obsstate_class(**{f: mask_field(getattr(obs, f), getattr(valid, f)) for f in obs._fields})
	
	@partial(jax.jit, static_argnames=['self'])
	def count_valid(self, mask: ObsState) -> Dict[str, int]:
		"""Count valid observations per field (useful for diagnostics)."""
		def count_field(m):
			if m is None: return 0
			return jnp.sum(m)
		return {f: count_field(getattr(mask, f)) for f in mask._fields}
	