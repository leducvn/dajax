# utils/obstime.py
import jax
import jax.numpy as jnp
from typing import Protocol, Dict, Union
from functools import partial
from dajax.models.base import ObsState

class ObsTimeMask(Protocol):
	"""Protocol for defining observation times"""
	mask: jax.Array
	def _generate(self, ntime: int) -> jax.Array:
		"""Generate boolean mask for time dimension"""
		...
	
	@partial(jax.jit, static_argnames=['self'])
	def select(self, field: jax.Array) -> jax.Array:
		"""Apply time mask to field."""
		#jax.debug.print('Time Shape: {x}',  x=field.shape)
		if field.ndim >= 2 and field.shape[0] == self.mask.size:
			#jax.debug.print('Time Shape: {x}, Mean: {y}',  x=field[self.mask, ...].shape, y=field[self.mask, ...].mean())
			return field[self.mask, ...]
		else: return field

class InitialTimeMask(ObsTimeMask):
	"""Select only the initial time (t=0)"""
	def __init__(self, ntime: int):
		self.ntime = ntime
		self.mask = self._generate(ntime)
	
	def _generate(self, ntime: int) -> jax.Array:
		mask = jnp.zeros(ntime, dtype=bool)
		return mask.at[0].set(True)

class DATimeMask(ObsTimeMask):
	"""Select all times except t=0 (default behavior for DA)"""
	def __init__(self, ntime: int):
		self.ntime = ntime
		self.mask = self._generate(ntime)
	
	def _generate(self, ntime: int) -> jax.Array:
		mask = jnp.ones(ntime, dtype=bool)
		return mask.at[0].set(False)

class FinalTimeMask(ObsTimeMask):
	"""Select only the final time"""
	def __init__(self, ntime: int):
		self.ntime = ntime
		self.mask = self._generate(ntime)
	
	def _generate(self, ntime: int) -> jax.Array:
		mask = jnp.zeros(ntime, dtype=bool)
		return mask.at[-1].set(True)

class TimeIndexMask(ObsTimeMask):
	"""Select specific time indices"""
	def __init__(self, indices: Union[int, list[int]], ntime: int):
		self.indices = indices if isinstance(indices, list) else [indices]
		self.ntime = ntime
		self.mask = self._generate(ntime)
	
	def _generate(self, ntime: int) -> jax.Array:
		mask = jnp.zeros(ntime, dtype=bool)
		return mask.at[jnp.array(self.indices)].set(True)

class TimeRangeMask(ObsTimeMask):
	"""Select a range of time indices"""
	def __init__(self, start: int, end: int, ntime: int):
		self.start = start
		self.end = end
		self.ntime = ntime
		self.mask = self._generate(ntime)
	
	def _generate(self, ntime: int) -> jax.Array:
		mask = jnp.zeros(ntime, dtype=bool)
		return mask.at[self.start:self.end+1].set(True)

class RegularTimeMask(ObsTimeMask):
	"""Select times at regular intervals"""
	def __init__(self, start: int, spacing: int, ntime: int):
		self.start = start
		self.spacing = spacing
		self.ntime = ntime
		self.mask = self._generate(ntime)
	
	def _generate(self, ntime: int) -> jax.Array:
		mask = jnp.zeros(ntime, dtype=bool)
		return mask.at[self.start::self.spacing].set(True)

class ObsTime:
	"""Class for handling observation times"""
	def __init__(self, masks: Dict[str, ObsTimeMask]):
		self.masks = masks
	
	@partial(jax.jit, static_argnames=['self'])
	def select(self, obs: ObsState) -> ObsState:
		template = type(obs)(**{field: field for field in obs._fields})
		return jax.tree_util.tree_map(
			lambda field_name: (self.masks[field_name].select(getattr(obs,field_name)) if field_name in self.masks else None),
			template)