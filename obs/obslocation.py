# utils/obslocation.py
import jax
import jax.numpy as jnp
from typing import Protocol, Dict, Union, Optional
from functools import partial
from dajax.models.base import ObsState

class ObsMask(Protocol):
	"""Protocol for defining observation locations for each obs type"""
	mask: jax.Array
	def _generate(self, grid_info: dict) -> jax.Array:
		"""Generate boolean mask based on model grid info"""
		...
	
	@partial(jax.jit, static_argnames=['self'])
	def select(self, field: jax.Array) -> jax.Array:
		"""Apply mask to field"""
		#jax.debug.print('Loc Shape: {x}',  x=field.shape)
		return field[..., self.mask]

class RandomMask(ObsMask):
	def __init__(self, key: jax.random.PRNGKey, fraction: float, grid_info: dict):
		self.fraction = fraction
		self.key = key
		self.mask = self._generate(grid_info)
	
	def _generate(self, grid_info: dict) -> jax.Array:
		if 'nlat' in grid_info and 'nlon' in grid_info:
			# 2D case
			nlat = grid_info['nlat']
			nlon = grid_info['nlon']
			shape = (nlat,nlon)
		elif 'ny' in grid_info and 'nx' in grid_info:
			# 2D case
			ny = grid_info['ny']
			nx = grid_info['nx']
			shape = (ny,nx)
		elif 'nx' in grid_info:
			# 1D case
			nx = grid_info['nx']
			shape = (nx,)
		else:
			raise ValueError("grid_info must contain either 'nx' for 1D or both 'nlat' and 'nlon' for 2D")
		return jax.random.uniform(self.key, shape) < self.fraction
		#nx = grid_info['nx']
		#return jax.random.uniform(self.key, (nx,)) < self.fraction

class RegularMask(ObsMask):
	def __init__(self, start: Union[int,tuple[int,int]], spacing: Union[int,tuple[int,int]], grid_info: dict):
		self.start = start
		self.spacing = spacing
		self.mask = self._generate(grid_info)
	
	def _generate(self, grid_info: dict) -> jax.Array:
		if 'nlat' in grid_info and 'nlon' in grid_info:
			# 2D case
			nlat = grid_info['nlat']
			nlon = grid_info['nlon']
			lat_start, lon_start = self.start
			lat_spacing, lon_spacing = self.spacing
			mask = jnp.zeros((nlat, nlon), dtype=bool)
			lat_indices = jnp.arange(lat_start, nlat, lat_spacing)
			lon_indices = jnp.arange(lon_start, nlon, lon_spacing)
			lat_mesh, lon_mesh = jnp.meshgrid(lat_indices, lon_indices, indexing='ij')
			return mask.at[lat_mesh, lon_mesh].set(True)
		elif 'ny' in grid_info and 'nx' in grid_info:
			# 2D case
			nlat = grid_info['ny']
			nlon = grid_info['nx']
			lat_start, lon_start = self.start
			lat_spacing, lon_spacing = self.spacing
			mask = jnp.zeros((nlat, nlon), dtype=bool)
			lat_indices = jnp.arange(lat_start, nlat, lat_spacing)
			lon_indices = jnp.arange(lon_start, nlon, lon_spacing)
			lat_mesh, lon_mesh = jnp.meshgrid(lat_indices, lon_indices, indexing='ij')
			return mask.at[lat_mesh, lon_mesh].set(True)
		elif 'nx' in grid_info:
			# 1D case
			nx = grid_info['nx']
			mask = jnp.zeros(nx, dtype=bool)
			return mask.at[self.start::self.spacing].set(True)
		else:
			raise ValueError("grid_info must contain either 'nx' for 1D or both 'nlat' and 'nlon' for 2D")

class AreaMask(ObsMask):
	def __init__(self, start: Union[int, tuple[int,int], tuple[float,float]], 
				end: Union[int, tuple[int,int], tuple[float,float]], 
				grid_info: dict, use_latlon: bool = False, select_inside: bool = True,
				levels: Optional[Union[int, list[int], tuple[int,int]]] = None):
		"""
		Create an area mask for selecting observations.
		Args:
			start: Starting point - either single int (1D), tuple of ints (2D indices), 
				or tuple of floats (2D lat/lon when use_latlon=True)
			end: Ending point - same format as start
			grid_info: Dictionary containing grid information
			use_latlon: If True, interpret start/end as (lat, lon) coordinates instead of indices
			select_inside: If True, select points inside the area; if False, select points outside
			levels: Optional level selection:
				- None: select all levels (default behavior)
				- int: select single level
				- list[int]: select specific levels [0, 2, 4]
				- tuple[int,int]: select level range (start_lev, end_lev) inclusive
		"""
		self.start = start
		self.end = end
		self.use_latlon = use_latlon
		self.select_inside = select_inside
		self.levels = levels
		self.mask = self._generate(grid_info)
	
	def _generate(self, grid_info: dict) -> jax.Array:
		# First generate the 2D spatial mask (existing logic)
		if 'nlat' in grid_info and 'nlon' in grid_info:
			nlat = grid_info['nlat']
			nlon = grid_info['nlon']
			
			if self.use_latlon:
				if not ('lat' in grid_info and 'lon' in grid_info):
					raise ValueError("grid_info must contain 'lat' and 'lon' arrays when use_latlon=True")
				lon = grid_info['lon']
				lon = jnp.where(lon >= 180, lon - 360, lon)
				lat = grid_info['lat']
				start_lon, start_lat = self.start
				end_lon, end_lat = self.end
				lon_start_idx = jnp.argmin(jnp.abs(lon - start_lon))
				lon_end_idx = jnp.argmin(jnp.abs(lon - end_lon))
				lat_start_idx = jnp.argmin(jnp.abs(lat - start_lat))
				lat_end_idx = jnp.argmin(jnp.abs(lat - end_lat))
				lat_start_idx, lat_end_idx = jnp.minimum(lat_start_idx, lat_end_idx), jnp.maximum(lat_start_idx, lat_end_idx)
				lon_start_idx, lon_end_idx = jnp.minimum(lon_start_idx, lon_end_idx), jnp.maximum(lon_start_idx, lon_end_idx)
			else:
				lat_start_idx, lon_start_idx = self.start
				lat_end_idx, lon_end_idx = self.end
			
			# Create 2D spatial mask
			mask_2d = jnp.zeros((nlat, nlon), dtype=bool)
			lat_indices = jnp.arange(lat_start_idx, lat_end_idx + 1)
			lon_indices = jnp.arange(lon_start_idx, lon_end_idx + 1)
			lat_mesh, lon_mesh = jnp.meshgrid(lat_indices, lon_indices, indexing='ij')
			mask_2d = mask_2d.at[lat_mesh, lon_mesh].set(True)
			if not self.select_inside: mask_2d = jnp.logical_not(mask_2d)
			
			# Extend to 3D if nlev exists and levels are specified
			if 'nlev' in grid_info and self.levels is not None:
				nlev = grid_info['nlev']
				mask_3d = jnp.zeros((nlev, nlat, nlon), dtype=bool)
				
				# Handle different level selection formats
				if isinstance(self.levels, int):
					# Single level
					mask_3d = mask_3d.at[self.levels, :, :].set(mask_2d)
				elif isinstance(self.levels, (list, jax.Array)):
					# List of specific levels
					for lev in self.levels: mask_3d = mask_3d.at[lev, :, :].set(mask_2d)
				elif isinstance(self.levels, tuple) and len(self.levels) == 2:
					# Range of levels (inclusive)
					start_lev, end_lev = self.levels
					for lev in range(start_lev, end_lev + 1): mask_3d = mask_3d.at[lev, :, :].set(mask_2d)
				else:
					raise ValueError("levels must be int, list[int], or tuple[int,int]")
				return mask_3d
			else:
				# No level specification - return 2D mask
				return mask_2d
			
		elif 'ny' in grid_info and 'nx' in grid_info:
			ny = grid_info['ny']
			nx = grid_info['nx']

			if self.use_latlon:
				raise ValueError("use_latlon=True not supported for grids without 'nlat'/'nlon' specification")
			y_start_idx, x_start_idx = self.start
			y_end_idx, x_end_idx = self.end
			mask_2d = jnp.zeros((ny, nx), dtype=bool)
			y_indices = jnp.arange(y_start_idx, y_end_idx + 1)
			x_indices = jnp.arange(x_start_idx, x_end_idx + 1)
			y_mesh, x_mesh = jnp.meshgrid(y_indices, x_indices, indexing='ij')
			mask_2d = mask_2d.at[y_mesh, x_mesh].set(True)
			if not self.select_inside: mask_2d = jnp.logical_not(mask_2d)
			return mask_2d
			
		elif 'nx' in grid_info:
			if self.use_latlon:
				raise ValueError("use_latlon=True not supported for 1D grids")
			nx = grid_info['nx']
			mask = jnp.zeros(nx, dtype=bool)
			mask = mask.at[self.start:self.end+1].set(True)
			if not self.select_inside: mask = jnp.logical_not(mask)
			return mask
		else:
			raise ValueError("grid_info must contain either 'nx' for 1D, 'ny' and 'nx' for 2D, or 'nlat' and 'nlon' for 2D lat/lon grids")
		
class LandMask(ObsMask):
	def __init__(self, grid_info: dict):
		self.mask = self._generate(grid_info)
	
	def _generate(self, grid_info: dict) -> jax.Array:
		ls = grid_info['landsea_mask']
		return ls >= 0.5

class ObsLocation:
	"""Class for handling observation locations"""
	def __init__(self, masks: Dict[str, ObsMask]):
		self.masks = masks
	
	@partial(jax.jit, static_argnames=['self'])
	def select(self, obs: ObsState) -> ObsState:
		template = type(obs)(**{field: field for field in obs._fields})
		return jax.tree_util.tree_map(
			lambda field_name: (self.masks[field_name].select(getattr(obs,field_name)) if field_name in self.masks else None),
			template)