# utils/obsoperator.py
import jax
import jax.numpy as jnp
from typing import Protocol, Dict, Tuple, Union
from functools import partial
from dajax.models.base import PhyState, ObsState

class ObsMap(Protocol):
	"""Protocol for mapping physical space to observation space"""
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		"""Map physical state to observation"""
		...

class IdentityMap(ObsMap):
	def __init__(self, field: str):
		self.field = field
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		return getattr(phystate, self.field)

class ExponentialMap(ObsMap):
	def __init__(self, field: str, scale: float):
		self.field = field
		self.scale = scale
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		return self.scale*jnp.exp(getattr(phystate, self.field))

class L1NormMap(ObsMap):
	def __init__(self, fields: Tuple[str, ...]):
		self.fields = fields
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		return sum(jnp.abs(getattr(phystate, f)) for f in self.fields)

class L2NormMap(ObsMap):
	def __init__(self, fields: Tuple[str, ...]):
		self.fields = fields
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		return jnp.sqrt(sum(getattr(phystate, f)**2 for f in self.fields))

class PressureLevelMap(ObsMap):
	"""Map a field from sigma levels to specific pressure level(s).
	Performs log-linear vertical interpolation from model sigma coordinates
	to fixed pressure levels.
	Args:
		field: Name of the 3D field to interpolate (e.g., 't', 'u', 'v', 'q')
		sigma_levels: Model sigma levels, shape (nlev,), ordered top-to-bottom 
							(i.e., sigma[0] is model top, sigma[-1] is near surface)
		target_pressure: Target pressure level(s) in hPa
							Can be float or array of floats for multiple levels
		ps_field: Name of surface pressure field in PhyState (default: 'ps')
		fill_value: Value for invalid points (underground or above model top)
	"""
	def __init__(self, field: str, sigma_levels: jax.Array, target_pressure: Union[float, jax.Array],
					ps_field: str = 'ps', logps: bool = True, fill_value: float = jnp.nan):
		self.field = field
		self.sigma = jnp.asarray(sigma_levels)  # (nlev,)
		self.target_p = jnp.atleast_1d(jnp.asarray(target_pressure))  # (n_targets,)
		self.ps_field = ps_field
		self.logps = logps
		self.fill_value = fill_value
		self.n_targets = len(self.target_p)
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		"""Interpolate field from sigma to pressure levels.
		Handles:
		- Single state: field (nlev, nlat, nlon), ps (nlat, nlon)
		- Trajectory: field (ntime, nlev, nlat, nlon), ps (ntime, nlat, nlon)
		Returns:
			Single target: (nlat, nlon) or (ntime, nlat, nlon)
			Multiple targets: (n_targets, nlat, nlon) or (ntime, n_targets, nlat, nlon)
		"""
		field = getattr(phystate, self.field)
		ps = getattr(phystate, self.ps_field)
		if self.logps: ps = 1000.*jnp.exp(ps)

		# Check if trajectory (has time dimension)
		is_trajectory = ps.ndim == 3  # (ntime, nlat, nlon)
		
		if is_trajectory:
			# Compute p_sigma: (ntime, nlev, nlat, nlon)
			p_sigma = self.sigma[None, :, None, None] * ps[:, None, :, :]
		else:
			# Compute p_sigma: (nlev, nlat, nlon)
			p_sigma = self.sigma[:, None, None] * ps[None, :, :]
		#jax.debug.print("Pmin: {x}, Pmax: {y}, Pmean: {z}", x=p_sigma.min(), y=p_sigma.max(), z=p_sigma.mean())
		# Interpolate to target pressure(s)
		result = self._interp_to_pressure(field, p_sigma, is_trajectory)
		
		# Squeeze if single target level
		if self.n_targets == 1:
			if is_trajectory: result = result[:, 0, :, :]  # (ntime, nlat, nlon)
			else: result = result[0, :, :]  # (nlat, nlon)
		return result
	
	def _interp_to_pressure(self, field: jax.Array, p_sigma: jax.Array, is_trajectory: bool) -> jax.Array:
		"""Log-linear interpolation in pressure.
		Args:
			field: (nlev, nlat, nlon) or (ntime, nlev, nlat, nlon)
			p_sigma: (nlev, nlat, nlon) or (ntime, nlev, nlat, nlon)
			is_trajectory: whether input has time dimension
		Returns:
			(n_targets, nlat, nlon) or (ntime, n_targets, nlat, nlon)
		"""
		log_p_sigma = jnp.log(p_sigma)
		log_target_p = jnp.log(self.target_p)
		
		def interp_column(log_p_col, field_col):
			"""Interpolate single column to all target levels."""
			return jnp.interp(log_target_p, log_p_col, field_col, left=self.fill_value, right=self.fill_value)
		
		if is_trajectory:
			# (ntime, nlev, nlat, nlon) -> (ntime, nlat, nlon, nlev)
			log_p_cols = log_p_sigma.transpose(0, 2, 3, 1)
			field_cols = field.transpose(0, 2, 3, 1)
			ntime, nlat, nlon, nlev = log_p_cols.shape
			
			# Reshape to (ntime*nlat*nlon, nlev) for vmap
			log_p_flat = log_p_cols.reshape(-1, nlev)
			field_flat = field_cols.reshape(-1, nlev)
			
			# Interpolate all columns
			result_flat = jax.vmap(interp_column)(log_p_flat, field_flat)
			
			# Reshape back: (ntime*nlat*nlon, n_targets) -> (ntime, n_targets, nlat, nlon)
			result = result_flat.reshape(ntime, nlat, nlon, self.n_targets)
			result = result.transpose(0, 3, 1, 2)
		else:
			# (nlev, nlat, nlon) -> (nlat, nlon, nlev)
			log_p_cols = log_p_sigma.transpose(1, 2, 0)
			field_cols = field.transpose(1, 2, 0)
			nlat, nlon, nlev = log_p_cols.shape
			
			# Reshape to (nlat*nlon, nlev) for vmap
			log_p_flat = log_p_cols.reshape(-1, nlev)
			field_flat = field_cols.reshape(-1, nlev)
			
			# Interpolate all columns
			result_flat = jax.vmap(interp_column)(log_p_flat, field_flat)
			
			# Reshape back: (nlat*nlon, n_targets) -> (n_targets, nlat, nlon)
			result = result_flat.reshape(nlat, nlon, self.n_targets)
			result = result.transpose(2, 0, 1)
		
		return result

class PressureLevelNormMap(ObsMap):
	"""Map norm of multiple fields at specific pressure level(s).
	Useful for wind speed observations: sqrt(u² + v²) at pressure levels.
	"""
	def __init__(self, fields: Tuple[str, ...], sigma_levels: jax.Array, target_pressure: Union[float, jax.Array],
					ps_field: str = 'ps', logps: bool = True, fill_value: float = jnp.nan, norm_type: str = 'l2'):
		self.fields = fields
		self.sigma = jnp.asarray(sigma_levels)
		self.target_p = jnp.atleast_1d(jnp.asarray(target_pressure))
		self.ps_field = ps_field
		self.logps = logps
		self.fill_value = fill_value
		self.norm_type = norm_type
		self.n_targets = len(self.target_p)
		
		# Create individual pressure level maps for each field
		self._maps = [PressureLevelMap(f, sigma_levels, target_pressure, 
												ps_field, logps, fill_value) for f in fields]
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> jax.Array:
		# Get each field at pressure level(s)
		field_values = [m.phy2obs(phystate) for m in self._maps]
		
		if self.norm_type == 'l2':
			return jnp.sqrt(sum(f**2 for f in field_values))
		elif self.norm_type == 'l1':
			return sum(jnp.abs(f) for f in field_values)
		else:
			raise ValueError(f"Unknown norm type: {self.norm_type}")

class ObsOperator:
	def __init__(self, mappings: Dict[str, ObsMap], obsstate_class: type):
		self.mappings = mappings
		self.obsstate_class = obsstate_class
	
	@partial(jax.jit, static_argnames=['self'])
	def phy2obs(self, phystate: PhyState) -> ObsState:
		template = self.obsstate_class(**{field: field for field in self.obsstate_class._fields})
		return jax.tree_util.tree_map(
			lambda field_name: self.mappings[field_name].phy2obs(phystate) if field_name in self.mappings else None,
      	template)