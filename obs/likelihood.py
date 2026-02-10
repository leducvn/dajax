#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from typing import Protocol, Dict, Union
from functools import partial
from dajax.models.base import ObsState

class Distribution(Protocol):
   """Protocol for probability distributions."""
   def sample(self, key: jax.Array, loc: jax.Array) -> jax.Array:
      """Sample from the distribution."""
      ...
   
   def scale(self, loc: jax.Array) -> jax.Array:
      """Get scale parameter matching the shape of loc."""
      ...

   @partial(jax.jit, static_argnames=['self','biasmin','biasmax'])
   def e2isample(self, key: jax.Array, loc: jax.Array, 
               biasmin: Union[float, jax.Array], biasmax: Union[float, jax.Array]) -> tuple[jax.Array, jax.Array]:
      """Generate lower and upper bounds for inequality observations."""
      obs = self.sample(key, loc)
      lower = obs - jnp.broadcast_to(biasmax, loc.shape)
      upper = obs - jnp.broadcast_to(biasmin, loc.shape)
      # For invalid obs: set bounds to 0 (neutral)
      invalid = jnp.isnan(loc)
      lower = jnp.where(invalid, 0.0, lower)
      upper = jnp.where(invalid, 0.0, upper)
      return lower, upper
   
   @partial(jax.jit, static_argnames=['self'])
   def e2iscale(self, loc: jax.Array) -> tuple[jax.Array, jax.Array]:
      """Get scale parameters for inequality observations."""
      scale = self.scale(loc)
      # For invalid obs: set scales to inf (neutral)
      invalid = jnp.isnan(loc)
      lower_scale = jnp.where(invalid, jnp.inf, scale)
      upper_scale = jnp.where(invalid, jnp.inf, -scale)
      return lower_scale, upper_scale  # +1 for lower bound (y>=theta), -1 for upper bound (y<=theta)
   
   @partial(jax.jit, static_argnames=['self','threshold'])
   def isample(self, key: jax.Array, loc: jax.Array, threshold: Union[float, jax.Array]) -> tuple[jax.Array, jax.Array]:
      """Sample observation and determine threshold and scale based on observation value.
      Args:
         threshold: Threshold value
      Returns:
         Tuple of (new_threshold, scale) where:
         - new_threshold has same shape as loc
         - scale has sign depending on whether obs >= threshold
      """
      theta = jnp.broadcast_to(threshold, loc.shape)
      obs = self.sample(key, loc)
      scale = self.scale(loc)
      scale_sign = jnp.where(obs >= theta, 1.0, -1.0)
      scale = scale*scale_sign
      # For invalid obs: set theta to 0 (matching obs=0)
      invalid = jnp.isnan(loc)
      theta = jnp.where(invalid, 0.0, theta)
      scale = jnp.where(invalid, jnp.inf, scale)
      return theta, scale

class Predefined(Distribution):
   """Distribution that always returns obs as given, useful for predefine y."""
   def __init__(self, scale: Union[float, jax.Array], bias: Union[float, jax.Array] = 0.0):
      self.scale_param = jnp.asarray(scale)
      self.bias = jnp.asarray(bias)
   
   @partial(jax.jit, static_argnames=['self'])
   def sample(self, key: jax.Array, loc: jax.Array) -> jax.Array:
      biased_loc = loc + jnp.broadcast_to(self.bias, loc.shape)
      return biased_loc
   
   @partial(jax.jit, static_argnames=['self'])
   def scale(self, loc: jax.Array) -> jax.Array:
      return jnp.broadcast_to(self.scale_param, loc.shape)

class Gaussian(Distribution):
   def __init__(self, scale: Union[float, jax.Array], bias: Union[float, jax.Array] = 0.0):
      self.scale_param = jnp.asarray(scale) #Standard deviation parameter
      self.bias = jnp.asarray(bias)
   
   @partial(jax.jit, static_argnames=['self'])
   def sample(self, key: jax.Array, loc: jax.Array) -> jax.Array:
      biased_loc = loc + jnp.broadcast_to(self.bias, loc.shape)
      return biased_loc + jax.random.normal(key,loc.shape)*self.scale_param
   
   @partial(jax.jit, static_argnames=['self'])
   def scale(self, loc: jax.Array) -> jax.Array:
      return jnp.broadcast_to(self.scale_param, loc.shape)

class Logistic(Distribution):
   def __init__(self, scale: Union[float, jax.Array], bias: Union[float, jax.Array] = 0.0):
      self.scale_param = jnp.asarray(scale)
      self.bias = jnp.asarray(bias)
   
   @partial(jax.jit, static_argnames=['self'])
   def sample(self, key: jax.Array, loc: jax.Array) -> jax.Array:
      u = jax.random.uniform(key, loc.shape)
      biased_loc = loc + jnp.broadcast_to(self.bias, loc.shape)
      return biased_loc + self.scale_param*jnp.log(u/(1-u))
   
   @partial(jax.jit, static_argnames=['self'])
   def scale(self, loc: jax.Array) -> jax.Array:
      return jnp.broadcast_to(self.scale_param, loc.shape)

class MInfinite(Distribution):
   """Distribution that always returns -infinity, useful for y < threshold observations."""
   def __init__(self, scale: Union[float, jax.Array]):
      self.scale_param = jnp.asarray(scale)

   @partial(jax.jit, static_argnames=['self'])
   def sample(self, key: jax.Array, loc: jax.Array) -> jax.Array:
      return jnp.full_like(loc, -jnp.inf)
   
   @partial(jax.jit, static_argnames=['self'])
   def scale(self, loc: jax.Array) -> jax.Array:
      return jnp.broadcast_to(self.scale_param, loc.shape)

class PInfinite(Distribution):
   """Distribution that always returns +infinity, useful for y > threshold observations."""
   def __init__(self, scale: Union[float, jax.Array]):
      self.scale_param = jnp.asarray(scale)

   @partial(jax.jit, static_argnames=['self'])
   def sample(self, key: jax.Array, loc: jax.Array) -> jax.Array:
      return jnp.full_like(loc, jnp.inf)
   
   @partial(jax.jit, static_argnames=['self'])
   def scale(self, loc: jax.Array) -> jax.Array:
      return jnp.broadcast_to(self.scale_param, loc.shape)

class ObsParam:
	"""Class for handling observation parameters"""
	def __init__(self, params: Dict[str, Union[int, jax.Array]]):
		self.params = params
   
	@partial(jax.jit, static_argnames=['self'])
	def get(self, field: str) -> Union[int, jax.Array]:
		"""Get parameters"""
		return self.params[field]

class Likelihood:
   """Class for handling observation likelihood model p(y|x)."""
   def __init__(self, distributions: Dict[str, Distribution]):
      """Initialize likelihood model.
      Args:
         distributions: Dictionary mapping field names to their distributions
         key: Initial random key for sampling
      """
      self.distributions = distributions
   
   @partial(jax.jit, static_argnames=['self'])
   def sample(self, key: jax.Array, state: ObsState) -> tuple[jax.Array, ObsState]:
      """Sample from p(y|x) given state x.
      Args:
         state: True state in observation space
      Returns:
         Sampled observation
      """
      key, *subkeys = jax.random.split(key, len(state._fields)+1)
      key_dict = {field: key for field, key in zip(state._fields, subkeys)}
      template = type(state)(**{field: field for field in state._fields})
      return key, jax.tree_util.tree_map(
         lambda field_name, field_val: (
            self.distributions[field_name].sample(key_dict[field_name], field_val)
            if field_name in self.distributions else None), template, state )
   
   @partial(jax.jit, static_argnames=['self'])
   def scale(self, state: ObsState) -> ObsState:
      """Get scale parameters for each field.
      Args:
         state: State to match shape
      Returns:
         Scale parameters in ObsState format
      """
      template = type(state)(**{field: field for field in state._fields})
      return jax.tree_util.tree_map(
         lambda field_name: (self.distributions[field_name].scale(getattr(state,field_name))
            if field_name in self.distributions else None), template)
   
   @partial(jax.jit, static_argnames=['self','biasmin','biasmax'])
   def e2isample(self, key: jax.Array, state: ObsState, biasmin: ObsParam, biasmax: ObsParam) -> tuple[jax.Array, ObsState, ObsState]:
      """Sample inequality bounds.
      Args:
         state: True state in observation space
         biasmin: Minimum bias for each field
         biasmax: Maximum bias for each field
      Returns:
         Tuple of (lower_bounds, upper_bounds) as ObsState
      """
      key, *subkeys = jax.random.split(key, len(state._fields)+1)
      key_dict = {field: key for field, key in zip(state._fields, subkeys)}
      template = type(state)(**{field: field for field in state._fields})
      def sample_bounds(field_name):
         if field_name in biasmin.params:
            return self.distributions[field_name].e2isample(key_dict[field_name], 
               getattr(state,field_name), biasmin.params[field_name], biasmax.params[field_name])
         return None, None
      bounds = jax.tree_util.tree_map(sample_bounds, template)
      return key, *tuple(type(state)(*x) for x in zip(*bounds))
   
   @partial(jax.jit, static_argnames=['self','biasmin','biasmax'])
   def e2iscale(self, state: ObsState, biasmin: ObsParam, biasmax: ObsParam) -> tuple[ObsState, ObsState]:
      """Get scale parameters for inequality observations.
      Returns:
         Tuple of (lower_scales, upper_scales) as ObsState
      """
      template = type(state)(**{field: field for field in state._fields})
      def get_scales(field_name):
         if (field_name in biasmin.params):
            return self.distributions[field_name].e2iscale(getattr(state,field_name))
         return None, None
      scales = jax.tree_util.tree_map(get_scales, template)
      return tuple(type(state)(*x) for x in zip(*scales))
   
   @partial(jax.jit, static_argnames=['self','threshold'])
   def isample(self, key: jax.Array, state: ObsState, threshold: ObsParam) -> tuple[jax.Array, ObsState, ObsState]:
      """Sample observations and determine thresholds and scales based on obs values.
      Args:
         state: True state in observation space
         threshold: Initial threshold values in ObsState format
      Returns:
         Tuple of (new_threshold, scale) as ObsState where scale has appropriate signs
      """
      key, *subkeys = jax.random.split(key, len(state._fields)+1)
      key_dict = {field: key for field, key in zip(state._fields, subkeys)}
      template = type(state)(**{field: field for field in state._fields})
      def sample_field(field_name):
         if (field_name in threshold.params):
            return self.distributions[field_name].isample(
               key_dict[field_name], getattr(state,field_name), threshold.params[field_name])
         return None, None
      results = jax.tree_util.tree_map(sample_field, template)
      return key, *tuple(type(state)(*x) for x in zip(*results))
   




