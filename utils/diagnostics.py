#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Tuple
from functools import partial
from dajax.models.base import ObsState

# Selector type definitions
Selector = Callable[[jax.Array], Dict[str,jax.Array]]

@jax.jit
def no_selector(field: jax.Array) -> Dict[str,jax.Array]:
   """Return field as a single part."""
   return {"all": field}

@jax.jit
def level_selector(field: jax.Array) -> Dict[str,jax.Array]:
   """Split field into levels."""
   return {f"level{k}": field[k] for k in range(field.shape[0])}

@jax.jit
def window_selector(window_size: int) -> Selector:
   """Create a selector that splits field into sliding windows."""
   def selector(field: jax.Array) -> Dict[str,jax.Array]:
      n = field.shape[0]
      return {f"window{i}": field[i:i+window_size] for i in range(0, n-window_size+1)}
   return selector

class Score:
   """Base class for score computation methods."""
   def __init__(self, name: str, field_selectors: Dict[str, Selector]):
      """Initialize score with name and field selectors.
      Args:
         name: Name of the score (e.g., 'rmse', 'me')
         field_selectors: Dictionary mapping field names to their selector functions
      """
      self.name = name
      self.field_selectors = field_selectors

   @partial(jax.jit, static_argnames=['self'])
   def compute_nobs(self, parts: Dict[str,jax.Array]) -> jax.Array:
      """Compute number of valid (non-NaN) observations for each part of the field.
      Args:
         parts: Input field array
      Returns:
         Array of observation counts for each part
      """
      #nobs = [jnp.sum(~jnp.isnan(part)) for part in parts.values()]
      #jax.debug.print("nobs: {x}, {y}", x=nobs, y=self.name)
      return jnp.array([jnp.sum(~jnp.isnan(part)) for part in parts.values()])

   def compute_val(self, parts: Dict[str,jax.Array], est_parts: Dict[str,jax.Array]) -> jax.Array:
      """Compute score values for a single field.  
      Args:
         parts: True field values
         est_parts: Estimated field values  
      Returns:
         Array of score values for each part
      """
      raise NotImplementedError

   @partial(jax.jit, static_argnames=['self'])
   def compute(self, truth: ObsState, estimate: ObsState) -> Tuple[ObsState, ObsState]:
      """Compute scores for all fields using tree_map."""
      template = type(truth)(**{field: field for field in truth._fields})
      nobs = jax.tree_util.tree_map(
         lambda field_name: (
            self.compute_nobs(self.field_selectors[field_name](getattr(truth,field_name))) 
            if (field_name in self.field_selectors) else None),
         template)
      values = jax.tree_util.tree_map(
         lambda field_name: (
            self.compute_val(self.field_selectors[field_name](getattr(truth,field_name)), 
                           self.field_selectors[field_name](getattr(estimate,field_name))) 
            if (field_name in self.field_selectors) else None),
         template)
      return nobs, values
   
   def average(self, nobs_list: List[ObsState], values_list: List[ObsState]) -> Tuple[ObsState, ObsState]:
      """Average multiple score results. 
      Args:
         nobs_list: List of observation count ObsStates
         values_list: List of score value ObsStates
      Returns:
         Tuple of averaged (nobs, values) as ObsStates
      """
      raise NotImplementedError

class RMSEScore(Score):
   """Root Mean Square Error score."""
   def __init__(self, field_selectors: Dict[str, Selector]):
      super().__init__(name='rmse', field_selectors=field_selectors)

   @partial(jax.jit, static_argnames=['self'])
   def compute_val(self, parts: Dict[str,jax.Array], est_parts: Dict[str,jax.Array]) -> jax.Array:
      """Compute RMSE, ignoring NaN values."""
      return jnp.array([jnp.sqrt(jnp.nanmean((parts[name]-est_parts[name])**2)) for name in parts.keys()])
   
   @partial(jax.jit, static_argnames=['self'])
   def average(self, nobs: ObsState, vals: ObsState) -> Tuple[ObsState, ObsState]:
      """Average RMSE scores, handling NaN in vals (from all-NaN parts)."""
      template = type(nobs)(**{field: field for field in nobs._fields})
      def avg_field(field_name):
         nobs_arrays = getattr(nobs, field_name)
         val_arrays = getattr(vals, field_name)
         nobs_sum = jnp.sum(nobs_arrays, axis=0)
         # Replace NaN vals with 0 for weighted sum (they have nobs=0 anyway)
         val_arrays_safe = jnp.where(jnp.isnan(val_arrays), 0.0, val_arrays)
         weighted_sum = jnp.sum(nobs_arrays*val_arrays_safe**2, axis=0)
         # Avoid division by zero
         avg_vals = jnp.where(nobs_sum>0, jnp.sqrt(weighted_sum/nobs_sum), jnp.nan)
         return nobs_sum, avg_vals
      
      avg_nobs = jax.tree_util.tree_map(
         lambda field_name: avg_field(field_name)[0] if field_name in self.field_selectors else None,
         template)
      avg_vals = jax.tree_util.tree_map(
         lambda field_name: avg_field(field_name)[1] if field_name in self.field_selectors else None,
         template)
      return avg_nobs, avg_vals

class MEScore(Score):
   """Mean Error score."""
   def __init__(self, field_selectors: Dict[str, Selector]):
      super().__init__(name='me', field_selectors=field_selectors)

   @partial(jax.jit, static_argnames=['self'])
   def compute_val(self, parts: Dict[str,jax.Array], est_parts: Dict[str,jax.Array]) -> jax.Array:
      """Compute ME, ignoring NaN values."""
      return jnp.array([jnp.nanmean(est_parts[name]-parts[name]) for name in parts.keys()])
   
   @partial(jax.jit, static_argnames=['self'])
   def average(self, nobs: ObsState, vals: ObsState) -> Tuple[ObsState, ObsState]:
      """Average ME scores, handling NaN in vals."""
      template = type(nobs)(**{field: field for field in nobs._fields})
      def avg_field(field_name):
         nobs_arrays = getattr(nobs, field_name)
         val_arrays = getattr(vals, field_name)
         nobs_sum = jnp.sum(nobs_arrays, axis=0)
         # Replace NaN vals with 0 for weighted sum
         val_arrays_safe = jnp.where(jnp.isnan(val_arrays), 0.0, val_arrays)
         weighted_sum = jnp.sum(nobs_arrays*val_arrays_safe, axis=0)
         # Avoid division by zero
         avg_vals = jnp.where(nobs_sum>0, weighted_sum/nobs_sum, jnp.nan)
         return nobs_sum, avg_vals
      
      avg_nobs = jax.tree_util.tree_map(
         lambda field: avg_field(field)[0] if field in self.field_selectors else None,
         template)
      avg_vals = jax.tree_util.tree_map(
         lambda field: avg_field(field)[1] if field in self.field_selectors else None,
         template)
      return avg_nobs, avg_vals

class Diagnostics:
   """Class for managing multiple scores."""
   def __init__(self, scores: List[Score]):
      """Initialize with list of score objects."""
      self.scores = {score.name: score for score in scores}

   @partial(jax.jit, static_argnames=['self'])
   def compute(self, truth: ObsState, estimate: ObsState) -> Dict[str,Tuple[ObsState,ObsState]]:
      """Compute all scores between truth and estimate states."""
      return {name: score.compute(truth,estimate) for name,score in self.scores.items()}
   
   @partial(jax.jit, static_argnames=['self'])
   def average(self, score_dict: Dict[str,Tuple[ObsState,ObsState]]) -> Dict[str,Tuple[ObsState,ObsState]]:
      """Average scores over multiple timesteps."""
      result = {}
      for name,score in self.scores.items():
         # Extract nobs and values lists for this score
         nobs = score_dict[name][0]
         vals = score_dict[name][1]
         # Compute average using score-specific method
         result[name] = score.average(nobs,vals)
      return result

   def print_comparison(self, score_results: List[Dict[str,Tuple[ObsState,ObsState]]], names: List[str]):
      """Print comparison of multiple score results.
      Args:
         score_results: List of score dictionaries from different experiments
         names: Names of the experiments for labeling
      """
      # For each score type
      for score_name in self.scores.keys():
         print(f"{score_name.upper()} Scores:")
         # Get first non-None field as reference
         first_result = score_results[0][score_name]
         nobs, values = first_result
         # For each field
         for field in values._fields:
            field_vals = getattr(values, field)
            if field_vals is not None:
               print(f"  {field}:")
               nlevels = field_vals.shape[0]
               # For each level
               for level in range(nlevels):
                  print(f"    level{level}:")
                  # For each experiment
                  for result_dict,name in zip(score_results,names):
                     nobs, vals = result_dict[score_name]
                     n = getattr(nobs,field)[level]
                     v = getattr(vals,field)[level]
                     print(f"      {name:10s}: {v:8.4f} [nobs: {n}]")