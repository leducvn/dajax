#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass

@dataclass
class FieldInfo:
   shape: Tuple[int, ...]                 
   description: str = ""     
   mean: Optional[float] = None
   std: Optional[float] = None
   min_val: Optional[float] = None
   max_val: Optional[float] = None
   norm_method: str = 'standard'  # 'standard' or 'minmax' or 'none'

@dataclass
class FieldTransform:
   """Configuration for field transformation."""
   source_fields: List[str]  # Source field names used in transformation
   transform_fn: Callable  # Function to transform the fields
   transform_name: str       # Name of the transform function (for serialization)
   output_name: str  # Name of the output field
   shape: Tuple[int, ...]  # Shape of output field (excluding batch dimension)
   description: str = ""  # Description of the transformation
   mean: Optional[float] = None  # For normalization
   std: Optional[float] = None  # For normalization
   min_val: Optional[float] = None  # For min-max normalization
   max_val: Optional[float] = None  # For min-max normalization
   norm_method: str = 'standard'  # 'standard' or 'minmax' or 'none'

   def transform(self, df: pd.DataFrame, recalculate_stats: bool = True) -> jnp.ndarray:
      """Apply transformation to create new field.
      Args:
         df: Input DataFrame containing source fields
      Returns:
         Transformed data as JAX array
      """
      # Extract source data and convert to numpy arrays
      source_data = [df[field].to_numpy() for field in self.source_fields]
      result = self.transform_fn(*source_data)
      # If result is a tuple (like from wind components), stack it
      if isinstance(result, tuple):
         result = tuple(jnp.array(r) if not isinstance(r, jnp.ndarray) else r for r in result)
         result = jnp.column_stack(result)
      elif not isinstance(result, jnp.ndarray): result = jnp.array(result)
      # Only recalculate statistics if requested and not already set
      if recalculate_stats:
         self.mean = float(jnp.mean(result))
         self.std = float(jnp.std(result))
         self.min_val = float(jnp.min(result))
         self.max_val = float(jnp.max(result))
      return result

   def normalize(self, data: jnp.ndarray) -> jnp.ndarray:
      """Normalize a field based on its configuration."""
      if self.norm_method == 'standard':
         return (data-self.mean)/self.std
      elif self.norm_method == 'minmax':
         return (data-self.min_val)/(self.max_val-self.min_val)
      elif self.norm_method == 'none':
         return data
   
   def denormalize(self, data: jnp.ndarray) -> jnp.ndarray:
      """Denormalize a field based on its configuration."""
      if self.norm_method == 'standard':
         return data*self.std + self.mean
      elif self.norm_method == 'minmax':
         return data*(self.max_val-self.min_val) + self.min_val
      elif self.norm_method == 'none':
         return data

class DatasetTransform:
   """Handles transformations for multiple fields."""
   def __init__(self, transform_fields: Dict[str, FieldTransform]):
      self.transform_fields = transform_fields

   def transform(self, df: pd.DataFrame, recalculate_stats: bool = True) -> Dict[str, jnp.ndarray]:
      """Transform multiple fields from a DataFrame.
      Args:
         df: Input DataFrame containing all required source fields
      Returns:
         Dictionary mapping field names to transformed JAX arrays
      """
      transformed_data = {}
      for name, transform in self.transform_fields.items():
         transformed_data[name] = transform.transform(df, recalculate_stats)
      return transformed_data

   def normalize(self, data_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
      """Normalize entire dataset."""
      norm_data = {}
      for name, data in data_dict.items():
         transform = self.transform_fields[name]
         norm_data[name] = transform.normalize(data)
      return norm_data

   def denormalize(self, data_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
      """Denormalize entire dataset."""
      denorm_data = {}
      for name, data in data_dict.items():
         transform = self.transform_fields[name]
         denorm_data[name] = transform.denormalize(data)
      return denorm_data

   def transform_to_info(self, sequence_length: int) -> Dict[str, FieldInfo]:
      """Convert transforms to FieldInfo dictionary."""
      info_fields = {}
      for name, field in self.transform_fields.items():
         shape = (sequence_length,) + field.shape
         info_fields[name] = FieldInfo(
            shape=shape,
            description=field.description,
            mean=field.mean,
            std=field.std,
            min_val=field.min_val,
            max_val=field.max_val,
            norm_method=field.norm_method
         )      
      return info_fields


