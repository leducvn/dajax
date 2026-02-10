#!/usr/bin/env python3
from dajax.utils.transform_registry import register_transform
import jax.numpy as jnp

# Register basic transforms directly here
@register_transform("log10")
def log10_transform(x):
   return jnp.log10(x)

@register_transform("identity")
def identity_transform(x):
   return jnp.array(x)

def initialize_transforms():
   """Initialize all transformation functions by importing their modules."""
   # Import all modules that contain transform functions
   # Since the transforms are registered when modules are imported,
   # we don't need to do anything special here
   from dajax.utils.meteorology import calculate_wind_components
   from dajax.utils.time import calculate_temporal_variables
   # Optionally return something to make sure the import isn't optimized away
   return True


