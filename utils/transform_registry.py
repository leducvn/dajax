#!/usr/bin/env python3
from typing import Dict, Callable, Any

# Global registry of transformation functions
_TRANSFORM_REGISTRY: Dict[str, Callable] = {}

def register_transform(name: str, func: Callable = None):
   """Register a transformation function."""
   def decorator(f):
      _TRANSFORM_REGISTRY[name] = f
      return f
   
   if func is not None:
      _TRANSFORM_REGISTRY[name] = func
      return func
   
   return decorator

def get_transform(name: str) -> Callable:
   """Get a registered transformation function by name."""
   if name not in _TRANSFORM_REGISTRY:
      raise ValueError(f"Transform function '{name}' not registered. Available transforms: {list(_TRANSFORM_REGISTRY.keys())}")
   return _TRANSFORM_REGISTRY[name]


