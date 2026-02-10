#!/usr/bin/env python3
import os, json, pickle
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from typing import Union, Optional, Dict, List, Any, Tuple, NamedTuple
from pathlib import Path

def write_states(filename: Union[str,Path], states: Dict[str,NamedTuple], time: jax.Array,
                  metadata: Optional[Dict[str, Any]] = None,
                  coords: Optional[Dict[str, jax.Array]] = None,
                  dims: Optional[Dict[str, Tuple[str, ...]]] = None) -> None:
   """Write multiple model state trajectories to a single NetCDF file.
   Args:
      filename: Output file path
      states: Dictionary mapping names to state objects
               e.g. {"truth": state_truth, "forecast": state_forecast}
      time: Time array
      metadata: Optional metadata dictionary
      coords: Optional coordinate arrays for each dimension
      dims: Optional mapping of field names to dimension names
   """
   fpath = Path(filename)
   if fpath.exists(): os.remove(fpath)
   ds = xr.Dataset()
   ds.coords['time'] = time
   if coords:
      for name, values in coords.items():
         if name != 'time': ds.coords[name] = values

   # Process each state
   for state_name, state in states.items():
      if dims is None:
         state_dims = {}
         for field in state._fields:
            array = getattr(state, field)
            field_dims = ['time'] + [f'd{i}' for i in range(1, array.ndim)]
            state_dims[field] = tuple(field_dims)
      else: state_dims = dims
      # Add variables with state name suffix
      for field in state._fields:
         array = getattr(state, field)
         array = jnp.asarray(array)
         data = xr.DataArray(array, dims=state_dims[field])
         ds[f"{field}_{state_name}"] = data
   if metadata: ds.attrs.update(metadata)
   ds.to_netcdf(fpath)

def write_scores(filename: Union[str, Path], scores: Dict[str, Dict[str, NamedTuple]], time: jax.Array,
                  metadata: Optional[Dict[str, Any]] = None,
                  coords: Optional[Dict[str, jax.Array]] = None,
                  dim_names: Optional[Dict[str, List[str]]] = None) -> None:
   """Write score trajectories to a NetCDF file.
   Args:
      filename: Output file path
      scores: Dictionary with structure {"forecast": {score_type: val}, "analysis": {score_type: val}}
               where val is the score values (without nobs)
      time: Time array
      metadata: Optional metadata dictionary
      coords: Optional coordinate arrays for each dimension
      dim_names: Optional mapping from field name to list of dimension names
                  e.g., {'u': ['level'], 'ps': ['part']}
   """
   fpath = Path(filename)
   if fpath.exists(): os.remove(fpath)
   ds = xr.Dataset()
   ds.coords['time'] = time
   if coords:
      for name, values in coords.items():
         if name != 'time': ds.coords[name] = values
   
   # Process each type of score (forecast/analysis)
   for score_category, score_dict in scores.items():
      for score_type, score_data in score_dict.items():
         for field in score_data._fields:
            array = getattr(score_data, field)
            if array is not None:
               array = jnp.asarray(array)
               # Use provided dim_names or default to field-specific names
               if dim_names and field in dim_names:
                  extra_dims = dim_names[field]
               else:
                  extra_dims = [f'{field}_d{i}' for i in range(1, array.ndim)]
               dims = ['time'] + extra_dims
               ds[f"{field}_{score_type}_{score_category}"] = xr.DataArray(array, dims=dims)
   if metadata: ds.attrs.update(metadata)
   ds.to_netcdf(fpath)

def save_parameters(param, filepath: str) -> None:
   """
   Save model parameters to disk.
   Args:
      param: Model parameters (NamedTuple with JAX arrays)
      filepath: Path to save the parameters to
   """
   # Create directory if it doesn't exist
   os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
   param_dict = {}
   for field_name in param._fields:
      value = getattr(param, field_name)
      # Convert JAX arrays to numpy for serialization
      if isinstance(value, jax.Array):
         param_dict[field_name] = {'data': np.array(value), 'shape': value.shape, 'dtype': str(value.dtype)}
      else: param_dict[field_name] = value
   # Save metadata about parameter structure
   param_dict['_metadata'] = {'param_type': type(param).__name__, 'fields': list(param._fields)}
   with open(filepath, 'wb') as f: pickle.dump(param_dict, f)

def load_parameters(param_class, filepath: str) -> Any:
   """
   Load model parameters from disk.
   Args:
      param_class: The parameter class to instantiate
      filepath: Path to load the parameters from
   Returns:
      Loaded parameters as an instance of param_class
   """
   # Check if file exists
   if not os.path.exists(filepath):
      raise FileNotFoundError(f"Parameter file not found: {filepath}")
   # Load the dictionary
   with open(filepath, 'rb') as f: param_dict = pickle.load(f)
   # Verify metadata if it exists
   if '_metadata' in param_dict:
      metadata = param_dict.pop('_metadata')
      if metadata['param_type'] != param_class.__name__:
         print(f"Warning: Loaded parameters have type {metadata['param_type']}, "
               f"but expected {param_class.__name__}")
   # Convert dictionary back to arrays
   reconstructed_dict = {}
   for field_name in param_class._fields:
      if field_name not in param_dict:
         raise ValueError(f"Missing field {field_name} in loaded parameters")
      value = param_dict[field_name]
      # Reconstruct JAX arrays
      if isinstance(value, dict) and 'data' in value:
         reconstructed_dict[field_name] = jnp.array(value['data'], dtype=value['dtype'])
      else:
         reconstructed_dict[field_name] = value
   # Construct parameter object
   return param_class(**reconstructed_dict)

def save_timeseries_preprocessor(preprocessor, filepath: str):
   """Save the entire preprocessor with all transforms and their statistics."""
   config_data = {
      'input_fields_info': {name: {
         'source_fields': transform.source_fields,
         'transform_name': transform.transform_name,  # Store name instead of function
         'output_name': transform.output_name,
         'shape': transform.shape,
         'description': transform.description,
         'mean': transform.mean,
         'std': transform.std,
         'min_val': transform.min_val,
         'max_val': transform.max_val,
         'norm_method': transform.norm_method
      } for name, transform in preprocessor.config.input_fields.items()},
      'output_fields_info': {name: {
         'source_fields': transform.source_fields,
         'transform_name': transform.transform_name,  # Store name instead of function
         'output_name': transform.output_name,
         'shape': transform.shape,
         'description': transform.description,
         'mean': transform.mean,
         'std': transform.std,
         'min_val': transform.min_val,
         'max_val': transform.max_val,
         'norm_method': transform.norm_method
      } for name, transform in preprocessor.config.output_fields.items()},
      'input_sequence_length': preprocessor.config.input_sequence_length,
      'output_sequence_length': preprocessor.config.output_sequence_length,
      'n_consecutive_inputs': preprocessor.config.n_consecutive_inputs,
      'time_column': preprocessor.config.time_column,
      'train_ratio': preprocessor.config.train_ratio,
      'missing_value_strategy': preprocessor.config.missing_value_strategy,
      'forecast_start_hour': preprocessor.config.forecast_start_hour
   }
   with open(filepath, 'wb') as f: pickle.dump(config_data, f)
   
def load_timeseries_preprocessor(filepath: str):
   """Load a complete preprocessor with all transforms and their statistics."""
   from dajax.utils.field import FieldTransform
   from dajax.utils.timeseries_preprocessor import TimeSeriesPreprocessor, DataConfig
   from dajax.utils.transform_registry import get_transform
   
   # Load config data
   with open(filepath, 'rb') as f: config_data = pickle.load(f)
   input_fields = {}
   for name, info in config_data['input_fields_info'].items():
      # Get the transform function by name from the registry
      transform_fn = get_transform(info['transform_name'])
      input_fields[name] = FieldTransform(
         source_fields=info['source_fields'],
         transform_fn=transform_fn,
         transform_name=info['transform_name'],
         output_name=info['output_name'],
         shape=info['shape'],
         description=info['description'],
         mean=info['mean'],
         std=info['std'],
         min_val=info['min_val'],
         max_val=info['max_val'],
         norm_method=info['norm_method']
      )
   output_fields = {}
   for name, info in config_data['output_fields_info'].items():
      transform_fn = get_transform(info['transform_name'])
      output_fields[name] = FieldTransform(
         source_fields=info['source_fields'],
         transform_fn=transform_fn,
         transform_name=info['transform_name'],
         output_name=info['output_name'],
         shape=info['shape'],
         description=info['description'],
         mean=info['mean'],
         std=info['std'],
         min_val=info['min_val'],
         max_val=info['max_val'],
         norm_method=info['norm_method']
      )
   
   # Create DataConfig and preprocessor
   config = DataConfig(
      input_fields=input_fields,
      output_fields=output_fields,
      input_sequence_length=config_data['input_sequence_length'],
      output_sequence_length=config_data['output_sequence_length'],
      n_consecutive_inputs=config_data['n_consecutive_inputs'],
      time_column=config_data['time_column'],
      train_ratio=config_data['train_ratio'],
      missing_value_strategy=config_data['missing_value_strategy'],
      forecast_start_hour=config_data['forecast_start_hour']
   )
   return TimeSeriesPreprocessor(config)

def write_trajectory(filename: Union[str, Path], state: NamedTuple, time: jax.Array,
                     metadata: Optional[Dict[str, Any]] = None,
                     coords: Optional[Dict[str, jax.Array]] = None,
                     dims: Optional[Dict[str, Tuple[str, ...]]] = None) -> None:
   """Write model state trajectory to a NetCDF file using xarray.
   Args:
      filename: Output file path
      state: Model state (can be State or ObsState)
      time: Time array
      metadata: Optional metadata dictionary
      coords: Optional coordinate arrays for each dimension
      dims: Optional mapping of field names to dimension names
   Example:
      # For Lorenz96
      dims = {'u': ('time', 'x')}
      coords = {'time': time, 'x': jnp.arange(nx)}
      write_trajectory('output.nc', state, time, dims=dims, coords=coords)
      # For SHQG
      dims = { 'u': ('time', 'lev', 'lat', 'lon'),
               'v': ('time', 'lev', 'lat', 'lon')}
      coords = {'time': time,
               'lev': jnp.arange(nlev),
               'lat': lat_array,
               'lon': lon_array}
      write_trajectory('output.nc', state, time, dims=dims, coords=coords)
      ```
   """
   fpath = Path(filename)
   if fpath.exists(): os.remove(fpath)
   ds = xr.Dataset()
   ds.coords['time'] = time
   if coords:
      for name, values in coords.items():
         if name != 'time': ds.coords[name] = values
   # Get default dimension names based on array shapes if dims not provided
   if dims is None:
      dims = {}
      for field in state._fields:
         array = getattr(state, field)
         field_dims = ['time'] + [f'd{i}' for i in range(1, array.ndim)]
         dims[field] = tuple(field_dims)
   
   # Add variables
   for field in state._fields:
      array = getattr(state, field)
      array = jnp.asarray(array) # Convert to numpy array
      da = xr.DataArray(array, dims=dims[field])
      ds[field] = da
   if metadata: ds.attrs.update(metadata)
   ds.to_netcdf(fpath)

def read_trajectory(filename: Union[str, Path], state_type: type,
                  time_slice: Optional[slice] = None) -> Tuple[NamedTuple, jax.Array]:
   """Read model state trajectory from a NetCDF file.
   Args:
      filename: Input file path
      state_type: Type of state to construct (State or ObsState class)
      time_slice: Optional slice for partial time reading
   Returns:
      tuple: (state, time)
   """
   ds = xr.open_dataset(filename)
   time = ds.time.values
   if time_slice: time = time[time_slice]
   
   fields = {}
   for field in state_type._fields:
      if field not in ds: raise KeyError(f"Field {field} not found in file {filename}")
      array = ds[field].values
      if time_slice: array = array[time_slice]
      fields[field] = jnp.array(array)
   ds.close()
   state = state_type(**fields)
   return state, jnp.array(time)

def write_state(filename: Union[str, Path], state: NamedTuple,
               metadata: Optional[Dict[str, Any]] = None,
               coords: Optional[Dict[str, jax.Array]] = None,
               dims: Optional[Dict[str, Tuple[str, ...]]] = None) -> None:
   """Write single model state to a NetCDF file.
   Similar to write_trajectory but for a single state without time dimension."""
   ds = xr.Dataset()
   if coords:
      for name, values in coords.items(): ds.coords[name] = values
   if dims is None:
      dims = {}
      for field in state._fields:
         array = getattr(state, field)
         field_dims = [f'd{i}' for i in range(array.ndim)]
         dims[field] = tuple(field_dims)
   
   # Add variables
   for field in state._fields:
      array = getattr(state, field)
      array = jnp.asarray(array) # Convert to numpy array
      da = xr.DataArray(array, dims=dims[field])
      ds[field] = da
   if metadata: ds.attrs.update(metadata)
   ds.to_netcdf(filename)

def read_state(filename: Union[str, Path], state_type: type) -> NamedTuple:
   """Read single model state from a NetCDF file."""
   ds = xr.open_dataset(filename)
   fields = {}
   for field in state_type._fields:
      if field not in ds: raise KeyError(f"Field {field} not found in file {filename}")
      fields[field] = jnp.array(ds[field].values)
   ds.close()
   return state_type(**fields)