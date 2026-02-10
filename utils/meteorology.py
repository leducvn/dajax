#!/usr/bin/env python3
import re
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Union, Tuple
from dajax.utils.transform_registry import register_transform

@register_transform("wind_components")
def calculate_wind_components(
	speed: Union[jnp.ndarray, np.ndarray, pd.Series], 
	direction: Union[jnp.ndarray, np.ndarray, pd.Series]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""Calculate u and v wind components from speed and direction.
	Uses meteorological convention: direction is where wind is coming FROM,
	measured clockwise from North.
	Args:
		speed: Wind speed as jax array, numpy array or pandas Series
		direction: Wind direction in degrees as jax array, numpy array or pandas Series
					(0=North, 90=East, 180=South, 270=West)
	Returns:
		Tuple of (u, v) where:
			u: Positive means wind FROM east, negative FROM west
			v: Positive means wind FROM south, negative FROM north
	"""
	# Convert to numpy arrays if pandas Series
	speed_arr = speed.to_numpy() if isinstance(speed, pd.Series) else speed
	direction_arr = direction.to_numpy() if isinstance(direction, pd.Series) else direction
	# Convert to jax arrays if needed
	if not isinstance(speed_arr, jnp.ndarray): speed_arr = jnp.array(speed_arr)
	if not isinstance(direction_arr, jnp.ndarray): direction_arr = jnp.array(direction_arr)
	direction_rad = jnp.deg2rad(direction_arr)
	# Convert from meteorological to mathematical angle and from "coming from" to "going to"
	math_angle = jnp.pi/2 - direction_rad + jnp.pi
	u = speed_arr*jnp.cos(math_angle)  # u component 
	v = speed_arr*jnp.sin(math_angle)  # v component
	return u, v

@register_transform("dewpoint_depression")
def calculate_dewpoint_depression(
	temperature: Union[jnp.ndarray, np.ndarray, pd.Series],
	dew_point: Union[jnp.ndarray, np.ndarray, pd.Series]
) -> jnp.ndarray:
	"""Calculate dewpoint depression.
	Args:
		temperature: Air temperature in Celsius
		dew_point: Dew point temperature in Celsius
	Returns:
		Dewpoint depression in Celsius
	"""
	# Convert to numpy arrays if pandas Series
	temp_arr = temperature.to_numpy() if isinstance(temperature, pd.Series) else temperature
	dew_arr = dew_point.to_numpy() if isinstance(dew_point, pd.Series) else dew_point
	# Convert to jax arrays if needed
	if not isinstance(temp_arr, jnp.ndarray): temp_arr = jnp.array(temp_arr)
	if not isinstance(dew_arr, jnp.ndarray): dew_arr = jnp.array(dew_arr)
	return temp_arr - dew_arr

def calculate_vapor_pressure(
	dew_point: Union[jnp.ndarray, np.ndarray, pd.Series]
) -> jnp.ndarray:
	"""Calculate water vapor pressure using dew point temperature.
	Uses Bolton's formula.
	Args:
		dew_point: Dew point temperature in Celsius
	Returns:
		Vapor pressure in hPa
	"""
	# Convert to numpy arrays if pandas Series
	dew_arr = dew_point.to_numpy() if isinstance(dew_point, pd.Series) else dew_point
	# Convert to jax arrays if needed
	if not isinstance(dew_arr, jnp.ndarray): dew_arr = jnp.array(dew_arr)
	return 6.112*jnp.exp(17.67*dew_arr/(dew_arr+243.5))

def calculate_saturated_vapor_pressure(
	temperature: Union[jnp.ndarray, np.ndarray, pd.Series]
) -> jnp.ndarray:
	"""Calculate saturated water vapor pressure using air temperature.
	Uses Bolton's formula.
	Args:
		temperature: Air temperature in Celsius
	Returns:
		Saturated vapor pressure in hPa
	"""
	# Convert to numpy arrays if pandas Series
	temp_arr = temperature.to_numpy() if isinstance(temperature, pd.Series) else temperature
	# Convert to jax arrays if needed
	if not isinstance(temp_arr, jnp.ndarray): temp_arr = jnp.array(temp_arr)
	return 6.112*jnp.exp(17.67*temp_arr/(temp_arr+243.5))

@register_transform("specific_humidity")
def calculate_specific_humidity(
	dew_point: Union[jnp.ndarray, np.ndarray, pd.Series], 
	altimeter: Union[jnp.ndarray, np.ndarray, pd.Series]
) -> jnp.ndarray:
	"""Calculate specific humidity from temperature, dew point and altimeter (surface pressure).
	Args:
		temperature: Air temperature in Celsius
		dew_point: Dew point temperature in Celsius
		altimeter: Surface pressure in hPa
	Returns:
		Specific humidity in kg/kg
	"""
	# Get vapor pressure from dew point
	e = calculate_vapor_pressure(dew_point)  # hPa
	# Convert altimeter to jax array if needed
	alt_arr = altimeter.to_numpy() if isinstance(altimeter, pd.Series) else altimeter
	if not isinstance(alt_arr, jnp.ndarray): alt_arr = jnp.array(alt_arr)
	# Calculate mixing ratio (kg/kg)
	# Use 0.622 = Rd/Rv (ratio of gas constants for dry air and water vapor)
	r = 0.622*e/(alt_arr-e)
	# Convert to specific humidity
	# q = r/(1 + r) is equivalent to e/(e + 1.608*p) where p is total pressure
	return r/(1+r)

@register_transform("relative_humidity")
def calculate_relative_humidity(
	temperature: Union[jnp.ndarray, np.ndarray, pd.Series], 
	dew_point: Union[jnp.ndarray, np.ndarray, pd.Series]
) -> jnp.ndarray:
	"""Calculate relative humidity from temperature and dew point.
	Args:
		temperature: Air temperature in Celsius
		dew_point: Dew point temperature in Celsius
	Returns:
		Relative humidity in percent (0-100)
	"""
	# Calculate vapor pressures
	e = calculate_vapor_pressure(dew_point)      # Actual vapor pressure
	es = calculate_saturated_vapor_pressure(temperature)  # Saturation vapor pressure
	# Calculate relative humidity
	rh = 100.0*e/es
	# Clip to valid range [0-100]
	return jnp.clip(rh, 0, 100)

def convert_visibility_single(vis_str: str) -> float:
	"""Convert a single visibility string to meters."""
	if pd.isna(vis_str): return np.nan
	# Convert to string if not already
	vis_str = str(vis_str).strip().lower()
	# Handle empty strings
	if not vis_str: return np.nan
	# Remove any leading symbols (>, <, =)
	vis_str = re.sub(r'^[><]=?', '', vis_str)
	# Try to extract number and unit using regex
	match = re.match(r'(\d+(?:/\d+)?(?:\.\d+)?)\s*(km|m|sm)?', vis_str)
	if not match: return np.nan
	value_str, unit = match.groups()
	# Handle fractions (like 1/4)
	if '/' in value_str:
		num, denom = map(float, value_str.split('/'))
		value = num / denom
	else: value = float(value_str)
	# Convert to meters based on unit
	if unit == 'km': return value * 1000
	elif unit == 'sm': return value * 1609.34
	elif unit == 'm' or unit is None: return value
	return np.nan

def convert_visibility(vis: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
	"""Convert visibility values to meters.
	Handles formats like:
	- "5000m"
	- ">10km"
	- "10 km"
	- "<1/4SM" (statute miles)
	- "3000"
	Args:
		vis: Array-like of visibility strings
	Returns:
		numpy array of visibility values in meters
	"""
	# Convert to pandas Series if not already
	if not isinstance(vis, pd.Series): vis = pd.Series(vis)
	# Apply conversion function to each element
	converted = vis.apply(convert_visibility_single)
	return converted.to_numpy()

def preprocess_visibility(df: pd.DataFrame) -> pd.DataFrame:
	"""Convert visibility fields to numeric values before handling missing values."""
	# Find all visibility columns - those that might need conversion
	vis_columns = [col for col in df.columns 
		if any(term in col.lower() for term in ['vis', 'main_visibility', 'visibility'])]
	for col in vis_columns:
		if col in df.columns and df[col].dtype == 'object':  # Only convert string columns
			df[col] = convert_visibility(df[col])
	return df