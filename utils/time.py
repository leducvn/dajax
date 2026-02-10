#!/usr/bin/env python3
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
from dajax.utils.transform_registry import register_transform

@register_transform("temporal_variables")
def calculate_temporal_variables(
	datetime_series: Union[pd.Series, np.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""Calculate temporal variables from datetime.
	Args:
		datetime_series: Pandas Series containing datetime values
	Returns:
		Tuple of (year_sin, year_cos, day_sin, day_cos) as JAX arrays containing:
			year_sin: Sine of annual cycle
			year_cos: Cosine of annual cycle
			day_sin: Sine of daily cycle
			day_cos: Cosine of daily cycle
	"""
	# Convert to datetime if string
	if isinstance(datetime_series, np.ndarray): datetime_series = pd.Series(datetime_series)
	if datetime_series.dtype == 'object': datetime_series = pd.to_datetime(datetime_series)
	# Calculate day of year (1-365)
	day_of_year = datetime_series.dt.dayofyear.to_numpy()
	# Calculate hour with minutes as decimal (0-24)
	hour_decimal = (datetime_series.dt.hour+datetime_series.dt.minute/60).to_numpy()
	# Convert to JAX arrays and calculate cycles
	day_of_year = jnp.array(day_of_year)
	hour_decimal = jnp.array(hour_decimal)
	# Annual cycle
	year_sin = jnp.sin(2*jnp.pi*day_of_year/365.25)
	year_cos = jnp.cos(2*jnp.pi*day_of_year/365.25)
	# Daily cycle
	day_sin = jnp.sin(2*jnp.pi*hour_decimal/24)
	day_cos = jnp.cos(2*jnp.pi*hour_decimal/24)
	return year_sin, year_cos, day_sin, day_cos

def create_positional_encoding(
	seq_length: int, 
	d_model: int,
	max_wavelength: int = 10000
) -> jnp.ndarray:
	"""Create positional encoding matrix for transformer models.
	Args:
		seq_length: Length of input sequence
		d_model: Dimension of the model embeddings
		max_wavelength: Maximum wavelength for the sinusoidal encoding
	Returns:
		Matrix of shape (seq_length, d_model) containing positional encodings
	"""
	position = jnp.arange(seq_length)[:, jnp.newaxis]
	div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(max_wavelength) / d_model))
	
	pe = jnp.zeros((seq_length, d_model))
	pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
	pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
	
	return pe

def create_timestamp_features(
	timestamps: pd.Series,
	include_time: bool = True,
	include_date: bool = True
) -> Dict[str, jnp.ndarray]:
	"""Create various timestamp features for time series analysis.
	Args:
		timestamps: Pandas Series containing datetime values
		include_time: Whether to include time-based features
		include_date: Whether to include date-based features
	Returns:
		Dictionary containing various temporal features as JAX arrays
	"""
	# Convert to datetime if needed
	if timestamps.dtype == 'object':
		timestamps = pd.to_datetime(timestamps)
		
	features = {}
	
	if include_time:
		# Time features
		features['hour_sin'] = jnp.array(jnp.sin(2 * jnp.pi * timestamps.dt.hour/24))
		features['hour_cos'] = jnp.array(jnp.cos(2 * jnp.pi * timestamps.dt.hour/24))
		features['minute_sin'] = jnp.array(jnp.sin(2 * jnp.pi * timestamps.dt.minute/60))
		features['minute_cos'] = jnp.array(jnp.cos(2 * jnp.pi * timestamps.dt.minute/60))
		
	if include_date:
		# Date features
		features['day_of_week_sin'] = jnp.array(jnp.sin(2 * jnp.pi * timestamps.dt.dayofweek/7))
		features['day_of_week_cos'] = jnp.array(jnp.cos(2 * jnp.pi * timestamps.dt.dayofweek/7))
		features['day_of_year_sin'] = jnp.array(jnp.sin(2 * jnp.pi * timestamps.dt.dayofyear/365.25))
		features['day_of_year_cos'] = jnp.array(jnp.cos(2 * jnp.pi * timestamps.dt.dayofyear/365.25))
		features['month_sin'] = jnp.array(jnp.sin(2 * jnp.pi * timestamps.dt.month/12))
		features['month_cos'] = jnp.array(jnp.cos(2 * jnp.pi * timestamps.dt.month/12))
	
	return features