#!/usr/bin/env python3
import jax, datetime
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from dajax.utils.field import FieldInfo, FieldTransform, DatasetTransform
from dajax.utils.time import calculate_temporal_variables
from dajax.utils.meteorology import (calculate_wind_components, 
                                    calculate_specific_humidity, calculate_relative_humidity,
                                    convert_visibility, preprocess_visibility)

@dataclass
class DataConfig:
   """Configuration for data preprocessing."""
   input_fields: Dict[str, FieldTransform]  # Fields to use as input
   output_fields: Dict[str, FieldTransform]  # Fields to use as output
   input_sequence_length: int  # Length of input sequence
   output_sequence_length: int  # Length of output sequence
   n_consecutive_inputs: int = 1  # Number of consecutive input sequences to use
   time_column: str = 'date_time'  # Column containing timestamps
   train_ratio: float = 0.95  # Ratio of data to use for training
   missing_value_strategy: str = 'interpolate'  # Strategy for handling missing values
   forecast_start_hour: int = 0  # Hour of day to start forecast sequences (0-23)
   
class TimeSeriesPreprocessor:
   """Preprocessor for time series data."""
   def __init__(self, config: DataConfig):
      """Initialize preprocessor with configuration."""
      self.config = config
      self.input = DatasetTransform(config.input_fields)
      self.output = DatasetTransform(config.output_fields)
         
   def _handle_outliers(self, df: pd.DataFrame, column: str, 
                     lower_quantile: float = 0.001, 
                     upper_quantile: float = 0.999) -> pd.DataFrame:
      """Detect and handle outliers in a column using quantile method."""
      data = df[column]
      if not np.issubdtype(data.dtype, np.number): return df  # Skip non-numeric columns
      lower_bound = data.quantile(lower_quantile)
      upper_bound = data.quantile(upper_quantile)
      outlier_mask = (data < lower_bound) | (data > upper_bound)
      if outlier_mask.any(): df.loc[outlier_mask, column] = np.nan
      return df
      
   def _handle_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
      """Process datetime column and fill missing values."""
      # Convert to datetime
      df[self.config.time_column] = pd.to_datetime(df[self.config.time_column])
      df[self.config.time_column] = df[self.config.time_column] - pd.Timedelta(hours=7)
      # Check for duplicate timestamps and handle them
      if df[self.config.time_column].duplicated().any():
         print(f"Warning: Found duplicate timestamps in {self.config.time_column}. Keeping first occurrence.")
         # Keep first occurrence of duplicated timestamps
         df = df.drop_duplicates(subset=[self.config.time_column], keep='first')
      # Sort by datetime
      df = df.sort_values(by=self.config.time_column)
      # Check for missing timestamps
      time_diff = df[self.config.time_column].diff()
      # Skip NaN values (first row) when finding most common time difference
      time_diff_no_na = time_diff.dropna()
      if len(time_diff_no_na) == 0:
         print("Warning: Not enough data points to determine time frequency.")
         return df
      modal_diff = time_diff.mode()[0]  # Most common time difference
      # Find gaps
      gaps = time_diff > modal_diff
      if gaps.any():
         # Create complete timestamp range
         full_range = pd.date_range(
               start=df[self.config.time_column].min(),
               end=df[self.config.time_column].max(),
               freq=modal_diff
         )
         # Reindex dataframe to include missing timestamps
         df = df.set_index(self.config.time_column)
         # Verify no duplicate indices before reindexing
         if df.index.duplicated().any():
            # This shouldn't happen since we already removed duplicates above
            # But just in case there's some edge case we missed
            print("Warning: Still found duplicate timestamps after initial cleanup.")
            df = df[~df.index.duplicated(keep='first')]
         df = df.reindex(full_range)
         df = df.reset_index()
         df = df.rename(columns={'index': self.config.time_column})
      return df
      
   def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
      """Handle missing values according to specified strategy."""
      # First handle datetime
      df = self._handle_datetime(df)
      # Handle outliers for each numeric column
      numeric_columns = df.select_dtypes(include=[np.number]).columns
      for column in numeric_columns: df = self._handle_outliers(df, column)
      # Then handle missing values
      if self.config.missing_value_strategy == 'interpolate':
         return df.interpolate(method='linear', limit_direction='both')
      elif self.config.missing_value_strategy == 'ffill':
         return df.fillna(method='ffill').fillna(method='bfill')  # Combine with bfill for edges
      elif self.config.missing_value_strategy == 'bfill':
         return df.fillna(method='bfill').fillna(method='ffill')  # Combine with ffill for edges
      else:
         raise ValueError(f"Unknown missing value strategy: {self.config.missing_value_strategy}")

   def _create_sequences_old(self, df: pd.DataFrame) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
      """
      Create input and output sequences from dataframe.
      Uses input_sequence_length as the stride between sequences.
      First sequence starts at forecast_start_hour.
      Args:
         df: Input dataframe
      Returns:
         Tuple of (input_sequences, output_sequences)
      """
      df[self.config.time_column] = pd.to_datetime(df[self.config.time_column])
      df['hour'] = df[self.config.time_column].dt.hour
      first_forecast_indices = df.index[df['hour'] == self.config.forecast_start_hour]
      if len(first_forecast_indices) == 0:
         raise ValueError(f"No data points found at forecast hour {self.config.forecast_start_hour}")
      first_start_idx = first_forecast_indices[0]
      # Make sure we have enough data for the input sequence before the first forecast hour
      if first_start_idx < self.config.input_sequence_length:
         # Not enough data before the first forecast hour, find the next one
         for idx in first_forecast_indices[1:]:
            if idx >= self.config.input_sequence_length:
               first_start_idx = idx
               break
         else:
            raise ValueError(f"Not enough data before any forecast hour for input sequence")
      first_start_idx = first_start_idx - self.config.input_sequence_length
      
      # Use input_sequence_length as the stride
      stride_steps = self.config.input_sequence_length
      # Generate all start indices using the stride
      valid_start_indices = [first_start_idx]
      current_idx = first_start_idx + stride_steps
      while current_idx + self.config.input_sequence_length + self.config.output_sequence_length <= len(df):
         valid_start_indices.append(current_idx)
         current_idx += stride_steps
      print(f"Created {len(valid_start_indices)} sequences with stride of {stride_steps} time steps")
      print(f"First sequence starts at index {valid_start_indices[0]}")
      if len(valid_start_indices) > 1:
         print(f"Stride between sequences: {valid_start_indices[1] - valid_start_indices[0]} time steps")
      
      # Transform all fields
      input_data = self.input.transform(df)
      output_data = self.output.transform(df)
      
      # Create sequences
      sequences_input = {}
      sequences_output = {}
      # Process input fields
      for field_name, field_config in self.config.input_fields.items():
         data = input_data[field_name]
         # Initialize array to store sequences for this field
         if len(data.shape) == 1:
            field_sequences = np.zeros((len(valid_start_indices), self.config.input_sequence_length, 1))
         else:
            # For 2D data
            n_features = data.shape[1]
            field_sequences = np.zeros((len(valid_start_indices), self.config.input_sequence_length, n_features))
         # Fill the sequences array
         for i, start_idx in enumerate(valid_start_indices):
            if len(data.shape) == 1:
               # For 1D data
               seq = data[start_idx:start_idx + self.config.input_sequence_length]
               field_sequences[i, :, 0] = seq
            else:
               # For 2D data
               seq = data[start_idx:start_idx + self.config.input_sequence_length, :]
               field_sequences[i, :, :] = seq
         sequences_input[field_name] = jnp.array(field_sequences)
      
      # Process output fields
      for field_name, field_config in self.config.output_fields.items():
         data = output_data[field_name]
         # Initialize array to store sequences for this field
         if len(data.shape) == 1:
            # For 1D data
            field_sequences = np.zeros((len(valid_start_indices), self.config.output_sequence_length, 1))
         else:
            # For 2D data
            n_features = data.shape[1]
            field_sequences = np.zeros((len(valid_start_indices), self.config.output_sequence_length, n_features))
         # Fill the sequences array
         for i, start_idx in enumerate(valid_start_indices):
            output_start = start_idx + self.config.input_sequence_length
            if len(data.shape) == 1:
               # For 1D data
               seq = data[output_start:output_start + self.config.output_sequence_length]
               field_sequences[i, :, 0] = seq
            else:
               # For 2D data
               seq = data[output_start:output_start + self.config.output_sequence_length, :]
               field_sequences[i, :, :] = seq
         sequences_output[field_name] = jnp.array(field_sequences)
      
      # Print hour of first sequence for verification
      input_start_hour = df.iloc[valid_start_indices[0]]['hour']
      output_start_hour = df.iloc[valid_start_indices[0] + self.config.input_sequence_length]['hour']
      print(f"First sequence: Input starts at hour {input_start_hour}, Output starts at hour {output_start_hour}")
      return sequences_input, sequences_output

   def _create_sequences(self, df: pd.DataFrame, target_months: List[int] = None) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
      """Create input and output sequences from dataframe.
      Creates sequences of n consecutive inputs where each input has input_sequence_length time steps.
      The output sequence starts immediately after the last input.
      Args:
         df: Input dataframe
         target_months: Optional list of months (as integers 1-12) to filter sequences by.
                        Only sequences where output starts in these months will be included.
      Returns:
         Tuple of (input_sequences, output_sequences) dictionaries
      """
      # Find first occurrence of forecast_start_hour
      df['hour'] = pd.to_datetime(df[self.config.time_column]).dt.hour
      istart = df.index[df['hour'] == self.config.forecast_start_hour][0]
      # Calculate total input length (n consecutive inputs)
      total_input_length = self.config.n_consecutive_inputs * self.config.input_sequence_length
      # Calculate how many complete sequences we can create
      # Each new sequence moves forward by one input_sequence_length
      max_idx = len(df) - total_input_length - self.config.output_sequence_length
      n_potential_sequences = (max_idx - istart) // self.config.input_sequence_length + 1
      
      # Transform all fields
      input_data = self.input.transform(df)
      output_data = self.output.transform(df)

      # If we need to filter by month, pre-calculate all valid sequence indices
      valid_seq_indices = []
      if target_months is not None and len(target_months) > 0:
         for seq_idx in range(n_potential_sequences):
            # Calculate starting index for this output
            sample_idx = istart + seq_idx*self.config.input_sequence_length
            output_start = sample_idx + total_input_length
            # Get the datetime value at the output start index
            output_start_time = df.iloc[output_start][self.config.time_column]
            output_start_month = output_start_time.month
            # Keep only sequences where output starts in one of the target months
            if output_start_month in target_months: valid_seq_indices.append(seq_idx)
         #print(f"Found {len(valid_seq_indices)} sequences with output starting in months {target_months}")
         if len(valid_seq_indices) == 0:
            raise ValueError(f"No sequences found with output starting in months {target_months}")
      else: valid_seq_indices = list(range(n_potential_sequences))

      # Create sequences
      sequences_input = {}
      sequences_output = {}
      # Process input fields
      for field_name, field_config in self.config.input_fields.items():
         data = input_data[field_name]
         field_sequences = []
         # For each training sample...
         for seq_idx in valid_seq_indices:
            # Calculate the starting index for this sequence
            # Each sequence starts input_sequence_length steps after the previous one
            sample_idx = istart + seq_idx*self.config.input_sequence_length
            # Get field shape (works for both 1D and multi-dimensional)
            field_shape = data.shape[1:] if len(data.shape) > 1 else (1,)
            # Create an array to hold n consecutive inputs for this field
            # Shape will be (n_consecutive_inputs, input_sequence_length, *field_shape)
            consecutive_inputs = np.zeros((self.config.n_consecutive_inputs, 
                                    self.config.input_sequence_length, *field_shape))
            # Fill in each of the n consecutive input segments
            for i in range(self.config.n_consecutive_inputs):
               segment_start = sample_idx + i*self.config.input_sequence_length
               segment_end = segment_start + self.config.input_sequence_length
               if len(data.shape) == 1:
                  # For 1D data, reshape to (sequence_length, 1)
                  segment_data = jax.device_get(data[segment_start:segment_end])
                  consecutive_inputs[i] = segment_data.reshape(-1, 1)
               else:
                  # For multi-dimensional data
                  consecutive_inputs[i] = jax.device_get(data[segment_start:segment_end])
            field_sequences.append(consecutive_inputs)
         # Convert to JAX arrays and store
         sequences_input[field_name] = jnp.array(field_sequences)
      
      # Process output fields
      for field_name, field_config in self.config.output_fields.items():
         data = output_data[field_name]
         field_sequences = []
         # For each sequence, extract the output following the last input
         for seq_idx in valid_seq_indices:
            # Calculate starting index for this output
            # Each output starts after n consecutive inputs
            sample_idx = istart + seq_idx*self.config.input_sequence_length
            output_start = sample_idx + total_input_length
            output_end = output_start + self.config.output_sequence_length
            # Get field shape
            field_shape = data.shape[1:] if len(data.shape) > 1 else (1,)
            if len(data.shape) == 1:
               # For 1D data, reshape to (sequence_length, 1)
               segment_data = jax.device_get(data[output_start:output_end])
               field_sequences.append(segment_data.reshape(-1, 1))
            else:
               # For multi-dimensional data
               field_sequences.append(jax.device_get(data[output_start:output_end]))
         # Stack all sequences for this field
         sequences_output[field_name] = jnp.array(field_sequences)
      return sequences_input, sequences_output

   def process(self, data: pd.DataFrame, target_months: List[int] = None) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]:
      """Process data and split into training and validation sets.
      Returns:
         Tuple of (train_data, val_data) where each is a tuple of (input_dict, output_dict)
      """
      # Sort by time
      df = data.sort_values(by=self.config.time_column).copy()
      # Convert visibility to numeric BEFORE handling missing values
      df = preprocess_visibility(df)
      # Handle missing values
      df = self._handle_missing_values(df)
      # Create sequences
      input_sequences, output_sequences = self._create_sequences(df, target_months)
      # Split into training and validation
      n_sequences = len(next(iter(input_sequences.values())))
      n_train = int(n_sequences*self.config.train_ratio)

      train_indices = np.random.choice(n_sequences, n_train, replace=False)
      # Use remaining indices for validation
      val_indices = np.array([i for i in range(n_sequences) if i not in train_indices])
      # Split data using the random indices
      train_input = {k: v[train_indices] for k, v in input_sequences.items()}
      train_output = {k: v[train_indices] for k, v in output_sequences.items()}
      val_input = {k: v[val_indices] for k, v in input_sequences.items()}
      val_output = {k: v[val_indices] for k, v in output_sequences.items()}
      
      #train_input = {k: v[:n_train] for k, v in input_sequences.items()}
      #train_output = {k: v[:n_train] for k, v in output_sequences.items()}
      #train_input = self.input.normalize(train_input)
      #train_output = self.output.normalize(train_output)
      #val_input = {k: v[n_train:] for k, v in input_sequences.items()}
      #val_output = {k: v[n_train:] for k, v in output_sequences.items()}
      #val_input = self.input.normalize(val_input)
      #val_output = self.output.normalize(val_output)
      return (train_input, train_output), (val_input, val_output)
   
   def prepare_forecast_forcings(self, df: pd.DataFrame, forecast_date_str: str, model_constructor: Optional[Callable] = None) -> Any:
      """
      Prepare forcings for forecasting from a dataframe.
      Args:
         df: Input DataFrame containing the necessary columns
         forecast_date_str: Forecast start date/time in YYYYMMDDHH format
         model_constructor: Optional function to create model forcing object (receives dict)
      Returns:
         If model_constructor is provided: Model-specific forcing object
         Otherwise: Dictionary of transformed input data
      """
      forecast_dt = datetime.datetime.strptime(forecast_date_str, '%Y%m%d%H')
      # Calculate the start date for input sequence
      # Calculate total input time needed: input_sequence_length * n_consecutive_inputs time steps
      total_time_steps = self.config.input_sequence_length * self.config.n_consecutive_inputs
      # Estimate time per step (assuming regular intervals)
      # Get the time column from config
      time_col = self.config.time_column
      # Calculate the typical time step duration
      time_series = pd.to_datetime(df[time_col])
      time_diffs = time_series.diff().dropna()
      median_diff = time_diffs.median()
      # Calculate the start datetime for the forecast input
      start_dt = forecast_dt - (median_diff * total_time_steps)
      # Filter the dataframe to include only the required time range
      # Include a small buffer before start_dt to ensure we have enough data
      buffer_dt = start_dt - (median_diff * 5)  # 5 extra steps as buffer
      df_filtered = df[(df[time_col] >= buffer_dt) & (df[time_col] < forecast_dt)].copy()
      # Make sure we have enough data
      if len(df_filtered) < total_time_steps:
         raise ValueError(f"Not enough data for forecasting. Need {total_time_steps} records but only have {len(df_filtered)}")
      # Sort by time to ensure proper sequence
      df_filtered = df_filtered.sort_values(by=time_col)
      # If we have more data than needed, take the most recent time steps
      if len(df_filtered) > total_time_steps:
         df_filtered = df_filtered.iloc[-total_time_steps:].reset_index(drop=True)
      
      # Handle missing values in the filtered data
      df_filtered = self._handle_missing_values(df_filtered)
      # Transform all input fields
      input_data = self.input.transform(df_filtered, recalculate_stats=False)
      # Take the most recent data needed for forecasting
      sequences = {}
      for name, data in input_data.items():
         # Get field shape
         field_shape = data.shape[1:] if len(data.shape) > 1 else (1,)
         # Reshape to expected input format
         reshaped_data = np.zeros((self.config.n_consecutive_inputs, self.config.input_sequence_length, *field_shape))
         for i in range(self.config.n_consecutive_inputs):
            start_idx = i * self.config.input_sequence_length
            end_idx = start_idx + self.config.input_sequence_length
            segment = data[start_idx:end_idx]
            if len(data.shape) == 1: reshaped_data[i, :, 0] = segment
            else: reshaped_data[i, :, :] = segment
         sequences[name] = jnp.array(reshaped_data)
      # If a model constructor is provided, use it to create the forcing object
      if model_constructor is not None: return model_constructor(**sequences)
      # Otherwise, return the dictionary of sequences
      return sequences

def main():
   # Example usage
   # Define transformations
   logvis = FieldTransform(
      source_fields=['main_visibility'],
      transform_fn=lambda x: jnp.log10(x),
      #source_fields=['vis'],
      #transform_fn=lambda x: jnp.log10(x),
      output_name='logvis',
      shape=(1,),
      description='Log of visibility'
   )
   temperature = FieldTransform(
      source_fields=['temperature'],
      transform_fn=lambda x: jnp.array(x),
      output_name='temperature',
      shape=(1,),
      description='Temperature'
   )
   dewpoint = FieldTransform(
      source_fields=['dew_point'],
      transform_fn=lambda x: jnp.array(x),
      output_name='dewpoint',
      shape=(1,),
      description='Dew point'
   )
   wind = FieldTransform(
      source_fields=['wind_speed', 'wind_direction_degrees'],
      transform_fn=calculate_wind_components,
      output_name='wind',
      shape=(2,),
      description='Wind U,V components'
   )
   time = FieldTransform(
      source_fields=['date_time'],
      transform_fn=calculate_temporal_variables,
      output_name='time',
      shape=(4,),
      description='Temporal cyclic features',
      norm_method = 'none'
   )
   
   # Create configuration
   config = DataConfig(
      input_fields={
         'visibility': logvis,
         'temperature': temperature,  # Direct field use
         'dewpoint': dewpoint,     # Direct field use
         'wind': wind,    # Transformed field
         'time': time        # Will give us temporal features
      },
      output_fields={
         'visibility': logvis  # Transformed field
      },
      input_sequence_length=48,  # 24-hour history
      output_sequence_length=60,    # 30-hour prediction
      n_consecutive_inputs = 10,
      train_ratio=0.95,
      time_column='date_time',
      forecast_start_hour=6  # Start forecasts at 06Z
   )
   
   # Create preprocessor
   preprocessor = TimeSeriesPreprocessor(config)
   # Load and process data
   df = pd.read_csv('metar-VVNB.csv')
   (train_input, train_output), (val_input, val_output) = preprocessor.process(df)
   print("Training input shapes:")
   for k, v in train_input.items(): print(f"{k}: {v.shape}")
   print("\nTraining output shapes:")
   for k, v in train_output.items(): print(f"{k}: {v.shape}")
   print("\nValidation input shapes:")
   for k, v in val_input.items(): print(f"{k}: {v.shape}")
   print("\nValidation output shapes:")
   for k, v in val_output.items(): print(f"{k}: {v.shape}")
   #print(val_input['visibility'][0])
   #print(val_input['temperature'][0])
   #print(val_input['wind'][0])
   #print(val_input['time'][0])
   #print(val_output['visibility'][0])

if __name__ == "__main__":
   main()