#!/usr/bin/env python3
import jax, sys, os
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dajax.mlmodels.lstm import LSTM, Config
from dajax.estimators.firstorder import FOBatchEstimator
from dajax.estimators.adam import Adam
from dajax.estimators.adabelief import AdaBelief
from dajax.estimators.adamw import AdamW
from dajax.losses.standard import L2Loss, L1Loss
from dajax.losses.penalty import ThresholdWeightedLoss
from dajax.losses.quantile import QuantileLoss
from dajax.losses.base import CompoundLoss
from dajax.utils.field import FieldTransform
from dajax.utils.timeseries_preprocessor import TimeSeriesPreprocessor, DataConfig
from dajax.utils.transform import initialize_transforms
from dajax.utils.transform_registry import get_transform
from dajax.utils.inout import save_parameters, save_timeseries_preprocessor

if len(sys.argv) > 2:
   station = sys.argv[1]
   forecast_start_hour = int(sys.argv[2])
initialize_transforms()
log10_transform = get_transform("log10")
identity_transform = get_transform("identity")
calculate_dewpoint_depression = get_transform("dewpoint_depression")
calculate_wind_components = get_transform("wind_components")
calculate_temporal_variables = get_transform("temporal_variables")

# Input and output variables
logvis = FieldTransform(
   source_fields=['main_visibility'],
   transform_fn=log10_transform,  # actual function
   transform_name="log10",        # string identifier
   output_name='logvis',
   shape=(1,),
   description='Log of visibility',
   norm_method = 'minmax'
)
temperature = FieldTransform(
   source_fields=['temperature'],
   transform_fn=identity_transform,
   transform_name="identity",
   output_name='temperature',
   shape=(1,),
   description='Temperature'
)
dewpoint = FieldTransform(
   source_fields=['dew_point'],
   transform_fn=identity_transform,
   transform_name="identity",
   output_name='dewpoint',
   shape=(1,),
   description='Dew point'
)
depression = FieldTransform(
   source_fields=['temperature','dew_point'],
   transform_fn=calculate_dewpoint_depression,
   transform_name="dewpoint_depression",
   output_name='depression',
   shape=(1,),
   description='Dew point depression'
)
pressure = FieldTransform(
   source_fields=['altimeter'],
   transform_fn=identity_transform,
   transform_name="identity",
   output_name='pressure',
   shape=(1,),
   description='Surface pressure'
)
wind = FieldTransform(
   source_fields=['wind_speed', 'wind_direction_degrees'],
   transform_fn=calculate_wind_components,
   transform_name="wind_components",
   output_name='wind',
   shape=(2,),
   description='Wind U,V components'
)
time = FieldTransform(
   source_fields=['date_time'],
   transform_fn=calculate_temporal_variables,
   transform_name="temporal_variables",
   output_name='time',
   shape=(4,),
   description='Temporal cyclic features',
   norm_method = 'none'
)
logvispred = FieldTransform(
   source_fields=['main_visibility'],
   transform_fn=log10_transform,  # actual function
   transform_name="log10",
   output_name='logvis',
   shape=(1,),
   description='Log of visibility',
   norm_method = 'minmax'
)
logvispred_q10 = FieldTransform(
   source_fields=['main_visibility'],
   transform_fn=log10_transform,  # actual function
   transform_name="log10",
   output_name='logvis_q10',
   shape=(1,),
   description='Log of visibility',
   norm_method = 'minmax'
)
logvispred_q50 = FieldTransform(
   source_fields=['main_visibility'],
   transform_fn=log10_transform,  # actual function
   transform_name="log10",
   output_name='logvis_q50',
   shape=(1,),
   description='Log of visibility',
   norm_method = 'minmax'
)
logvispred_q90 = FieldTransform(
   source_fields=['main_visibility'],
   transform_fn=log10_transform,  # actual function
   transform_name="log10",
   output_name='logvis_q90',
   shape=(1,),
   description='Log of visibility',
   norm_method = 'minmax'
)

# Configuration
dataconfig = DataConfig(
   input_fields={
      'visibility': logvis,
      'temperature': temperature,  # Direct field use
      'dewpoint': dewpoint,     # Direct field use
      #'depression': depression,
      #'pressure': pressure,     # Direct field use
      'wind': wind,    # Transformed field
      'time': time        # Will give us temporal features
   },
   output_fields={
      'vispred': logvispred  # Transformed field
      #'vispred_q10': logvispred_q10,  # Transformed field
      #'vispred_q50': logvispred_q50,  # Transformed field
      #'vispred_q90': logvispred_q90,  # Transformed field
   },
   input_sequence_length=48,  # 24-hour history
   output_sequence_length=60,    # 30-hour prediction
   n_consecutive_inputs = 6,
   train_ratio=0.9,
   time_column='date_time',
   forecast_start_hour=forecast_start_hour # Start forecasts at 
)
months = [12,1,2,3]

# Preprocessor
preprocessor = TimeSeriesPreprocessor(dataconfig)
# Training data
df = pd.read_csv('./data/metar-'+station+'.csv', low_memory=False)
(train_input, train_output), (val_input, val_output) = preprocessor.process(df, months)
# Model
input_fields = preprocessor.input.transform_to_info(dataconfig.input_sequence_length)
output_fields = preprocessor.output.transform_to_info(dataconfig.output_sequence_length)
print(input_fields)
config = Config(hidden_sizes=[256,128,],  # Two-layer LSTM as specified
               input_fields=input_fields, output_fields=output_fields)
model = LSTM(config=config)

# Check and adjust training data
ndevice = 1 #jax.local_device_count()
first_train_key = next(iter(train_input.keys()))
n_train_samples = train_input[first_train_key].shape[0]
train_remainder = n_train_samples % ndevice
if train_remainder != 0:
   # Calculate how many samples to keep (round down to nearest multiple of device count)
   n_train_keep = n_train_samples - train_remainder
   # Trim all training arrays to an even multiple of device count
   train_input = {k: v[:n_train_keep] for k, v in train_input.items()}
   train_output = {k: v[:n_train_keep] for k, v in train_output.items()}
   print(f"Adjusted training samples from {n_train_samples} to {n_train_keep} to be divisible by {ndevice} devices")
# Check and adjust validation data
first_val_key = next(iter(val_input.keys()))
n_val_samples = val_input[first_val_key].shape[0]
val_remainder = n_val_samples % ndevice
if val_remainder != 0:
   # Calculate how many samples to keep
   n_val_keep = n_val_samples - val_remainder
   # Trim all validation arrays
   val_input = {k: v[:n_val_keep] for k, v in val_input.items()}
   val_output = {k: v[:n_val_keep] for k, v in val_output.items()}
   print(f"Adjusted validation samples from {n_val_samples} to {n_val_keep} to be divisible by {ndevice} devices")

# Normalization
model._normalized_mode = False
if model._normalized_mode:
   train_input = preprocessor.input.normalize(train_input)
   train_output = preprocessor.output.normalize(train_output)
   val_input = preprocessor.input.normalize(val_input)
   val_output = preprocessor.output.normalize(val_output)
train_forcings = model._Forcing(**train_input)
train_truth = model._PhyState(**train_output)
val_forcings = model._Forcing(**val_input)
val_truth = model._PhyState(**val_output)
#print(train_input['wind'].shape, train_output['vispred'].shape)

# Training
loss = CompoundLoss({'vispred': L2Loss()})
#loss = CompoundLoss({'vispred': ThresholdWeightedLoss(threshold_low=jnp.log10(2000),
                        #threshold_high=jnp.log10(8000), below_weight=5., mode='below')})
#loss = CompoundLoss({'vispred_q10': QuantileLoss(0.1),
#                     'vispred_q50': QuantileLoss(0.5),
#                     'vispred_q90': QuantileLoss(0.9),})
optimizers = {
   'Adam': Adam(learning_rate=0.001, max_iter=10000, patience=30),
   'AdamW': AdamW(learning_rate=0.001, max_iter=10000, patience=30),
   #'AdaBelief': AdaBelief(learning_rate=0.001, max_iter=10000, patience=30),
}

# Initial state and parameters
prev_loss = np.inf
for seed in range(100):
   key = jax.random.PRNGKey(seed)
   param0 = model.create_default_param(key)
   #param0 = load_parameters(model._Param, './parameter/'+station+str(forecast_start_hour)+'.pkl')
   state0 = model.default_state(param0)
   
   #(train_input, train_output), (val_input, val_output) = preprocessor.process(df)
   #train_forcings = model._Forcing(**train_input)
   #train_truth = model._PhyState(**train_output)
   #val_forcings = model._Forcing(**val_input)
   #val_truth = model._PhyState(**val_output)
   
   # Run estimation
   for name, optimizer in optimizers.items():
      estimator = FOBatchEstimator(model, loss, optimizer)
      val_loss0 = estimator._loss_fn(param0, state0, val_forcings, val_truth)
      param, diagnostics = estimator.estimate(
         param0=param0, state0=state0, forcings=train_forcings, truth=train_truth,
         val_forcings=val_forcings, val_truth=val_truth
      )
      val_loss = estimator._loss_fn(param, state0, val_forcings, val_truth)
      print(seed, name, val_loss0, val_loss)
      if val_loss < prev_loss:
         prev_loss = val_loss
         best_param = param
         print('New best parameter:', seed, name, val_loss)
param = best_param

# Output
save_path = './parameter/'+station+str(forecast_start_hour)+'_transform.pkl'
save_timeseries_preprocessor(preprocessor, save_path)
save_path = './parameter/'+station+str(forecast_start_hour)+'.pkl'
save_parameters(param, save_path)
print(f"\nTraining completed in {diagnostics['niter']} iterations")
#print(f"Best validation loss: {diagnostics['best_val_loss'][:diagnostics['niter']]}")
#print(f"Validation loss: {diagnostics['val_loss'][:diagnostics['niter']]}")
