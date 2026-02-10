#!/usr/bin/env python3
import jax, sys, os, datetime, psycopg2
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dajax.mlmodels.lstm import LSTM, Config
from dajax.utils.field import FieldTransform
from dajax.utils.timeseries_preprocessor import TimeSeriesPreprocessor, DataConfig
from dajax.utils.transform import initialize_transforms
from dajax.utils.inout import load_parameters, load_timeseries_preprocessor

def connect_to_db():
   """Connect to the PostgreSQL database."""
   try:
      conn = psycopg2.connect( host="172.19.20.51", database="amdb", user="amprd", password="PrdAm#2019")
      return conn
   except Exception as e:
      print(f"Error connecting to database: {e}")
      return None

def format_date_for_query(date_str):
   """Convert date string format (YYYYMMDDHH) to database format."""
   year = int(date_str[0:4])
   month = int(date_str[4:6])
   day = int(date_str[6:8])
   hour = int(date_str[8:10])
   # Create datetime object
   dt = datetime.datetime(year, month, day, hour, 0)
   return dt.strftime('%Y-%m-%d %H:%M:%S')

def get_forecast_data(station_id: str, forecast_date: str, days_back: int = 12):
   """
   Get data from PostgreSQL for forecast preparation with robust handling of missing timestamps.
   Args:
      station_id: The station ID (sid in the database)
      forecast_date: The starting date for forecast in format YYYYMMDDHH
      days_back: Number of days of history to retrieve
      
   Returns:
      DataFrame with complete time series data at 30-minute intervals
   """
   # Format the forecast date for query
   forecast_dt = format_date_for_query(forecast_date)
   # Calculate the start date (days_back days before forecast date)
   start_date = datetime.datetime.strptime(forecast_dt, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=days_back)
   start_dt = start_date.strftime('%Y-%m-%d %H:%M:%S')
   
   # Connect to database
   conn = connect_to_db()
   if not conn:
      raise Exception("Failed to connect to database")
   try:
      cur = conn.cursor()
      query = """
      SELECT sid, date, hhmm, dd, ff, t, td, vis, pmsl
      FROM metar_obs
      WHERE sid = %s AND date >= %s AND date <= %s
      ORDER BY date, hhmm
      """
      cur.execute(query, (station_id, start_dt, forecast_dt))
      rows = cur.fetchall()
      if not rows:
         raise ValueError(f"No data found for station {station_id} between {start_dt} and {forecast_dt}")
         
      # Create DataFrame
      columns = ['sid', 'date', 'hhmm', 'wind_direction_degrees', 
               'wind_speed', 'temperature', 'dew_point', 
               'main_visibility', 'altimeter']
      df = pd.DataFrame(rows, columns=columns)
      # Replace -9999 values with NaN
      for col in df.columns:
         # Skip non-numeric columns like 'sid', 'date'
         if df[col].dtype.kind in 'iuf': df.loc[df[col] == -9999, col] = np.nan
      # Convert date and hhmm to datetime
      df['hour'] = df['hhmm'].apply(lambda x: int(x/100))
      df['minute'] = df['hhmm'].apply(lambda x: x % 100)
      df['wind_speed'] = df['wind_speed'].apply(lambda x: x/10)
      df['temperature'] = df['temperature'].apply(lambda x: x/10)
      df['dew_point'] = df['dew_point'].apply(lambda x: x/10)
      
      # Create proper datetime column
      df['date_time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h') + pd.to_timedelta(df['minute'], unit='m')
      # Sort by datetime
      df = df.sort_values('date_time')
      # Drop unnecessary columns
      df = df.drop(['hhmm', 'hour', 'minute'], axis=1)
      # Check data coverage before proceeding
      time_range = (df['date_time'].max() - df['date_time'].min()).total_seconds() / 3600  # in hours
      if time_range < days_back * 24 * 0.75:  # Check if we have at least 75% of the expected time range
         print(f"Warning: Data covers only {time_range:.1f} hours out of {days_back*24} hours requested")
      
      # Create complete timestamp range with 30-minute intervals
      complete_range = pd.date_range(
         start=df['date_time'].min(),
         end=df['date_time'].max(),
         freq='30T'
      )
      # Check if we have enough data points for meaningful interpolation
      if len(df) < len(complete_range) * 0.5:  # Less than 50% of expected points
         print(f"Warning: Sparse data - only {len(df)} points out of {len(complete_range)} expected")
      # Reindex dataframe to include all timestamps
      df = df.set_index('date_time')
      # Verify no duplicate indices before reindexing
      if df.index.duplicated().any():
         print("Warning: Found duplicate timestamps after initial cleanup.")
         df = df[~df.index.duplicated(keep='first')]
      df = df.reindex(complete_range)
      df = df.reset_index()
      df = df.rename(columns={'index': 'date_time'})
      
      # Handle missing values with appropriate strategy
      # First handle outliers if needed
      # Then interpolate
      df = df.interpolate(method='linear', limit_direction='both')
      # Final check: ensure we have enough data for forecast
      if df.isna().any().any():
         print("Warning: DataFrame still contains NaN values after interpolation")
         # Count how many columns have NaNs
         nan_columns = df.columns[df.isna().any()].tolist()
         print(f"Columns with NaNs: {nan_columns}")
         # Fill remaining NaNs with appropriate defaults or methods
         df = df.fillna(method='ffill').fillna(method='bfill')  # Use forward fill then backward fill
      return df
   except Exception as e:
      print(f"Error querying database: {e}")
      raise
   finally:
      if conn: conn.close()

def plot_forecast(forecast, initial_date, station, output_dir):
   """
   Plot forecast results with uncertainty.
   Args:
      forecast: Model predictions
      initial_date: Starting date of the forecast (YYYYMMDDHH format)
      station: Station identifier
      output_dir: Directory to save output
   """
   os.makedirs(output_dir, exist_ok=True)
   # Convert forecast date string to datetime object
   forecast_dt = datetime.datetime.strptime(initial_date, '%Y%m%d%H')
   # Create timeline for x-axis
   timeline = [forecast_dt + datetime.timedelta(minutes=30*i) for i in range(len(forecast.vispred[:,0]))]
   # Convert from log10 visibility to actual visibility
   vis_q50 = 10**np.array(forecast.vispred[:,0])
   invalid = vis_q50 > 10000.; vis_q50[invalid] = 10000.
   print(vis_q50)
   
   # Create figure
   plt.figure(figsize=(15, 8))
   # Plot median forecast with uncertainty range
   plt.plot(timeline, vis_q50, 'r-', linewidth=2, label='Median Forecast')
   #plt.fill_between(timeline, vis_q10, vis_q90, alpha=0.3, color='red', label='80% Prediction Interval')
   # Add thresholds for visibility categories
   plt.axhline(y=1500, color='b', linestyle='--', label='Low Visibility (1500m)')
   plt.axhline(y=6000, color='g', linestyle='--', label='Moderate Visibility (6000m)')
   # Format the plot
   plt.xlabel('Time')
   plt.ylabel('Visibility (m)')
   plt.title(f'Visibility Forecast for {station} starting {forecast_dt.strftime("%Y-%m-%d %H:%M")}')
   plt.grid(True, alpha=0.3)
   plt.legend()
   # Format x-axis labels for better readability
   plt.gcf().autofmt_xdate()
   # Save figure
   filename = f'{station}{initial_date[8:10]}.png'
   plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
   plt.close()
   
   # Save raw data
   df = pd.DataFrame({
      'datetime': timeline,
      'vis_q50': vis_q50.flatten(),
   })
   filename = f'{station}{initial_date[8:10]}.csv'
   df.to_csv(os.path.join(output_dir, filename), index=False)

if len(sys.argv) > 2:
   station = sys.argv[1]
   initial_date = sys.argv[2]
forecast_start_hour = int(initial_date[8:10])
parameter_dir = os.environ['HOME']+'/visexp/lstm/parameter'
forecast_dir = os.environ['HOME']+'/visexp/lstm/forecast/'+initial_date[:6]+'/'+initial_date[:8]
initialize_transforms()

# Preprocessor
preprocessor_file = os.path.join(parameter_dir, station+str(forecast_start_hour)+'_transform.pkl')
preprocessor = load_timeseries_preprocessor(preprocessor_file)
#preprocessor = TimeSeriesPreprocessor(dataconfig)

# Model
input_fields = preprocessor.input.transform_to_info(preprocessor.config.input_sequence_length)
output_fields = preprocessor.output.transform_to_info(preprocessor.config.output_sequence_length)
#print(input_fields)
config = Config(hidden_sizes=[256,128,],  # Two-layer LSTM as specified
               input_fields=input_fields, output_fields=output_fields)
model = LSTM(config=config)

# Forcings
df = get_forecast_data(station, initial_date, preprocessor.config.n_consecutive_inputs)
forcings = preprocessor.prepare_forecast_forcings(df, initial_date, model._Forcing)
print(input_fields)
#print(10**forcings.visibility)
#print(forcings.wind.shape)
#print(forcings.time)
# Normalization
model._normalized_mode = False
if model._normalized_mode: forcings = preprocessor.input.normalize(forcings)

# Forecast
parameter_file = os.path.join(parameter_dir, station+str(forecast_start_hour)+'.pkl')
param = load_parameters(model._Param, parameter_file)
state0 = model.default_state(param)
final_state, _ = model.integrate(state0, param, preprocessor.config.n_consecutive_inputs, forcings=forcings, save_freq=1)
forecast = model.mod2phy(final_state, param)
os.makedirs(os.path.dirname(os.path.abspath(forecast_dir)), exist_ok=True)
plot_forecast(forecast, initial_date, station, forecast_dir)
