#!/usr/bin/env python3
import jax, os
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
from dajax.models.speedy import SPEEDY, Config, State, ObsState, Param
from dajax.models.speedy.util import TimeInfo
from dajax.schemes.etkf import ETKF
from dajax.schemes.ida import IDA
from dajax.obs.observation import ObsSetting, Observation
from dajax.obs.likelihood import Gaussian, Logistic
from dajax.obs.obsoperator import IdentityMap, ExponentialMap, PressureLevelMap
from dajax.obs.obslocation import RandomMask, RegularMask
from dajax.obs.obstime import DATimeMask
from dajax.utils.diagnostics import RMSEScore, MEScore, Diagnostics, no_selector, level_selector
from dajax.utils.inout import write_states, write_scores
from dajax.core.daosse import DAConfig, IDAOSSE

seeds = list(range(3))
nmembers = [900]
rhos = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
rhos = [4.0,]
biases = [-1.0,0.0,1.0]
biases = [0.0]
fractions = [0.02,0.04,0.08,0.16,0.32,0.64]
#fractions = [0.32]
dt = 2400.
tspinup_model = 365*24*3600 # 1 year
tspinup_da = 1*24*3600 # 1 day
tend = 2*365*24*3600 # 10 years
dawindow = 9
obsfreq = 9
uverr = 1.*jnp.sqrt(3)/jnp.pi
terr = 1.*jnp.sqrt(3)/jnp.pi
qerr = 0.1*jnp.sqrt(3)/jnp.pi
pserr = 1.*jnp.sqrt(3)/jnp.pi # hPa
uvt_levels = [925., 850., 700., 500., 300., 200., 100.] # hPa, 1000 hPa always below surface
q_levels = [925., 850., 700., 500., 300.] # hPa
save_freq = 1
trunc = 30
kx = 8
time0 = TimeInfo.create(year=2000, month=1, day=1)
base_dir = Path(f"./results")

for nmember in nmembers:
   for rho in rhos:
      for bias in biases:
         for fraction in fractions:
            for seed in seeds:
               # Files
               output_dir = base_dir / f"seed{seed}"
               output_dir.mkdir(parents=True, exist_ok=True)
               states_file = output_dir / f"n{nmember}_r{rho}_b{bias}_f{fraction}_states.nc"
               scores_file = output_dir / f"n{nmember}_r{rho}_b{bias}_f{fraction}_scores.nc"
               if os.path.isfile(states_file) and os.path.isfile(scores_file): continue

               # Config
               daconfig = DAConfig(dawindow=dawindow, obsfreq=obsfreq, nmember=nmember, rho=rho)
               config = Config.create(trunc=trunc, dt=dt, kx=kx)

               # Model
               law = SPEEDY.default_param()
               nature = SPEEDY(config=config)
               model = SPEEDY(config=config)

               # Key
               key = jax.random.PRNGKey(seed)
               key, key0, key1, key2 = jax.random.split(key, 4)

               # Observation system
               mask = RandomMask(key0,fraction=fraction,grid_info=nature.grid_info)
               timemask = DATimeMask(ntime=int(dawindow/obsfreq)+1)
               settings = {'u': ObsSetting(distribution = Logistic(scale=uverr,bias=bias),
                                          mapper = PressureLevelMap('u', nature.vertical_grid.fsg, uvt_levels),
                                          mask = mask, timemask = timemask, biasmin = 0.0, biasmax = 0.0),
                           'v': ObsSetting(distribution = Logistic(scale=uverr,bias=bias),
                                          mapper = PressureLevelMap('v', nature.vertical_grid.fsg, uvt_levels),
                                          mask = mask, timemask = timemask, biasmin = 0.0, biasmax = 0.0),
                           't': ObsSetting(distribution = Logistic(scale=terr,bias=bias),
                                          mapper = PressureLevelMap('t', nature.vertical_grid.fsg, uvt_levels),
                                          mask = mask, timemask = timemask, biasmin = 0.0, biasmax = 0.0),
                           'q': ObsSetting(distribution = Logistic(scale=qerr,bias=bias),
                                          mapper = PressureLevelMap('q', nature.vertical_grid.fsg, q_levels),
                                          mask = mask, timemask = timemask, biasmin = 0.0, biasmax = 0.0),
                           'ps': ObsSetting(distribution = Logistic(scale=pserr,bias=bias),
                                          mapper = ExponentialMap('ps', 1000.),
                                          mask = mask, timemask = timemask, biasmin = 0.0, biasmax = 0.0),}
               observation = Observation(settings, obsstate_class=ObsState)

               # Diagnostics
               me = MEScore({'u': level_selector,'v': level_selector,'t': level_selector,'q': level_selector,'ps': no_selector})
               rmse = RMSEScore({'u': level_selector,'v': level_selector,'t': level_selector,'q': level_selector,'ps': no_selector})
               diagnostics = Diagnostics([me,rmse])

               # DA scheme
               scheme = IDA(rho=rho)  

               # DA system
               da_system = IDAOSSE(nature, model, observation, scheme, diagnostics, daconfig)
               param = da_system.ensemble.random_param(key0, law, noise_scale=0.0) 

               # Spinup
               true = nature.default_state(law, time0)
               true = nature.random_state(key1, law, true, noise_scale=1.)
               nstep = int(tspinup_model/dt)
               true, _ = nature.integrate(true, law, nstep)
               ens = da_system.ensemble.random_state(key2, param, true, noise_scale=1.)
               ncycle = int(tspinup_da/(dawindow*dt))
               key, state, score = da_system.cycle(key, true, ens, law, param, ncycle)
               diagnostics.print_comparison([score.forecast, score.analysis], ['Forecast', 'Analysis'])
               #sys.exit(0)
               # DA cycles
               ncycle = int(tend/(dawindow*dt))
               key, state_trajectory, score_trajectory = da_system.cycle(key, state.truth, state.analysis, law, param, ncycle, save_freq)

               # Diagnostics
               forecast_scores = score_trajectory.forecast
               analysis_scores = score_trajectory.analysis
               forecast_score = diagnostics.average(forecast_scores)
               analysis_score = diagnostics.average(analysis_scores)
               diagnostics.print_comparison([forecast_score, analysis_score], ['Forecast', 'Analysis'])

               # Output
               time = jnp.arange(ncycle//save_freq)*save_freq*daconfig.dawindow*config.dt
               lon = nature.grid_info['lon']
               lat = model.transformer.lat_full
               coords = {'time': time, 'lev': jnp.arange(kx), 'lat': lat, 'lon': lon}
               dims = {'u': ('time', 'lev', 'lat', 'lon'),
                        'v': ('time', 'lev', 'lat', 'lon'),
                        't': ('time', 'lev', 'lat', 'lon'),
                        'q': ('time', 'lev', 'lat', 'lon'),
                        'ps': ('time', 'lat', 'lon'),}
               states = {"truth": state_trajectory.truth,
                        "forecast": state_trajectory.forecast,
                        "analysis": state_trajectory.analysis}
               write_states(str(states_file), states, time, coords=coords, dims=dims)
               dim_names = {
                  'u': ['level'],
                  'v': ['level'],
                  't': ['level'],
                  'q': ['qlevel'],
                  'ps': ['surface'],}
               scores = {"forecast": {score_type: score[1] for score_type, score in score_trajectory.forecast.items()},
                        "analysis": {score_type: score[1] for score_type, score in score_trajectory.analysis.items()}}
               write_scores(str(scores_file), scores, time, dim_names=dim_names)
