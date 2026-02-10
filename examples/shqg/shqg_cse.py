#!/usr/bin/env python3
import jax, os
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
from dajax.models.shqg import SHQG, Config, State, ObsState, Param
from dajax.schemes.ida import IDA
from dajax.obs.observation import ObsSetting, Observation
from dajax.obs.likelihood import Logistic, Predefined, MInfinite
from dajax.obs.obsoperator import IdentityMap, L2NormMap
from dajax.obs.obslocation import RandomMask, RegularMask, AreaMask
from dajax.obs.obstime import InitialTimeMask, DATimeMask
from dajax.utils.diagnostics import RMSEScore, MEScore, Diagnostics, no_selector
from dajax.utils.inout import write_states, write_scores
from dajax.core.daosse import DAConfig, IDACSE

seeds = list(range(3,6))
nmembers = [100]
thresholds = [17.0]
rho = 1.0
biases = [0.0]
fractions = [0.32]
dt = 6*3600.
tspinup_model = 365*24*3600 # 1 year
tspinup_da = 1*24*3600 # 1 day
tend = 10*365*24*3600 # 10 years
dawindow = 4
obsfreq = 4
obserr = 4.*jnp.sqrt(3)/jnp.pi
ctlwindow = 8
ctlfreq = 1
ctlobserr = 0.01*4.*jnp.sqrt(3)/jnp.pi
rescale = 10.
save_freq = 1
T = 21
Tgrid = 31
nlev = 3
controlled_start = (125.0,25.0); controlled_end = (150.0,50.0)
base_dir = Path(f"./results")

for nmember in nmembers:
   for threshold in thresholds:
      for bias in biases:
         for fraction in fractions:
            for seed in seeds:
               if seed == 3: unmodified_start = (125.0,25.0); unmodified_end = (150.0,50.0)
               elif seed == 4: unmodified_start = (100.0,0.0); unmodified_end = (175.0,75.0)
               elif seed == 5: unmodified_start = (0.0,0.0); unmodified_end = (180.0,90.0)
               
               # Files
               output_dir = base_dir / f"seed{seed}"
               output_dir.mkdir(parents=True, exist_ok=True)
               states_file = output_dir / f"n{nmember}_r{rho}_states.nc"
               scores_file = output_dir / f"n{nmember}_r{rho}_scores.nc"
               #if os.path.isfile(states_file) and os.path.isfile(scores_file): continue

               # Config
               daconfig = DAConfig(dawindow=dawindow, obsfreq=obsfreq, nmember=nmember, obserr=obserr, rho=rho)
               ctlconfig = DAConfig(dawindow=ctlwindow, obsfreq=ctlfreq, nmember=nmember, obserr=ctlobserr, rho=0.)
               config = Config(T=T, Tgrid=Tgrid, nlev=nlev, dt=dt)

               # Model
               law = SHQG.default_param()
               nature = SHQG(config=config, param=law)
               model = SHQG(config=config, param=law)

               # Key
               key = jax.random.PRNGKey(seed)
               key, key0, key1, key2 = jax.random.split(key, 4)

               # Observations
               mask = RandomMask(key0, fraction=fraction, grid_info=nature.grid_info)
               mask2 = RegularMask(start=(0,0), spacing=(3,3), grid_info=nature.grid_info)
               time_mask = DATimeMask(ntime=int(dawindow/obsfreq)+1)
               settings = {'u': ObsSetting(distribution = Logistic(scale=obserr,bias=bias),
                                          mapper = IdentityMap('u'), mask = mask,
                                          timemask = time_mask, biasmin = 0.0, biasmax = 0.0),
                           'v': ObsSetting(distribution = Logistic(scale=obserr,bias=bias),
                                          mapper = IdentityMap('v'), mask = mask,
                                          timemask = time_mask, biasmin = 0.0, biasmax = 0.0)}
                           #'u2': ObsSetting(distribution = Logistic(scale=obserr,bias=bias),
                                          #mapper = IdentityMap('u'), mask = mask2,
                                          #timemask = time_mask, threshold = threshold),
                           #'v2': ObsSetting(distribution = Logistic(scale=obserr,bias=bias),
                                          #mapper = IdentityMap('v'), mask = mask2,
                                          #timemask = time_mask, threshold = threshold)}
               observation = Observation(settings, obsstate_class=ObsState)
               
               # Control
               unmodified_mask = AreaMask(start=unmodified_start, end=unmodified_end,
	               grid_info=nature.grid_info, use_latlon=True, select_inside=False)
               controlled_mask = AreaMask(start=controlled_start, end=controlled_end,
	               grid_info=nature.grid_info, use_latlon=True, select_inside=True, levels=2)
               initial_mask = InitialTimeMask(ntime=int(ctlwindow/ctlfreq)+1)
               da_mask = DATimeMask(ntime=int(ctlwindow/ctlfreq)+1)
               #mask3 = RegularMask(start=(0,0), spacing=(3,3), grid_info=nature.grid_info)
               settings = {'u': ObsSetting(distribution = Predefined(scale=rescale*ctlobserr,bias=bias),
                                          mapper = IdentityMap('u'), mask = unmodified_mask,
                                          timemask = initial_mask, biasmin = 0.0, biasmax = 0.0),
                           'v': ObsSetting(distribution = Predefined(scale=rescale*ctlobserr,bias=bias),
                                          mapper = IdentityMap('v'), mask = unmodified_mask,
                                          timemask = initial_mask, biasmin = 0.0, biasmax = 0.0),
                           'wind': ObsSetting(distribution = MInfinite(scale=ctlobserr),
                                          mapper = L2NormMap(('u','v')), mask = controlled_mask,
                                          timemask = da_mask, threshold = threshold)}
               control = Observation(settings, obsstate_class=ObsState)
               
               # Diagnostics
               me = MEScore({'u': no_selector,'v': no_selector})
               rmse = RMSEScore({'u': no_selector,'v': no_selector})
               diagnostics = Diagnostics([me,rmse])

               # DA scheme
               scheme = IDA(rho=rho)
               ctlscheme = IDA(rho=0.)

               # DA system
               da_system = IDACSE(nature, model, observation, scheme, diagnostics, daconfig, control, ctlscheme, ctlconfig)
               param = da_system.ensemble.random_param(key0, law, noise_scale=0.0) 

               # Spinup
               true = nature.default_state(law)
               true = nature.random_state(key1, law, true, noise_scale=1.e-7)
               nstep = int(tspinup_model/dt)
               true, _ = nature.integrate(true, law, nstep)
               ens = da_system.ensemble.random_state(key2, param, true, noise_scale=1.e-6)
               ncycle = int(tspinup_da/(dawindow*dt))
               key, state, score = da_system.cycle(key, true, ens, law, param, ncycle)
               diagnostics.print_comparison([score.forecast, score.analysis], ['Forecast', 'Analysis'])
               
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
               lon = jnp.where(lon >= 180, lon - 360, lon)
               coords = {'time': time, 'lev': jnp.arange(nlev), 'lat': nature.grid_info['lat'], 'lon': lon}
               dims = {'u': ('time', 'lev', 'lat', 'lon'),
                        'v': ('time', 'lev', 'lat', 'lon')}
               states = {"truth": state_trajectory.truth,
                        "forecast": state_trajectory.forecast,
                        "analysis": state_trajectory.analysis}
               write_states(str(states_file), states, time, coords=coords, dims=dims)
               scores = {"forecast": {score_type: score[1] for score_type, score in score_trajectory.forecast.items()},
                        "analysis": {score_type: score[1] for score_type, score in score_trajectory.analysis.items()}}
               write_scores(str(scores_file), scores, time, coords=coords)
