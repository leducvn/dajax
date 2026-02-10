#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from typing import Generic, NamedTuple, Tuple, Dict, Optional
from functools import partial
from abc import abstractmethod
from dajax.models.base import Model, State, PhyState, ObsState, Param, Config
from dajax.obs.observation import Observation
from dajax.schemes.base import Scheme
from dajax.utils.diagnostics import Diagnostics
from dajax.utils.ensemble import Ensemble

class DAConfig(NamedTuple):
   """Configuration for data assimilation system."""
   dawindow: int       # Number of timesteps in DA window
   obsfreq: int        # Frequency of observations
   nmember: int        # Number of ensemble members
   obserr: float
   rho: float

class DAState(NamedTuple):
   """State container for DA system including diagnostics."""
   truth: State
   forecast: State
   analysis: State

class DAScore(NamedTuple):
   """Container for DA diagnostics scores for forecast and analysis."""
   forecast: Dict[str,Tuple[ObsState,ObsState]]  # Score type -> Result for forecast
   analysis: Dict[str,Tuple[ObsState,ObsState]]  # Score type -> Result for analysis

class BaseDAOSSE(Generic[State, PhyState, ObsState, Param, Config]):
   """Data Assimilation System combining model and DA scheme."""
   def __init__(self, nature: Model[State, PhyState, Param, Config], model: Model[State, PhyState, Param, Config], 
               observation: Observation, scheme: Scheme[State, ObsState], diagnostics: Diagnostics, config: DAConfig):
      self.nature = nature
      self.model = model
      self.observation = observation
      self.scheme = scheme
      self.diagnostics = diagnostics
      self.config = config
      self.ensemble = Ensemble(model, config.nmember, observation)
   
   def verify(self, dastate: DAState, law0: Param, param0: Param) -> DAScore:
      """Compute diagnostics scores for one DA window."""
      phytruth = self.nature.mod2phy(dastate.truth, law0)
      phystatef = self.ensemble.mod2phy(dastate.forecast, param0)
      phystatea = self.ensemble.mod2phy(dastate.analysis, param0)
      obstruth = self.observation.Hforward(phytruth)
      obsstatef = self.ensemble.Hforward(phystatef)
      obsstatea = self.ensemble.Hforward(phystatea)
      obsstatef = self.scheme.mean(obsstatef)
      obsstatea = self.scheme.mean(obsstatea)
      forecast_scores = self.diagnostics.compute(obstruth, obsstatef)
      analysis_scores = self.diagnostics.compute(obstruth, obsstatea)
      return DAScore(forecast=forecast_scores, analysis=analysis_scores)

   @partial(jax.jit, static_argnames=['self', 'ncycle', 'save_freq'])
   def cycle(self, key: jax.Array, true0: State, state0: State, law0: Param, param0: Param, ncycle: int, save_freq: Optional[int] = None) -> Tuple[jax.Array, DAState, DAScore]:
      def step(carry, _):
         key, true_state, ens_state = carry
         key, state = self.forward(key, true_state, ens_state, law0, param0)
         score = self.verify(state, law0, param0)
         return (key, state.truth, state.analysis), (state, score)
      
      if save_freq is None:
         # Only return final states
         (key, true_final, ens_final), (_, score) = jax.lax.scan(step, (key, true0, state0), jnp.arange(ncycle))
         state = DAState(true_final, ens_final, ens_final)
         score = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), score) # Average scores over cycles
         return key, state, score
      else:
         # Inner loop: run for save_freq cycles
         def inner_loop(carry, _):
            key, true_state, ens_state = carry
            # Run save_freq steps but discard intermediate results
            def inner_step(carry, _):
               key, true_state, ens_state = carry
               key, state = self.forward(key, true_state, ens_state, law0, param0)
               return (key, state.truth, state.analysis), None
            (key, true_new, ens_new), _ = jax.lax.scan(inner_step, (key, true_state, ens_state), jnp.arange(save_freq-1))
            key, state = self.forward(key, true_new, ens_new, law0, param0)
            score = self.verify(state, law0, param0)
            #jax.debug.print('Fcst: {x}, Ana: {y}', x=score.forecast['rmse'][1][0], y=score.analysis['rmse'][1][0])
            #statemean = DAState(state.truth, self.scheme.mean(state.forecast), self.scheme.mean(state.analysis))
            phytruth = self.nature.mod2phy(state.truth, law0)
            phystatef = self.ensemble.mod2phy(state.forecast, param0)
            phystatea = self.ensemble.mod2phy(state.analysis, param0)
            phystatef = self.scheme.mean(phystatef)
            phystatea = self.scheme.mean(phystatea)
            statemean = DAState(phytruth, phystatef, phystatea)
            return (key, state.truth, state.analysis), (statemean, score)
         # Outer loop: save states every save_freq cycles
         (key, _, _), (state_trajectory, score_trajectory) = jax.lax.scan(inner_loop, (key, true0, state0), jnp.arange(ncycle//save_freq))
         return key, state_trajectory, score_trajectory
   
   @abstractmethod
   def forward(self, key: jax.Array, true0: State, state0: State, law0: Param, param0: Param) -> tuple[jax.Array, DAState]:
      """Run one DA window. Must be implemented by subclasses."""
      pass

class DAOSSE(BaseDAOSSE[State, PhyState, ObsState, Param, Config]):
   """Data Assimilation System with conventional observations."""
   def __init__(self, nature: Model[State, PhyState, Param, Config], model: Model[State, PhyState, Param, Config], 
               observation: Observation, scheme: Scheme[State, ObsState], diagnostics: Diagnostics, config: DAConfig):
      super().__init__(nature, model, observation, scheme, diagnostics, config)

   @partial(jax.jit, static_argnames=['self'])
   def forward(self, key: jax.Array, truth0: State, state0: State, law0: Param, param0: Param) -> tuple[jax.Array, DAState]:
      """Run one DA window."""
      truth1, trajectory = self.nature.integrate(truth0, law0, self.config.dawindow, self.config.obsfreq)
      phytruth = self.nature.mod2phy(trajectory, law0)
      obstruth = self.observation.Hforward(phytruth)
      key, obs = self.observation.sample(key, obstruth)
      obserr = self.observation.scale(obstruth)
      #jax.debug.print("u: {x}, u2: {y}", x=obs.u, y=obs.u2)

      state1f, trajectory = self.ensemble.integrate(state0, param0, self.config.dawindow, self.config.obsfreq)
      phystate = self.ensemble.mod2phy(trajectory, param0)
      obsstate = self.ensemble.Hforward(phystate)
      state1a = self.scheme.forward(state1f, obsstate, obs, obserr)
      return key, DAState(truth1, state1f, state1a)
   
class IDAOSSE(BaseDAOSSE[State, PhyState, ObsState, Param, Config]):
   """Data Assimilation System with inequality observations."""
   def __init__(self, nature: Model[State, PhyState, Param, Config], model: Model[State, PhyState, Param, Config], 
               observation: Observation, scheme: Scheme[State, ObsState], diagnostics: Diagnostics, config: DAConfig):
      super().__init__(nature, model, observation, scheme, diagnostics, config)

   @partial(jax.jit, static_argnames=['self'])
   def forward(self, key: jax.Array, truth0: State, state0: State, law0: Param, param0: Param) -> tuple[jax.Array, DAState]:
      """Run one DA window."""
      truth1, trajectory = self.nature.integrate(truth0, law0, self.config.dawindow, self.config.obsfreq)
      phytruth = self.nature.mod2phy(trajectory, law0)
      obstruth = self.observation.Hforward(phytruth)
      key, lower_bound, upper_bound = self.observation.e2isample(key, obstruth)
      lower_scale, upper_scale = self.observation.e2iscale(obstruth)
      obs = self.observation.combine_states(lower_bound, upper_bound)
      obserr = self.observation.combine_states(lower_scale, upper_scale)
      key, bound, scale = self.observation.isample(key, obstruth)
      obs = self.observation.combine_states(obs, bound)
      obserr = self.observation.combine_states(obserr, scale)

      state1f, trajectory = self.ensemble.integrate(state0, param0, self.config.dawindow, self.config.obsfreq)
      phystate = self.ensemble.mod2phy(trajectory, param0)
      obsstate = self.ensemble.Hforward(phystate)
      obsstate2 = self.observation.double_states(obsstate)
      state1a = self.scheme.forward(state1f, obsstate2, obs, obserr)
      return key, DAState(truth1, state1f, state1a)
   
class IDACSE(BaseDAOSSE[State, PhyState, ObsState, Param, Config]):
   """Data Assimilation System and Control System."""
   def __init__(self, nature: Model[State, PhyState, Param, Config], model: Model[State, PhyState, Param, Config], 
               observation: Observation, scheme: Scheme[State, ObsState], diagnostics: Diagnostics, config: DAConfig,
               control: Observation, ctlscheme: Scheme[State, ObsState], ctlconfig: DAConfig):
      super().__init__(nature, model, observation, scheme, diagnostics, config)
      self.control = control
      self.ctlscheme = ctlscheme
      self.ctlconfig = ctlconfig
      self.ctlensemble = Ensemble(model, ctlconfig.nmember, control)

   @partial(jax.jit, static_argnames=['self'])
   def forward(self, key: jax.Array, truth0: State, state0: State, law0: Param, param0: Param) -> tuple[jax.Array, DAState]:
      """Run one DA window."""
      truth1, trajectory = self.nature.integrate(truth0, law0, self.config.dawindow, self.config.obsfreq)
      phytruth = self.nature.mod2phy(trajectory, law0)
      obstruth = self.observation.Hforward(phytruth)
      key, lower_bound, upper_bound = self.observation.e2isample(key, obstruth)
      lower_scale, upper_scale = self.observation.e2iscale(obstruth)
      obs = self.observation.combine_states(lower_bound, upper_bound)
      obserr = self.observation.combine_states(lower_scale, upper_scale)
      key, bound, scale = self.observation.isample(key, obstruth)
      obs = self.observation.combine_states(obs, bound)
      obserr = self.observation.combine_states(obserr, scale)

      state1f, trajectory = self.ensemble.integrate(state0, param0, self.config.dawindow, self.config.obsfreq)
      phystate = self.ensemble.mod2phy(trajectory, param0)
      obsstate = self.ensemble.Hforward(phystate)
      obsstate2 = self.observation.double_states(obsstate)
      state1a = self.scheme.forward(state1f, obsstate2, obs, obserr)

      """Run one control horizon. First integrate the nature to get obs."""
      mean1a = self.scheme.mean(state1a)
      _, trajectory = self.nature.integrate(mean1a, law0, self.ctlconfig.dawindow, self.ctlconfig.obsfreq)
      phytruth = self.nature.mod2phy(trajectory, law0)
      obstruth = self.control.Hforward(phytruth)
      key, lower_bound, upper_bound = self.control.e2isample(key, obstruth)
      lower_scale, upper_scale = self.control.e2iscale(obstruth)
      obs = self.control.combine_states(lower_bound, upper_bound)
      obserr = self.control.combine_states(lower_scale, upper_scale)
      key, bound, scale = self.control.isample(key, obstruth)
      obs = self.control.combine_states(obs, bound)
      obserr = self.control.combine_states(obserr, scale)

      _, trajectory = self.ctlensemble.integrate(state1a, param0, self.ctlconfig.dawindow, self.ctlconfig.obsfreq)
      phystate = self.ctlensemble.mod2phy(trajectory, param0)
      obsstate = self.ctlensemble.Hforward(phystate)
      obsstate2 = self.control.double_states(obsstate)
      truth2, state2a = self.ctlscheme.ctlforward(truth1, state1a, obsstate2, obs, obserr)
      # Create pseudo-ensemble from truth1 for verification
      truth1_ensemble = jax.tree_util.tree_map(
         lambda x: jnp.broadcast_to(x, (self.ctlconfig.nmember,) + x.shape) if x is not None else None,
         truth1)
      return key, DAState(truth2, truth1_ensemble, state2a)

def main():
   """Example usage with Lorenz96 model."""
   #jax.config.update("jax_enable_x64", True)
   from dajax.models.lorenz96 import Lorenz96, Config, State, ObsState, Param
   from dajax.schemes.etkf import ETKF
   from dajax.schemes.ida import IDA
   from dajax.obs.observation import ObsSetting, Observation
   from dajax.obs.likelihood import Gaussian, Logistic
   from dajax.obs.obsoperator import IdentityMap
   from dajax.obs.obslocation import RandomMask, RegularMask
   from dajax.utils.diagnostics import RMSEScore, MEScore, Diagnostics, no_selector
   from dajax.utils.inout import write_states, write_scores

   seed = 1
   dt = 0.05
   tspinup1 = 1.; tspinup2 = 1.
   tend = 300.
   save_freq = 1
   daconfig = DAConfig(dawindow=1, obsfreq=1, nmember=20, obserr=0.55, rho=0.7)

   # Key
   key = jax.random.PRNGKey(seed)
   key, key0, key1, key2 = jax.random.split(key, 4)

   # Model
   config = Config(nx=40, dt=dt)
   law = Param(F=8.0)
   nature = Lorenz96(config=config)
   param = Param(F=law.F+0.0*jax.random.normal(key0,(daconfig.nmember,1)))
   model = Lorenz96(config=config)

   # Observation system
   settings = {'u': ObsSetting(distribution = Logistic(scale=daconfig.obserr),
                              mapper = IdentityMap('u'),
                              mask = RandomMask(key0,fraction=0.9,grid_info=model.grid_info), #mask = RegularMask(start=1,spacing=2,grid_info=model.grid_info),
                              biasmin = -0.0, biasmax = 0.0),
               'u2': ObsSetting(distribution = Logistic(scale=daconfig.obserr),
                              mapper = IdentityMap('u'),
                              mask = RegularMask(start=0,spacing=1,grid_info=model.grid_info),
                              threshold = 3.0),}
   observation = Observation(settings, obsstate_class=ObsState)
   
   # Diagnostics
   me = MEScore({'u': no_selector, 'u2': no_selector})
   rmse = RMSEScore({'u': no_selector, 'u2': no_selector})
   diagnostics = Diagnostics([me,rmse])

   # DA scheme
   #scheme = ETKF(rho=daconfig.rho)  
   scheme = IDA(rho=daconfig.rho)  

   # DA system
   #da_system = DAOSSE(nature, model, observation, scheme, diagnostics, daconfig)
   da_system = IDAOSSE(nature, model, observation, scheme, diagnostics, daconfig)

   # Spinup
   true = model.default_state(law)
   true = model.random_state(key1, law, true, noise_scale=0.1)
   nstep = int(tspinup1/dt)
   true, _ = nature.integrate(true, law, nstep)
   ens = da_system.ensemble.random_state(key2, param, true, noise_scale=1.5)
   #ens = State(u=true.u[None,:]+3.5*jax.random.normal(key2,(daconfig.nmember,config.nx)))
   ncycle = int(tspinup2/(daconfig.dawindow*dt))
   key, state, score = da_system.cycle(key, true, ens, law, param, ncycle)
   diagnostics.print_comparison([score.forecast, score.analysis], ['Forecast', 'Analysis'])
   #import sys
   #sys.exit(0)
   # DA cycles
   ncycle = int(tend/(daconfig.dawindow*dt))
   key, state_trajectory, score_trajectory = da_system.cycle(key, state.truth, state.analysis, law, param, ncycle, save_freq)

   # Diagnostics
   forecast_scores = score_trajectory.forecast
   analysis_scores = score_trajectory.analysis
   forecast_score = diagnostics.average(forecast_scores)
   analysis_score = diagnostics.average(analysis_scores)
   diagnostics.print_comparison([forecast_score, analysis_score], ['Forecast', 'Analysis'])

   # Output
   time = jnp.arange(ncycle//save_freq)*save_freq*daconfig.dawindow*config.dt
   coords = {'time': time, 'x': jnp.arange(config.nx)}
   dims = {'u': ('time', 'x')}
   states = {"truth": state_trajectory.truth,
            "forecast": state_trajectory.forecast,
            "analysis": state_trajectory.analysis}
   write_states('states.nc', states, time, coords=coords, dims=dims)
   scores = {"forecast": {score_type: score[1] for score_type, score in score_trajectory.forecast.items()},
            "analysis": {score_type: score[1] for score_type, score in score_trajectory.analysis.items()}}
   write_scores('scores.nc', scores, time, coords=coords)

if __name__ == "__main__":
   main()