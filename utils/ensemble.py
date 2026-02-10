#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from functools import partial
from typing import Generic, Tuple, Optional
from dajax.models.mixin import Forcing
from dajax.models.base import Model, State, PhyState, ObsState, Param, Config
from dajax.obs.observation import Observation

class Ensemble(Generic[State, PhyState, ObsState, Param, Config]):
   def __init__(self, model: Model[State, PhyState, Param, Config], nmember: int, observation: Optional[Observation] = None):
      self.model = model
      self.nmember = nmember
      self.observation = observation
      # Get number of devices and compute sharding
      self.ndevice = jax.local_device_count()
      # Determine members per device
      self.members_per_device = (nmember+self.ndevice-1)//self.ndevice
      
      # Create both vmap and pmap versions
      # vmap for single device
      self._random_param_vmap = jax.vmap(model.random_param, in_axes=(0, None, None))
      self._random_state_vmap = jax.vmap(model.random_state, in_axes=(0, 0, None,None))
      self._forward_vmap = jax.vmap(model.forward, in_axes=(0, 0))
      self._integrate_vmap = jax.vmap(model.integrate, in_axes=(0, 0, None, None))
      self._mod2phy_vmap = jax.vmap(model.mod2phy, in_axes=(0, 0))
      self._phy2mod_vmap = jax.vmap(model.phy2mod, in_axes=(0, 0, 0))
      # pmap for multi-device (with vmap inside each device)
      self._random_param_pmap = jax.pmap(self._random_param_vmap, axis_name='device')
      self._random_state_pmap = jax.pmap(self._random_state_vmap, axis_name='device')
      self._forward_pmap = jax.pmap(self._forward_vmap, axis_name='device')
      self._integrate_pmap = jax.pmap(self._integrate_vmap, axis_name='device')
      self._mod2phy_pmap = jax.pmap(self._mod2phy_vmap, axis_name='device')
      self._phy2mod_pmap = jax.pmap(self._phy2mod_vmap, axis_name='device')
      if observation is not None:
         self._Hforward_vmap = jax.vmap(observation.Hforward)
         self._Hforward_pmap = jax.pmap(self._Hforward_vmap, axis_name='device')
      
      # Special version for ML batch training - non-batched state/param, batched forcings
      self._integrate_forcings_vmap = jax.vmap(model.integrate, in_axes=(None, None, None, None, 0))
      self._integrate_forcings_pmap = jax.pmap(self._integrate_forcings_vmap, axis_name='device')


   def _reshape_for_pmap(self, state: State) -> State:
      """Reshape state for pmap processing"""
      return jax.tree_map(
         lambda x: x.reshape(self.ndevice,self.members_per_device,*x.shape[1:]) if x is not None else None, state)
   
   def _reshape_from_pmap(self, state: State) -> State:
      """Reshape state back from pmap processing"""
      return jax.tree_map(
         lambda x: x.reshape(self.nmember, *x.shape[2:]) if x is not None else None, state)
   
   def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
      # Create random keys for each member
      keys = jax.random.split(key, self.nmember)
      if self.ndevice > 1:
         # Reshape keys and parameters for pmap
         keys_reshaped = keys.reshape(self.ndevice, self.members_per_device, 2)
         result = self._random_param_pmap(keys_reshaped, base, noise_scale)
         return self._reshape_from_pmap(result)
      else: return self._random_param_vmap(keys, base, noise_scale)
   
   #@partial(jax.jit, static_argnums=(0,))
   def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float) -> State:
      # Create random keys for each member
      keys = jax.random.split(key, self.nmember)
      if self.ndevice > 1:
         # Reshape keys and parameters for pmap
         keys_reshaped = keys.reshape(self.ndevice, self.members_per_device, 2)
         param_reshaped = self._reshape_for_pmap(param)
         result = self._random_state_pmap(keys_reshaped, param_reshaped, base, noise_scale)
         return self._reshape_from_pmap(result)
      else: return self._random_state_vmap(keys, param, base, noise_scale)
   
   @partial(jax.jit, static_argnums=(0,))
   def forward(self, state: State, param: Param) -> State:
      if self.ndevice > 1:
         state_reshaped = self._reshape_for_pmap(state)
         param_reshaped = self._reshape_for_pmap(param)
         result = self._forward_pmap(state_reshaped, param_reshaped)
         return self._reshape_from_pmap(result)
      else: return self._forward_vmap(state, param)

   @partial(jax.jit, static_argnums=(0,3,4))
   def integrate(self, state: State, param: Param, nstep: int, save_freq: Optional[int] = None) -> Tuple[State,State]:
      if self.ndevice > 1:
         state_reshaped = self._reshape_for_pmap(state)
         param_reshaped = self._reshape_for_pmap(param)
         final_state, trajectory = self._integrate_pmap(state_reshaped, param_reshaped, nstep, save_freq)
         return (self._reshape_from_pmap(final_state), self._reshape_from_pmap(trajectory) if trajectory is not None else None)
      else: return self._integrate_vmap(state, param, nstep, save_freq)
   
   @partial(jax.jit, static_argnums=(0,))
   def mod2phy(self, state: State, param: Param) -> PhyState:
      if self.ndevice > 1:
         state_reshaped = self._reshape_for_pmap(state)
         param_reshaped = self._reshape_for_pmap(param)
         result = self._mod2phy_pmap(state_reshaped, param_reshaped)
         return self._reshape_from_pmap(result)
      else: return self._mod2phy_vmap(state, param)

   @partial(jax.jit, static_argnums=(0,))
   def phy2mod(self, state: PhyState, ref_state: State, param: Param) -> PhyState:
      if self.ndevice > 1:
         state_reshaped = self._reshape_for_pmap(state)
         ref_state_reshaped = self._reshape_for_pmap(ref_state)
         param_reshaped = self._reshape_for_pmap(param)
         result = self._phy2mod_pmap(state_reshaped, ref_state_reshaped, param_reshaped)
         return self._reshape_from_pmap(result)
      else: return self._phy2mod_vmap(state, ref_state, param)

   @partial(jax.jit, static_argnums=(0,))
   def Hforward(self, phystate: PhyState) -> ObsState:
      if self.ndevice > 1:
         phystate_reshaped = self._reshape_for_pmap(phystate)
         result = self._Hforward_pmap(phystate_reshaped)
         return self._reshape_from_pmap(result)
      else: return self._Hforward_vmap(phystate)
   
   #@partial(jax.jit, static_argnums=(0,1,2,3,4))
   def integrate_forcings(self, state: State, param: Param, nstep: int, save_freq: Optional[int] = None, 
                              forcings: Optional[Forcing] = None) -> Tuple[State,State]:
      if self.ndevice > 1:
         forcings_reshaped = self._reshape_for_pmap(forcings)
         state_replicated = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.ndevice,)+x.shape) if x is not None else None, 
            state
         )
         param_replicated = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.ndevice,)+x.shape) if x is not None else None, 
            param
         )
         nstep_arr = jnp.array([nstep] * self.ndevice)  # [ndevice]
         if save_freq is not None: save_freq_arr = jnp.array([save_freq] * self.ndevice)  # [ndevice]
         else: save_freq_arr = None
         # Call the pmap version with properly shaped arguments
         final_state, trajectory = self._integrate_forcings_pmap(
            state_replicated, param_replicated, nstep_arr, save_freq_arr, forcings_reshaped
         )
         # Take only the first device's output for state and param since they're replicated
         final_state_result = jax.tree_map(lambda x: x[0] if x is not None else None, final_state)
         # For trajectory, reshape back from device-distributed to flat batch
         if trajectory is not None: trajectory_result = self._reshape_from_pmap(trajectory)
         else: trajectory_result = None
         return (final_state_result, trajectory_result)
      else: return self._integrate_forcings_vmap(state, param, nstep, save_freq, forcings)
