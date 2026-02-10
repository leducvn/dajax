#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from typing import Protocol, TypeVar, Optional, runtime_checkable
from dajax.models.base import State, PhyState, ObsState
# Although we use State in forward(), it is a generic hint, and can be State or PhyState

@runtime_checkable
class Scheme(Protocol[State, ObsState]):
   """Protocol defining the interface for data assimilation schemes.
   This protocol defines the minimum interface that DA schemes must implement to be
   compatible with the data assimilation system.
   """
   def forward(self, state: State, obsstate: ObsState, obs: ObsState, obserr: ObsState) -> State:
      """Perform one step of the data assimilation scheme.
      Args:
         state: Current ensemble state (generic: can be State or PhyState)
         obsstate: Ensemble state in observation space
         obs: Observations
         obserr: Observation error standard deviations
      Returns:
         Updated ensemble state after assimilation
      """
      ...
   
   def ctlforward(self, truth: State, state: State, obsstate: ObsState, obs: ObsState, obserr: ObsState) -> tuple[State,State]:
      """Perform one step of the control scheme.
      Args:
         truth: True state (generic: can be State or PhyState)
         state: Current ensemble state (generic: can be State or PhyState)
         obsstate: Ensemble state in observation space
         obs: Observations
         obserr: Observation error standard deviations
      Returns:
         Updated true and ensemble states after control
      """
      ...

   StateType = TypeVar('StateType', State, PhyState, ObsState)

   @staticmethod
   @jax.jit
   def mean(state: StateType) -> StateType:
      return jax.tree_util.tree_map(
         lambda x: jnp.mean(x, axis=0) if x is not None else None, state)

   @staticmethod
   @jax.jit
   def spread(state: StateType) -> StateType:
      return jax.tree_util.tree_map(
         lambda x: jnp.std(x, axis=0) if x is not None else None, state)
   
   @staticmethod
   @jax.jit
   def get_member(state: StateType, k: int) -> StateType:
      return jax.tree_util.tree_map(lambda x: x[k] if x is not None else None, state)
   
   @staticmethod
   @jax.jit
   def split_ensemble(state: StateType) -> list[StateType]:
      first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
      nmember = first_field.shape[0]
      return [Scheme.get_member(state,k) for k in range(nmember)]
   
   @staticmethod
   @jax.jit
   def combine_ensemble(states: list[StateType]) -> StateType:
      return jax.tree_util.tree_map(
         lambda *xs: jnp.stack(xs, axis=0) if xs[0] is not None else None, *states)
   
   @staticmethod
   @jax.jit
   def normalize(factor: float, state: StateType, mean: StateType, spread: Optional[StateType] = None) -> StateType:
      def normalize_field(x, m, s=None):
         if x is None: return None
         is_ensemble = x.ndim > m.ndim
         if spread is None:
            return (x-m[None,...] if is_ensemble else x-m)/factor
         else:
            return (x-m[None,...])/(factor*s[None,...]) if is_ensemble else (x-m)/(factor*s)
      if spread is None:
         return jax.tree_util.tree_map(normalize_field, state, mean)
      else:
         return jax.tree_util.tree_map(normalize_field, state, mean, spread)
   
   @staticmethod
   @jax.jit
   def scale_ensemble(weights: StateType, state: StateType) -> StateType:
      return jax.tree_util.tree_map(
         lambda w,x: x*w[None,...] if x is not None else None, weights, state)
   
   @staticmethod
   @jax.jit
   def weighted_sum(weights: jax.Array, state: StateType) -> StateType:
      return jax.tree_util.tree_map(
         lambda x: jnp.einsum('i,i...->...', weights, x) if x is not None else None, state)
   #def weighted_sumv2(weights: jax.Array, states: list[StateType]) -> StateType:
      #return jax.tree_util.tree_reduce(lambda x, y: x + y, [state*w for state,w in zip(states,weights)])
   
   @staticmethod
   @jax.jit
   def covariance(state: StateType) -> jax.Array:
      first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
      nmember = first_field.shape[0]
      def field_covariance(field_data):
         if field_data is None: return jnp.zeros((nmember,nmember))
         flat_data = field_data.reshape(nmember,-1)
         return jnp.einsum('ij,kj->ik', flat_data, flat_data)
         #return jnp.dot(flat_data,flat_data.T)
      return jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(field_covariance, state))
   