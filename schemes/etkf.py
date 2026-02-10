#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
from dajax.schemes.base import Scheme
from dajax.models.base import State, ObsState

class ETKF(Scheme[State, ObsState]):
   """Ensemble Transform Kalman Filter implementation."""
   def __init__(self, rho: float):
      self.rho = rho
      # JIT compile nonlinear function once during initialization
      self._weight_ana = jax.jit(self._make_weight_ana())
      self._compute_ana = jax.jit(self._make_compute_ana())
      self._weight_ens = jax.jit(self._make_weight_ens())
      self._compute_ens = jax.jit(self._make_compute_ens())
   
   def _make_weight_ana(self):
      def weight_ana(state: ObsState, obs: ObsState) -> jax.Array:
         # Both state and obs are normalized fields
         def logp(w: jax.Array, state: ObsState, obs: ObsState):
            Jb = 0.5*jnp.dot(w,w)
            s = self.weighted_sum(w,state) - obs
            Jo = 0.5*s.dot(s)
            return Jb+Jo
         logp = jax.jit(logp)
         #dlogp = jax.grad(logp)
         #def dlogp(w: jax.Array, state: ObsState, obs: ObsState):
            #s = -obs
            #for k in range(len(w)): s = s + state[k]*w[k]
            #g = w*1.
            #for k in range(len(w)): g = g.at[k].add(s.dot(state[k]))
            #return g

         first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
         nmember = first_field.shape[0]
         w = jnp.zeros((nmember))
         Jold = logp(w, state, obs)
         res = minimize(fun=logp, x0=w, args=(state, obs), method='BFGS', tol=1.e-12) #, options={'disp':False})
         #jax.debug.print("J: {x}, Success: {y}, Status: {z}", x=res.fun, y=res.success, z=res.status)
         Jnew = logp(res.x, state, obs)
         return jax.lax.cond(Jnew < Jold, lambda _: res.x, lambda _: w, operand=None)

         #optimizer = optax.adam(learning_rate=1e-1)
         #opt_state = optimizer.init(w)
         # Optimization loop
         #def step(carry, _):
            #w, opt_state = carry
            #loss = logp(w, state, obs)
            #grads = dlogp(w, state, obs)
            #updates, opt_state = optimizer.update(grads, opt_state)
            #w = optax.apply_updates(w, updates)
            #return (w, opt_state), loss
         #init_carry = (w, opt_state)
         #(wopt, _), losses = jax.lax.scan(step, init_carry, xs=jnp.arange(1000))
         #jax.debug.print("Loss: {x}, Grad: {y}", x=logp(wopt,state,obs), y=jnp.sum(dlogp(wopt,state,obs**2)))
         #Jnew = logp(wopt, state, obs)
         #return jax.lax.cond(Jnew < Jold, lambda _: wopt, lambda _: w, operand=None)

         #YTY = self.covariance(state)
         #E, V = jnp.linalg.eigh(YTY)
         #E = E[::-1]; V = V[:,::-1]
         #E = jnp.where(E < 1e-12, 0., E)
         #E = E.at[-1].set(0.)
         #Y = self.split_ensemble(state)
         #for k in range(nmember): w = w.at[k].set(obs.dot(Y[k]))
         #wopt = jnp.zeros((nmember))
         #for k in range(nmember): wopt = wopt.at[k].set(jnp.dot(w,V[:,k]))
         #wopt = wopt/(1.0+E)
         #wopt = jnp.sum(wopt[None,:]*V, axis=1)
         #Jnew = logp(wopt, state, obs)
         #jax.debug.print("Jold: {y}, Jnew: {z}", y=Jold, z=Jnew)
         #return wopt
         #return jax.lax.cond(Jnew < Jold, lambda _: wopt, lambda _: w, operand=None)
      return weight_ana
   
   def _make_compute_ana(self):
      def compute_ana(w: jax.Array, state: State, mean: State) -> State:
         # state is the square root of B. This works with ObsState as well.
         return mean + self.weighted_sum(w,state)
      return compute_ana
   
   def _make_weight_ens(self):
      def weight_ens(state: ObsState) -> jax.Array:
         # state are normalized fields
         first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
         nmember = first_field.shape[0]
         YTY = self.covariance(state)
         E, V = jnp.linalg.eigh(YTY)
         E = E[::-1]; V = V[:,::-1]
         E = jnp.where(E < 0., 0., E)
         E = E.at[-1].set(0.)
         # inflation
         #nrank = jnp.sum(E >= 1.E-12)
         Lambda = 1.0/jnp.sqrt(1.0+E)
         Lmean = jnp.mean(Lambda)
         #FL = (1.+self.rho*(1/Lmean-1))*Lambda
         FL = (1.+self.rho*(1-Lmean))*Lambda
         # Analysis ensemble
         w = jnp.sqrt(nmember-1)*V*FL[None,:]
         w = jnp.dot(V,w.T)
         return w
      return weight_ens
   
   def _make_compute_ens(self):
      def compute_ens(w: jax.Array, state: State, mean: State) -> State:
         # state is the square root of B, mean is the analysis
         return jax.tree_util.tree_map(
            lambda x, m: m[None,...] + jnp.einsum('ij,i...->j...', w, x) if m is not None else None, state, mean)
      return compute_ens
   
   @partial(jax.jit, static_argnames=['self', ])
   def forward(self, state: State, obsstate: ObsState, obs: ObsState, obserr: ObsState) -> State:
      nmember = getattr(state, state._fields[0]).shape[0]
      factor = jnp.sqrt(nmember-1)
      obsmean = self.mean(obsstate)
      inv = self.normalize(1., obs, obsmean, obserr)
      obspert = self.normalize(factor, obsstate, obsmean, obserr)
      mean = self.mean(state)
      pert = self.normalize(factor, state, mean)
      w = self._weight_ana(obspert, inv)
      ana = self._compute_ana(w, pert, mean)
      w = self._weight_ens(obspert)
      anastate = self._compute_ens(w, pert, ana)
      return anastate

