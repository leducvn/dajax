#!/usr/bin/env python3
import jax, jaxopt
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
from dajax.schemes.base import Scheme
from dajax.models.base import State, PhyState, ObsState
# Although we use State in forward(), it is a generic hint, and can be State or PhyState

class IDA(Scheme[State, ObsState]):
   """Interval Data Assimilation implementation."""
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
            Jo = s.log_expit().sum()
            return Jb-Jo
         logp = jax.jit(logp)
         #dlogp = jax.grad(logp)

         first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
         nmember = first_field.shape[0]
         w = jnp.zeros((nmember))
         Jold = logp(w, state, obs)
         res = minimize(fun=logp, x0=w, args=(state, obs), method='BFGS', tol=1.e-12)
         Jnew = logp(res.x, state, obs)
         #jax.debug.print("Jold: {w}, Jnew: {x}, Success: {y}, Status: {z}", w=Jold, x=Jnew, y=res.success, z=res.status)
         return jax.lax.cond(Jnew < Jold, lambda _: res.x, lambda _: w, operand=None)
         #solver = jaxopt.LBFGS(fun=logp, maxiter=1000, tol=1.e-6)
         #res = solver.run(w, state, obs)
         #Jnew = logp(res.params, state, obs)
         #jax.debug.print("Jold: {w}, Jnew: {x}", w=Jold, x=Jnew)
         #return jax.lax.cond(Jnew < Jold, lambda _: res.params, lambda _: w, operand=None)
      return weight_ana
   
   def _make_compute_ana(self):
      def compute_ana(w: jax.Array, state: State, mean: State) -> State:
         # state is the square root of B. This works with ObsState as well.
         return mean + self.weighted_sum(w,state)
      return compute_ana
   
   def _make_weight_ens(self):
      def weight_ens(w: jax.Array, state: ObsState, obs: ObsState) -> jax.Array:
         # state are normalized fields
         first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
         nmember = first_field.shape[0]
         s = self.weighted_sum(w,state) - obs
         s = s.expit()
         s = s*(1.-s)
         state = self.scale_ensemble(s.sqrt(),state)
         YTY = self.covariance(state)
         E, V = jnp.linalg.eigh(YTY)
         E = E[::-1]; V = V[:,::-1]
         E = jnp.where(E < 0., 0., E)
         E = E.at[-1].set(0.)
         # inflation
         Lambda = 1.0/jnp.sqrt(1.0+E)
         Lmean = jnp.mean(Lambda)
         #FL = (1.+self.rho*(1/Lmean-1))*Lambda
         FL = (1.+self.rho*(1-Lmean))*Lambda
         # Analysis ensemble
         w = jnp.sqrt(nmember-1)*V*FL[None,:]
         w = jnp.dot(V, w.T)
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
      w = self._weight_ens(w, obspert, inv)
      anastate = self._compute_ens(w, pert, ana)
      return anastate
   
   @partial(jax.jit, static_argnames=['self', ])
   def ctlforward(self, truth: State, state: State, obsstate: ObsState, obs: ObsState, obserr: ObsState) -> tuple[State,State]:
      nmember = getattr(state, state._fields[0]).shape[0]
      factor = jnp.sqrt(nmember-1)
      obsmean = self.mean(obsstate)
      inv = self.normalize(1., obs, obsmean, obserr)
      obspert = self.normalize(factor, obsstate, obsmean, obserr)
      mean = self.mean(state)
      pert = self.normalize(factor, state, mean)
      w = self._weight_ana(obspert, inv)
      ana = self._compute_ana(w, pert, mean)
      inc = mean - ana # Note we use mean-ana not ana-mean to utilize normalize() later
      truthnew = truth - inc
      statenew = self.normalize(1., state, inc)
      return truthnew, statenew

