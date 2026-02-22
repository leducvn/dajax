#!/usr/bin/env python3
"""
Robust Ensemble Transform Kalman Filter with Huber norm.

The Huber norm provides robustness to outliers by using:
- L2 norm (quadratic) for small innovations |Δy| ≤ δ
- L1 norm (linear) for large innovations |Δy| > δ

The optimization uses the Majorization-Minimization (MM) algorithm,
which iteratively solves quadratic surrogates (ETKF problems) to minimize the Huber cost.
"""
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
from typing import NamedTuple, Optional
from dajax.schemes.base import Scheme
from dajax.models.base import State, ObsState

class HuberConfig(NamedTuple):
    """Configuration for Huber ETKF."""
    delta: float = 2.0          # Huber threshold (in units of normalized innovation)
    rho: float = 1.0            # Inflation parameter
    max_iter: int = 100         # Maximum MM iterations
    tol: float = 1e-6           # Convergence tolerance for cost function ratio
    gtol: float = 1e-6          # Convergence tolerance for gradient norm
    Hexact: bool = True         # Use exact Hessian
    L2guess: bool = True        # Use L2 (ETKF) solution as initial guess

class HuberETKF(Scheme[State, ObsState]):
    """Robust Ensemble Transform Kalman Filter with Huber norm.
    
    This filter handles outliers by using the Huber cost function instead of
    the standard L2 norm. The Huber cost is quadratic for small innovations
    and linear for large innovations, reducing the influence of outliers.
    
    The analysis is found iteratively using the Majorization-Minimization
    algorithm. Each iteration solves a standard ETKF problem with scaled
    observation errors. The iteration terminates when:
    - Cost function ratio |J_new - J_old|/|J_old| < tol, or
    - Gradient norm < gtol, or
    - Maximum iterations reached
    
    The ensemble is generated using the Hessian at the analysis point.
    
    Args:
        config: HuberConfig with algorithm parameters
        
    Alternatively, can be initialized with individual parameters:
        delta: Huber threshold
        rho: Inflation parameter
        max_iter: Maximum MM iterations (use 1 for simple QC procedure)
        tol: Convergence tolerance for cost function ratio
        gtol: Convergence tolerance for gradient norm
    """
    
    def __init__(self, config: Optional[HuberConfig] = None, 
                delta: float = 2.0, rho: float = 1.0, max_iter: int = 100,
                tol: float = 1e-6, gtol: float = 1e-6):
        if config is not None:
            self.config = config
        else:
            self.config = HuberConfig(delta=delta, rho=rho, max_iter=max_iter, tol=tol, gtol=gtol)
        
        self.delta = self.config.delta
        self.rho = self.config.rho
        self.max_iter = self.config.max_iter
        self.tol = self.config.tol
        self.gtol = self.config.gtol
        self.Hexact = self.config.Hexact
        self.L2guess = self.config.L2guess
        
        # JIT compile core functions
        self._weight_ana_etkf = jax.jit(self._make_weight_ana_etkf())
        self._weight_ana = jax.jit(self._make_weight_ana())
        self._compute_ana = jax.jit(self._make_compute_ana())
        self._weight_ens = jax.jit(self._make_weight_ens())
        self._compute_ens = jax.jit(self._make_compute_ens())
    
    def _make_weight_ana_etkf(self):
        """Create ETKF analysis weight solver (reused from ETKF scheme)."""
        def weight_ana_etkf(state: ObsState, obs: ObsState) -> jax.Array:
            # Both state and obs are normalized fields
            def logp(w: jax.Array, state: ObsState, obs: ObsState):
                Jb = 0.5 * jnp.dot(w, w)
                s = self.weighted_sum(w, state) - obs
                Jo = 0.5 * s.dot(s)
                return Jb + Jo
            logp = jax.jit(logp)

            first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
            nmember = first_field.shape[0]
            w = jnp.zeros((nmember))
            Jold = logp(w, state, obs)
            res = minimize(fun=logp, x0=w, args=(state, obs), method='BFGS', tol=1.e-12)
            Jnew = logp(res.x, state, obs)
            return jax.lax.cond(Jnew < Jold, lambda _: res.x, lambda _: w, operand=None)
        return weight_ana_etkf
    
    def _make_weight_ana(self):
        """Create the iterative Huber analysis weight computation using MM algorithm."""
        delta = self.delta
        max_iter = self.max_iter
        tol = self.tol
        gtol = self.gtol
        L2guess = self.L2guess
        
        def weight_ana(obspert: ObsState, inv: ObsState) -> jax.Array:
            """Compute analysis weights using Huber norm via MM algorithm.
            
            At each iteration:
            1. Compute effective observation error based on current innovation
            2. Scale obspert and inv by effective observation error
            3. Solve ETKF problem with scaled observations
            4. Check convergence (cost ratio < tol or gradient norm < gtol)
            
            Args:
                obspert: Normalized ensemble perturbations in obs space
                inv: Normalized innovation (y - H(xf_mean)) / sigma_o
                
            Returns:
                wmean: Analysis weights [nmember]
            """
            first_field = next(x for x in jax.tree_util.tree_leaves(obspert) if x is not None)
            nmember = first_field.shape[0]
            # Create zero ObsState for normalization (same structure as inv)
            zero_obs = jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x) if x is not None else None, inv)
            
            def compute_eff_obserr(w: jax.Array, obspert: ObsState, inv: ObsState) -> ObsState:
                """Compute effective observation error scaling based on innovation."""
                s = self.weighted_sum(w, obspert) - inv
                s_abs = jax.tree_util.tree_map(
                    lambda x: jnp.abs(x) if x is not None else None, s)
                eff_obserr = jax.tree_util.tree_map(
                    lambda x: jnp.where(x > delta, jnp.sqrt(x / delta), 1.0) if x is not None else None, s_abs)
                return eff_obserr
            
            def huber_cost(w: jax.Array, obspert: ObsState, inv: ObsState) -> float:
                """Compute Huber cost function value."""
                Jb = 0.5 * jnp.dot(w, w)
                s = self.weighted_sum(w, obspert) - inv
                # Compute Jo with Huber norm
                def huber_jo(x):
                    if x is None: return 0.0
                    abs_x = jnp.abs(x)
                    inside = abs_x <= delta
                    jo = jnp.where(inside, 0.5 * x**2, delta * abs_x - 0.5 * delta**2)
                    return jnp.sum(jo)
                Jo = sum(huber_jo(x) for x in jax.tree_util.tree_leaves(s))
                return Jb + Jo
            
            def huber_grad(w: jax.Array, obspert: ObsState, inv: ObsState) -> jax.Array:
                """Compute gradient of Huber cost function."""
                s = self.weighted_sum(w, obspert) - inv
                # Gradient: w - sum_k (dJ/ds_k * obspert_k)
                # where dJ/ds = s for |s|<=delta, delta*sign(s) for |s|>delta
                def grad_contribution(s_field, obspert_field):
                    if s_field is None or obspert_field is None:
                        return jnp.zeros(nmember)
                    abs_s = jnp.abs(s_field)
                    inside = abs_s <= delta
                    ds = jnp.where(inside, s_field, delta * jnp.sign(s_field))
                    # obspert_field has shape [nmember, ...], ds has shape [...]
                    # We need sum over observation dimensions
                    return jnp.sum(obspert_field * ds[None, ...], axis=tuple(range(1, obspert_field.ndim)))
                grad = w.copy()
                for s_field, obspert_field in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(obspert)):
                    grad = grad - grad_contribution(s_field, obspert_field)
                return grad
            
            def mm_iteration_body(carry):
                """One iteration of the MM algorithm."""
                wmean, J_old, niter = carry
                # Compute effective observation error
                eff_obserr = compute_eff_obserr(wmean, obspert, inv)
                # Scale obspert and inv by effective observation error
                scaled_inv = self.normalize(1., inv, zero_obs, eff_obserr)
                scaled_obspert = self.normalize(1., obspert, zero_obs, eff_obserr)
                # Solve ETKF problem with scaled observations
                wmean_new = self._weight_ana_etkf(scaled_obspert, scaled_inv)
                # Compute new cost and gradient for convergence check
                J_new = huber_cost(wmean_new, obspert, inv)
                grad = huber_grad(wmean_new, obspert, inv)
                grad_norm = jnp.sqrt(jnp.dot(grad, grad))
                # Check convergence
                ratio = jnp.abs(J_new - J_old) / (jnp.abs(J_old) + 1e-12)
                return wmean_new, J_new, niter + 1, ratio, grad_norm
            
            def cond_fn(carry):
                """Check if we should continue iterating."""
                _, _, niter, ratio, grad_norm = carry
                not_converged = (ratio >= tol) & (grad_norm >= gtol)
                not_max_iter = niter < max_iter
                return not_converged & not_max_iter
            
            def body_fn(carry):
                """Wrapper for iteration body."""
                wmean, J_old, niter, _, _ = carry
                wmean_new, J_new, niter_new, ratio, grad_norm = mm_iteration_body((wmean, J_old, niter))
                return wmean_new, J_new, niter_new, ratio, grad_norm
            
            # Initialize weights
            if L2guess: wmean_init = self._weight_ana_etkf(obspert, inv)
            else: wmean_init = jnp.zeros((nmember))
            # Initial cost
            J_init = huber_cost(wmean_init, obspert, inv)
            niter_init = 0
            ratio_init = 1.
            grad_norm_init = 1.
            # Run MM iterations with early termination
            wmean_final, J_final, niter_final, _, _ = jax.lax.while_loop(
                cond_fn, body_fn, 
                (wmean_init, J_init, niter_init, ratio_init, grad_norm_init)
            )
            #jax.debug.print("niter: {x}, Jold: {y}, Jnew: {z}", x=niter_final, y=J_init, z=J_final)
            # Use final result if cost decreased, else use initial ETKF solution
            wmean_out = jax.lax.cond(J_final < J_init, lambda: wmean_final, lambda: wmean_init)
            return wmean_out
        return weight_ana
    
    def _make_compute_ana(self):
        """Create analysis computation from weights."""
        def compute_ana(w: jax.Array, state: State, mean: State) -> State:
            return mean + self.weighted_sum(w, state)
        return compute_ana
    
    def _make_weight_ens(self):
        """The Hessian of the Huber cost function at the analysis point is used
        to generate the analysis ensemble. For observations inside the threshold,
        the Hessian contribution is 1. For observations outside, it's delta/|s|.
        This is equivalent to scaling by sqrt(delta/|s|) = 1/eff_obserr.
        """
        delta = self.delta
        rho = self.rho
        Hexact = self.Hexact
        
        def weight_ens(w: jax.Array, state: ObsState, obs: ObsState) -> jax.Array:
            """Compute analysis ensemble transform weights.
            Uses the true Hessian of the Huber cost at the analysis point.
            For Huber norm:
            - Inside (|s| <= delta): d²J/ds² = 1
            - Outside (|s| > delta): d²J/ds² = delta/|s|
            The scaling factor is sqrt(d²J/ds²) = sqrt(delta/|s|) for outside.
            """
            first_field = next(x for x in jax.tree_util.tree_leaves(state) if x is not None)
            nmember = first_field.shape[0]
            
            # Compute innovation at analysis point
            s = self.weighted_sum(w, state) - obs
            s_abs = jax.tree_util.tree_map(lambda x: jnp.abs(x) if x is not None else None, s)
            if Hexact:
                # Compute exact Hessian scaling: 1 for inside, 0 for outside
                hessian_scale = jax.tree_util.tree_map(
                    lambda x: jnp.where(x <= delta, 1.0, 0.0) if x is not None else None, s_abs)
            else:
                # Compute Hessian scaling: sqrt(delta/|s|) for outside, 1 for inside
                hessian_scale = jax.tree_util.tree_map(
                    lambda x: jnp.where(x>delta, jnp.sqrt(delta/x), 1.0) if x is not None else None, s_abs)
            # Scale ensemble perturbations by Hessian scaling
            state = self.scale_ensemble(hessian_scale, state)
            
            # Compute covariance in ensemble space
            YTY = self.covariance(state)
            E, V = jnp.linalg.eigh(YTY)
            E = E[::-1]; V = V[:, ::-1]
            E = jnp.where(E < 0., 0., E)
            E = E.at[-1].set(0.)
            # Inflation
            Lambda = 1.0 / jnp.sqrt(1.0 + E)
            Lmean = jnp.mean(Lambda)
            FL = (1. + rho * (1 - Lmean)) * Lambda
            # Analysis ensemble transform
            w = jnp.sqrt(nmember - 1) * V * FL[None, :]
            w = jnp.dot(V, w.T)
            return w
        
        return weight_ens
    
    def _make_compute_ens(self):
        """Create analysis ensemble computation."""
        def compute_ens(w: jax.Array, state: State, mean: State) -> State:
            """Transform ensemble perturbations to analysis ensemble.
            
            Args:
                w_ens: Ensemble transform weights [nmember, nmember]
                state: Normalized ensemble perturbations (sqrt(B)/(nmember-1))
                ana: Analysis mean
                
            Returns:
                Analysis ensemble state
            """
            return jax.tree_util.tree_map(
                lambda x, m: m[None,...] + jnp.einsum('ij,i...->j...', w, x) if m is not None else None, state, mean)
        return compute_ens
    
    @partial(jax.jit, static_argnames=['self'])
    def forward(self, state: State, obsstate: ObsState, obs: ObsState, obserr: ObsState) -> State:
        """Perform one Huber ETKF analysis step.
        
        Args:
            state: Ensemble state (model or physical space) [nmember, ...]
            obsstate: Ensemble in observation space [nmember, nobs]
            obs: Observations [nobs]
            obserr: Observation error standard deviations [nobs]
            
        Returns:
            Analysis ensemble state
        """
        nmember = getattr(state, state._fields[0]).shape[0]
        factor = jnp.sqrt(nmember - 1)
        # Compute means
        obsmean = self.mean(obsstate)
        mean = self.mean(state)
        # Normalize innovation and ensemble perturbations
        inv = self.normalize(1., obs, obsmean, obserr)
        obspert = self.normalize(factor, obsstate, obsmean, obserr)
        pert = self.normalize(factor, state, mean)
        # Compute analysis weights using Huber MM algorithm
        w = self._weight_ana(obspert, inv)
        # Compute analysis mean
        ana = self._compute_ana(w, pert, mean)
        # Compute ensemble transform weights using true Hessian
        w = self._weight_ens(w, obspert, inv)
        # Generate analysis ensemble
        anastate = self._compute_ens(w, pert, ana)
        
        return anastate
