#!/usr/bin/env python3
import sys, os, netCDF4
from numpy import *
from scipy.special import expit, log_expit
from scipy.optimize import minimize
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dajax.models.lorenz96 import Lorenz96, Config, State, ObsState
from dajax.schemes.base import Scheme
from dajax.schemes.etkf import ETKF
from dajax.schemes.ida import IDA
from dajax.utils.ensemble import Ensemble
from dajax.obs.observation import ObsSetting, Observation
from dajax.obs.likelihood import Gaussian
from dajax.obs.obsoperator import IdentityMap
from dajax.obs.obslocation import RegularMask

missing = -9999.
def L95_nlterm(u,N):
   N[:] = u[im1]*(u[ip1]-u[im2])+f

def M(u):
   if model == 'KS':
      uhat[:] = fft.fft(u)
      N0[:] = g*fft.fft(u**2)
      ahat[:] = eLdt2*uhat + 0.5*dt*q*N0
      Na[:] = g*fft.fft(fft.ifft(ahat).real**2)
      bhat[:] = eLdt2*uhat + 0.5*dt*q*Na
      Nb[:] = g*fft.fft(fft.ifft(bhat).real**2)
      chat[:] = eLdt2*ahat + 0.5*dt*q*(2*Nb-N0)
      Nc[:] = g*fft.fft(fft.ifft(chat).real**2)
      uhat[:] = eLdt*uhat + dt*(f1*N0+f2*2*(Na+Nb)+f3*Nc)
      return fft.ifft(uhat).real
   elif model == 'L95':
      uhat[:] = u
      L95_nlterm(uhat,N0)
      ahat[:] = eLdt2*uhat + 0.5*dt*q*N0
      L95_nlterm(ahat,Na)
      bhat[:] = eLdt2*uhat + 0.5*dt*q*Na
      L95_nlterm(bhat,Nb)
      chat[:] = eLdt2*ahat + 0.5*dt*q*(2*Nb-N0)
      L95_nlterm(chat,Nc)
      uhat[:] = eLdt*uhat + dt*(f1*N0+f2*2*(Na+Nb)+f3*Nc)
      return uhat.real
   pass

def logp(w, y, Y, s, g):
   Jb = 0.5*dot(w,w)
   s[:] = dot(Y,w)-y
   #Jo = 0.5*dot(s,s)
   Jo = -sum(log_expit(s))
   return Jb+Jo

def dlogp(w, y, Y, s, g):
   g[:] = w
   s[:] = dot(Y,w)-y
   #g += dot(Y.T,s)
   s[:] = 1.-expit(s)
   g -= dot(Y.T,s)
   return g

def jaxlogp(w: jax.Array, state: ObsState, obs: ObsState, scheme: Scheme):
   Jb = 0.5*jnp.dot(w,w)
   s = scheme.weighted_sum(w,state) - obs
   Jo = s.log_expit().sum()
   return Jb-Jo
jaxdlogp = jax.grad(jaxlogp)

def write_netcdf(filename, time, ytrue, yfcst, yana, yfspr, yaspr):
   os.system('rm -f '+filename)
   nc = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
   ntime,nvar,nx = ytrue.shape
   nc.createDimension('x', nx)
   nc.createDimension('var', nvar)
   nc.createDimension('time', ntime)
   var = nc.createVariable('time', 'f4', ('time'))
   var[:] = time.astype(float32)
   
   var = nc.createVariable('ytrue', 'f4', ('time','var','x'))
   var[:] = ytrue.astype(float32)
   var = nc.createVariable('yfcst', 'f4', ('time','var','x'))
   var[:] = yfcst.astype(float32)
   var = nc.createVariable('yana', 'f4', ('time','var','x'))
   var[:] = yana.astype(float32)
   var = nc.createVariable('yfspr', 'f4', ('time','var','x'))
   var[:] = yfspr.astype(float32)
   var = nc.createVariable('yaspr', 'f4', ('time','var','x'))
   var[:] = yaspr.astype(float32)
   nc.close()

# Model parameters
nmember = 20
model = 'L95'; nx = 40
if model == 'KS': dt = 0.25
else: dt = 0.05
nroot = 64
domain = arange(nx).astype(int32)
land = domain >= 0 # entire domain
nland = sum(land)

# Parameters
Tspinup = 60*4*dt # one month
Tend = 4*365*4*dt
sigmao = 1.0
#bias = 0.0
#delta = 0.0
rho = 0.5 # inflation parameter
cycle = 1; nslot = 1
bias = zeros((nslot,nx))

root_dir = '/home/leduc/project/ida'
config = Config(nx=nx, dt=dt, F=8.0)
jaxmodel = Lorenz96(config=config)
grid_info = jaxmodel.grid_info
key = jax.random.PRNGKey(0)
settings = {'u': ObsSetting(distribution = Gaussian(scale=sigmao),
                           mapper = IdentityMap('u'),
                           mask = RegularMask(start=0,spacing=1,grid_info=grid_info))}
observation = Observation(settings, obsstate_class=ObsState)
scheme = IDA(rho=rho)
ensemble = Ensemble(jaxmodel, observation, nmember)

# Models
jroot = exp(pi*1j*(arange(nroot/2)+0.5)/(nroot/2))
if model == 'KS':
   alfa = 1.; gamma = 1.
   #alfa = -0.3; gamma = 0.
   Lx = 32*pi
   dx = Lx/nx
   x = dx*arange(nx)
   # Coefficients
   kwave = zeros((nx))
   kwave[:nx/2] = arange(nx/2)
   kwave[nx/2] = 0.
   kwave[nx/2+1:] = arange(-nx/2+1,0)
   kwave *= 2*pi/Lx
   g = -0.5*1j*kwave
   # Linear operator
   Ldt = (alfa*kwave**2-gamma*kwave**4)*dt
elif model == 'L95':
   f = 8
   Lx = nx
   dx = 1.
   x = arange(nx)
   # Coefficients
   im2 = (x-2)%nx
   im1 = (x-1)%nx
   ip1 = (x+1)%nx
   # Linear operator
   Ldt = -ones((nx))*dt
pass

# ETDRK4 matrices
Ldt2 = 0.5*Ldt
eLdt = exp(Ldt)
eLdt2 = exp(Ldt2)
q = zeros((nx))
f1 = zeros((nx))
f2 = zeros((nx))
f3 = zeros((nx))
for k in range(nx):
   z = Ldt2[k]+jroot
   q[k] = mean((exp(z)-1)/z).real
   z = Ldt[k]+jroot
   f1[k] = mean((exp(z)*(z**2-3*z+4)-(z+4))/z**3).real
   f2[k] = mean((exp(z)*(z-2)+(z+2))/z**3).real
   f3[k] = mean((exp(z)*(4-z)-(z**2+3*z+4))/z**3).real
# Temporary model variables
N0 = zeros((nx),dtype=complex128)
Na = zeros((nx),dtype=complex128)
Nb = zeros((nx),dtype=complex128)
Nc = zeros((nx),dtype=complex128)
uhat = zeros((nx),dtype=complex128)
ahat = zeros((nx),dtype=complex128)
bhat = zeros((nx),dtype=complex128)
chat = zeros((nx),dtype=complex128)

# Derived parameters
nvar = 1
nm = nx
no = nslot*nx
ntime = int(Tend/(cycle*dt))+1
time = linspace(0,(ntime-1)*cycle*dt/(4*dt),ntime)

# Spinup
xtrue = random.uniform(0,1,nx)
xmean = 1.*xtrue; xstd = xtrue**2
for i in range(int(Tspinup/dt)):
   xtrue[:] = M(xtrue)
   xmean += xtrue; xstd += xtrue**2
xmean /= int(Tspinup/dt); xstd /= int(Tspinup/dt)
xstd[:] = sqrt(xstd-xmean**2)
# True
xt = zeros((nslot+1,nx))
xt[-1] = xtrue
# Obs
observed = zeros((nvar,nslot,nx),dtype=bool)
for ivar in range(nvar):
   for itime in range(nslot): observed[ivar,itime] = land
m = sum(observed)
print('Number of obs:', m)
yo = zeros((nvar,nslot,nx))
sigmay = zeros(yo.shape)
# Initial perturbations
xa = zeros((nslot+1,nmember+1,nx))
xa[-1,-1] = xt[-1]
#xa[-1,-1] = xmean
for k in range(nmember): xa[-1,k] = xa[-1,-1] + xstd*random.normal(0,1,nx)

# Variables
xf = zeros(xa.shape)
yf = zeros((nvar,nslot,nmember+1,nx))
y = zeros((m))
Y = zeros((m,nmember))
s = zeros((m))
# Temporary ETKF variables
w = zeros((nmember,nmember))
wmean = zeros((nmember))
d = zeros((nmember)); E  = zeros((nmember)); V = zeros((nmember,nmember))
Gamma = zeros((nmember)); Lambda = zeros((nmember))
FG = zeros((nmember)); FL = zeros((nmember))
# Output
ytrue = zeros((ntime,nvar,nx))
yfcst = zeros((ytrue.shape))
yana = zeros((ytrue.shape))
yfspr = zeros((ytrue.shape))
yaspr = zeros((ytrue.shape))

# Filter
for i in range(ntime):
   # True
   xt[0] = xt[-1]
   for j in range(nslot):
      xt[j+1] = xt[j]
      for k in range(int(cycle/nslot)): xt[j+1] = M(xt[j+1])
   true0 = State(u=jnp.array(xt[0]))
   true1, trajectory = jaxmodel.integrate(true0, cycle, cycle//nslot)
   xt[-1] = array(true1.u)
   # Obs
   ytrue[i,0] = xt[-1]
   yo[0] = xt[1:] + random.normal(0,sigmao,no).reshape(nslot,nx)
   sigmay[0] = sigmao
   phytrue = jaxmodel.mod2phy(trajectory)
   obstrue = observation.Hforward(phytrue)
   key, obs = observation.sample(key, obstrue)
   #yo[0] = array(obs.u)
   #obs = ObsState(u=jnp.array(yo[0]))
   #print(yo[0]-xt[1:])
   #print(array(obs.u)-xt[-1])
   #if i == 2: sys.exit(0)

   # Compare
   #xt[1] = M(xt[0])
   #true0 = State(u=jnp.array(xt[0]))
   #obs = ObsState(u=jnp.array(yo[0]))
   #true1 = jaxmodel.forward(true0)
   #true1, trajectory = jaxmodel.integrate(true0, cycle, cycle//nslot)
   #print(cycle, cycle//nslot)
   #print(xt[1])
   #print(array(true1.u))
   #sys.exit(0)

   # Forward
   xf[0,:-1] = xa[-1,:-1]
   for k in range(nmember):
      for j in range(nslot):
         xf[j+1,k] = xf[j,k]
         for l in range(int(cycle/nslot)): xf[j+1,k] = M(xf[j+1,k])
   state0 = State(u=jnp.array(xf[0,:-1]))
   state1f, trajectory = ensemble.integrate(state0, cycle, cycle//nslot)
   xf[-1,:-1] = array(state1f.u)
   xf[:,-1] = mean(xf[:,:-1],axis=1)
   yfcst[i,0] = xf[-1,-1]
   yfspr[i,0] = std(xf[-1,:-1],axis=0)

   # Obs space
   yf[0,:,:-1] = xf[1:,:-1]
   yf[:,:,-1] = mean(yf[:,:,:-1],axis=2)
   yfmean = yf[:,:,-1]
   for k in range(nmember):
      yfview = yf[:,:,k] 
      Y[:,k] = (yfview[observed]-yfmean[observed])/sqrt(nmember-1)/sigmay[observed]
   y[:] = (yo[observed]-yfmean[observed])/sigmay[observed]
   for k in range(nmember): xf[:,k] = (xf[:,k]-xf[:,-1])/sqrt(nmember-1)

   # Compare
   #state0 = State(u=jnp.array(xf[0,:-1]))
   #state1f, trajectory = ensemble.integrate(state0, cycle, cycle//nslot)
   #phystate = ensemble.mod2phy(trajectory)
   #obsstate = ensemble.Hforward(phystate)
   #factor = jnp.sqrt(nmember-1)
   #obsmean = scheme.mean(obsstate)
   #obserr = observation.scale(obs)
   #inv = scheme.normalize(1., obs, obsmean, obserr)
   #obspert = scheme.normalize(factor, obsstate, obsmean, obserr)
   #statemean = scheme.mean(state1f)
   #pert = scheme.normalize(factor, state1f, statemean)
   #print(y)
   #print(array(inv.u))
   #print(xf[1,-1])
   #print(array(statemean.u))
   #sys.exit(0)
   #for k in range(nmember): xf[:,k] = (xf[:,k]-xf[:,-1])/sqrt(nmember-1)
   #print(Y[:,1])
   #print(array(obspert.u[1]))
   #print(xf[1,1])
   #print(array(pert.u[1]))
   #sys.exit(0)

   # Maximum likelihood for analysis
   wmean[:] = 0.
   #wmean[:] = random.normal(0,sigmao,nmember)
   Jold = logp(wmean, y, Y, s, d)
   res = minimize(logp, wmean, args=(y, Y, s, d), method='L-BFGS-B', jac=dlogp, tol=1.e-12, options={'disp':False})
   Jnew = logp(res.x, y, Y, s, d)
   #print('LBFGSB:', i, Jold, Jnew)
   if Jnew < Jold: wmean[:] = res.x
   xa[:,-1] = xf[:,-1]
   for k in range(nmember): xa[:,-1] += wmean[k]*xf[:,k]
   phystate = ensemble.mod2phy(trajectory)
   obsstate = ensemble.Hforward(phystate)
   factor = jnp.sqrt(nmember-1)
   obsmean = scheme.mean(obsstate)
   obserr = observation.scale(obs)
   inv = scheme.normalize(1., obs, obsmean, obserr)
   obspert = scheme.normalize(factor, obsstate, obsmean, obserr)
   statemean = scheme.mean(state1f)
   pert = scheme.normalize(factor, state1f, statemean)
   # Check logp
   #print(logp(wmean, array(inv.u[0]), array(obspert.u[:,0,:].T), s, d))
   #print(jaxlogp(jnp.array(wmean), obspert, inv, scheme))
   #print(dlogp(wmean, array(inv.u[0]), array(obspert.u[:,0,:].T), s, d))
   #print(jaxdlogp(jnp.array(wmean), obspert, inv, scheme))
   # Check BFGS
   #wmean[:] = 0.
   #y = array(inv.u[0]); Y = array(obspert.u[:,0,:].T)
   #Jold = logp(wmean, y, Y, s, d)
   #res = minimize(logp, wmean, args=(y, Y, s, d), method='L-BFGS-B', jac=dlogp, tol=1.e-12, options={'disp':False})
   #Jnew = logp(res.x, y, Y, s, d)
   #print('LBFGSB:', Jold, Jnew)
   #jaxw = scheme._weight_ana(obspert, inv)
   #print(res.x)
   #print(array(jaxw))
   # Check Hessian
   y = array(inv.u[0]); Y = array(obspert.u[:,0,:].T)
   jaxw = scheme._weight_ana(obspert, inv)
   s[:] = dot(Y,array(jaxw))-y
   s[:] = expit(s)
   s[:] = sqrt(s*(1.-s))
   #print('s:',s)
   for k in range(nmember): Y[:,k] *= s
   print(Y[:,1])
   E[:], V[:,:] = linalg.eigh(dot(Y.T,Y))
   E[:] = E[::-1]; V[:,:] = V[:,::-1]
   invalid = E < 0.; E[invalid] = 0.
   E[-1] = 0.
   #print(E)
   s = scheme.weighted_sum(jaxw,obspert) - inv
   s = s.expit()
   s = s*(1.-s)
   s = s.sqrt()
   #print('s:',s.u)
   state = scheme.scale_ensemble(s,obspert)
   print(state.u[1,0])
   #print(state.u[1])
   YTY = scheme.covariance(state)
   E, V = jnp.linalg.eigh(YTY)
   E = E[::-1]; V = V[:,::-1]
   E = jnp.where(E < 0., 0., E)
   E = E.at[-1].set(0.)
   #print(E)
   sys.exit(0)
   jaxw = scheme._weight_ana(obspert, inv)
   ana = scheme._compute_ana(jaxw, pert, statemean)
   xa[-1,-1] = array(ana.u)

   # Compare
   #jaxw = scheme._weight_ana(obspert, inv)
   #print(res.x)
   #print(array(jaxw))
   #ana = scheme._compute_ana(jaxw, pert, statemean)
   #print(array(xa[1,-1]))
   #print(array(ana.u))
   #sys.exit(0)

   # Eigen-decomposition
   E[:], V[:,:] = linalg.eigh(dot(Y.T,Y))
   E[:] = E[::-1]; V[:,:] = V[:,::-1]
   invalid = E < 1.E-12; E[invalid] = 0.
   E[-1] = 0.
   
   # Compare
   #YTY = scheme.covariance(obspert)
   #jaxE, jaxV = jnp.linalg.eigh(YTY)
   #jaxE = jaxE[::-1]; jaxV = jaxV[:,::-1]
   #jaxE = jnp.where(jaxE < 1e-12, 0., jaxE)
   #jaxE = jaxE.at[-1].set(0.)
   #print(E)
   #print(array(jaxE))
   #sys.exit(0)
   
   # inflation
   for k in range(nmember):
      if E[k] < 1.E-12: break
   nrank = k
   Gamma[:nrank] = sqrt(E[:nrank]) # singular values or gamma
   Gamma[nrank:] = 0.
   Lambda[:] = 1./sqrt(1.+Gamma**2) # lambda
   Lmean = Lambda.mean()
   #FL[:] = (1.+rho*(1/Lmean-1))*Lambda
   FL[:] = (1.+rho*(1-Lmean))*Lambda
   
   # Compare
   #jaxGamma = jnp.sqrt(jaxE)
   #jaxLambda = 1.0/jnp.sqrt(1.0+jaxGamma**2)
   #jaxLmean = jnp.mean(jaxLambda)
   #jaxFL = (1.+rho*(1-jaxLmean))*jaxLambda
   #print(FL)
   #print(array(jaxFL))
   #sys.exit(0)

   # Analysis ensemble
   for k in range(nmember):
      w[:,k] = 0.; w[k,k] = sqrt(nmember-1)
      d[:] = dot(V.T,w[:,k])
      #d *= Lambda
      d *= FL
      w[:,k] = 0.
      for k2 in range(nmember): w[:,k] += d[k2]*V[:,k2]
   jaxw = scheme._weight_ens(obspert)
   w = array(jaxw)
   # Compare
   #jaxw = scheme._weight_ens(obspert)
   #print(w[:,1])
   #print(array(jaxw)[:,1])
   #sys.exit(0)

   for k in range(nmember):
      xa[:,k] = xa[:,-1]
      for k2 in range(nmember): xa[:,k] += w[k2,k]*xf[:,k2]
   anastate = scheme._compute_ens(jaxw, pert, ana)
   xa[-1,:-1] = array(anastate.u)
   yana[i,0] = xa[-1,-1]
   yaspr[i,0] = std(xa[-1,:-1],axis=0)

   # Compare
   #anastate = scheme._compute_ens(jaxw, pert, ana)
   #print(array(xa[1,1]))
   #print(array(anastate.u[1]))
   #sys.exit(0)

   # Diagnostic
   ntotal = 0; rmsef = 0.; rmsea = 0.
   for ivar in range(nvar):
      ntotal += nland
      rmsef += sum((yfcst[i,ivar,land]-ytrue[i,ivar,land])**2)
      rmsea += sum((yana[i,ivar,land]-ytrue[i,ivar,land])**2)
   rmsef = sqrt(rmsef/ntotal)
   rmsea = sqrt(rmsea/ntotal)
   print('DA:', i, rmsef, rmsea)
#write_netcdf(output_file, time, ytrue, yfcst, yana, yfspr, yaspr)

