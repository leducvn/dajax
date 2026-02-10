"""
SPEEDY atmospheric model for JAX.

Simplified Parameterizations, primitivE-Equation DYnamics model
converted from Fortran to JAX.
"""

from .main import SPEEDY
from .state import (
    State, PhyState, ObsState, SpectralState, Param, Config,
    ForcingState, LandState, SeaState, SolarState, CachedState, DiagState
)
from .util import TimeInfo

__all__ = [
    # Main class
    'SPEEDY',
    # States
    'State',
    'SpectralState',
    'PhyState',
    'ObsState',
    'Param',
    'Config',
    'TimeInfo',
    # Sub-states (if needed externally)
    'ForcingState',
    'LandState',
    'SeaState',
    'SolarState',
    'CachedState',
    'DiagState',
]