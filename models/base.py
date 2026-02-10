import jax
import jax.numpy as jnp
from typing import Protocol, TypeVar, runtime_checkable
from typing import NamedTuple, Optional, Tuple, Union, ClassVar
from dajax.models.mixin import Forcing

State = TypeVar('State', bound=NamedTuple)
PhyState = TypeVar('PhyState', bound=NamedTuple)
ObsState = TypeVar('ObsState', bound=NamedTuple)
Param = TypeVar('Param', bound=NamedTuple)
Config = TypeVar('Config', bound=NamedTuple)

def _apply_unary_elementwise(op, state: NamedTuple) -> NamedTuple:
   """Apply unary operation elementwise on state."""
   return jax.tree_util.tree_map(
      lambda x: op(x) if x is not None else None, state)

def _sum_elements(state: NamedTuple) -> float:
   """Compute sum of all elements in a state."""
   sums = jax.tree_util.tree_map(
      lambda x: jnp.sum(x) if x is not None else 0., state)
   return jnp.sum(jnp.array(jax.tree_util.tree_leaves(sums)))

def _state_op(op, state1: NamedTuple, state2: NamedTuple) -> NamedTuple:
   """Apply operation between two states."""
   if type(state1) != type(state2):
      raise TypeError(f"Cannot operate between {type(state1)} and {type(state2)}")
   return jax.tree_util.tree_map(
      lambda x, y: op(x,y) if (x is not None and y is not None) else None, state1, state2)

def _scalar_op(op, state: NamedTuple, scalar) -> NamedTuple:
   """Apply operation between state and scalar."""
   return jax.tree_util.tree_map(
      lambda x: op(x,scalar) if x is not None else None, state)

def _inner_product(state1: NamedTuple, state2: NamedTuple) -> float:
   """Compute inner product between two states."""
   if type(state1) != type(state2):
      raise TypeError(f"Cannot compute inner product between {type(state1)} and {type(state2)}")
   products = jax.tree_util.tree_map(
      lambda x, y: jnp.sum(x*y) if (x is not None and y is not None) else 0., state1, state2)
   return jnp.sum(jnp.array(jax.tree_util.tree_leaves(products)))

def add_operators(cls):
   """Decorator to add arithmetic operators to a NamedTuple class."""
   @jax.jit
   def add(self, other):
      if hasattr(other, '_fields'): return _state_op(jnp.add, self, other)
      return _scalar_op(jnp.add, self, other)

   @jax.jit
   def sub(self, other):
      if hasattr(other, '_fields'): return _state_op(jnp.subtract, self, other)
      return _scalar_op(jnp.subtract, self, other)

   @jax.jit
   def mul(self, other):
      if hasattr(other, '_fields'): return _state_op(jnp.multiply, self, other)
      return _scalar_op(jnp.multiply, self, other)

   @jax.jit
   def truediv(self, other):
      if hasattr(other, '_fields'): return _state_op(jnp.divide, self, other)
      return _scalar_op(jnp.divide, self, other)

   @jax.jit
   def pow(self, other):
      if hasattr(other, '_fields'): return _state_op(jnp.power, self, other)
      return _scalar_op(jnp.power, self, other)
   
   @jax.jit
   def neg(self):
      return _scalar_op(jnp.multiply, self, -1.0)

   @jax.jit
   def abs(self):
      return jax.tree_util.tree_map(
         lambda x: jnp.abs(x) if x is not None else None, self)

   @jax.jit
   def radd(self, other):
      return self.__add__(other)

   @jax.jit
   def rsub(self, other):
      return (-self).__add__(other)

   @jax.jit
   def rmul(self, other):
      return self.__mul__(other)

   @jax.jit
   def rtruediv(self, other):
      if hasattr(other, '_fields'):
         return _state_op(jnp.divide, other, self)
      return jax.tree_util.tree_map(
         lambda x: other/x if x is not None else None, self)
   
   @jax.jit
   def sum(self):
      return _sum_elements(self)
   
   @jax.jit
   def sqrt(self):
      return _apply_unary_elementwise(jnp.sqrt, self)
   
   @jax.jit
   def log_expit(self):
      """Compute log(expit(x)) elementwise using log_sigmoid for stability."""
      return _apply_unary_elementwise(jax.nn.log_sigmoid, self)

   @jax.jit
   def expit(self):
      """Compute expit(x) elementwise using sigmoid."""
      return _apply_unary_elementwise(jax.nn.sigmoid, self)

   @jax.jit
   def dot(self, other):
      return _inner_product(self, other)


   # Add all operators to the class
   cls.__add__ = add
   cls.__sub__ = sub
   cls.__mul__ = mul
   cls.__truediv__ = truediv
   cls.__pow__ = pow
   
   cls.__neg__ = neg
   cls.__abs__ = abs
   cls.__radd__ = radd
   cls.__rsub__ = rsub
   cls.__rmul__ = rmul
   cls.__rtruediv__ = rtruediv
   cls.sum = sum
   cls.sqrt = sqrt
   cls.log_expit = log_expit
   cls.expit = expit
   cls.dot = dot

   return cls

@runtime_checkable
class Model(Protocol[State, PhyState, Param, Config]):
   """Protocol defining the interface for models in the data assimilation system.
   
   This protocol defines the minimum interface that models must implement to be
   compatible with the data assimilation system. Models can implement additional
   methods and attributes as needed.
   
   Models can be one of two types regarding parameter initialization:
   1. Independent parameters: Parameters can be created without model initialization
      These models should implement default_param as a staticmethod
   2. Dependent parameters: Parameters require model initialization
      These models should set requires_instance_param = True and implement create_default_param
   """
   config: Config
   requires_instance_param: ClassVar[bool] = False  # Default to independent parameters
   
   @staticmethod
   def default_param() -> Optional[Param]:
      """Get default deterministic parameters for independent parameter models.
      Returns None for models requiring instance-based parameter creation."""
      ...
      
   def create_default_param(self) -> Optional[Param]:
      """Get default deterministic parameters for dependent parameter models.
      Returns None for models using static parameter creation."""
      ...
      
   def random_param(self, key: jax.Array, base: Param, noise_scale: float) -> Param:
      """Initialize parameters with optional reference parameters"""
      ...
      
   def default_state(self, param: Param) -> State:
      """Get default deterministic state"""
      ...
   
   def random_state(self, key: jax.Array, param: Param, base: State, noise_scale: float) -> State:
      """Initialize a single state with optional reference state"""
      ...
   
   def forward(self, state: State, param: Param) -> State:
      """Advance the model state by one timestep."""
      ...
   
   def integrate(self, state: State, param: Param, nstep: int, save_freq: Optional[int] = None, 
                  forcings: Optional[Forcing] = None) -> Tuple[State,State]:
      """Integrate the model forward in time."""
      ...
   
   def mod2phy(self, state: State, param: Param) -> PhyState:
      """Map model space to physical space."""
      ...
   
   def phy2mod(self, phystate: PhyState, ref_state: State, param: Param) -> State:
      """Convert physical state back to full model state.
      ref_state provides auxiliary fields (forcing, cached, time, etc.)
      """
      ...

   @property
   def state_info(self) -> dict[str, Tuple[int, ...]]:
      """Returns shape information about the state space."""
      ...
   
   @property
   def grid_info(self) -> dict[str, Union[int, jax.Array]]:
      """Returns shape information about the physical space."""
      ...
