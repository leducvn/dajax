from typing import Generic, TypeVar, NamedTuple

# Type variables for different forcing types
Forcing = TypeVar('Forcing', bound=NamedTuple)
ForcingConfig = TypeVar('ForcingConfig', bound=NamedTuple)

class ForcedModelMixin(Generic[Forcing, ForcingConfig]):
   """Mixin class providing common functionality for models with forcings.
   
   This mixin provides:
   1. Standard methods for handling forcings
   2. Utilities for time management
   3. Methods for forcing data validation and preprocessing
   
   Example usage:
   ```python
   class Forcing(NamedTuple):
      rain: jax.Array    # Precipitation
      pet: jax.Array    # Evaporation
   
   class ForcingConfig(NamedTuple):
      start_date: Optional[int] = None
      end_date: Optional[int] = None
      
   class HyMod(Model[State, PhyState, Param, Config], ForcedModelMixin[Forcing, ForcingConfig]):
      def __init__(self, config: Config, forcing_config: ForcingConfig):
         super().__init__()
         self.forcing_config = forcing_config
   ```
   """

   forcing_config: ForcingConfig
   
   def load_forcing(self) -> Forcing:
      """Load forcing data from file with proper validation."""
      raise NotImplementedError
   
   def preprocess_forcing(self, forcing: Forcing) -> Forcing:
      """Preprocess forcing data (e.g., interpolation, unit conversion)."""
      raise NotImplementedError


