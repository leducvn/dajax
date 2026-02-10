# DAJax: Data Assimilation with JAX

⚠️ **DEVELOPMENT STATUS**: This system is actively under development. The API and structure are subject to change. Use at your own discretion and expect breaking changes in future updates.

## Overview

DAJax is a high-performance data assimilation system built with Python JAX for scientific computing and numerical weather prediction applications.

**Key Features:**
- Multiple dynamical models (Lorenz96, SHQG, SPEEDY)
- Data assimilation schemes (ETKF, IDA)
- Flexible observation systems with various distributions and operators
- Control theory applications through Interval Data Assimilation (IDA)
- GPU acceleration via JAX
- Modular architecture for experimentation

⚠️ **Note**: This is research software under active development. Features, APIs, and interfaces may change without notice.

## Project Structure
```
dajax/
├── core/           # Core DA system implementations (OSSE, IDAOSSE)
├── models/         # Dynamical models
│   ├── lorenz96/   # Lorenz 96 model
│   ├── ks/         # Kuramoto-Sivashinsky model
│   ├── shqg/       # Spherical quasi-geostrophic model
│   └── speedy/     # SPEEDY atmospheric model
├── obs/            # Observation system components
│   ├── observation.py      # Main observation system
│   ├── likelihood.py       # Distribution classes (Gaussian, Logistic)
│   ├── obsoperator.py      # Observation operators
│   ├── obslocation.py      # Spatial masks (Random, Regular, Area)
│   └── obstime.py          # Temporal masks
├── schemes/        # Data assimilation schemes
│   ├── etkf/       # Ensemble Transform Kalman Filter
│   └── ida/        # Interval Data Assimilation
├── utils/          # Utilities (diagnostics, I/O, visualization)
├── data/           # Model data and parameters (symlink or local)
│   ├── hymod/      # Hydrological model data
│   ├── shqg/       # SHQG model data
│   └── speedy/     # SPEEDY model data
└── examples/       # Example scripts and tutorials
    ├── shqg/       # SHQG examples
    ├── lorenz96/   # Lorenz96 examples
    └── speedy/     # SPEEDY examples
```

## Installation

### Requirements

- Python 3.12+
- JAX (GPU support recommended for large-scale experiments)
- NumPy
- SciPy
- xarray
- netCDF4
- matplotlib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/leducvn/dajax.git
cd dajax
```

2. Install dependencies:
```bash
pip install jax jaxlib numpy scipy xarray netcdf4 matplotlib
```

For GPU support (recommended):
```bash
# CUDA 12.x
pip install jax[cuda12]
# Or CUDA 11.x
pip install jax[cuda11]
```

3. Set up the data directory:

The `data` directory may be a symlink to a larger data storage location. If you're setting up from scratch:
```bash
# Option 1: Use symlink to existing data
ln -s /path/to/your/data ./data

# Option 2: Create local data directories
mkdir -p data/{hymod,shqg,speedy}
```

## Quick Start

### Basic OSSE with SHQG Model
```python
import jax
import jax.numpy as jnp
from dajax.models.shqg import SHQG, Config
from dajax.schemes.ida import IDA
from dajax.obs.observation import ObsSetting, Observation
from dajax.obs.likelihood import Logistic
from dajax.obs.obsoperator import IdentityMap
from dajax.obs.obslocation import RandomMask
from dajax.obs.obstime import DATimeMask
from dajax.core.daosse import DAConfig, IDAOSSE

# Model configuration
config = Config(T=21, Tgrid=31, nlev=3, dt=6*3600)
law = SHQG.default_param()
nature = SHQG(config=config, param=law)
model = SHQG(config=config, param=law)

# DA configuration
daconfig = DAConfig(
    dawindow=4,      # DA window length
    obsfreq=4,       # Observation frequency
    nmember=100,     # Ensemble size
    obserr=4.0,      # Observation error
    rho=1.0          # Inflation factor
)

# Observation system
key = jax.random.PRNGKey(0)
mask = RandomMask(key, fraction=0.32, grid_info=nature.grid_info)
timemask = DATimeMask(ntime=int(daconfig.dawindow/daconfig.obsfreq)+1)

settings = {
    'u': ObsSetting(
        distribution=Logistic(scale=4.0, bias=0.0),
        mapper=IdentityMap('u'),
        mask=mask,
        timemask=timemask
    )
}
observation = Observation(settings, obsstate_class=nature.ObsState)

# DA scheme
scheme = IDA(rho=1.0)

# DA system
da_system = IDAOSSE(nature, model, observation, scheme, diagnostics, daconfig)

# Run experiments (see examples/ for complete workflows)
```

For complete working examples, see the `examples/` directory.

## Key Components

### Models

Currently supported models:
- **Lorenz96**: Classic chaotic system for DA testing
- **Kuramoto-Sivashinsky**: Nonlinear PDE model
- **SHQG**: Spherical quasi-geostrophic atmospheric model
- **SPEEDY**: Simplified atmospheric GCM

Each model implements:
- `integrate()`: Time integration
- `default_state()`: Initial state generation
- `random_state()`: Perturbed state generation
- Grid information for observation operators

### Observation System

Highly modular observation framework:
- **Distributions**: 
  - Gaussian: Traditional Gaussian observations
  - Logistic: For inequality constraints
- **Operators**: Identity mapping, interpolation, derived quantities
- **Spatial Masks**: Random, regular grid, area-based selection
- **Temporal Masks**: Configurable observation windows

### Data Assimilation Schemes

- **ETKF**: Ensemble Transform Kalman Filter for traditional Gaussian DA
- **IDA**: Interval Data Assimilation for inequality constraints and control applications

### Control Theory Applications

The IDA scheme supports control modifications through:
- **Selectors**: Determine which grid points to modify
- **Applicators**: Determine how to apply modifications
- Control on initial conditions via `ctlforward()` method
- Flexible experimentation with selective vs. global modifications

## Important Notes

### Numerical Stability

⚠️ **Known Issues:**
- Be cautious with cubic interpolation for small values (~10^-10) in SHQG

### Data Organization

The `data/` directory may be a symbolic link to a separate storage location. Ensure your data path is correctly set up before running experiments.

## Examples

See the `examples/` directory for complete experiment scripts:
- OSSE (Observing System Simulation Experiments)
- Twin experiments
- Control theory applications

Each example includes configuration, spinup, DA cycling, and output generation.

## Output Format

Results are saved in netCDF format containing:
- **States**: truth, forecast, analysis trajectories
- **Diagnostics**: RMSE, ME, and other scores
- **Metadata**: Coordinates (time, lat, lon, lev), configuration parameters

## Development Status

This project is under active development. Current focus areas:
- Extending 2D capabilities to 3D with vertical level selection
- Enhanced control modification strategies
- Integration with machine learning models
- Performance optimization for large-scale applications

**Expect:**
- API changes without backward compatibility
- New features and modules
- Refactoring of existing components
- Documentation updates

## Contributing

This is currently a research project. If you find bugs or have suggestions, please contact the development team.

## Citation

If you use DAJax in your research, please cite appropriately (to be added).

## License

[To be determined]

## Contact

[To be added]

## Acknowledgments

[To be determined]