# Genetic Algorithm for Biomechanical Movement Optimization

A Python-based optimization framework using Genetic Algorithms and CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to optimize biomechanical movements in OpenSim models. This project simulates and optimizes muscle activation patterns for various movements including jumping, counter-movement jumps, and flips.

## Features

- **Multiple Optimization Algorithms**
  - Genetic Algorithm with customizable mutation, crossover, and selection
  - CMA-ES (Covariance Matrix Adaptation Evolution Strategy) solver
  
- **Biomechanical Movement Types**
  - Jumper: Basic vertical jump optimization
  - cmjJumper: Counter-movement jump optimization
  - Flipper: Backflip/flip movement optimization

- **Parallel Processing**
  - Multi-worker support for faster simulations
  - Automatic CPU core detection or manual worker configuration

- **Flexible Muscle Control**
  - Muscle group-based control system
  - Time-based activation patterns
  - Support for 8 major muscle groups:
    - Quadriceps
    - Hamstrings
    - Glutes
    - Calves
    - Tibialis Anterior
    - Trunk Flexors
    - Trunk Extensors
    - Hip Flexors

## Requirements

This project requires:
- Python 3.x
- OpenSim (opensim-python)
- NumPy
- CMA (for CMA-ES solver)
- Multiprocessing support

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Talhauzumcu/GeneticAlgorithm.git
cd GeneticAlgorithm
```
2. Install Miniconda (if not already installed):
    - Download from [Miniconda official site](https://docs.conda.io/en/latest/miniconda.html)
    - Follow the installation instructions for your operating system

3. Create and activate a conda environment:
```bash
conda create -n opensim-env python=3.x
conda activate opensim-env
```

4. Install OpenSim:
```bash
conda install -c opensim-org opensim
```

5. Install the remaining dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to start optimization:

```bash
python main.py
```

### Configuration

Edit the parameters in `main.py` to customize the optimization:

```python
POPULATION_SIZE = 200          # Number of individuals in each generation
GENERATION_COUNT = 1000        # Number of generations to evolve
MUTATION_RATE = 0.02          # Probability of mutation (0-1)
INTEGRATION_DURATION = 2       # Simulation time in seconds
OVERLAP = 5                    # Elite individuals to carry forward
RANDOM = 0                     # Random individuals to introduce per generation
N_WORKERS = 8                  # Number of parallel workers
```

### Using Different Movement Types

Choose your movement type in `main.py`:

```python
from objects import Jumper, cmjJumper, Flipper

# Select one:
pop_object = Jumper      # For basic jump
pop_object = cmjJumper   # For counter-movement jump
pop_object = Flipper     # For flip movements
```

### Using Genetic Algorithm

```python
from geneticSolver import geneticSolver

genetic_solver = geneticSolver(
    model_path='./models/your_model.osim',
    population_object=Jumper,
    initial_population=None,  # or provide custom population
    population_size=200,
    generation_count=1000,
    mutation_rate=0.02,
    overlap=5,
    random=0,
    initial_time=0.0,
    final_time=2.0,
    n_workers=8,
    termination_function=your_termination_function
)
```

### Using CMA-ES Solver

```python
from cmaSolver import cmaSolver

cma_solver = cmaSolver(
    model_path='./models/your_model.osim',
    population_object=Jumper,
    population_size=200,
    generation_count=1000,
    initial_time=0.0,
    final_time=2.0,
    n_workers=8,
    termination_function=your_termination_function,
    sigma0=0.3  # Initial standard deviation
)
```

## Project Structure

```
.
├── main.py                      # Main entry point
├── geneticSolver.py             # Genetic algorithm implementation
├── cmaSolver.py                 # CMA-ES solver implementation
├── objects.py                   # Movement type definitions (Jumper, Flipper, etc.)
├── simulator.py                 # OpenSim simulation wrapper
├── modelController.py           # Model manipulation utilities
├── modify_model.py              # Model modification tools
├── resimulate.py               # Re-simulation utilities
├── utils.py                     # Helper functions
├── requirements.txt             # Python dependencies
├── models/                      # OpenSim model files
├── initial_population/          # Starting population files
├── GeneticSolverJsonResults/    # Output results from genetic solver
├── resimulatedJsonResults/      # Re-simulation results
└── results/                     # General results directory
```

## Output

Results are saved as JSON files containing:
- Generation number
- Individual ID
- Fitness score
- Muscle activation patterns (genom)

Example filename: `flipper_gen100_15_1.84.json`
- `gen100`: Generation 100
- `15`: Individual ID 15
- `1.84`: Fitness score of 1.84

## Custom Termination Functions

Define custom termination conditions for your simulations:

```python
def termination_function(model, state):
    pelvis_ty = model.getCoordinateSet().get('pelvis_ty').getValue(state)
    return pelvis_ty < 0.25  # Stop if pelvis height below 0.25
```
## Authors

- Talha Uzumcu

