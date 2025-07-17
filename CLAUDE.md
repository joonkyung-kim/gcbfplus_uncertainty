# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GCBF+ (Graph Control Barrier Function Plus) is a JAX-based implementation of a neural graph control barrier function framework for distributed safe multi-agent control. The project implements the T-RO paper "GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control" by Zhang et al.

## Key Commands

### Environment Setup
```bash
# Create conda environment
conda create -n gcbfplus python=3.10
conda activate gcbfplus

# Install JAX (follow official JAX installation guide first)
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Training
```bash
# Train GCBF+ model (example for DoubleIntegrator)
python train.py --algo gcbf+ --env DoubleIntegrator -n 8 --area-size 4 --loss-action-coef 1e-4 --n-env-train 16 --lr-actor 1e-5 --lr-cbf 1e-5 --horizon 32

# Use settings from settings.yaml for paper reproduction
python train.py --algo gcbf+ --env DoubleIntegrator -n 8 --area-size 4 --loss-action-coef 1e-4 --n-env-train 16 --lr-actor 1e-5 --lr-cbf 1e-5 --horizon 32 --steps 1000
```

### Testing
```bash
# Test trained model
python test.py --path <path-to-log> --epi 5 --area-size 4 -n 16 --obs 0

# Test nominal controller
python test.py --env SingleIntegrator -n 16 --u-ref --epi 1 --area-size 4 --obs 0

# Test CBF-QP algorithms
python test.py --env SingleIntegrator -n 16 --algo dec_share_cbf --epi 1 --area-size 4 --obs 0 --alpha 1
```

### Configuration
- Use `settings.yaml` for hyperparameter configurations per environment
- Environment-specific parameters are stored in `settings.yaml`
- Pre-trained models are available in `pretrained/` directory

## Architecture

### Core Components

**Multi-Agent Controllers** (`gcbfplus/algo/`):
- `base.py`: Abstract base class `MultiAgentController` defining the interface
- `gcbf_plus.py`: Main GCBF+ implementation with graph neural networks and CBF learning
- `gcbf.py`: Standard GCBF implementation
- `centralized_cbf.py`: Centralized CBF-QP controller
- `dec_share_cbf.py`: Decentralized shared CBF-QP controller

**Neural Network Modules** (`gcbfplus/algo/module/`):
- `cbf.py`: Control Barrier Function network implementation
- `policy.py`: Actor policy network (DeterministicPolicy)
- `value.py`: Value function networks
- `distribution.py`: Probability distribution utilities

**Environments** (`gcbfplus/env/`):
- `base.py`: Abstract `MultiAgentEnv` base class
- Environment implementations: `single_integrator.py`, `double_integrator.py`, `dubins_car.py`, `linear_drone.py`, `crazyflie.py`
- `obstacle.py`: Obstacle handling utilities
- `plot.py`: Visualization utilities

**Training Infrastructure** (`gcbfplus/trainer/`):
- `trainer.py`: Main training loop and environment interaction
- `buffer.py`: Experience replay buffer (`MaskedReplayBuffer`)
- `data.py`: Data structures (Rollout, etc.)
- `utils.py`: Training utilities and helpers

**Neural Networks** (`gcbfplus/nn/`):
- `gnn.py`: Graph neural network implementations
- `mlp.py`: Multi-layer perceptron implementations
- `utils.py`: Network utility functions

**Utilities** (`gcbfplus/utils/`):
- `graph.py`: Graph data structures (`GraphsTuple`)
- `typing.py`: Type definitions
- `utils.py`: General utility functions

### Key Design Patterns

1. **Graph-based representation**: All environments use `GraphsTuple` for state representation
2. **JAX-based implementation**: Heavy use of JAX for JIT compilation and automatic differentiation
3. **Modular architecture**: Clear separation between environments, algorithms, and neural networks
4. **Abstract base classes**: `MultiAgentController` and `MultiAgentEnv` define interfaces

### Dependencies

Primary dependencies:
- JAX/Flax for neural networks and automatic differentiation
- Jraph for graph neural networks
- JaxProxQP for quadratic programming
- Optax for optimization
- Wandb for experiment tracking

### Directory Structure

- `gcbfplus/`: Main package source code
- `pretrained/`: Pre-trained models organized by environment
- `logs/`: Training logs and checkpoints
- `media/`: Visualization media files
- `train.py`: Main training script
- `test.py`: Testing and evaluation script
- `settings.yaml`: Hyperparameter configurations

### Common Environments

Supported environments:
- `SingleIntegrator`: 2D single integrator dynamics
- `DoubleIntegrator`: 2D double integrator dynamics  
- `DubinsCar`: Dubins car model
- `LinearDrone`: Linear drone dynamics
- `CrazyFlie`: CrazyFlie quadrotor model

Each environment supports:
- Variable number of agents
- Configurable area size
- Obstacle avoidance
- LiDAR sensing (configurable rays)