# DeepRL

Implementation of some Deep Reinforcement Learning algorithms and environments.

## Description

The goal of this project is to have complete modularity with the algorithms and models used.

The implementations are completely made in PyTorch. 

The environments used can either be single-agent using the Gymnasium
API, or multi-agents using the PettingZoo Parallel API. Most algorithms also support action masking.

## Getting Started

### Technologies used

* Python>=3.10
* PyTorch
* Install all the requirements using `pip install -r requirements.txt`

### Usage

* Change the algorithm and the environment in the `main.py` file.

### Algorithms

The following algorithms are currently available:
* PPO (discrete [supports action masking] and continuous actions)
* A2C (discrete [supports action masking] and continuous actions)
* DQN (discrete actions)

### Environments

The following environments have been implemented:
* Snake
* Minesweeper

Any Gymnasium Env or PettingZoo ParallelEnv can be used.

## Authors

Timothé Watteau (@timothewt)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
