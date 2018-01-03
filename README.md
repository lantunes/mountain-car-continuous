Mountain Car Continuous
=======================

This repository contains implementations of algorithms that solve (or attempt to solve) the continuous mountain car
problem, which is based on continuous states and actions. The continuous mountain car environment is provided by the 
OpenAI Gym (MountainCarContinuous-v0). The code in this repo makes use of the Tensorflow 1.1 library.

The following algorithms are implemented:
 
REINFORCE with Stochastic Policy Gradient:
- located in the rl.reinforce module
- uses a Gaussian Policy Gradient
- mu and sigma are both learned (sigma does not depend on the state, while mu does)
- the input features for mu are learned using a linear layer of neurons given the raw state
- a vector of parameters is defined for sigma, which are adjusted using the Gaussian Policy Gradient, and are finally 
summed to yield a scalar, of which the exponential is used as the value for sigma (the standard deviation) 
- the optimization is done in minibatches, and the batches are shuffled
- the class of interest is rl.reinforce.agent.TFNeuralNetStochasticPolicyAgent
- the entry point is the simulator.py script in the rl.reinforce module

- the plot below represents the average total reward for each episode over 10 trials:

<img src="https://raw.githubusercontent.com/lantunes/mountain-car-continuous/master/util/gaussian-policy-gradient-10-runs.png" width="50%"/>

(MountainCarContinuous-v0 defines "solving" as getting an average reward of 90.0 over 100 consecutive trials.)