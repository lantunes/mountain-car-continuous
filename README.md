Mountain Car Continuous
=======================

This repository contains implementations of algorithms that solve (or attempt to solve) the continuous mountain car
problem, which is based on continuous states and actions. The continuous mountain car environment is provided by the 
OpenAI Gym (MountainCarContinuous-v0). The code in this repo makes use of the Tensorflow 1.1 library.

The following algorithms are implemented:
 
Stochastic Policy Gradient with REINFORCE:
- located in the rl.reinforce module
- uses a Gaussian Policy Gradient
- mu and sigma are learned separately (i.e. sigma does not depend on the state, while mu does)
- the input features for mu are learned using a linear layer of neurons
- a vector of parameters is defined for sigma, which are adjusted using the Gaussian Policy Gradient, and are finally 
summed to yield a scalar, of which the exponential is used as the value for sigma (the standard deviation) 
- the class of interest is rl.reinforce.agent.TFNeuralNetStochasticPolicyAgent
- the optimization is done in minibatches, and the batches are shuffled

- the plot below represents the average reward after 10 trials:

![Avg. Reward](https://raw.githubusercontent.com/lantunes/mountain-car-continuous/master/util/gaussian-policy-gradient-10-runs.png)

(MountainCarContinuous-v0 defines "solving" as getting average reward of 90.0 over 100 consecutive trials.) <!-- .element height="50%" width="50%" -->