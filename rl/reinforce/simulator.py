import gym

from rl.reinforce.agent import *

# MountainCarContinuous-v0 defines "solving" as getting an average reward of 90.0 over 100 consecutive trials.
env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space

MAX_EPISODES = 200
MAX_STEPS = 1000

agent = TFNeuralNetStochasticPolicyAgent(env, num_input=state_dim, init_learning_rate=5e-5, min_learning_rate=1e-10,
                                         learning_rate_N_max=2000, shuffle=True, batch_size=24, sigma=None)

# render an animation of each step in an episode
render = False

if __name__ == "__main__":

    for episode_counter in range(MAX_EPISODES):
        state = env.reset()
        total_rewards = 0
        sigmas = []

        done = False
        for step_counter in range(MAX_STEPS):
            if render:
                env.render()
            action, sigma = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)

            total_rewards += reward
            sigmas.append(sigma)
            agent.store_rollout(state, action, reward)

            state = next_state
            if done:
                break

        agent.update_model(episode_counter)

        print("{},{:.2f}".format(episode_counter, total_rewards))
