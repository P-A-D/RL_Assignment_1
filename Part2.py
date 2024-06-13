import copy
import numpy as np
import matplotlib.pyplot as plt


class Environment:
    """The environment class that gives rewards to the agent. Its rewards can be adjusted over time steps."""
    def __init__(self, change_type='gradual', gradual_mode=True):
        self.change_type = change_type
        self.gradual_mode = gradual_mode
        self.reward_distribution_means = np.random.normal(loc=0.0, scale=1.0, size=10)

    def _get_reward(self, action):
        """Protected method that samples a reward from the reward distribution."""
        return np.random.normal(loc=self.reward_distribution_means[action], scale=1.0)

    def _adjust(self):
        """Adjusts the reward distribution means."""
        if self.change_type == 'gradual':
            if self.gradual_mode:
                self.reward_distribution_means += np.random.normal(loc=0.0, scale=0.001 ** 2, size=10)
            else:
                self.reward_distribution_means = 0.5 * self.reward_distribution_means + np.random.normal(loc=0.0,
                                                                                                         scale=0.01 ** 2,
                                                                                                         size=10)
        else:
            if np.random.random() < 0.005:
                self.reward_distribution_means = np.random.permutation(self.reward_distribution_means)

    def get_reward(self, action):
        """Gets the agent's rewards and adjusts the reward distribution means."""
        reward = self._get_reward(action)
        self._adjust()
        return reward


class ASEGreedy:
    """Epsilon greedy agent class."""
    def __init__(self, epsilon, environment, decreasing_step_size=True, update_method='incremental', alpha=0.1,
                 iterations=20000, optimistic=False, decay_rate=0.01):
        """If epsilon=0 then this agent behaves greedily"""
        # loading arguments
        self.epsilon = epsilon
        self.environment = environment
        self.update_method = update_method
        self.decreasing_step_size = decreasing_step_size
        if optimistic:
            self.action_value_estimates = np.ones(10) * 10
        else:
            self.action_value_estimates = np.zeros(10)
        self.alpha = alpha
        self.iterations = iterations
        self.decay_rate = decay_rate

        # initializing class variables
        self.k = 10
        self.action_sequence = []
        self.reward_sequence = []
        self.action_counts = np.zeros(10)
        self.action_reward_record = [[0]] * self.k

    def select_action(self):
        """This method selects an action for the agent."""
        if np.random.rand() < self.epsilon:  # explore
            action_index = np.random.randint(self.k)
            self.action_counts[action_index] += 1
        else:  # exploit
            action_index = np.argmax(self.action_value_estimates)
            self.action_counts[action_index] += 1
        self.action_sequence.append(action_index)
        return action_index

    def learn_from_consequences(self, action):
        """This function determines the reward agent observes by taking an action,
         and then updates its action value estimates."""
        # observe the received reward
        received_reward = self.environment.get_reward(action)
        self.reward_sequence.append(received_reward)
        self.action_reward_record[action].append(received_reward)

        # update action value estimates
        if self.update_method == 'avg':
            self.action_value_estimates[action] = (self.action_value_estimates[action] * self.action_counts[
                action] + received_reward) / (self.action_counts[action] + 1)
        elif self.update_method == 'incremental':
            self.action_value_estimates[action] = self.action_value_estimates[action] + (
                        received_reward - self.action_value_estimates[action]) / self.action_counts[action]
        else:
            add = 0
            for i in range(len(self.action_reward_record[action])):
                add += self.alpha * (1 - self.alpha) ** (len(self.action_reward_record[action]) - i) * \
                       self.action_reward_record[action][i]
            self.action_value_estimates[action] = self.action_value_estimates[action] * (1 - self.alpha) ** \
                                                  self.action_counts[action] + add

        if self.decreasing_step_size:
            self.epsilon *= (1 - self.decay_rate)

    def learn(self):
        """Trains the agent for the given number of iterations."""
        for _ in range(self.iterations):
            action = self.select_action()
            self.learn_from_consequences(action)
        return self.reward_sequence


class MovingEvaluator:
    """This class trains an agent for 1000 different initialization."""
    def __init__(self, epsilon, env_change_type='gradual', env_gradual_mode=True, run_count=1000, **kwargs):
        self.epsilon = epsilon
        self.change_type = env_change_type
        self.gradual_mode = env_gradual_mode
        self.run_count = run_count
        self.kwargs = kwargs

        self.reward_sequences = []
        self.terminal_average_rewards = None

    def run(self, preset_env=None):
        """Trains an agent with the given algorithm 1000 times and aggregates the algorithm's performance"""
        # trainings
        if preset_env is None:
            environment = Environment(change_type=self.change_type, gradual_mode=self.gradual_mode)
        else:
            environment = preset_env
        for _ in range(self.run_count):
            env = copy.deepcopy(environment)
            agent = ASEGreedy(epsilon=self.epsilon, environment=env, **self.kwargs)
            reward_sequence = agent.learn()
            self.reward_sequences.append(reward_sequence)

        # aggregation
        self.terminal_average_rewards = np.array(self.reward_sequences).mean(axis=0)
        return self.terminal_average_rewards

    def visualize(self):
        """Plots the results of the agent."""
        # terminal rewards histogram
        fig = plt.figure()
        plt.grid()
        plt.hist(self.terminal_average_rewards, bins=50)
        plt.axvline(x=self.terminal_average_rewards.mean(), color='r', linestyle='dashed')
        plt.xlabel('Reward')
        plt.ylabel('Count')
        plt.title(f'Epsilon={self.epsilon} - {"decreasing step size" if self.kwargs["decreasing_step_size"] else "constant step size"} - Mean={round(self.terminal_average_rewards.mean(), 2)}')
        plt.show()

        # terminal rewards box plot
        fig = plt.figure()
        plt.grid()
        plt.title(f'Epsilon={self.epsilon} - {"decreasing step size" if self.kwargs["decreasing_step_size"] else "constant step size"} - Median={round(np.median(self.terminal_average_rewards), 2)}')
        plt.axhline(y=np.median(self.terminal_average_rewards), color='r', linestyle='dashed')
        plt.boxplot(self.terminal_average_rewards)
        plt.show()