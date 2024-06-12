import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    """Evaluator for the part 1 of the projects"""
    def __init__(self, agent, run_count=1000, gradient=False, **kwargs):
        # loading arguments into parameters
        self.agent = agent
        self.run_count = run_count
        self.gradient = gradient
        self.kwargs = kwargs

        # initializing class variables
        self.k = 10
        self.optimal_percentage_sequences = []
        self.average_reward_sequences = []

        # visualization variables
        self.rewards = None
        self.percents = None

    def run(self):
        """Initializes and trains an agent with the given algorithm n times and aggregates the algorithm's performance"""
        # trainings
        if self.gradient:
            for _ in range(self.run_count):
                reward_distribution_means = np.random.normal(size=(self.k,))
                agent = self.agent(**self.kwargs, reward_distribution_means=reward_distribution_means)
                optimal_percentage_sequence = agent.learn()
                self.optimal_percentage_sequences.append(optimal_percentage_sequence)

            # aggregation
            self.percents = np.array(self.optimal_percentage_sequences).mean(axis=0) * 100
        else:
            for _ in range(self.run_count):
                reward_distribution_means = np.random.normal(size=(self.k,))
                agent = self.agent(**self.kwargs, reward_distribution_means=reward_distribution_means)
                avg_reward_acquired_sequence, optimal_percentage_sequence = agent.learn()
                self.average_reward_sequences.append(avg_reward_acquired_sequence)
                self.optimal_percentage_sequences.append(optimal_percentage_sequence)

            # aggregation
            self.rewards = np.array(self.average_reward_sequences).mean(axis=0)
            self.percents = np.array(self.optimal_percentage_sequences).mean(axis=0) * 100
        return self.percents, self.rewards

    def visualize(self):
        if not self.gradient:
            fig = plt.figure()
            plt.plot(self.rewards)
            plt.xlabel("Time step")
            plt.ylabel("Average rewards")
            plt.title(f"epsilon={self.kwargs['epsilon']}")
            plt.grid()
            plt.show()

        fig = plt.figure()
        plt.plot(self.percents)
        plt.xlabel("Time step")
        plt.ylabel(f"Optimal action percentage - max={round(self.percents.max(), 1)}%")
        if not self.gradient:
            plt.title(f"epsilon={self.kwargs['epsilon']}")
        else:
            plt.title(f"alpha={self.kwargs['alpha']}")
        plt.ylim(0, 100)
        plt.grid()
        plt.show()


class EpsilonGreedyAgent:
    """EpsilonGreedyAgent for the part 1 of the projects"""
    def __init__(self, epsilon, reward_distribution_means, update_method='avg', action_values=np.zeros(10), alpha=0.1,
                 iterations=1000):
        """If epsilon=0 then this agent behaves greedily"""
        # loading arguments
        self.epsilon = epsilon
        self.reward_distribution_means = reward_distribution_means
        self.update_method = update_method
        self.action_value_estimates = action_values
        self.alpha = alpha
        self.iterations = iterations

        # initializing class variables
        self.k = 10
        self.action_sequence = []
        self.optimal_percentage = []
        self.avg_reward_acquired = []
        self.action_counts = np.zeros(10)
        self.action_reward_record = [[0.0]] * self.k
        self.best_action = np.argmax(self.reward_distribution_means)

    def select_action(self):
        """There are no states, so action rewards do not depend on the current state."""
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
         and then updates the action value estimates of the agent."""
        # calculate the received reward
        received_reward = np.random.normal(loc=self.reward_distribution_means[action], scale=1.0)
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

    def evaluate(self):
        # find the average acquired reward of the agent for the current time step
        best_action_index = np.argmax(self.action_value_estimates)
        average_acquired_reward = (1 - self.epsilon) * self.action_value_estimates[best_action_index] + \
                                  np.sum(
                                      np.delete(self.action_value_estimates, best_action_index)) * self.epsilon / self.k
        self.avg_reward_acquired.append(average_acquired_reward)

        # find the percentage of optimal actions
        self.optimal_percentage.append(
            np.sum(np.array(self.action_sequence) == self.best_action) / len(self.action_sequence))

    def learn(self):
        """Trains the agent for the given number of iterations."""
        for _ in range(self.iterations):
            action = self.select_action()
            self.learn_from_consequences(action)
            self.evaluate()
        return self.avg_reward_acquired, self.optimal_percentage


class GradientAgent:
    def __init__(self, alpha, run_count=1000, time_steps=5000):
        self.alpha = alpha
        self.run_count = run_count
        self.max_time_steps = time_steps

        self.reward_means = None
        self.preferences = np.zeros(10)
        self.action_probabilities = None
        self.action_sequence = []
        self.time_step = 0
        self.average_reward = 0
        self.percents = []
        self.percent_sequences = []

        self.results = None

    def reset(self):
        self.reward_means = np.random.normal(size=(10,))
        self.preferences = np.zeros(10)
        self.action_probabilities = None
        self.action_sequence = []
        self.time_step = 0
        self.average_reward = 0
        self.percents = []

    def select_action(self):
        self.action_probabilities = np.exp(self.preferences)/(np.exp(self.preferences).sum())
        action = np.random.choice(10, p=self.action_probabilities)
        self.action_sequence.append(action)
        self.time_step += 1
        return action

    def learn_from_consequences(self, action):
        received_reward = np.random.normal(loc=self.reward_means[action], scale=1.0)
        onehot = np.zeros(10)
        onehot[action] = 1
        self.preferences += self.alpha * (received_reward - self.average_reward)*(onehot - self.action_probabilities)
        self.average_reward += (received_reward - self.average_reward)/self.time_step

    def evaluate(self):
        self.percents.append(np.sum(np.array(self.action_sequence) == np.argmax(self.reward_means))/self.time_step)

    def learn(self):
        for _ in range(self.run_count):
            self.reset()
            for _ in range(self.max_time_steps):
                action = self.select_action()
                self.learn_from_consequences(action)
                self.evaluate()
            self.percent_sequences.append(self.percents)
        self.results = np.array(self.percent_sequences).mean(axis=0)*100
        return self.results

    def plot(self):
        fig = plt.figure()
        plt.grid(zorder=0)
        plt.xlabel("Time steps")
        plt.ylabel(f"Percentage of optimal actions - max={self.results.max():.2f}")
        plt.ylim(0, 100)
        plt.title(f"alpha={self.alpha}")
        plt.plot(self.results)
        plt.show()


if __name__ == "__main__":
    asd = GradientAgent(alpha=0.1, run_count=1000, time_steps=50)
    x = asd.learn()
    asd.plot()
