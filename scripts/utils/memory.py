import numpy as np

class ExperienceBuffer():
#Store the data from the episodes

    def __init__(self):
        self.num_episodes = 0
        self.num_experiences = 0
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.safety_costs_buffer = []
        self.next_states_buffer = []
        self.discounted_future_rewards_buffer = []
        self.discounted_future_safety_costs_buffer = []
        self.advantages_buffer = []
        self.safety_advantages_buffer = []

        self.total_episode_reward = []
        self.total_episode_cost = []

        self.episode_length = 0
        self.episode_states_buffer = []
        self.episode_actions_buffer = []
        self.episode_rewards_buffer = []
        self.episode_safety_costs_buffer = []
        self.episode_next_states_buffer = []
        self.episode_discounted_future_rewards_buffer = []
        self.episode_discounted_future_safety_costs_buffer = []

    def get_experiences(self):
        return np.vstack(self.states_buffer), np.asarray(self.actions_buffer), np.asarray(self.rewards_buffer), np.asarray(self.safety_costs_buffer), np.vstack(self.next_states_buffer), np.asarray(self.discounted_future_rewards_buffer), np.asarray(self.discounted_future_safety_costs_buffer)

    def get_episode_experiences(self):
        return np.vstack(self.episode_states_buffer), np.asarray(self.episode_actions_buffer), np.asarray(self.episode_rewards_buffer), np.asarray(self.episode_safety_costs_buffer), np.vstack(self.episode_next_states_buffer), np.asarray(self.episode_discounted_future_rewards_buffer), np.asarray(self.episode_discounted_future_safety_costs_buffer)

    def get_advantages(self):
        return np.asarray(self.advantages_buffer), np.asarray(self.safety_advantages_buffer)

    def get_number_of_experiences(self):
        return self.num_experiences

    def get_number_of_episodes(self):
        return self.num_episodes

    def get_discounted_future_returns(self, returns, discount_factor):
        discounted_future_returns = [0]*len(returns)
        r = 0
        for t in reversed(range(len(returns))):
            r = returns[t] + discount_factor * r
            discounted_future_returns[t] = r
        return discounted_future_returns

    def add_experience(self, state, action, reward, safety_cost, next_state):
        self.episode_states_buffer.append(state)
        self.episode_actions_buffer.append(action)
        self.episode_rewards_buffer.append(reward)
        self.episode_safety_costs_buffer.append(safety_cost)
        self.episode_next_states_buffer.append(next_state)
        self.episode_length += 1

    def add_advantages(self, advantages, safety_advantages):
        self.advantages_buffer += list(advantages)
        self.safety_advantages_buffer += list(safety_advantages)

    def clear_episode_buffer(self):
        self.episode_length = 0
        self.episode_states_buffer = []
        self.episode_actions_buffer = []
        self.episode_rewards_buffer = []
        self.episode_safety_costs_buffer = []
        self.episode_next_states_buffer = []
        self.episode_discounted_future_rewards_buffer = []
        self.episode_discounted_future_safety_costs_buffer = []

    def add_episode(self, reward_discount_factor, safety_discount_factor):
        self.episode_discounted_future_rewards_buffer = self.get_discounted_future_returns(self.episode_rewards_buffer, reward_discount_factor)
        self.episode_discounted_future_safety_costs_buffer = self.get_discounted_future_returns(self.episode_safety_costs_buffer, safety_discount_factor)
        self.states_buffer += self.episode_states_buffer
        self.actions_buffer += self.episode_actions_buffer
        self.rewards_buffer += self.episode_rewards_buffer
        self.safety_costs_buffer += self.episode_safety_costs_buffer
        self.next_states_buffer += self.episode_next_states_buffer
        self.discounted_future_rewards_buffer += self.episode_discounted_future_rewards_buffer
        self.discounted_future_safety_costs_buffer += self.episode_discounted_future_safety_costs_buffer
        self.num_episodes += 1
        self.num_experiences += self.episode_length

    def clear_buffer(self):
        self.num_episodes = 0
        self.num_experiences = 0
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.safety_costs_buffer = []
        self.next_states_buffer = []
        self.discounted_future_rewards_buffer = []
        self.discounted_future_safety_costs_buffer = []
        self.advantages_buffer = []
        self.safety_advantages_buffer = []
