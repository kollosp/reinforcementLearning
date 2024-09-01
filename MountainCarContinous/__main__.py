import gym, os
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
parser = argparse.ArgumentParser(description="Perform training and use MLP in reinforcement learning task: Mountain Car Continous")
parser.add_argument("-m", "--mode", help="Select application mode: training (0) or animation (1)", choices=["0", "1"])
parser.add_argument("-n", "--new", help="Add this flag if you want to reinitialize model", action='store_true')
parser.add_argument("-i", "--iterations", help="If mode is training(0) it specifies number of learning iterations "
                                               "alternatively number of animations to be displayed", default=10, type=int)
args = parser.parse_args()
ENVIRONMENT_TITLE = "MountainCarContinuous-v0"


class Agent:
    def __init__(self):
        self.optimizer_ = keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn_ = keras.losses.binary_crossentropy
        self.model_ = self.create_model()

    def create_model(self):
        """
        Function that should be overloaded in subclasses
        :return:
        """
        return keras.models.Sequential([
            keras.layers.Dense(4, activation="elu", input_shape=[2]),
            keras.layers.Dense(1, activation="sigmoid")
        ])

    @property
    def optimizer(self):
        return self.optimizer_
    @property
    def loss_fn(self):
        return self.loss_fn_
    @property
    def model(self):
        return self.model_

    def save_weights(self,path='best_model_weights.h5'):
        self.model.save_weights(path)

    def load_weights(self,path='best_model_weights.h5', not_raise=False):
        if os.path.exists(path):
            self.model.load_weights(path)
        else:
            if not not_raise:
                raise RuntimeError("Cannot load from file that doesn't exit")

    def __call__(self, observation):
        """
        Function that should be overloaded in subclasses
        :return:
        """
        # left_probability = 1 -> action 0 (move left)
        # left_probability = 0 -> action 1 (move right)
        left_probability = self(observation[np.newaxis])
        # randomized action
        action = tf.random.uniform([1, 1]) > left_probability
        y_target = 0 if tf.cast(action, tf.float32) == 1 else 1
        y_target = tf.constant([[y_target]])

        return int(action[0,0].numpy()), left_probability, y_target

    def play_one_step(self, observation):
        """
        Function computes next action and gradients
        :param observations:
        :return:
        """
        with tf.GradientTape() as tape:
            # let's assume that performed action was correct.
            action, left_probability, y_target = self(observation)
            loss = tf.reduce_mean(self.loss_fn(y_target, left_probability))
        grads = tape.gradient(loss, self.model.trainable_variables)
        return action, grads

    def clone(self):
        copy = keras.models.clone_model(self.model)
        copy.set_weights(self.model.get_weights())
        return copy

    def apply_gradients(self, all_grads, all_discounted_rewards):
        all_mean_grads = []
        for var_index in range(len(self.model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [discounted_reward * all_grads[episode_index][step][var_index]
                 for episode_index, discounted_rewards in enumerate(all_discounted_rewards)
                 for step, discounted_reward in enumerate(discounted_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        self.model.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))

class Animation:
    def __init__(self, environment_name, model, render_mode="rgb_array"):
        self.environment_name = environment_name
        self.agent = agent
        self.render_mode = "rgb_array"
        self.env = gym.make(self.environment_name, render_mode=self.render_mode)

    def render(self):
        _ = self.env.render()
        if self.render_mode == "human":
            time.sleep(0.05)


    @staticmethod
    def discount_rewards(rewards, discount_factor):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
        return discounted

    @staticmethod
    def discount_and_normalize(all_rewards, discount_factor):
        all_discounted = [Animation.discount_rewards(rewards, discount_factor)
                          for rewards in all_rewards]
        flat = np.concatenate(all_discounted)
        reward_mean = flat.mean()
        reward_std = flat.std()

        return [(discount - reward_mean) / reward_std
                for discount in all_discounted]

    def play_multiple_episodes(self, episodes, max_steps):
        all_rewards = []
        all_grads = []

        for episode in range(episodes):
            episode_rewards = []
            episode_grads = []

            observation = self.env.reset()
            observation = observation[0]


            for step in range(max_steps):
                action, grads = self.agent.play_one_step(observation)
                observation, reward, terminated, truncated = self.env.step(action)
                print("i", observation)
                episode_rewards.append(reward)
                episode_grads.append(grads)
                if truncated or terminated:
                    break

            all_rewards.append(episode_rewards)
            all_grads.append(episode_grads)

        return all_rewards, all_grads

    def fit_tf_model(self, iterations, episodes_per_update, max_steps, discount_factor=0.95):

        avg_rewards = []
        best_model = None
        current_max_reword = 0
        current_max_reword_index = 0
        for i in range(iterations):
            all_rewards, all_grads = self.play_multiple_episodes(episodes_per_update, max_steps)
            avg_rewards.append(np.mean([sum(rewards) for rewards in all_rewards]))

            all_discounted_rewards = Animation.discount_and_normalize(all_rewards, discount_factor)
            self.agent.apply_gradients(all_grads, all_discounted_rewards)

            print(f"Iteration {i}: avg_rewards:", avg_rewards[-1])

            if current_max_reword < avg_rewards[-1]:
                current_max_reword = avg_rewards[-1]
                best_model = self.agent.clone()

                print(f"Iteration {i}: best model changed. Last model at iteration {current_max_reword_index}",
                      avg_rewards[-1])
                current_max_reword_index = i
        return avg_rewards, best_model

if __name__ == "__main__":
    agent = Agent()
    animation = Animation(environment_name=ENVIRONMENT_TITLE, model=agent)

    if args.mode is None or args.mode == 0:
        print("Going into training mode.")
        if not args.new:
            print("Loading last model.")
            agent.load_weights(not_raise = True)

        all_rewards, best_model = animation.fit_tf_model(iterations=args.iterations,
                                                         episodes_per_update=10,
                                                         max_steps=500,
                                                         discount_factor=0.95)
        agent.save_weights()
        plt.plot(list(range(len(all_rewards))), all_rewards)
        plt.show()
        pass
    else:
        pass