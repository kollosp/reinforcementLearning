import gym, os
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
parser = argparse.ArgumentParser(description="Perform training and use MLP in reinforcement learning task: CarPole")
parser.add_argument("-m", "--mode", help="Select application mode: training (0) or animation (1)", choices=["0", "1"])
parser.add_argument("-n", "--new", help="Add this flag if you want to reinitialize model", action='store_true')
parser.add_argument("-i", "--iterations", help="If mode is training(0) it specifies number of learning iterations "
                                               "alternatively number of animations to be displayed", default=10, type=int)
args = parser.parse_args()

print(args)

n_inputs = 4

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[4]),
    keras.layers.Dense(1, activation="sigmoid")
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2,-1,-1):
        discounted[step] += discounted[step+1] * discount_factor
    return discounted

def discount_and_normalize(all_rewards, discount_factor):
    all_discounted = [discount_rewards(rewards, discount_factor)
                      for rewards in all_rewards]
    flat = np.concatenate(all_discounted)
    reward_mean = flat.mean()
    reward_std = flat.std()

    return [(discount - reward_mean) / reward_std
            for discount in all_discounted]


def play_one_step(env, observation, model, loss_fn):
    with tf.GradientTape() as tape:
        # left_probability = 1 -> action 0 (move left)
        # left_probability = 0 -> action 1 (move right)
        left_probability = model(observation[np.newaxis])
        # randomized action
        action = tf.random.uniform([1,1]) > left_probability

        # let's assume that performed action was correct.
        y_target = 0 if tf.cast(action, tf.float32) == 1 else 1
        y_target = tf.constant([[y_target]])
        loss = tf.reduce_mean(loss_fn(y_target, left_probability))
    grads = tape.gradient(loss, model.trainable_variables)
    observation, reward, terminated, truncated, info = env.step(int(action[0,0].numpy()))
    return observation, reward, terminated, truncated, grads

def play_multiple_episodes(env, episodes, max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []

    for episode in range(episodes):
        episode_rewards = []
        episode_grads = []

        observation = env.reset()
        observation = observation[0]

        for step in range(max_steps):
            observation, reward, terminated, truncated, grads = play_one_step(env, observation, model, loss_fn)
            episode_rewards.append(reward)
            episode_grads.append(grads)
            if truncated or terminated:
                break

        all_rewards.append(episode_rewards)
        all_grads.append(episode_grads)

    return all_rewards, all_grads


def fit_tf_model(iterations, episodes_per_update, max_steps, discount_factor=0.95):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    avg_rewards = []
    best_model = None
    current_max_reword = 0
    current_max_reword_index = 0
    for i in range(iterations):
        all_rewards, all_grads = play_multiple_episodes(env,episodes_per_update, max_steps, model, loss_fn)
        avg_rewards.append(np.mean([sum(rewards) for rewards in all_rewards]))

        all_discounted_rewards = discount_and_normalize(all_rewards, discount_factor)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [discounted_reward * all_grads[episode_index][step][var_index]
                for episode_index, discounted_rewards in enumerate(all_discounted_rewards)
                for step, discounted_reward in enumerate(discounted_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
        print(f"Iteration {i}: avg_rewards:", avg_rewards[-1])

        if current_max_reword < avg_rewards[-1]:
            current_max_reword = avg_rewards[-1]
            best_model = keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())
            print(f"Iteration {i}: best model changed. Last model at iteration {current_max_reword_index}", avg_rewards[-1])
            current_max_reword_index = i
    return avg_rewards, best_model

def basic_policy(observation):
    # 0 - left
    # 1 - right
    pole_angle = observation[2]
    return 1 if pole_angle > 0 else 0

def run_episodes(episodes, render_mode="rgb_array"):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    all_rewards = []

    for episode in range(episodes):
        total_reword = 0
        observation = env.reset()
        observation = observation[0]
        for f in range(200):
            action = basic_policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            print(observation)
            total_reword += reward
            _ = env.render()

            if render_mode == "human":
                time.sleep(0.05)

            if terminated or truncated:
                all_rewards.append(total_reword)
                print(f"Episode {episode} Done. Points: {total_reword}")

                break
    return all_rewards, best_model



if __name__ == "__main__":
    if args.mode is None or args.mode == 0:
        print("Going into training mode.")
        # all_rewards = run_episodes(episodes=200)
        if os.path.exists("best_model_weights.h5") and not args.new:
            print("Loading last model.")
            model.load_weights('best_model_weights.h5')
        all_rewards, best_model = fit_tf_model(iterations=args.iterations, episodes_per_update=10, max_steps=500, discount_factor=0.95)
        best_model.save_weights('best_model_weights.h5')
        plt.plot(list(range(len(all_rewards))), all_rewards)
        plt.show()
    else:
        print("Going into animation mode.")
        if os.path.exists("best_model_weights.h5"):
            print("Loading last model.")
            model.load_weights('best_model_weights.h5')
        else:
            raise "You have to train model first!"

        env = gym.make("CartPole-v1", render_mode="human")
        observation = env.reset()
        observation = observation[0]

        for iterations in range(args.iterations):
            observation = env.reset()
            observation = observation[0]
            total_reword = 0
            for step in range(500):
                left_probability = model(observation[np.newaxis])
                # randomized action
                action = tf.random.uniform([1, 1]) > left_probability
                observation, reward, terminated, truncated, info = env.step(int(action[0, 0].numpy()))
                total_reword += reward
                _ = env.render()
                time.sleep(0.05)

                if terminated or truncated:
                    print(f"Episode {iterations} Done. Points: {total_reword}")
                    break