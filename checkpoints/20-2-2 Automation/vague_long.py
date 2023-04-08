import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
import os
import time
from itertools import product
from tqdm import tqdm
import pdb

import matplotlib
matplotlib.use('Agg')

def reward_dist(n: int) -> float:
    # Gaussian with FWHM of 2.
    return np.exp(- (n**2) / (4 / np.log(2)))


class World:
    def __init__(self, n_states: int, 
                 n_signals: int, n_actions: int, 
                 reward_mod=(1,1), seed: int = 0) -> None:
        self.setup = (n_signals, n_actions)
        self.pos, self.neg = reward_mod
        self.positive, self.negative = reward_mod
        self.n_states = n_states
        self.state = 0
        self.random = np.random.RandomState(seed)

    def get_state(self) -> int:
        self.state = self.random.randint(self.n_states)
        return self.state

    def evaluate(self, action: int) -> int:
        step = self.n_states / self.setup[0]
        correct = self.state // step
        return self.pos if action == correct else -self.neg


class Sender:
    def __init__(self, n_stimuli: int, n_signals: int, q_not: float = 1e-6) -> None:
        # n_stimuli: number of possible states in the world,
        #            each corresponding to a stimulus
        # n_signals: number of signals that can be sent in response,
        #            usually equal to the number of states in the world
        # q_not:     initial signal propensity values. Final value of null signal.
        self.n_signals = n_signals + 1      # +1 here represents null signal.
        self.signal_weights = np.zeros((self.n_signals, n_stimuli))
        self.signal_weights.fill(q_not)
        self.last_situation = (0, 0)

    def get_signal(self, stimulus: int) -> int:
        # exponential calculation
        num = np.exp(self.signal_weights[:, stimulus])
        den = np.sum(np.exp(self.signal_weights[:, stimulus]))
        probabilities = num / den
        signal = np.random.choice(self.n_signals, p=probabilities)
        if signal == self.n_signals-1:
            # null action
            return -1
        self.last_situation = (stimulus, signal)
        return signal

    def update(self, reward: int) -> None:
        # I am capping weight values at 308 due to overflow errors.
        stimulus, signal = self.last_situation
        self.signal_weights[signal, stimulus] += reward

        # after updating the first weight, we must reinforce the surrouding weights
        # using a gaussian distribution with a height of 1 and a width of 2
        # so that stimulus+2 and stimulus-2 are updated with 1/2 the reward.
        for i in range(1, 4):
            r = reward * reward_dist(i)

            # reward right
            if stimulus + i < self.signal_weights.shape[1]:
                q_last = self.signal_weights[signal, stimulus + i]
                self.signal_weights[signal, stimulus +
                                    i] = min(q_last + r, 308)

            # reward left
            if stimulus - i >= 0:
                q_last = self.signal_weights[signal, stimulus - i]
                self.signal_weights[signal, stimulus -
                                    i] = min(q_last + r, 308)

                            
class Receiver:
    def __init__(self, n_signals, n_actions, q_not: float = 1e-6) -> None:
        # n_signals: number of signals that can be sent in response,
        #            usually equal to the number of states in the world
        # n_actions: number of actions that can be taken in response,
        #            usually equal to the number of states in the world
        # q_not:     initial action propensity value
        self.n_actions = n_actions
        self.action_weights = np.zeros((n_signals, n_actions))
        self.action_weights.fill(q_not)
        self.last_situation = (0, 0)

    def get_action(self, signal: int) -> int:
        # exponential calculation
        num = np.exp(self.action_weights[signal, :])
        den = np.sum(np.exp(self.action_weights[signal, :]))
        probabilities = num / den
        action = np.random.choice(self.n_actions, p=probabilities)
        self.last_situation = (signal, action)

        return action
    
    def update(self, reward: int) -> None:
        signal, action = self.last_situation
        q_last = self.action_weights[signal, action]
        self.action_weights[signal, action] = min(q_last + reward, 308)

class History:
    def __init__(self, epochs, states, signals, actions):
        self.send_hist = np.zeros((epochs // 25, signals+1, states))
        self.reci_hist = np.zeros((epochs // 25, signals, actions))
        self.epochs = epochs
        self.ep = 0
        # TODO: Genralize functions to work mid-run

    def add(self, send_weights, reci_weights):
        self.send_hist[self.ep] = send_weights
        self.reci_hist[self.ep] = reci_weights
        self.ep += 1

    def make_gif(self, fps, seed, filename_base): 
        for i in range(epochs // 25):
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            plt.tight_layout(pad=3)

            sns.heatmap(
                np.exp(self.send_hist[i]) /
                np.exp(self.send_hist[i]).sum(axis=0),
                square=True, cbar=False, annot=True, fmt='.1f', ax=axs[0])
            axs[0].set_ylabel('messages')
            axs[0].set_xlabel('world states')
            axs[0].set_title(f'Sender\'s weights')

            sns.heatmap(
                np.exp(self.reci_hist[i]) /
                np.exp(self.reci_hist[i]).sum(axis=0),
                square=True, cbar=False, annot=True, fmt='.2f', ax=axs[1])
            axs[1].set_xlabel('actions')
            axs[1].set_ylabel('messages')
            axs[1].set_title(f'Receiver\'s weights')
            
            
            fig.suptitle(f'Rollout {i*25}')
            plt.savefig(f"images/game_{i*25}.png")
            plt.close(fig)

        images = []
        for filename in [f'images/game_{j*25}.png' for j in range(epochs // 25)]:
            images.append(imageio.imread(filename))
        if not os.path.exists(f'gifs/{seed}'):
            os.mkdir(f'gifs/{seed}')
        imageio.mimsave(f'gifs/{seed}/{filename_base}.gif', images, fps=fps)
        # no return

    def make_graph(self, seed):
        fig, axs = plt.subplots(2, 3, figsize=(10, 8), sharey=True)

        ran = range(0, self.epochs, 25)
        axs[0, 0].plot(ran, self.send_hist[:, 0, 0], label='action 0')
        axs[0, 0].plot(ran, self.send_hist[:, 1, 0], label='action 1')
        axs[0, 0].plot(ran, self.send_hist[:, 2, 0], label='null action')
        axs[0, 0].set_title('state 0')

        axs[0, 1].plot(ran, self.send_hist[:, 0, 5], label='action 0')
        axs[0, 1].plot(ran, self.send_hist[:, 1, 5], label='action 1')
        axs[0, 1].plot(ran, self.send_hist[:, 2, 5], label='null action')
        axs[0, 1].set_title('state 5')

        axs[0, 2].plot(ran, self.send_hist[:, 0, 9], label='action 0')
        axs[0, 2].plot(ran, self.send_hist[:, 1, 9], label='action 1')
        axs[0, 2].plot(ran, self.send_hist[:, 2, 9], label='null action')
        axs[0, 2].set_title('state 9')

        axs[1, 0].plot(ran, self.send_hist[:, 0, 10], label='action 0')
        axs[1, 0].plot(ran, self.send_hist[:, 1, 10], label='action 1')
        axs[1, 0].plot(ran, self.send_hist[:, 2, 10], label='null action')
        axs[1, 0].set_title('state 10')

        axs[1, 1].plot(ran, self.send_hist[:, 0, 14], label='action 0')
        axs[1, 1].plot(ran, self.send_hist[:, 1, 14], label='action 1')
        axs[1, 1].plot(ran, self.send_hist[:, 2, 14], label='null action')
        axs[1, 1].set_title('state 14')

        axs[1, 2].plot(ran, self.send_hist[:, 0, 19], label='action 0')
        axs[1, 2].plot(ran, self.send_hist[:, 1, 19], label='action 1')
        axs[1, 2].plot(ran, self.send_hist[:, 2, 19], label='null action')
        axs[1, 2].set_title('state 19')

        fig.suptitle(f'Sum action propensities over {self.epochs} epochs')
        fig.savefig(f'gifs/{seed}/weights.png')
        plt.close(fig)


multipliers = range(1, 6) # 1, 2, 3, 4, 5
negatives = [m*(10**x) for m in multipliers for x in range(-3, 1)]
weights = [m*(10**x) for m in multipliers for x in range(-6, 2)]

# constants
positive = 0.01          # reward for correct action
epochs = 20_000          # Number of epochs to train for
world_states = 20        # number of world states. evenly split among signals
signals = 2              # number of signals sender can send (not including null)
actions = 2              # number of actions reciever can respond with
gif_fps = 10             # frames per second for gif

# world states should be evenly divisible by action and signals
assert world_states % signals == world_states % actions == 0

for i, (negative, weight) in tqdm(enumerate(product(negatives, weights)), total=len(negatives)*len(weights)):
    # print(f"{negative}, {weight}")
    if i < 99:
        continue
    rew = (positive, negative)
    seed = int(time.time())
    S = Sender(world_states, signals, weight)
    R = Receiver(signals, actions, weight)
    W = World(world_states, signals, actions, rew, seed)
    H = History(epochs, world_states, signals, actions)
    past_rewards = slow = 0
    for epoch in range(epochs):
        stimulus = W.get_state()
        signal = S.get_signal(stimulus)
        if signal != -1:
            action = R.get_action(signal)
            reward = W.evaluate(action)
            past_rewards += reward
            S.update(reward)
            R.update(reward)
        # else null action

        if epoch % 25 == 0:
            # save history
            H.add(S.signal_weights, R.action_weights)
            

        if epoch % 100 == 0:
            # print(f'Epoch {epoch}, last 100 epochs reward: {past_rewards/100:e}')
            slow = past_rewards / 100
            past_rewards = 0

    # pdb.set_trace()
    # now we decide if this is a run to flag
    if ((np.argmax(S.signal_weights[:, 9]) == 2 or 
        np.argmax(S.signal_weights[:, 10]) == 2 or
        slow < 0) and (np.argmax(S.signal_weights[:, 5]) != 2 or 
        np.argmax(S.signal_weights[:, 14]) != 2)):
        H.make_gif(gif_fps, seed, f'{world_states}-{actions}-{signals}-game')
        H.make_graph(seed)
        with open(f'gifs/{seed}/params.txt', 'w') as f:
            f.write(f'negative: {negative},\ninitial weight: {weight}\nfinal average reward: {slow}')
        print(f"Saved with negative: {negative} and initial weight: {weight}")
