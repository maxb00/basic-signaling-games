import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio

# https://tomekkorbak.com/2019/10/08/lewis-signaling-games/
class World:
    def __init__(self, n_states: int, seed: int) -> None:
        self.n_states = n_states
        self.state = 0
        self.random = np.random.RandomState(seed)

    def get_state(self) -> int:
        self.state = self.random.randint(self.n_states)
        return self.state

    def evaluate(self, action: int) -> int:
        return 1 if action == self.state else 0
    

class Sender:
    def __init__(self, n_stimuli: int, n_signals: int, q_not: float = 1e-6) -> None:
        # n_stimuli: number of possible states in the world, 
        #            each corresponding to a stimulus
        # n_signals: number of signals that can be sent in response,
        #            usually equal to the number of states in the world
        # q_not:     initial probabilities of sending each signal before a reward
        self.n_signals = n_signals
        self.signal_weights = np.zeros((n_stimuli, n_signals))
        self.signal_weights.fill(q_not)
        self.last_situation = (0, 0)

    def get_signal(self, stimulus: int) -> int:
        # exponential calculation
        # num = np.exp(self.signal_weights[stimulus, :])
        # den = np.sum(np.exp(self.signal_weights[stimulus, :]))
        # simple sum
        num = self.signal_weights[stimulus, :]
        den = np.sum(self.signal_weights[stimulus, :])
        probabilities = num / den
        signal = np.random.choice(self.n_signals, p=probabilities)
        self.last_situation = (stimulus, signal)
        return signal
    
    def update(self, reward: int) -> None:
        stimulus, signal = self.last_situation
        self.signal_weights[stimulus, signal] += reward


class Receiver:
    def __init__(self, n_signals, n_actions, q_not: float = 1e-6) -> None:
        # n_signals: number of signals that can be sent in response,
        #            usually equal to the number of states in the world
        # n_actions: number of actions that can be taken in response,
        #            usually equal to the number of states in the world
        # q_not:     initial probabilities of taking each action before a reward
        self.n_actions = n_actions
        self.action_weights = np.zeros((n_signals, n_actions))
        self.action_weights.fill(q_not)
        self.last_situation = (0, 0)

    def get_action(self, signal: int) -> int:
        # exponential calculation
        # num = np.exp(self.action_weights[signal, :])
        # den = np.sum(np.exp(self.action_weights[signal, :]))
        # simple sum
        num = self.action_weights[signal, :]
        den = np.sum(self.action_weights[signal, :])
        probabilities = num / den
        action = np.random.choice(self.n_actions, p=probabilities)
        self.last_situation = (signal, action)
        return action
    
    def update(self, reward: int) -> None:
        signal, action = self.last_situation
        self.action_weights[signal, action] += reward




# testing
if __name__ == "__main__":
    # constants
    epochs = 1000
    seed = 0
    states = 3
    actions = 3
    signals = 3

    def make_gif(filename_base):
        images = []
        nm = filename_base.split('-')[-1]
        for filename in [f'{nm}_{i}.png' for i in range(epochs) if i % 25 == 0]:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{filename_base}.gif', images)


    # begin experiment
    S, R = Sender(states, signals), Receiver(signals, actions)
    W = World(states, seed)
    past_rewards = 0
    for epoch in range(epochs):
        stimulus = W.get_state()
        signal = S.get_signal(stimulus)
        action = R.get_action(signal)
        reward = W.evaluate(action)
        past_rewards += reward
        S.update(reward)
        R.update(reward)

        if epoch % (epochs // 5) == 0:
            plt.tight_layout(pad=0)
            plot = sns.heatmap(
                np.exp(R.action_weights) /
                np.exp(R.action_weights).sum(axis=0),
                square=True, cbar=False, annot=True, fmt='.1f'
            ).get_figure()
            plt.xlabel('messages')
            plt.ylabel('actions')
            plt.title(f'Receiver\'s weights, rollout {epoch}')
            plt.savefig(f"receiver_{epoch}.png")
            plt.clf()

            plot = sns.heatmap(
                np.exp(S.signal_weights) /
                np.exp(S.signal_weights).sum(axis=0),
                square=True, cbar=False, annot=True, fmt='.1f'
            ).get_figure()
            plt.xlabel('world states')
            plt.ylabel('messages')
            plt.title(f'Sender\'s weights, rollout {epoch}')
            plt.savefig(f"sender_{epoch}.png")
            plt.clf()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, last 100 epochs reward: {past_rewards/100}')
            print(stimulus, signal, action, reward)
            past_rewards = 0

    make_gif(f'{states}-{actions}-{signals}-sender')
    make_gif(f'{states}-{actions}-{signals}-receiver')
    print("Observation to message mapping:")
    print(S.signal_weights.argmax(1))
    print("Message to action mapping:")
    print(R.action_weights.argmax(1))
