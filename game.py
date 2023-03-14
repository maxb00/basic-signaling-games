import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
import pdb
import warnings

def reward_dist(i: int) -> float:
    return np.exp(- (i**2) / (4 / np.log(2)))

def early_stop(epochs, rewards, threshold=0.95):
    return np.sum(rewards[-epochs:]) / epochs > threshold

B = pdb.set_trace


np.seterr(all='warn')
warnings.filterwarnings('error')

# https://tomekkorbak.com/2019/10/08/lewis-signaling-games/
class World:
    def __init__(self, n_states: int, 
                 n_signals: int, n_actions: int, seed: int) -> None:
        self.setup = (n_signals, n_actions)
        self.n_states = n_states
        self.state = 0
        self.random = np.random.RandomState(seed)

    def get_state(self) -> int:
        self.state = self.random.randint(self.n_states)
        return self.state

    def evaluate(self, action: int) -> int:
        # action will be 1 or 2
        # action 1 (0) will score on states 0-9
        if self.state < 10:
            return 1 if action == 0 else -1
        # action 2 will score on states 10-19
        return 1 if action == 1 else -1

        
    

class Sender:
    def __init__(self, n_stimuli: int, n_signals: int, q_not: float = 1e-6) -> None:
        # n_stimuli: number of possible states in the world, 
        #            each corresponding to a stimulus
        # n_signals: number of signals that can be sent in response,
        #            usually equal to the number of states in the world
        # q_not:     initial probabilities of sending each signal before a reward
        self.n_signals = n_signals + 1      # +1 here represents null signal.
        self.signal_weights = np.zeros((self.n_signals, n_stimuli))
        self.signal_weights.fill(q_not)
        self.last_situation = (0, 0)


    def get_signal(self, stimulus: int) -> int:
        # exponential calculation
        try:
            num = np.exp(self.signal_weights[:, stimulus])
            den = np.sum(np.exp(self.signal_weights[:, stimulus]))
            probabilities = num / den
            signal = np.random.choice(self.n_signals, p=probabilities)
            if signal == self.n_signals-1:
                # null action
                return -1
        except Warning:
            B()
    
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
                self.signal_weights[signal, stimulus + i] = min(q_last + r, 308)

            # reward left
            if stimulus - i >= 0:
                q_last = self.signal_weights[signal, stimulus - i] 
                self.signal_weights[signal, stimulus - i] = min(q_last + r, 308)

        # for i in range(self.signal_weights.shape[0]):
        #     r = reward * reward_dist(abs(i - signal))
        #     self.signal_weights[i, signal] += r




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
        try:
            num = np.exp(self.action_weights[signal, :])
            den = np.sum(np.exp(self.action_weights[signal, :]))
            probabilities = num / den
            action = np.random.choice(self.n_actions, p=probabilities)
            self.last_situation = (signal, action)
        except Warning:
            B()

        return action
    
    def update(self, reward: int) -> None:
        signal, action = self.last_situation
        q_last = self.action_weights[signal, action]
        self.action_weights[signal, action] = min(q_last + reward, 308)




# testing
if __name__ == "__main__":
    # constants
    epochs = 10_000
    seed = 0
    states = 20
    actions = 2
    signals = 2

    def make_gif(filename_base):
        images = []
        nm = filename_base.split('-')[-1]
        for filename in [f'{nm}_{i}.png' for i in range(epochs) if i % 25 == 0]:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{filename_base}.gif', images)


    # begin experiment
    S, R = Sender(states, signals, 1), Receiver(signals, actions, 1)
    W = World(states, signals, actions, seed)
    past_rewards = 0
    for epoch in range(epochs):
        stimulus = W.get_state()
        signal = S.get_signal(stimulus)
        # B()
        if signal != -1:
            action = R.get_action(signal)
            reward = W.evaluate(action)
            past_rewards += reward
            S.update(reward)
            R.update(reward)
        # else null action
        

        if epoch % 25 == 0:
            try:
                plt.tight_layout(pad=0)
                plot = sns.heatmap(
                    np.exp(R.action_weights) /
                    np.exp(R.action_weights).sum(axis=0),
                    square=True, cbar=False, annot=True, fmt='.1f'
                ).get_figure()
                plt.xlabel('actions')
                plt.ylabel('messages')
                plt.title(f'Receiver\'s weights, rollout {epoch}')
                plt.savefig(f"receiver_{epoch}.png")
                plt.clf()
            except Warning:
                print('A')
                B()

            try:
                plot = sns.heatmap(
                    np.exp(S.signal_weights) /
                    np.exp(S.signal_weights).sum(axis=0),
                    square=True, cbar=False, annot=True, fmt='.1f'
                ).get_figure()
                plt.ylabel('messages')
                plt.xlabel('world states')
                plt.title(f'Sender\'s weights, rollout {epoch}')
                plt.savefig(f"sender_{epoch}.png")
                plt.clf()
            except Warning:
                print('B')
                B()
            

            

        if epoch % 100 == 0:
            # B()
            print(f'Epoch {epoch}, last 100 epochs reward: {past_rewards/100}')
            # print(stimulus, signal, action, reward)
            past_rewards = 0

    else: # yes, this is a for/else
        # this block runs if the loop wasn't broken
        # for checking end of game status
        B()

    make_gif(f'{states}-{actions}-{signals}-sender')
    make_gif(f'{states}-{actions}-{signals}-receiver')
    print("Observation to message mapping:")
    print(S.signal_weights.argmax(1))
    print("Message to action mapping:")
    print(R.action_weights.argmax(1))
