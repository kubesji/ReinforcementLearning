import gym
import random
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from collections import deque

EPISODES, e = 1000, 0
eps, eps_decay, eps_min = 1.0, 0.002, 0.05

class Agent:

    def __init__(self, n_actions, obs_shape, lr, batch_size=64, gamma=0.99):
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma

        def _create_model():
            model = Sequential()

            model.add(Input(self.obs_shape[0]))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.15))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.15))
            model.add(Dense(32, activation="relu"))
            model.add(Dropout(0.1))
            model.add(Dense(self.n_actions))

            model.compile(optimizer=Adam(learning_rate=self.lr), loss="mse")

            return model

        self.target_model = _create_model()
        self.train_model = _create_model()
        self._update_target()

        self.memory = deque(maxlen=100000)

        self.cnt = 0
        self._UPDATE_EVERY = 500

    def _update_target(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def add_experience(self, state, action, reward, done, new_state):
        self.memory.append((state, action, reward, done, new_state))

    def _get_experience(self):
        return random.sample(self.memory, self.batch_size)

    def get_action(self, state):
        state = state[np.newaxis, :]
        return np.argmax(self.train_model.predict(state, verbose=0, batch_size=1)[0])

    def learn(self):
        if len(self.memory) < 5*self.batch_size:
            return

        batch = self._get_experience()

        current_states = np.array([b[0] for b in batch])
        future_states = np.array([b[4] for b in batch])

        Q_current = self.train_model.predict(current_states, verbose=0, batch_size=self.batch_size)
        Q_future = self.target_model.predict(future_states, verbose=0, batch_size=self.batch_size)

        for i, (s, a, r, d, n_s) in enumerate(batch):
            new_q = r + (1-int(d)) * self.gamma * np.max(Q_future[i, :])
            Q_current[i, a] = new_q

        self.train_model.fit(current_states, Q_current, epochs=1, shuffle=0, batch_size=self.batch_size, verbose=0)

        self.cnt += 1
        if self.cnt >= self._UPDATE_EVERY:
            self.cnt = 0
            self._update_target()

env = gym.make("CartPole-v1")

n_a = env.action_space.n
obs_dim = env.observation_space.shape
history = []

agent = Agent(n_a, obs_dim, 0.001)

while e < EPISODES:
    s = env.reset()
    score = 0
    d = False

    while not d:
        if random.random() > eps:
            a = agent.get_action(s)
        else:
            a = random.randint(0, n_a - 1)

        n_s, r, d, _ = env.step(a)

        #if d and score < 500: reward = -10
        agent.learn()
        agent.add_experience(s, a, r, d, n_s)

        s = n_s
        score += r

    history.append(score)

    e += 1
    avg = np.mean(history[-25:])
    if e % 25 == 0:
        print(f"E: {e}, eps {eps:.2f} -> reward in last 25 episodes {avg:.2f}")

    # If last 25 episodes scored more than 450 on average, env is solved
    if avg > 475:
        print(f"Resolved in episode {e}")
        break

    # Get new epsilon - either decaued number or epsilon_min
    eps = max([eps - eps_decay, eps_min])

print("Done")