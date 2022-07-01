import gym
import pandas as pd
import numpy as np

env = gym.make("FrozenLake-v1")

SEED = 27
EPISODES, e = 500000, 0
eps, eps_decay, eps_min = 1.0, 0.99995, 0.01
alpha, gamma = 0.01, 0.99
score_history = [0]*EPISODES

n_actions, n_obs_space = env.action_space.n, env.observation_space.n

Q = np.random.rand(n_obs_space, n_actions)

while e < EPISODES:
    state = env.reset()

    done = False
    score = 0

    while not done:

        if np.random.random() > eps:
            action = np.argmax(Q[state,:])
        else:
            action = np.random.randint(0, n_actions-1)

        new_state, r, done, info = env.step(action)

        Q_old = Q[state, action]
        Q_max = np.max(Q[new_state, :])

        new_q = (1-alpha) * Q_old + alpha * (r + (1-int(done)) * gamma * Q_max)
        Q[state, action] = new_q

        state = new_state
        score += r

    score_history[e] = score
    if e % 1000 == 0 and e > 0:
        avg = np.mean(score_history[e-100:e])
        print(f"E: {e} -> reward {avg}, eps {eps:.3f}")

    e += 1
    eps = max(eps_min, eps*eps_decay)