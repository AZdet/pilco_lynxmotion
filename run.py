import numpy as np
import gym
from pilco import PILCO
import torch


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=True):
    X = []
    Y = []
    x = env.reset()
    for timestep in range(timesteps):
        if render:
            env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done:
                break
            if render:
                env.render()
        if verbose:
            print("Action: ", u)
            print("State : ", x_new)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
        if done:
            break
    return np.stack(X), np.stack(Y)


class myPendulum():
    def __init__(self):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(
            low=0, high=0.01*high)  # only difference
        self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

np.random.seed(3)
SUBS = 3
bf = 30
maxiter = 50
max_action = 2.0
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])
m_init = np.reshape([-1.0, 0, 0.0], (1, 3))
S_init = np.diag([0.01, 0.05, 0.01])
T = 40
T_sim = T
J = 0 # 4
N = 8
restarts = 2

if __name__ == "__main__":
    env = myPendulum()

    # initial random rollouts to generate a dataset for dynamics model
    X, Y = rollout(env, None, timesteps=T, random=True,
                   SUBS=SUBS, render=False, verbose=False)
    for i in range(1, J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True,
                         SUBS=SUBS, verbose=False, render=False)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    
    # build PILCO model 
    pilco = PILCO(X, Y, horizon=T, m_init=m_init, S_init=S_init)

    for rollouts in range(N):
        # main PILCO loop
        print("**** ITERATION no", rollouts, " ****")
        # learn dynamic model
        pilco.optimize_models(maxiter=maxiter)
        # improve policy
        pilco.optimize_policy(maxiter=maxiter)
        # collect more data
        X_new, Y_new = rollout(env, pilco, timesteps=T_sim,
                               verbose=False, render=False, SUBS=SUBS)

        # Update dataset
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        pilco.mgpr.set_XY(X, Y)
