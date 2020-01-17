import numpy as np
import torch
import pandas as pd
import time

from mgpr import MGPR


class LinearController(torch.nn.Module):
    def __init__(self, state_dim, control_dim):
        super(LinearController, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.random.rand(control_dim, state_dim), dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.tensor(np.random.randn(1, control_dim), dtype=torch.float32))

    def forward(self, x):
        '''
        Linear controller W * m = b

        Args
        mean (m) and variance (s) of the state

        Return 
        mean (M) and variance (S) of the action
        '''
        m, s = x
        M = m @ self.W.T + self.b  # mean output
        S = self.W @ s @ self.W.T  # output variance
        V = self.W.T  # input output covariance
        return M, S, V


class ExponentialReward:
    def __init__(self, state_dim, W=None, target=None):
        self.state_dim = state_dim
        if W is not None:
            self.W = np.reshape(W, (state_dim, state_dim))
        else:
            self.W = np.eye(state_dim)
        if target is not None:
            self.target = np.reshape(target, (1, state_dim))
        else:
            self.target = np.zeros((1, state_dim))
        self.W = torch.tensor(self.W, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
    
    def compute_reward(self, m, s):
        '''
        Calculating expectation of rewards, given mean and variance of state distribution, along with the target State and a weight matrix.
        Args 
        m : [1, k], mean of x
        s : [k, k], cov of x

        Returns
        M : [1, 1], mean reward weighted by W
        '''
        # see (3.43), note reward is -cost
        SW = s @ self.W
        S1 = torch.solve(self.W.T, (torch.eye(self.state_dim) + SW).T)[0].T
        reward = torch.exp(-1/2*(m - self.target) @ S1 @ (m-self.target).T) / (torch.det(torch.eye(self.state_dim) + SW))
        if reward != reward:
            raise RuntimeError('get numeric issuses, NAN in reward!')
        return reward.view(1, 1)


class PILCO:
    def __init__(self, X, Y, horizon=30,
                 m_init=None, S_init=None):
        self.mgpr = MGPR(X, Y)

        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]
        self.horizon = horizon

        self.controller = LinearController(self.state_dim, self.control_dim)
        self.reward = ExponentialReward(self.state_dim)

        if m_init is None or S_init is None:
            # default initial state for the rollouts is the first state in the dataset.
            self.m_init = X[0:1, 0:self.state_dim]
            self.S_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.S_init = S_init
        self.m_init = torch.tensor(self.m_init, dtype=torch.float32)
        self.S_init = torch.tensor(self.S_init, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.controller.parameters())

    def optimize_models(self, maxiter=200):
        '''
        Optimize GP models
        '''
        self.mgpr.optimize(max_iter=maxiter)
        self.mgpr.eval()
        # print learned dynamics model parameters
        lengthscales = {}
        variances = {}
        noises = {}
        i = 0
        for model in self.mgpr.models:
            lengthscales['GP' + str(i)] = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
            variances['GP' + str(i)] = np.array([model.covar_module.outputscale.item()])
            noises['GP' + str(i)] = np.array([model.likelihood.noise.item()])
            i += 1
        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))

    def optimize_policy(self, maxiter=50):
        '''
        Optimize controller's parameter's
        '''
        self.mgpr.eval()
        start = time.time()
        for i in range(maxiter):
            self.optimizer.zero_grad()
            reward = self.compute_reward() # policy evaluation
            loss = -reward
            loss.backward() # policy improvement by policy gradient 
            self.optimizer.step()
        end = time.time()
        print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (
            end - start, self.compute_reward()))

    def compute_action(self, x_m):
        x_m = torch.tensor(x_m, dtype=torch.float32)
        x_s = torch.zeros((self.state_dim, self.state_dim), dtype=torch.float32)
        return self.controller((x_m, x_s))[0].detach().numpy()

    def predict(self, m_x, s_x, n):
        '''
        predict n steps with learned model
        '''
        reward = 0
        for _ in range(n):
            m_x, s_x = self.propagate(m_x, s_x)
            reward += self.reward.compute_reward(m_x, s_x)
        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        ''' 
        propagate from one state distribution to the next one with controller and GP models
        '''
        # from state x to control u 
        m_u, s_u, c_xu = self.controller((m_x, s_x))
        # joint distribution of x and u
        m = torch.cat([m_x, m_u], axis=1)
        s1 = torch.cat([s_x, s_x@c_xu], axis=1)
        s2 = torch.cat([(s_x@c_xu).T, s_u], axis=1)
        s = torch.cat([s1, s2], axis=0)
        # go to next state by moment matching
        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        
        S_x = S_dx + s_x + s1@C_dx + C_dx.T @ s1.T

        M_x.reshape(1, self.state_dim)
        S_x.reshape(self.state_dim, self.state_dim)
        return M_x, S_x

    def compute_reward(self):
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return reward
