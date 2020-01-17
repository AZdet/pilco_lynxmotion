import gpytorch
import torch



class GPR(gpytorch.models.ExactGP):
    '''
    GP model for one column of y
    '''
    def __init__(self, X, Y, likelihood):
        super(GPR, self).__init__(X, Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=X.shape[1],
                lengthscale_prior=gpytorch.priors.GammaPrior(1, 10)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(1.5, 2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)


class MGPR:
    '''
    a collections of GPR, one for each column of Y
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_outputs = Y.shape[1]
        self.num_dims = X.shape[1]
        self.num_datapoints = X.shape[0]
        self.models = []
        # build a GPR for each dimension of Y
        for i in range(self.num_outputs):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.models.append(GPR(X, Y[:, i:i+1], likelihood))
        # prepare optimizer for each GPR
        self.optimizers = []
        for model in self.models:
            optimizer = torch.optim.Adam(model.parameters())
            self.optimizers.append(optimizer)

    def set_XY(self, X, Y):
        '''
        set updated dataset for models
        '''
        for i in range(len(self.models)):
            self.models[i].set_train_data(X, Y[:, i:i+1], strict=False)

    def eval(self):
        '''
        enter evaluate mode(pytorch concept)
        '''
        for model in self.models:
            likelihood = model.likelihood
            model.eval()
            likelihood.eval()

    def optimize(self, max_iter=1000):
        for model, optimizer in zip(self.models, self.optimizers):
            # enter train mode
            likelihood = model.likelihood
            model.train()
            likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            training_iter = max_iter
            train_x = model.train_inputs[0]
            train_y = model.train_targets
            for i in range(training_iter):
                optimizer.zero_grad()
                # predict from model
                output = model(train_x)
                # compute "loss": likelihood
                loss = -mll(output, train_y.view(-1))
                loss.backward()
                optimizer.step()

    @property
    def K(self):
        return torch.stack(
            [model.covar_module.base_kernel(
                self.X).evaluate() for model in self.models]
        )

    @property
    def noise(self):
        return torch.stack(
            [model.likelihood.noise for model in self.models]
        )

    @property
    def variance(self):
        return torch.stack(
            [model.covar_module.outputscale for model in self.models]
        ).view(-1, 1)

    @property
    def lengthscales(self):
        return torch.stack(
            [model.covar_module.base_kernel.lengthscale for model in self.models]
        ).view(self.num_outputs, -1)

    def predict_on_noisy_inputs(self, m, s):
        '''
        predict next state distribution with current [x, u] joint Gaussian distribution
        via moment matching

        ! a lot of math, see https://www.semanticscholar.org/paper/Efficient-reinforcement-learning-using-Gaussian-Deisenroth/af304fe978cfed58d576b5b1660710f1bfffb3f1 for reference
        '''
        self.eval()
        # prediction mode, no gradient is propagated
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # predict based on a noisy Gaussian input with mean=m, cov=s
            K = self.K
            eye = torch.eye(self.X.shape[0])
            L = torch.cholesky(K + self.noise[:, None] * eye)
            iK = torch.cholesky_solve(L, eye)
            beta = torch.cholesky_solve(self.Y.T[:, :, None], L)[:, :, 0]
            # predict given factorizations
            s = s[None, None, :, :].repeat(
                self.num_outputs, self.num_outputs, 1, 1)

            inp = (self.X - m)[None, :].repeat(self.num_outputs, 1, 1)

            # Calculate M and V: mean and inv(s) * input-output covariance
            iL = torch.diag_embed(1/self.lengthscales)
            iN = inp @ iL
            B = iL @ s[0, ...] @ iL + torch.eye(self.num_dims)

            # B is symmetric so it is the same
            t = torch.transpose(
                B.inverse() @ torch.transpose(iN, 1, 2), 1, 2
            )

            lb = torch.exp(-torch.sum(iN * t, -1)/2) * beta
            tiL = t @ iL
            c = self.variance / torch.sqrt(torch.det(B)).view(-1, 1)
            # (2.66), (2.68)
            M = (torch.sum(lb, -1).view(-1, 1) * c) 
            V = torch.matmul(torch.transpose(tiL, 1, 2),
                            lb[:, :, None])[..., 0] * c

            # Calculate S: Predictive Covariance
            R = s @ torch.diag_embed(
                1/self.lengthscales[None, :, :]**2 +
                1/self.lengthscales[:, None, :]**2
            ) + torch.eye(self.num_dims)

            X = inp[None, :, :, :]/(self.lengthscales[:, None, None, :])**2
            X2 = -inp[:, None, :, :]/(self.lengthscales[None, :, None, :])**2
            Q = R.inverse() @ s / 2
            Xs = torch.sum(X @ Q * X, -1)
            X2s = torch.sum(X2 @ Q * X2, -1)
            maha = -2 * torch.matmul(X @ Q, torch.transpose(X2, 2, 3)) + \
                Xs[:, :, :, None] + X2s[:, :, None, :]
            k = torch.log(self.variance) - \
                torch.sum(iN**2, -1)/2
            L = torch.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
            S = (beta[:, None, None, :].repeat(1, self.num_outputs, 1, 1) @ L @
                beta[None, :, :, None].repeat(self.num_outputs, 1, 1, 1))[:, :, 0, 0]

            diagL = torch.diagonal(L).T
            S = S - torch.diag(torch.sum(iK * diagL, [1, 2]))
            S = S / torch.sqrt(torch.det(R))
            S = S + torch.diag(self.variance)
            S = S - M @ M.t()

        return M.t(), S, V.t()


if __name__ == "__main__":
    # useage
    import math
    # torch.linspace(0, 1, 100).reshape(-1, 1)#
    train_x = torch.linspace(0, 1, 20).reshape(-1, 2)
    train_y = torch.cat((torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size())
                         * 0.2, torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2), dim=1)

    model = MGPR(train_x, train_y)
    # enter train mode
    model.optimize()
    model.predict_on_noisy_inputs(train_x.mean(), train_x.T @ train_x)
