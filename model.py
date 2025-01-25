import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

# clear the param store in case we're in a REPL
pyro.clear_param_store()

def model(data):
    medal, theta, results = data
    T = len(results[0][0])
    # number of embedded parameters
    n_embedding = 4
    # intial value of Country Effect (CE)
    CE_prev = torch.tensor(1.0)
    # hyperparameters that control the Beta prior
    alp_CE_0, beta_CE_0 = torch.tensor(5.0), torch.tensor(5.0)
    # hyperparameters that control the Uniform distribution
    u_CE_l0, u_CE_u0 = torch.tensor(-1.0), torch.tensor(1.0)
    # hyperparameters that control the Dirichlet distribution
    concent_CE_0 = torch.full((4,), 1.0)
    # hyperparameters that control the Normal distribution
    mu_const_0 = torch.tensor(0.0)
    sig2_const_0 = torch.tensor(4.0)
    # hyper parameters that control the HalfCauchy distribution
    scale_CE_0 = torch.tensor(1.0)
    # hyperparameters that control the Beta prior
    alp_P_0, beta_P_0 = torch.tensor(5.0), torch.tensor(5.0)
    # hyperparameters that control the Dirichlet distribution
    concent_P_0 = torch.full((n_embedding,), 1.0)
    # hyper parameters that control the HalfCauchy distribution
    scale_P_0 = torch.tensor(1.0)
    # sample phi from the Beta prior
    phi = pyro.sample("phi", dist.Beta(alp_CE_0, beta_CE_0))
    # sample w
    w = pyro.sample("w", dist.Dirichlet(concent_CE_0))
    # sample alpha
    alpha = pyro.sample("alpha", dist.Normal(mu_const_0, sig2_const_0))
    # sample sig2_CE
    sig2_CE = pyro.sample("sig2_CE", dist.HalfCauchy(scale_CE_0))
    # sample rho
    rho = pyro.sample("rho", dist.Beta(alp_P_0, beta_P_0))
    # sample beta_embed
    beta_embed = pyro.sample("beta_embed", dist.Dirichlet(concent_P_0))
    # sample sig2_P
    sig2_P = pyro.sample("sig2_P", dist.HalfCauchy(scale_P_0))
    # loop over the observed time series
    for t in range(T):
        # loop over countries
        for country in range(len(results)):
            CE_t = pyro.sample(f"CE_{t}_{country}", dist.Normal(phi*CE_prev + (torch.mul(w, medal)), sig2_CE**0.5))
            # loop over sports for each country
            prob_win_dict = {"Gold":torch.tensor(), "Silver":[], "Bronze":[]}
            for sport in range(len(results[country])):
                P_t = pyro.sample(f"P_{t}_{country}_{sport}", dist.Normal(rho*CE_t + torch.mul(beta_embed, theta), sig2_P**0.5))

            for sport in range(len(results[country])):    
                pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=results[country][t])
            CE_prev = CE_t
        CE_t = pyro.sample(f"CE_{t}", dist.Normal(phi*CE_prev + gamma*(torch.mul(w, medal)), sig2_CE**0.5))
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
        CE_prev = CE_t

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    medal, theta, results = data
    n_embedding = 4
    alpha_CE_q = pyro.param("alpha_CE_q", torch.tensor(5.0),
                         constraint=constraints.positive)
    beta_CE_q = pyro.param("beta_CE_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    mu_const_q = pyro.param("mu_const_q", torch.tensor(0.0),
                         constraint=constraints.positive)
    sig2_const_q = pyro.param("sig2_const_q", torch.tensor(4.0),
                        constraint=constraints.positive)
    scale_CE_q = pyro.param("scale_CE_q", torch.tensor(1.0),
                        constraint=constraints.positive)
    alp_P_q = pyro.param("alp_P_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    beta_P_q = pyro.param("beta_P_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    concent_P_q = pyro.param("concent_P_q", torch.full((n_embedding,), 1.0),
                        constraint=constraints.positive)
    scale_P_q = pyro.param("scale_P_q", torch.tensor(1.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))
    # sample alpha
    alpha = pyro.sample("alpha", dist.Normal(mu_const_q, sig2_const_q))
    # sample sig2_CE
    sig2_CE = pyro.sample("sig2_CE", dist.HalfCauchy(scale_CE_q))
    # sample rho
    rho = pyro.sample("rho", dist.Beta(alp_P_q, beta_P_q))
    # sample beta_embed
    beta_embed = pyro.sample("beta_embed", dist.Dirichlet(concent_P_q))
    # sample sig2_P
    sig2_P = pyro.sample("sig2_P", dist.HalfCauchy(scale_P_q))

# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the Beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nBased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))