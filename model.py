import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import numpyro.distributions as numpyro_dist
import numpyro as npyro
import pickle

# clear the param store in case we're in a REPL
pyro.clear_param_store()

def model(data):
    medal, theta, results, rankings = data
    T = len(results)
    n_countries = len(results[0])
    # number of embedded parameters
    n_embedding = 4
    # intial value of Country Effect (CE)
    CE_prev = torch.full((n_countries,), 1.0)
    CE_t = torch.full((n_countries,), 1.0)
    P_t = torch.full((n_countries,), 1.0)
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
        n_countries = len(results[t])
        # loop over countries
        for country in range(n_countries):
            CE_t[country] = pyro.sample(f"CE_{t}_{country}", dist.Normal(phi*CE_prev[country] + (torch.mul(w, medal[t][country])), sig2_CE**0.5))
            # loop over sports for each country
            for sport in range(len(results[t][country])):
                P_t[country] = pyro.sample(f"P_{t}_{country}_{sport}", dist.Normal(rho*CE_t[country] + torch.mul(beta_embed, theta[t][country][sport]), sig2_P**0.5))

            # Use NumPyro's Plackett-Luce for ranking model
            for event_ranking in range(ranking[t][sport]):
                for r, ranking in enumerate(event_ranking):  # Loop through observed rankings
                    remaining_players = torch.arange(n_countries).tolist()
                    for i, player in enumerate(ranking):  # Sequential ranking
                        logits = torch.log(P_t[remaining_players])  # Strengths of unranked players
                        probs = torch.softmax(logits, dim=0)
                        # Use NumPyro's PlackettLuce distribution for sampling rankings
                        npyro.sample(f"rank_{r}_{i}", numpyro_dist.PlackettLuce(probs), obs=torch.tensor(player))
                        remaining_players.remove(player)  # Remove the ranked player

        CE_prev = CE_t

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    medal, theta, results = data
    n_embedding = 4
    alp_CE_q = pyro.param("alpha_CE_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    beta_CE_q = pyro.param("beta_CE_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    concent_CE_q = pyro.param("concent_CE_q", torch.full((4,), 1.0),
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
    # sample phi from the Beta prior
    pyro.sample("phi", dist.Beta(alp_CE_q, beta_CE_q))
    # sample w
    pyro.sample("w", dist.Dirichlet(concent_CE_q))
    # sample alpha
    pyro.sample("alpha", dist.Normal(mu_const_q, sig2_const_q))
    # sample sig2_CE
    pyro.sample("sig2_CE", dist.HalfCauchy(scale_CE_q))
    # sample rho
    pyro.sample("rho", dist.Beta(alp_P_q, beta_P_q))
    # sample beta_embed
    pyro.sample("beta_embed", dist.Dirichlet(concent_P_q))
    # sample sig2_P
    pyro.sample("sig2_P", dist.HalfCauchy(scale_P_q))

data = pickle.load(open("data.pkl", "rb"))

# setup the optimizer
adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
n_steps = 200
for step in range(n_steps):
    svi.step(data)
    if step % 10 == 0:
        print('.', end='')

# grab the learned variational parameters
phi = pyro.param("phi").item()
w = pyro.param("w").item()
alpha = pyro.param("alpha").item()
sig2_CE = pyro.param("sig2_CE").item()
rho = pyro.param("rho").item()
beta_embed = pyro.param("beta_embed").item()
sig2_P = pyro.param("sig2_P").item()


print("\nBased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f")