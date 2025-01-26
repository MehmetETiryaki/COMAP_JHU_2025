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
import pandas as pd


# clear the param store in case we're in a REPL
pyro.clear_param_store()

def get_medal_tensor(input_file):
    num_unique_times = 31
    num_countries = 234

    medals_tensor = torch.zeros((num_unique_times, num_countries, 3))

    for row in pd.read_csv(input_file).iterrows():
        year = row[1]["Year"]
        country = row[1]["NOC"]
        medal = row[1]["Medal"]

        if medal == "Gold":
            medals_tensor[year][country][0] += 1
        elif medal == "Silver":
            medals_tensor[year][country][1] += 1
        elif medal == "Bronze":
            medals_tensor[year][country][2] += 1

    return medals_tensor

def get_theta_tensor(input_file):
    df = pd.read_csv(input_file)

    num_unique_times = 31
    num_countries = 234
    num_sports = 76

    theta_tensor = torch.zeros((num_unique_times, num_countries, num_sports, 3))

    for row in df.iterrows():
        year = row[1]["Year"]
        country = row[1]["NOC"]
        sport = row[1]["Sport"]
        medal = row[1]["Medal"]

        if medal == "Gold":
            theta_tensor[year][country][sport][0] += 1
        elif medal == "Silver":
            theta_tensor[year][country][sport][1] += 1
        elif medal == "Bronze":
            theta_tensor[year][country][sport][2] += 1

    return theta_tensor

def get_results_tensor(input_file):

    df = pd.read_csv(input_file)
    num_unique_times = 31
    num_sports = 76
    max_num_events = 137

    results_tensor = torch.full((num_unique_times, num_sports, max_num_events, 3), -1)

    for row in df.iterrows():
        year = row[1]["Year"]
        country = row[1]["NOC"]
        sport = row[1]["Sport"]
        event = row[1]["Event"]

        if row[1]["Medal"] == "Gold":
            results_tensor[year][sport][event][0] = country
        elif row[1]["Medal"] == "Silver":
            results_tensor[year][sport][event][1] = country
        elif row[1]["Medal"] == "Bronze":
            results_tensor[year][sport][event][2] = country

    return results_tensor

def PlackettLuce_LogLikelihood(P_t, ranking_obs):
    log_likelihood = 0.0
    remaining = [i for i in range(len(P_t))]
    for medal in range(len(ranking_obs)):
        # Use torch.log instead of math.log
        log_likelihood += torch.log(P_t[ranking_obs[medal]]) - torch.log(torch.sum(P_t[remaining]))
        remaining.remove(ranking_obs[medal])
    return log_likelihood


def model(data):
    medal, theta, result = data
    # medal.shape = [T, n_countries, 3]   (for the w dot-product)
    # theta.shape = [T, n_countries, num_sports, n_embedding]
    # result[t][sport] = list of events; each event is a ranking list (like [0,1,2])

    T = medal.shape[0]
    n_countries = medal.shape[1]
    num_sports = theta.shape[2]
    n_embedding = theta.shape[3]  # e.g. 3

    # 1. Global hyperparameters and prior samples
    # -------------------------------------------
    # Beta priors
    phi = pyro.sample("phi", dist.Beta(5.0, 5.0))         # correlation for CE
    rho = pyro.sample("rho", dist.Beta(5.0, 5.0))         # correlation for P
    # Dirichlet priors
    w = pyro.sample("w", dist.Dirichlet(torch.ones(3)))   # shape [3]
    beta_embed = pyro.sample("beta_embed", dist.Dirichlet(torch.ones(n_embedding)))
    # Normal, HalfCauchy
    alpha = pyro.sample("alpha", dist.Normal(0.0, 2.0))   # mean of CE at t=0
    sig2_CE = pyro.sample("sig2_CE", dist.HalfCauchy(1.0))
    sig2_P  = pyro.sample("sig2_P",  dist.HalfCauchy(1.0))

    # 2. Initialize and iterate over time
    # -----------------------------------
    CE_prev = torch.zeros(n_countries)  # shape [n_countries]

    for t in range(T):

        # --- Vectorized sampling of CE_t (shape [n_countries]) ---
        if t == 0:
            # All countries share the same prior Normal(alpha, sqrt(sig2_CE))
            # expand(...) + .to_event(1) -> treat them as a vector of shape [n_countries]
            CE_t = pyro.sample(
                f"CE_{t}",
                dist.Normal(alpha, sig2_CE.sqrt())
                    .expand([n_countries])
                    .to_event(1)
            )
        else:
            # mean for each country = phi * CE_{t-1} + (medal[t-1][country] dot w)
            # medal[t-1] shape: [n_countries, 3], w shape: [3] => broadcast => [n_countries]
            mean_t = phi * CE_prev + torch.matmul(medal[t-1], w)  # shape [n_countries]
            CE_t = pyro.sample(
                f"CE_{t}",
                dist.Normal(mean_t, sig2_CE.sqrt())
                    .to_event(1)  
            )

        # --- Vectorized sampling of P_t (shape [n_countries, num_sports]) ---
        # For each (country, sport), loc = rho * CE_t[country] + beta_embed dot theta[t][country][sport].
        # We can compute this via a tensor sum over the embedding dimension.
        #   theta[t] has shape [n_countries, num_sports, n_embedding].
        #   beta_embed has shape [n_embedding].
        # We'll do a .sum(dim=-1) after multiplying, giving shape [n_countries, num_sports].
        # Then we broadcast-add rho*CE_t (of shape [n_countries, 1]).
        embed_term = (theta[t] * beta_embed).sum(dim=-1)  # shape [n_countries, num_sports]
        loc = rho * CE_t.unsqueeze(-1) + embed_term       # shape [n_countries, num_sports]

        # Now sample from a LogNormal with that loc and some scale = sqrt(sig2_P)
        # We can do a single pyro.sample with .to_event(2) to treat the entire [n_countries, n_sports] block
        P_t = pyro.sample(
            f"P_{t}",
            dist.LogNormal(loc, sig2_P.sqrt())
                .to_event(2)
        )
        # P_t is shape [n_countries, n_sports]

        # 3. Plackett-Luce likelihood per event
        # -------------------------------------
        # Each "sport" might have multiple events at time t. We skip missing events, indicated by [-1, ...].
        for sport_idx in range(num_sports):
            event_list = result[t][sport_idx]
            for event_idx, ranking_obs in enumerate(event_list):
                if ranking_obs[0] == -1:
                    continue

                # P_t[:, sport_idx] => shape [n_countries]
                ll = PlackettLuce_LogLikelihood(P_t[:, sport_idx], ranking_obs)
                pyro.factor(f"likelihood_{t}_{sport_idx}_{event_idx}", ll)

        # Move forward in time
        CE_prev = CE_t

"""
def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    medal, theta, results = data
    n_embedding = 3
    alp_CE_q = pyro.param("alpha_CE_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    beta_CE_q = pyro.param("beta_CE_q", torch.tensor(5.0),
                        constraint=constraints.positive)
    concent_CE_q = pyro.param("concent_CE_q", torch.full((3,), 1.0),
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
"""

guide = pyro.infer.autoguide.AutoNormal(model)

medals_tensor = get_medal_tensor("toy.csv")
theta_tensor = get_theta_tensor("toy.csv")
results_tensor = get_results_tensor("toy.csv")

print("Data loaded")

data = (medals_tensor, theta_tensor, results_tensor)

# setup the optimizer
adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())


# do gradient steps
n_steps = 400
for step in range(n_steps):
    svi.step(data)
    if step % 10 == 0:
        print(f"Step {step} loss = {svi.evaluate_loss(data)}")

param_store = pyro.get_param_store()
for name, value in param_store.named_parameters():
    print(f"Parameter Name: {name}, Value: {value}")
