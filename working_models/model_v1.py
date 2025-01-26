import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.optim import ReduceLROnPlateau
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

    # Convert ranking_obs to a tensor for indexing
    #ranking_obs = torch.tensor(ranking_obs, device=P_t.device)

    # Track counts of medals already won by each country
    country_medal_count = torch.zeros_like(P_t)

    for medal in ranking_obs:
        # Adjust the probability for the current ranking
        adjusted_prob = P_t / (1 + country_medal_count)

        # Calculate log probability of the current ranking
        log_likelihood += torch.log(adjusted_prob[medal]) - torch.log(torch.sum(adjusted_prob))

        # Increment the count for the selected country
        country_medal_count[medal] += 1

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
    alpha = pyro.sample("alpha", dist.MultivariateNormal(torch.zeros(n_countries), torch.eye(n_countries)))   # mean of CE at t=0
    sig2_CE = torch.tensor(1.0) #pyro.sample("sig2_CE", dist.HalfCauchy(1.0))
    sig2_P  = torch.tensor(1.0) #pyro.sample("sig2_P",  dist.HalfCauchy(1.0))

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
                dist.Normal(0, sig2_CE.sqrt())
                    .expand([n_countries])
                    .to_event(1)
            )
        else:
            # mean for each country = phi * CE_{t-1} + (medal[t-1][country] dot w)
            # medal[t-1] shape: [n_countries, 3], w shape: [3] => broadcast => [n_countries]
            mean_t = phi * CE_prev + alpha#+ torch.matmul(medal[t-1], w)  # shape [n_countries]
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
        loc = rho * CE_t.unsqueeze(-1) #+ embed_term       # shape [n_countries, num_sports]

        # Now sample from a LogNormal with that loc and some scale = sqrt(sig2_P)
        # We can do a single pyro.sample with .to_event(2) to treat the entire [n_countries, n_sports] block
        P_t = pyro.sample(
            f"P_{t}",
            dist.LogNormal(loc.expand(-1, num_sports), sig2_P.sqrt())
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

def guide(data):
    medal, theta, result = data
    T = medal.shape[0]
    n_countries = medal.shape[1]
    num_sports = theta.shape[2]
    n_embedding = theta.shape[3]

    # 1. Global hyperparameters and variational distributions
    # -------------------------------------------------------
    # Learnable parameters for phi and rho (Beta priors)
    phi_alpha = pyro.param("phi_alpha", torch.tensor(5.0), constraint=constraints.greater_than(1.0))
    phi_beta = pyro.param("phi_beta", torch.tensor(5.0), constraint=constraints.greater_than(1.0))
    #phi = pyro.sample("phi", dist.Beta(phi_loc * phi_scale, (1 - phi_loc) * phi_scale))
    phi = pyro.sample("phi", dist.Beta(phi_alpha, phi_beta))

    rho_alpha = pyro.param("rho_alpha", torch.tensor(5.0), constraint=constraints.greater_than(1.0))
    rho_beta = pyro.param("rho_beta", torch.tensor(5.0), constraint=constraints.greater_than(1.0))
    #rho = pyro.sample("rho", dist.Beta(rho_loc * rho_scale, (1 - rho_loc) * rho_scale))
    rho = pyro.sample("rho", dist.Beta(rho_alpha, rho_beta))

    alpha_loc = pyro.param("alpha_loc", torch.zeros(n_countries))
    #alpha_scale = pyro.param("alpha_scale", torch.eye(n_countries), constraint=constraints.positive)
    alpha = pyro.sample("alpha", dist.MultivariateNormal(alpha_loc, torch.eye(n_countries)))

    # Learnable parameters for Dirichlet priors
    w_concentration = pyro.param("w_concentration", torch.ones(3), constraint=constraints.positive)
    w = pyro.sample("w", dist.Dirichlet(w_concentration))

    beta_embed_concentration = pyro.param("beta_embed_concentration", torch.ones(n_embedding), constraint=constraints.positive)
    beta_embed = pyro.sample("beta_embed", dist.Dirichlet(beta_embed_concentration))

    # Learnable parameters for Normal and HalfCauchy priors
    #alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    #alpha_scale = pyro.param("alpha_scale", torch.tensor(1.0), constraint=constraints.positive)
    #alpha = pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale))

    #sig2_CE_scale = pyro.param("sig2_CE_scale", torch.tensor(1.0), constraint=constraints.positive)
    sig2_CE = torch.tensor(1.0) #pyro.sample("sig2_CE", dist.HalfCauchy(sig2_CE_scale))

    #sig2_P_scale = pyro.param("sig2_P_scale", torch.tensor(1.0), constraint=constraints.positive)
    sig2_P = torch.tensor(1.0) #pyro.sample("sig2_P", dist.HalfCauchy(sig2_P_scale))

    # 2. Time-varying variables
    # -------------------------
    CE_prev_loc = pyro.param("CE_prev_loc", torch.zeros(n_countries))
    CE_prev_scale = pyro.param("CE_prev_scale", torch.ones(n_countries), constraint=constraints.positive)
    CE_prev = pyro.sample("CE_0", dist.Normal(CE_prev_loc, CE_prev_scale).to_event(1))

    P_0_loc = pyro.param("P_0_loc", torch.zeros((n_countries, num_sports)))
    P_0_scale = pyro.param("P_0_scale", torch.ones((n_countries, num_sports)), constraint=constraints.positive)
    pyro.sample("P_0", dist.LogNormal(P_0_loc, P_0_scale).to_event(2))

    for t in range(1, T):
        # CE_t: Learnable mean and scale for each time step
        CE_loc = pyro.param(f"CE_loc_{t}", torch.zeros(n_countries))
        CE_scale = pyro.param(f"CE_scale_{t}", torch.ones(n_countries), constraint=constraints.positive)
        pyro.sample(f"CE_{t}", dist.Normal(CE_loc, CE_scale).to_event(1))

        # P_t: Learnable parameters for sports' embedding and variances
        P_loc = pyro.param(f"P_loc_{t}", torch.zeros((n_countries, num_sports)))
        P_scale = pyro.param(f"P_scale_{t}", torch.ones((n_countries, num_sports)), constraint=constraints.positive)
        pyro.sample(f"P_{t}", dist.LogNormal(P_loc, P_scale).to_event(2))


if __name__ == "__main__":

    medals_tensor = get_medal_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned.csv")
    theta_tensor = get_theta_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned.csv")
    results_tensor = get_results_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned.csv")

    print("Data loaded")

    data = (medals_tensor, theta_tensor, results_tensor)

    # setup the optimizer
    adam_params = {"lr": 0.1}
    optimizer = Adam(adam_params)

    #optimizer = ReduceLROnPlateau({"lr": 0.1})
    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # do gradient steps
    n_steps = 100
    for step in range(n_steps):
        loss = svi.step(data)
        """
        # Check if loss improvement is less than the tolerance
        if 1.03 > loss/previous_loss > 0.97:
            patience_counter += 1
        else:
            patience_counter = 0  # Reset if improvement is significant

        # Adjust learning rate if patience is exceeded
        if patience_counter >= patience:
            # Reduce the learning rate
            new_lr = optimizer.get_state()["param_groups"][0]["lr"] * lr_decay
            optimizer.set_state({"param_groups": [{"lr": new_lr}]})

            print(f"Adjusted learning rate to {new_lr}")
            patience_counter = 0  # Reset patience counter
        previous_loss = loss
        """
        if step % 10 == 0:
            print(f"Step {step} loss = {svi.evaluate_loss(data)}")
            #print(pyro.param("alpha_loc")[220], pyro.param("alpha_loc").mean())
            print(pyro.param("phi_alpha"))
        

    #save the model parameters
    pyro.get_param_store().save("trained_model_params.pth")

    param_store = pyro.get_param_store()
    with open('trained_model_params.txt', 'w') as f:
        for name, value in param_store.named_parameters():
            f.write(f"{name}: {value}\n")

