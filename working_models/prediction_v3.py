import numpy as np
import pandas as pd
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

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

def PlackettLuce_pred(P):

    ranking_pred = []
    # Track counts of medals already won by each country
    country_medal_count = torch.zeros_like(P)

    for medal in range(3):
        # Adjust the probability for the current ranking
        adjusted_prob = P / (1 + country_medal_count)

        # Predict the winner of the current medal
        prediction = pyro.sample(f"country_{medal}", dist.Multinomial(1, adjusted_prob))
        country = torch.nonzero(prediction, as_tuple=True)[0].item()

        # Increment the count for the selected country
        country_medal_count[medal] += 1

        ranking_pred.append(country)
    return ranking_pred

def monte_carlo_simulation(n_samples, data, T_last):
    medal, theta, result = data
    # Simulate the next time step (T) using Monte Carlo
    simulated_results = []
    #T = medal.shape[0]
    n_countries = medal.shape[1]
    num_sports = theta.shape[2]
    n_embedding = theta.shape[3]

    pyro.get_param_store().load("trained_model_params.pth")

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))
    
    for _ in range(n_samples):
        # Initialize pyro.param for the simulation if needed
        CE_last = pyro.sample(f"CE_last", dist.Normal(pyro.param(f"CE_loc_{T_last}"),pyro.param(f"CE_scale_{T_last}")).to_event(1))
        P_last = pyro.sample(f"P_last", dist.Normal(pyro.param(f"P_loc_{T_last}"),pyro.param(f"P_scale_{T_last}")).to_event(1))
        
        # Sample from the model for the next time step (T)
        phi = pyro.sample("phi", dist.Beta(pyro.param("phi_alpha"), pyro.param("phi_beta")))
        rho = pyro.sample("rho", dist.Beta(pyro.param("rho_alpha"), pyro.param("rho_beta")))
        w = pyro.sample("w", dist.MultivariateNormal(pyro.param("w_loc"),pyro.param("w_scale")))
        beta_embed = pyro.sample("beta_embed", dist.MultivariateNormal(pyro.param("beta_embed_loc"),pyro.param("beta_embed_scale")))
        alpha = pyro.sample("alpha", dist.MultivariateNormal(pyro.param("alpha_loc"),pyro.param("alpha_scale")))
        
        # Simulate the CE for the next time step (T)
        CE_pred_loc = phi * CE_last + alpha + torch.matmul(medal[T_last], w)  # shape [n_countries]
        CE_pred = pyro.sample(
            f"CE_pred",
            dist.Normal(CE_pred_loc, torch.ones(n_countries)).to_event(1)
        )
        
        # Simulate the probabilities for the next time step (P_T)
        embed_term = (theta[T_last] * beta_embed).sum(dim=-1)  # shape [n_countries, num_sports]
        P_pred_loc = rho * CE_pred.unsqueeze(-1) + embed_term  # shape [n_countries, num_sports]
        P_pred = pyro.sample(
            f"P_pred_loc",
            dist.LogNormal(P_pred_loc, torch.ones((n_countries, num_sports))).to_event(2)
        )
        for index, value in enumerate(P_pred[:65]):
            print(index, value)
        
        # Use the Plackett-Luce likelihood to generate event outcomes based on P_T
        medal_count_pred = torch.zeros((n_countries, 3), dtype=torch.int)
        #result = torch.zeros((num_sports, ), dtype=torch.int)
        for sport_idx in range(num_sports):
            event_list = result[T_last][sport_idx]
            for event_idx, ranking_obs in enumerate(event_list):
                if ranking_obs[0] == -1:
                    continue
                ranking_pred = PlackettLuce_pred(P_pred[:, sport_idx])
                if sport_idx == 65:
                    print(ranking_pred)
                for m, country in enumerate(ranking_pred):
                    medal_count_pred[country, m] += 1
        for i_country, result_array in enumerate(medal_count_pred):
            print(i_country, result_array)

if __name__ == "__main__":

    medals_tensor = get_medal_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned.csv")
    theta_tensor = get_theta_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned.csv")
    results_tensor = get_results_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned.csv")

    print("Data loaded")

    data = (medals_tensor, theta_tensor, results_tensor)

    monte_carlo_simulation(1, data, 30)