import pyro
from matplotlib import pyplot as plt
import math
from model_v3 import get_theta_tensor
import numpy as np


pyro.clear_param_store()
pyro.get_param_store().load("trained_model_params.pth")

italy_power_mean = []
china_power_mean = []

italy_ce_mean = []
china_ce_mean = []

for i in range(1, 30):
    P_loc_param = pyro.get_param_store()[f"P_loc_{i}"]
    P_loc_param = P_loc_param.detach().numpy()
    P_scale_param = pyro.get_param_store()[f"P_scale_{i}"]
    P_scale_param = P_scale_param.detach().numpy()
    P_loc_param_table_tennis = P_loc_param[:, 65]
    P_scale_param_table_tennis = P_scale_param[:, 65]
    P_loc_param_table_tennis_italy = P_loc_param_table_tennis[102]
    P_scale_param_table_tennis_italy = P_scale_param_table_tennis[102]
    P_loc_param_table_tennis_china = P_loc_param_table_tennis[42]
    P_scale_param_table_tennis_china = P_scale_param_table_tennis[42]
    P_mean_italy = math.exp(P_loc_param_table_tennis_italy + 0.5 * P_scale_param_table_tennis_italy ** 2)
    P_mean_china = math.exp(P_loc_param_table_tennis_china + 0.5 * P_scale_param_table_tennis_china ** 2)
    italy_power_mean.append(P_mean_italy)
    china_power_mean.append(P_mean_china)

rho_alpha = pyro.get_param_store()["rho_alpha"]
rho_alpha = rho_alpha.detach().numpy()
rho_beta = pyro.get_param_store()["rho_beta"]
rho_beta = rho_beta.detach().numpy()
rho_mean = rho_alpha / (rho_alpha + rho_beta)

beta_embed_loc = pyro.get_param_store()["beta_embed_loc"]
beta_embed_loc = beta_embed_loc.detach().numpy()
beta_mean = beta_embed_loc

theta_tensor = get_theta_tensor("summerOly_athletes_team_id_no_high_irregulars_numerical_experience_combined_reassigned_home.csv")

theta_beta_mult_italy_list = []
theta_beta_mult_china_list = []

for i in range(1,30):
    theta_beta_mult_italy = np.dot(beta_mean, theta_tensor[i][102][65].detach().numpy())
    theta_beta_mult_china = np.dot(beta_mean, theta_tensor[i][42][65].detach().numpy())
    theta_beta_mult_italy_list.append(theta_beta_mult_italy)
    theta_beta_mult_china_list.append(theta_beta_mult_china)

italy_ce_rho_mean = []
china_ce_rho_mean = []
for i in range(1, 30):
    CE_loc_param = pyro.get_param_store()[f"CE_loc_{i}"]
    CE_loc_param = CE_loc_param.detach().numpy()

    CE_loc_param_italy = CE_loc_param[102]
    CE_loc_param_china = CE_loc_param[42]

    rho_ce_italy = rho_mean * CE_loc_param_italy
    rho_ce_china = rho_mean * CE_loc_param_china

    italy_ce_rho_mean.append(CE_loc_param_italy)
    china_ce_rho_mean.append(CE_loc_param_china)

plt.plot(range(1, 30), italy_power_mean, label="Italy Power")
plt.plot(range(1, 30), china_power_mean, label="China Power")
plt.plot(range(1, 30), theta_beta_mult_italy_list, label="Italy Theta Beta")
plt.plot(range(1, 30), theta_beta_mult_china_list, label="China Theta Beta")
plt.plot(range(1, 30), italy_ce_rho_mean, label="Italy CE Rho")
plt.plot(range(1, 30), china_ce_rho_mean, label="China CE Rho")
plt.xlabel("Time")
plt.legend()
plt.show()




