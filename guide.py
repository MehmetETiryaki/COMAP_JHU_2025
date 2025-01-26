def guide(data):
    medal, theta, result = data

    T = medal.shape[0]
    n_countries = medal.shape[1]
    num_sports = theta.shape[2]
    n_embedding = theta.shape[3]  # e.g., 3

    # 1. Global variational parameters
    # --------------------------------
    # Variational Beta distributions for phi and rho
    phi_alpha = pyro.param("phi_alpha", torch.tensor(5), constraint=constraints.greater_than(1.0))
    phi_beta = pyro.param("phi_beta", torch.tensor(5), constraint=constraints.greater_than(1.0))
    pyro.sample("phi", dist.Beta(phi_alpha, phi_beta))

    rho_alpha = pyro.param("rho_alpha", torch.tensor(5), constraint=constraints.greater_than(1.0))
    rho_beta = pyro.param("rho_beta", torch.tensor(5), constraint=constraints.greater_than(1.0))
    pyro.sample("rho", dist.Beta(rho_alpha, rho_beta))

    # Variational Dirichlet distributions
    w_concentration = pyro.param("w_concentration", torch.ones(3), constraint=constraints.positive)
    pyro.sample("w", dist.Dirichlet(w_concentration))

    beta_embed_concentration = pyro.param(
        "beta_embed_concentration", torch.ones(n_embedding), constraint=constraints.positive
    )
    pyro.sample("beta_embed", dist.Dirichlet(beta_embed_concentration))

    # Variational Normal distribution for alpha
    alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    alpha_scale = pyro.param("alpha_scale", torch.tensor(1.0), constraint=constraints.positive)
    pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale))

    # Variational HalfCauchy distributions for sig2_CE and sig2_P
    sig2_CE_scale = pyro.param("sig2_CE_scale", torch.tensor(1.0), constraint=constraints.positive)
    pyro.sample("sig2_CE", dist.HalfCauchy(sig2_CE_scale))

    sig2_P_scale = pyro.param("sig2_P_scale", torch.tensor(1.0), constraint=constraints.positive)
    pyro.sample("sig2_P", dist.HalfCauchy(sig2_P_scale))

    # 2. Time-dependent variables
    # ---------------------------
    CE_prev_loc = pyro.param(
        "CE_prev_loc", torch.zeros(n_countries)
    )  # Initial CE mean (t=0)
    CE_prev_scale = pyro.param(
        "CE_prev_scale", torch.ones(n_countries), constraint=constraints.positive
    )  # Initial CE scale

    for t in range(T):
        if t == 0:
            # Variational parameters for CE at t=0
            CE_t_loc = pyro.param(f"CE_{t}_loc", CE_prev_loc)
            CE_t_scale = pyro.param(f"CE_{t}_scale", CE_prev_scale, constraint=constraints.positive)

            pyro.sample(
                f"CE_{t}",
                dist.Normal(CE_t_loc, CE_t_scale).to_event(1),  # Vectorized across countries
            )
        else:
            # Variational parameters for CE at t>0
            CE_t_loc = pyro.param(
                f"CE_{t}_loc",
                torch.zeros(n_countries),  # Initialize to zero mean
            )
            CE_t_scale = pyro.param(
                f"CE_{t}_scale",
                torch.ones(n_countries),  # Initialize to unit variance
                constraint=constraints.positive,
            )
            pyro.sample(
                f"CE_{t}",
                dist.Normal(CE_t_loc, CE_t_scale).to_event(1),  # Vectorized across countries
            )

        # Variational parameters for P_t
        P_t_loc = pyro.param(
            f"P_{t}_loc",
            torch.zeros(n_countries, num_sports),  # Initialize to zero mean
        )
        P_t_scale = pyro.param(
            f"P_{t}_scale",
            torch.ones(n_countries, num_sports),  # Initialize to unit variance
            constraint=constraints.positive,
        )
        pyro.sample(
            f"P_{t}",
            dist.LogNormal(P_t_loc, P_t_scale).to_event(2),  # Vectorized across all dimensions
        )