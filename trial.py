def guide(data):
    medal, theta, results = data
    T = medal.shape[0]
    n_countries = medal.shape[1]
    num_sports = theta.shape[2]

    # Global variables
    # ...

    # Local variables
    with pyro.plate("time_plate", T, dim=-3):
        with pyro.plate("country_plate", n_countries, dim=-2):
            ce_loc = pyro.param("CE_loc", torch.zeros(T, n_countries))
            ce_scale = pyro.param("CE_scale", torch.ones(T, n_countries),
                                  constraint=constraints.positive)
            pyro.sample("CE", dist.Normal(ce_loc, ce_scale))

            with pyro.plate("sport_plate", num_sports, dim=-1):
                p_loc = pyro.param("P_loc", torch.zeros(T, n_countries, num_sports))
                p_scale = pyro.param("P_scale", torch.ones(T, n_countries, num_sports),
                                     constraint=constraints.positive)
                pyro.sample("P", dist.LogNormal(p_loc, p_scale))
