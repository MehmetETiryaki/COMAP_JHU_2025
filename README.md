# COMAP 2025 Problem C: Olympic Medal Modeling

Our (JHU) work on Problem C of COMAP's 2025 Mathematical Contest in Modeling (MCM): predicting Summer Olympic medal outcomes by country.

## Approach

Each country gets a latent strength in each sport at each Olympics, and that strength changes from one Games to the next. Event results (gold, silver, bronze) are modeled with a Plackett-Luce ranking likelihood and fit with stochastic variational inference (SVI).

Across versions, the model adds terms for:

- past medal counts
- home advantage
- athlete experience

## Structure

```
data/raw/          Original contest data
data/processed/    Cleaned datasets, one file per preprocessing step
preprocessing/     Data cleaning pipeline (team events, IDs, experience, home country)
models/            Model versions v0-v7, prediction and plotting scripts
models/params/     Trained parameters (.pth)
plots/             Exploratory plots
notebooks/         Analysis and plotting notebooks
archive/           Early drafts
```
