"""
momentum_pymc.py

Bayesian model comparison: are within-tournament rounds iid (Model A),
or does residual performance follow an AR(1) process with player-specific
persistence (Model B)?

Per SG component:
    Model A (independence):
        curr_resid ~ Normal(0, sigma)

    Model B (hierarchical AR(1)):
        rho_pop ~ Normal(0, 0.5)
        tau_rho ~ HalfNormal(0.3)
        rho_p   ~ Normal(rho_pop, tau_rho)        # non-centered, per player
        curr_resid ~ Normal(rho_p[player] * prev_resid, sigma)

    sigma ~ HalfNormal(2)

Pairs are within-tournament consecutive rounds for the same player:
(R1->R2, R2->R3, R3->R4) for made-cut, (R1->R2) for missed-cut.

Model comparison via PSIS-LOO. Per-player rho posteriors saved for
the shrinkage plot.

Inputs:
    momentum_residuals  built by build_residuals.py

Outputs (./momentum_results/):
    idata_A_<component>.nc
    idata_B_<component>.nc

Run:
    cd ~/Desktop/golf-trading && python3 momentum_pymc.py
    python3 momentum_pymc.py --components sg_putt
    python3 momentum_pymc.py --draws 1000 --tune 1000
"""

import argparse
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

DB_PATH    = "./golf.db"
OUT_DIR    = "momentum_results"
COMPONENTS = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']


def load_residuals():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM momentum_residuals", conn)
    conn.close()
    return df


def build_pairs(df, component):
    """Construct within-tournament (prev_resid, curr_resid) pairs per player."""
    col = f'resid_{component}'
    sub = df[df[col].notna()].copy()
    sub = sub.sort_values(['dg_id', 'event_id', 'year', 'tour', 'round_num'])
    g = sub.groupby(['dg_id', 'event_id', 'year', 'tour'], sort=False)
    sub['prev_round'] = g['round_num'].shift(1)
    sub['prev_resid'] = g[col].shift(1)
    mask = ((sub['round_num'] - sub['prev_round'] == 1)
            & sub['prev_resid'].notna())
    out = sub.loc[mask, ['dg_id', col, 'prev_resid']].rename(
        columns={col: 'curr_resid'}
    ).reset_index(drop=True)
    return out


def fit_model_A(curr_resid, args):
    with pm.Model():
        sigma = pm.HalfNormal('sigma', 2.0)
        pm.Normal('y', mu=0.0, sigma=sigma, observed=curr_resid)
        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept, random_seed=42,
            idata_kwargs={'log_likelihood': True},
            progressbar=True,
        )
    return idata


def fit_model_B(curr_resid, prev_resid, player_idx, players, args):
    with pm.Model(coords={'player': players}):
        sigma   = pm.HalfNormal('sigma', 2.0)
        rho_pop = pm.Normal('rho_pop', 0.0, 0.5)
        tau_rho = pm.HalfNormal('tau_rho', 0.3)

        z   = pm.Normal('z', 0.0, 1.0, dims='player')
        rho = pm.Deterministic('rho', rho_pop + z * tau_rho, dims='player')

        mu = rho[player_idx] * prev_resid
        pm.Normal('y', mu=mu, sigma=sigma, observed=curr_resid)

        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept, random_seed=42,
            idata_kwargs={'log_likelihood': True},
            progressbar=True,
        )
    return idata


def fit_component(df, component, args):
    print(f"\n{'=' * 60}")
    print(f"Component: {component}")
    print('=' * 60)

    pairs = build_pairs(df, component)
    if len(pairs) == 0:
        print(f"  No pairs for {component}. Skipping.")
        return

    players = sorted(pairs['dg_id'].unique().tolist())
    p_idx = {p: i for i, p in enumerate(players)}
    pairs['player_idx'] = pairs['dg_id'].map(p_idx)

    curr = pairs['curr_resid'].values.astype(float)
    prev = pairs['prev_resid'].values.astype(float)
    pidx = pairs['player_idx'].values.astype(int)

    print(f"  {len(pairs):,} pairs across {len(players):,} players.")
    print(f"  curr_resid: mean={curr.mean():+.3f}, std={curr.std():.3f}")
    print(f"  Empirical Pearson corr(prev, curr): "
          f"{np.corrcoef(prev, curr)[0, 1]:+.4f}")

    print("\n  Fitting Model A (independence)...")
    t0 = time.time()
    idata_A = fit_model_A(curr, args)
    print(f"  Model A: {time.time() - t0:.1f}s")

    print("\n  Fitting Model B (hierarchical AR(1))...")
    t0 = time.time()
    idata_B = fit_model_B(curr, prev, pidx, players, args)
    print(f"  Model B: {time.time() - t0:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    idata_A.to_netcdf(f"{OUT_DIR}/idata_A_{component}.nc")
    idata_B.to_netcdf(f"{OUT_DIR}/idata_B_{component}.nc")

    print("\n  Model B posterior summary:")
    summary = az.summary(
        idata_B, var_names=['rho_pop', 'tau_rho', 'sigma'], round_to=4
    )
    print(summary.to_string())

    print("\n  LOO comparison:")
    try:
        comp = az.compare({'A': idata_A, 'B': idata_B}, ic='loo')
        print(comp.to_string())
    except Exception as e:
        print(f"  az.compare failed: {e}")
        loo_A = az.loo(idata_A)
        loo_B = az.loo(idata_B)
        print(f"    A: elpd_loo = {loo_A.elpd_loo:.1f} +/- {loo_A.se:.1f}")
        print(f"    B: elpd_loo = {loo_B.elpd_loo:.1f} +/- {loo_B.se:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', nargs='+', default=COMPONENTS,
                        choices=COMPONENTS,
                        help='Which SG components to fit')
    parser.add_argument('--draws',         type=int,   default=1000)
    parser.add_argument('--tune',          type=int,   default=1000)
    parser.add_argument('--chains',        type=int,   default=4)
    parser.add_argument('--target_accept', type=float, default=0.9)
    args = parser.parse_args()

    print("Loading momentum_residuals...")
    df = load_residuals()
    print(f"  {len(df):,} rounds in residual table.")

    t_total = time.time()
    for comp in args.components:
        fit_component(df, comp, args)

    print(f"\nAll components done in {(time.time() - t_total)/60:.1f} min.")
    print(f"Posteriors saved to ./{OUT_DIR}/")


if __name__ == '__main__':
    main()