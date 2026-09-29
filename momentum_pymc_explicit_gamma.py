"""
momentum_pymc_explicit_gamma.py

Diagnostic version of momentum_pymc_re.py: samples the player-tournament
random effect gamma_{p,T} explicitly (non-centered) instead of
marginalizing it analytically. This is the specification described
in section 7 of the write-up that produced the funnel pathology
(R-hat = 1.15 on tau_gamma, ~17% divergent transitions on OTT).

Use this to generate trace plots that show the bad convergence,
NOT for inference.

Spec for Model B (the AR(1) version):
    sigma     ~ HalfNormal(2)
    rho_pop   ~ Normal(0, 0.5)
    tau_rho   ~ HalfNormal(0.3)
    rho_p     ~ Normal(rho_pop, tau_rho)            # non-centered
    tau_gamma ~ HalfNormal(0.5)                     # original loose prior
    gamma_c   ~ Normal(0, tau_gamma)                # non-centered, per cluster
    y_{c,j}   ~ Normal(rho_{p(c)} * prev_{c,j} + gamma_c, sigma)

Default: OTT only (the worst offender). Pass --components to override.

Run:
    cd ~/Desktop/golf-trading && python3 momentum_pymc_explicit_gamma.py
"""

import argparse
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

DB_PATH    = "./golf.db"
OUT_DIR    = "momentum_results_explicit_gamma"
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
    keep_cols = ['dg_id', 'event_id', 'year', 'tour', col, 'prev_resid']
    out = sub.loc[mask, keep_cols].rename(
        columns={col: 'curr_resid'}
    ).reset_index(drop=True)
    return out


def assign_cluster_ids(pairs):
    """Tag each pair with a (player, tournament) cluster id."""
    pairs = pairs.sort_values(
        ['dg_id', 'event_id', 'year', 'tour']
    ).reset_index(drop=True)

    pt_keys = (pairs['dg_id'].astype(str)    + '_' +
               pairs['event_id'].astype(str) + '_' +
               pairs['year'].astype(str)     + '_' +
               pairs['tour'])
    cluster_codes, _ = pd.factorize(pt_keys, sort=False)
    pairs = pairs.copy()
    pairs['cluster_id'] = cluster_codes
    return pairs


def fit_model_B_explicit(pairs, players, args):
    """
    Model B with gamma sampled explicitly (non-centered).
    This is the specification that funneled.
    """
    n_clusters = pairs['cluster_id'].nunique()
    cluster_idx = pairs['cluster_id'].values.astype(int)
    player_idx  = pairs['player_idx'].values.astype(int)
    prev = pairs['prev_resid'].values.astype(float)
    curr = pairs['curr_resid'].values.astype(float)

    print(f"  {len(pairs):,} pairs, {n_clusters:,} clusters, "
          f"{len(players):,} players.")
    print(f"  Sampling {n_clusters:,} explicit gamma latents.")

    with pm.Model(coords={'player': players,
                          'cluster': np.arange(n_clusters)}):
        sigma   = pm.HalfNormal('sigma', 2.0)
        rho_pop = pm.Normal('rho_pop', 0.0, 0.5)
        tau_rho = pm.HalfNormal('tau_rho', 0.3)

        z   = pm.Normal('z', 0.0, 1.0, dims='player')
        rho = pm.Deterministic('rho', rho_pop + z * tau_rho, dims='player')

        # Loose prior on tau_gamma to reproduce the original funnel.
        tau_gamma = pm.HalfNormal('tau_gamma', 0.5)

        # Explicit non-centered gamma: this is what funnels.
        gamma_z = pm.Normal('gamma_z', 0.0, 1.0, dims='cluster')
        gamma   = pm.Deterministic('gamma',
                                   gamma_z * tau_gamma,
                                   dims='cluster')

        mu = rho[player_idx] * prev + gamma[cluster_idx]
        pm.Normal('y', mu=mu, sigma=sigma, observed=curr)

        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept, random_seed=42,
            progressbar=True,
        )
    return idata


def fit_component(df, component, args):
    print(f"\n{'=' * 60}")
    print(f"Component: {component} (explicit gamma)")
    print('=' * 60)

    pairs = build_pairs(df, component)
    if len(pairs) == 0:
        print(f"  No pairs for {component}. Skipping.")
        return

    players = sorted(pairs['dg_id'].unique().tolist())
    p_idx = {p: i for i, p in enumerate(players)}
    pairs['player_idx'] = pairs['dg_id'].map(p_idx)
    pairs = assign_cluster_ids(pairs)

    print(f"  curr_resid: mean={pairs['curr_resid'].mean():+.3f}, "
          f"std={pairs['curr_resid'].std():.3f}")

    print("\n  Fitting Model B (explicit non-centered gamma)...")
    t0 = time.time()
    idata = fit_model_B_explicit(pairs, players, args)
    print(f"  Model B (explicit): {time.time() - t0:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/idata_B_explicit_{component}.nc"
    idata.to_netcdf(out_path)
    print(f"  Saved: {out_path}")

    print("\n  Posterior summary (headline parameters):")
    summary = az.summary(
        idata,
        var_names=['rho_pop', 'tau_rho', 'tau_gamma', 'sigma'],
        round_to=4,
    )
    print(summary.to_string())

    n_div = int(idata.sample_stats['diverging'].sum().item())
    total = int(idata.sample_stats['diverging'].size)
    print(f"\n  Divergences: {n_div:,} / {total:,} "
          f"({100 * n_div / total:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', nargs='+', default=['sg_ott'],
                        choices=COMPONENTS,
                        help='Which SG components to fit (default: sg_ott)')
    parser.add_argument('--draws',         type=int,   default=1000)
    parser.add_argument('--tune',          type=int,   default=1000)
    parser.add_argument('--chains',        type=int,   default=4)
    parser.add_argument('--target_accept', type=float, default=0.95)
    args = parser.parse_args()

    print("Loading momentum_residuals...")
    df = load_residuals()
    print(f"  {len(df):,} rounds in residual table.")

    t_total = time.time()
    for comp in args.components:
        fit_component(df, comp, args)

    print(f"\nDone in {(time.time() - t_total)/60:.1f} min.")
    print(f"Posteriors saved to ./{OUT_DIR}/")


if __name__ == '__main__':
    main()