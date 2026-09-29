"""
momentum_pymc.py

Bayesian model comparison: are within-tournament rounds iid (Model A),
or does residual performance follow an AR(1) process with player-specific
persistence (Model B)?

Both models include a player-tournament random effect gamma_{p,T} with
a shrinkage prior. This absorbs anything that's constant within a
(player, tournament) pair (course fit, weather suiting their game, mental
state for the week, etc.) so that rho is identified only by genuine
round-to-round dynamics, not by within-tournament shared offsets.

The gamma random effects are marginalized analytically. Within each
(p, T) cluster, the joint distribution of pairs is multivariate Normal
with a compound symmetric covariance:
    Sigma_k = sigma^2 * I_k + tau_gamma^2 * J_k
where J_k is the k x k matrix of ones. This gives the same posterior
on (rho_pop, tau_rho, tau_gamma, sigma) as explicitly representing the
~33K gamma parameters, but avoids the funnel pathology and converges
quickly. We do not retain individual gamma estimates (we never used them).

Per SG component:
    Both models share:
        sigma     ~ HalfNormal(2)
        tau_gamma ~ HalfNormal(0.2)

    Model A (independence):
        y_c ~ MVN(0, sigma^2 * I + tau_gamma^2 * J)

    Model B (hierarchical AR(1)):
        rho_pop ~ Normal(0, 0.5)
        tau_rho ~ HalfNormal(0.3)
        rho_p   ~ Normal(rho_pop, tau_rho)        # non-centered, per player
        y_c ~ MVN(rho_p * prev_c, sigma^2 * I + tau_gamma^2 * J)

Pairs are within-tournament consecutive rounds for the same player:
(R1->R2, R2->R3, R3->R4) for made-cut, (R1->R2) for missed-cut.

Model comparison via PSIS-LOO (cluster-level: each (player, tournament)
contributes one log-likelihood term). Per-player rho posteriors saved
for the shrinkage plot.

Inputs:
    momentum_residuals  built by build_residuals.py

Outputs (./momentum_results_re/):
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
import pytensor.tensor as pt
import arviz as az
import xarray as xr

DB_PATH    = "./golf.db"
OUT_DIR    = "momentum_results_re"
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


def prepare_size_groups(pairs):
    """
    Group pairs by their (player, tournament) cluster size.
    Within each size group, pairs are reshaped to (n_clusters, size).

    Made-cut clusters have size 3 (R1->R2, R2->R3, R3->R4).
    Missed-cut clusters have size 1 (R1->R2 only).
    """
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

    sizes_per_cluster = pairs.groupby('cluster_id').size()

    size_groups = {}
    for size in sorted(sizes_per_cluster.unique()):
        cluster_ids = sizes_per_cluster[sizes_per_cluster == size].index.tolist()
        sub = pairs[pairs['cluster_id'].isin(cluster_ids)].sort_values(
            'cluster_id'
        ).reset_index(drop=True)
        n = len(cluster_ids)
        size_groups[int(size)] = {
            'curr_resid': sub['curr_resid'].values.reshape(n, size).astype(float),
            'prev_resid': sub['prev_resid'].values.reshape(n, size).astype(float),
            'player_idx': sub['player_idx'].values.reshape(n, size).astype(int),
            'n_clusters': n,
        }
    return size_groups


def cs_cov(sigma, tau_gamma, k):
    """Compound symmetric covariance: sigma^2 * I_k + tau_gamma^2 * J_k."""
    return sigma**2 * pt.eye(k) + tau_gamma**2 * pt.ones((k, k))


def fit_model_A(size_groups, args):
    """y_c ~ MVN(0, sigma^2*I + tau_gamma^2*J), gamma marginalized."""
    with pm.Model():
        sigma     = pm.HalfNormal('sigma', 2.0)
        tau_gamma = pm.HalfNormal('tau_gamma', 0.2)

        for size, data in size_groups.items():
            curr = data['curr_resid']
            n = data['n_clusters']
            if size == 1:
                pm.Normal(f'y_size{size}',
                          mu=0.0,
                          sigma=pt.sqrt(sigma**2 + tau_gamma**2),
                          observed=curr[:, 0])
            else:
                pm.MvNormal(f'y_size{size}',
                            mu=pt.zeros((n, size)),
                            cov=cs_cov(sigma, tau_gamma, size),
                            observed=curr)

        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept, random_seed=42,
            idata_kwargs={'log_likelihood': True},
            progressbar=True,
        )
    return idata


def fit_model_B(size_groups, players, args):
    """y_c ~ MVN(rho_p * prev_c, sigma^2*I + tau_gamma^2*J), gamma marginalized."""
    with pm.Model(coords={'player': players}):
        sigma   = pm.HalfNormal('sigma', 2.0)
        rho_pop = pm.Normal('rho_pop', 0.0, 0.5)
        tau_rho = pm.HalfNormal('tau_rho', 0.3)

        z   = pm.Normal('z', 0.0, 1.0, dims='player')
        rho = pm.Deterministic('rho', rho_pop + z * tau_rho, dims='player')

        tau_gamma = pm.HalfNormal('tau_gamma', 0.2)

        for size, data in size_groups.items():
            curr = data['curr_resid']
            prev = data['prev_resid']
            pidx = data['player_idx']

            mu = rho[pidx] * prev  # shape (n_clusters, size)

            if size == 1:
                pm.Normal(f'y_size{size}',
                          mu=mu[:, 0],
                          sigma=pt.sqrt(sigma**2 + tau_gamma**2),
                          observed=curr[:, 0])
            else:
                pm.MvNormal(f'y_size{size}',
                            mu=mu,
                            cov=cs_cov(sigma, tau_gamma, size),
                            observed=curr)

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

    size_groups = prepare_size_groups(pairs)
    n_clusters = sum(d['n_clusters'] for d in size_groups.values())

    print(f"  {len(pairs):,} pairs across {len(players):,} players "
          f"and {n_clusters:,} (player, tournament) groups.")
    print(f"  Cluster sizes: " +
          ', '.join(f"size {s}: {d['n_clusters']:,}"
                    for s, d in size_groups.items()))

    all_curr = pairs['curr_resid'].values.astype(float)
    all_prev = pairs['prev_resid'].values.astype(float)
    print(f"  curr_resid: mean={all_curr.mean():+.3f}, "
          f"std={all_curr.std():.3f}")
    print(f"  Empirical Pearson corr(prev, curr): "
          f"{np.corrcoef(all_prev, all_curr)[0, 1]:+.4f}")

    print("\n  Fitting Model A "
          "(independence + marginalized PT random effect)...")
    t0 = time.time()
    idata_A = fit_model_A(size_groups, args)
    print(f"  Model A: {time.time() - t0:.1f}s")

    print("\n  Fitting Model B "
          "(hierarchical AR(1) + marginalized PT random effect)...")
    t0 = time.time()
    idata_B = fit_model_B(size_groups, players, args)
    print(f"  Model B: {time.time() - t0:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    idata_A.to_netcdf(f"{OUT_DIR}/idata_A_{component}.nc")
    idata_B.to_netcdf(f"{OUT_DIR}/idata_B_{component}.nc")

    print("\n  Model B posterior summary:")
    summary = az.summary(
        idata_B,
        var_names=['rho_pop', 'tau_rho', 'tau_gamma', 'sigma'],
        round_to=4,
    )
    print(summary.to_string())

    print("\n  LOO comparison:")
    # Pool log-likelihoods across the three size groups (y_size1/2/3)
    # into one combined obs dim so az.loo treats every cluster equally.
    def stack_loglik(idata):
        ll = idata.log_likelihood
        size_vars = [v for v in ll.data_vars if v.startswith('y_size')]
        stacked = xr.concat(
            [ll[v].rename({d: 'cluster' for d in ll[v].dims
                           if d not in ('chain', 'draw')})
             for v in size_vars],
            dim='cluster',
        )
        new_idata = az.InferenceData(
            posterior=idata.posterior,
            log_likelihood=xr.Dataset({'y': stacked}),
        )
        return new_idata

    try:
        idata_A_pooled = stack_loglik(idata_A)
        idata_B_pooled = stack_loglik(idata_B)
        comp = az.compare(
            {'A': idata_A_pooled, 'B': idata_B_pooled},
            ic='loo',
        )
        print(comp.to_string())
    except Exception as e:
        print(f"  az.compare failed: {e}")


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