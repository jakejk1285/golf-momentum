"""
prior_sensitivity.py

Prior sensitivity analysis for momentum_pymc_re.py (marginalized RE).

Reruns Model B on a single SG component (default: OTT) under three
alternative prior specifications and compares posteriors to the baseline.

Specifications:
    baseline    rho_pop ~ N(0, 0.5^2),  tau_gamma ~ Half-N(0.2)
    rho_tight   rho_pop ~ N(0, 0.2^2),  tau_gamma ~ Half-N(0.2)
    rho_loose   rho_pop ~ N(0, 1.0^2),  tau_gamma ~ Half-N(0.2)
    tg_loose    rho_pop ~ N(0, 0.5^2),  tau_gamma ~ Half-N(0.5)

Outputs (./momentum_results_sensitivity/):
    idata_B_<spec>_<component>.nc

Run:
    cd ~/Desktop/golf-trading && python3 prior_sensitivity.py
    python3 prior_sensitivity.py --components sg_total
    python3 prior_sensitivity.py --components sg_ott sg_total
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
BASELINE_DIR = "momentum_results_re"
OUT_DIR    = "momentum_results_sensitivity"
COMPONENTS = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']

SPECS = {
    'rho_tight': {'rho_pop_sd': 0.2, 'tau_gamma_sd': 0.2},
    'rho_loose': {'rho_pop_sd': 1.0, 'tau_gamma_sd': 0.2},
    'tg_loose':  {'rho_pop_sd': 0.5, 'tau_gamma_sd': 0.5},
}


def load_residuals():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM momentum_residuals", conn)
    conn.close()
    return df


def build_pairs(df, component):
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
    return sigma**2 * pt.eye(k) + tau_gamma**2 * pt.ones((k, k))


def fit_model_B_with_priors(size_groups, players, rho_pop_sd,
                             tau_gamma_sd, args):
    """Model B with configurable priors on rho_pop and tau_gamma."""
    with pm.Model(coords={'player': players}):
        sigma   = pm.HalfNormal('sigma', 2.0)
        rho_pop = pm.Normal('rho_pop', 0.0, rho_pop_sd)
        tau_rho = pm.HalfNormal('tau_rho', 0.3)

        z   = pm.Normal('z', 0.0, 1.0, dims='player')
        rho = pm.Deterministic('rho', rho_pop + z * tau_rho, dims='player')

        tau_gamma = pm.HalfNormal('tau_gamma', tau_gamma_sd)

        for size, data in size_groups.items():
            curr = data['curr_resid']
            prev = data['prev_resid']
            pidx = data['player_idx']

            mu = rho[pidx] * prev

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
            progressbar=True,
        )
    return idata


def get_summary_row(idata, label):
    """Extract headline parameter summary as a single row."""
    s = az.summary(
        idata,
        var_names=['rho_pop', 'tau_rho', 'tau_gamma', 'sigma'],
        round_to=4,
    )
    return {
        'spec':      label,
        'rho_pop':   s.loc['rho_pop',   'mean'],
        'rho_lo':    s.loc['rho_pop',   'hdi_3%'],
        'rho_hi':    s.loc['rho_pop',   'hdi_97%'],
        'rho_ess':   s.loc['rho_pop',   'ess_bulk'],
        'rho_rhat':  s.loc['rho_pop',   'r_hat'],
        'tau_rho':   s.loc['tau_rho',   'mean'],
        'tau_gamma': s.loc['tau_gamma', 'mean'],
        'sigma':     s.loc['sigma',     'mean'],
    }


def fit_component(df, component, args):
    print(f"\n{'=' * 70}")
    print(f"Component: {component}")
    print('=' * 70)

    pairs = build_pairs(df, component)
    if len(pairs) == 0:
        print(f"  No pairs for {component}. Skipping.")
        return None

    players = sorted(pairs['dg_id'].unique().tolist())
    p_idx = {p: i for i, p in enumerate(players)}
    pairs['player_idx'] = pairs['dg_id'].map(p_idx)
    size_groups = prepare_size_groups(pairs)

    n_clusters = sum(d['n_clusters'] for d in size_groups.values())
    print(f"  {len(pairs):,} pairs, {n_clusters:,} clusters, "
          f"{len(players):,} players.")

    rows = []

    # --- Baseline: load existing fit ---
    baseline_path = f"{BASELINE_DIR}/idata_B_{component}.nc"
    if os.path.exists(baseline_path):
        print(f"\n  Baseline: loading {baseline_path}")
        idata_baseline = az.from_netcdf(baseline_path)
        rows.append(get_summary_row(idata_baseline, 'baseline'))
    else:
        print(f"\n  Baseline file not found at {baseline_path}.")
        print(f"  Run momentum_pymc_re.py --components {component} first,")
        print(f"  or this script will refit from scratch.")
        print(f"\n  Refitting baseline (rho_pop SD=0.5, tau_gamma SD=0.2)...")
        t0 = time.time()
        idata_baseline = fit_model_B_with_priors(
            size_groups, players,
            rho_pop_sd=0.5, tau_gamma_sd=0.2,
            args=args,
        )
        print(f"  baseline: {time.time() - t0:.1f}s")
        os.makedirs(OUT_DIR, exist_ok=True)
        idata_baseline.to_netcdf(f"{OUT_DIR}/idata_B_baseline_{component}.nc")
        rows.append(get_summary_row(idata_baseline, 'baseline'))

    # --- Sensitivity specs ---
    for spec_name, spec in SPECS.items():
        print(f"\n  Spec: {spec_name}  "
              f"rho_pop_sd={spec['rho_pop_sd']}, "
              f"tau_gamma_sd={spec['tau_gamma_sd']}")
        t0 = time.time()
        idata = fit_model_B_with_priors(
            size_groups, players,
            rho_pop_sd=spec['rho_pop_sd'],
            tau_gamma_sd=spec['tau_gamma_sd'],
            args=args,
        )
        print(f"  {spec_name}: {time.time() - t0:.1f}s")

        os.makedirs(OUT_DIR, exist_ok=True)
        idata.to_netcdf(f"{OUT_DIR}/idata_B_{spec_name}_{component}.nc")
        rows.append(get_summary_row(idata, spec_name))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', nargs='+', default=['sg_ott'],
                        choices=COMPONENTS)
    parser.add_argument('--draws',         type=int,   default=1000)
    parser.add_argument('--tune',          type=int,   default=1000)
    parser.add_argument('--chains',        type=int,   default=4)
    parser.add_argument('--target_accept', type=float, default=0.9)
    args = parser.parse_args()

    print("Loading momentum_residuals...")
    df = load_residuals()
    print(f"  {len(df):,} rounds in residual table.")

    t_total = time.time()
    all_results = {}
    for comp in args.components:
        result = fit_component(df, comp, args)
        if result is not None:
            all_results[comp] = result

    print(f"\n{'=' * 70}")
    print("PRIOR SENSITIVITY SUMMARY")
    print('=' * 70)
    for comp, df_result in all_results.items():
        print(f"\n{comp}:")
        print(df_result.to_string(index=False, float_format='%.4f'))

        baseline_rho = df_result.loc[
            df_result['spec'] == 'baseline', 'rho_pop'
        ].values[0]
        deltas = df_result['rho_pop'] - baseline_rho
        df_result_with_delta = df_result.copy()
        df_result_with_delta['delta_rho_pop'] = deltas
        max_delta = deltas.abs().max()
        print(f"\n  max |delta rho_pop|: {max_delta:.4f}")

    print(f"\nDone in {(time.time() - t_total)/60:.1f} min.")
    print(f"Posteriors saved to ./{OUT_DIR}/")


if __name__ == '__main__':
    main()