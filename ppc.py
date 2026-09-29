"""
ppc.py

Posterior predictive check for the marginalized hierarchical AR(1)
model in momentum_pymc_re.py.

For each posterior draw of (rho_pop, tau_rho, tau_gamma, sigma):
  1. Sample rho_p per player from N(rho_pop, tau_rho^2).
  2. Sample gamma per (player, tournament) cluster from N(0, tau_gamma^2).
  3. Simulate curr_resid given prev_resid:
        curr ~ N(rho_p[player] * prev + gamma[cluster], sigma)
  4. Compute test statistics on the simulated dataset.

Test statistics:
  - lag-1 Pearson correlation between (prev_resid, curr_resid) over all pairs
  - within-cluster variance averaged across clusters
  - marginal SD of curr_resid
  - share of |curr_resid| > 3 (tail mass)

The check passes if the observed value sits inside the bulk of the
posterior predictive distribution. A value in the tail indicates the
model is misspecified for that statistic.

Outputs (./momentum_results_ppc/):
    ppc_<component>.npz             arrays of simulated stats per draw
    plots/ppc_lag1_<component>.png  histogram with observed line

Run:
    cd ~/Desktop/golf-trading && python3 ppc.py
    python3 ppc.py --components sg_total
    python3 ppc.py --n_draws 200
"""

import argparse
import os
import sqlite3
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

DB_PATH    = "./golf.db"
IDATA_DIR  = "momentum_results_re"
OUT_DIR    = "momentum_results_ppc"
COMPONENTS = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']

DISPLAY_NAMES = {
    'sg_ott':   'Off-the-Tee',
    'sg_app':   'Approach',
    'sg_arg':   'Around-the-Green',
    'sg_putt':  'Putting',
    'sg_total': 'Total',
}

# Match plot_diagnostics.py styling
NAVY       = '#1f3a5f'
LIGHT_BLUE = '#a6c8e0'
WARM_GRAY  = '#7a7a7a'
ACCENT     = '#c0392b'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'legend.frameon':    True,
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '#cccccc',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#333333',
    'axes.linewidth':    0.9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#e0e0e0',
    'grid.linewidth':    0.6,
    'grid.linestyle':    '-',
    'grid.alpha':        0.8,
    'figure.dpi':        100,
    'savefig.dpi':       180,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.2,
})


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


def assign_indices(pairs):
    """Add player_idx and cluster_id columns."""
    pairs = pairs.sort_values(
        ['dg_id', 'event_id', 'year', 'tour']
    ).reset_index(drop=True).copy()

    players = sorted(pairs['dg_id'].unique().tolist())
    p_idx = {p: i for i, p in enumerate(players)}
    pairs['player_idx'] = pairs['dg_id'].map(p_idx)

    pt_keys = (pairs['dg_id'].astype(str)    + '_' +
               pairs['event_id'].astype(str) + '_' +
               pairs['year'].astype(str)     + '_' +
               pairs['tour'])
    cluster_codes, _ = pd.factorize(pt_keys, sort=False)
    pairs['cluster_id'] = cluster_codes
    return pairs, players


def compute_stats(prev, curr, cluster_ids):
    """Compute test statistics on a (prev, curr) dataset."""
    lag1_corr = float(np.corrcoef(prev, curr)[0, 1])

    cluster_var = float(
        pd.Series(curr).groupby(cluster_ids).var(ddof=0).mean()
    )

    marginal_sd = float(curr.std(ddof=1))
    tail_share  = float(np.mean(np.abs(curr) > 3.0))

    return {
        'lag1_corr':    lag1_corr,
        'cluster_var':  cluster_var,
        'marginal_sd':  marginal_sd,
        'tail_share':   tail_share,
    }


def simulate_curr(prev, player_idx, cluster_idx, rho_p, gamma, sigma, rng):
    """Simulate curr_resid given prev_resid under Model B."""
    mu = rho_p[player_idx] * prev + gamma[cluster_idx]
    return mu + rng.normal(0.0, sigma, size=len(prev))


def run_ppc_component(component, n_draws, rng):
    print(f"\n{'=' * 70}")
    print(f"Component: {component}")
    print('=' * 70)

    idata_path = f"{IDATA_DIR}/idata_B_{component}.nc"
    if not os.path.exists(idata_path):
        print(f"  Missing {idata_path}. Skipping.")
        return None
    idata = az.from_netcdf(idata_path)

    df = load_residuals()
    pairs = build_pairs(df, component)
    if len(pairs) == 0:
        print(f"  No pairs for {component}. Skipping.")
        return None

    pairs, players = assign_indices(pairs)
    n_players  = len(players)
    n_clusters = pairs['cluster_id'].nunique()

    prev        = pairs['prev_resid'].values.astype(float)
    curr_obs    = pairs['curr_resid'].values.astype(float)
    player_idx  = pairs['player_idx'].values.astype(int)
    cluster_idx = pairs['cluster_id'].values.astype(int)

    print(f"  {len(pairs):,} pairs, {n_players:,} players, "
          f"{n_clusters:,} clusters.")

    obs_stats = compute_stats(prev, curr_obs, cluster_idx)
    print(f"  Observed: lag1_corr={obs_stats['lag1_corr']:+.4f}, "
          f"cluster_var={obs_stats['cluster_var']:.4f}, "
          f"marginal_sd={obs_stats['marginal_sd']:.4f}, "
          f"tail_share={obs_stats['tail_share']:.4f}")

    # Stack chains x draws into one flat dim
    posterior = idata.posterior
    rho_pop_s   = posterior['rho_pop'].values.flatten()
    tau_rho_s   = posterior['tau_rho'].values.flatten()
    tau_gamma_s = posterior['tau_gamma'].values.flatten()
    sigma_s     = posterior['sigma'].values.flatten()

    n_post = len(rho_pop_s)
    if n_draws > n_post:
        n_draws = n_post
    sel = rng.choice(n_post, size=n_draws, replace=False)

    sim_lag1     = np.empty(n_draws)
    sim_cvar     = np.empty(n_draws)
    sim_msd      = np.empty(n_draws)
    sim_tail     = np.empty(n_draws)

    print(f"  Simulating {n_draws:,} posterior predictive datasets...")
    t0 = time.time()
    for j, idx in enumerate(sel):
        rho_pop_j   = rho_pop_s[idx]
        tau_rho_j   = tau_rho_s[idx]
        tau_gamma_j = tau_gamma_s[idx]
        sigma_j     = sigma_s[idx]

        rho_p = rng.normal(rho_pop_j, tau_rho_j, size=n_players)
        gamma = rng.normal(0.0, tau_gamma_j, size=n_clusters)
        curr_sim = simulate_curr(
            prev, player_idx, cluster_idx, rho_p, gamma, sigma_j, rng,
        )
        s = compute_stats(prev, curr_sim, cluster_idx)
        sim_lag1[j] = s['lag1_corr']
        sim_cvar[j] = s['cluster_var']
        sim_msd[j]  = s['marginal_sd']
        sim_tail[j] = s['tail_share']

        if (j + 1) % 100 == 0:
            print(f"    {j + 1}/{n_draws} draws "
                  f"({time.time() - t0:.1f}s elapsed)")

    print(f"  Done in {time.time() - t0:.1f}s.")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        f"{OUT_DIR}/ppc_{component}.npz",
        sim_lag1=sim_lag1, sim_cvar=sim_cvar,
        sim_msd=sim_msd,  sim_tail=sim_tail,
        obs_lag1=obs_stats['lag1_corr'],
        obs_cvar=obs_stats['cluster_var'],
        obs_msd=obs_stats['marginal_sd'],
        obs_tail=obs_stats['tail_share'],
    )

    print_ppc_summary(component, obs_stats,
                      sim_lag1, sim_cvar, sim_msd, sim_tail)

    plot_lag1_ppc(component, obs_stats['lag1_corr'], sim_lag1)

    return {
        'obs':  obs_stats,
        'sim_lag1': sim_lag1,
        'sim_cvar': sim_cvar,
        'sim_msd':  sim_msd,
        'sim_tail': sim_tail,
    }


def ppc_pvalue(observed, simulated):
    """Two-sided posterior predictive p-value: how extreme is observed?"""
    p_one = float(np.mean(simulated >= observed))
    return min(p_one, 1.0 - p_one) * 2.0


def print_ppc_summary(component, obs, sim_lag1, sim_cvar, sim_msd, sim_tail):
    print(f"\n  Posterior predictive summary ({component}):")
    rows = [
        ('lag1_corr',   obs['lag1_corr'],
         sim_lag1.mean(), sim_lag1.std(), sim_lag1),
        ('cluster_var', obs['cluster_var'],
         sim_cvar.mean(), sim_cvar.std(), sim_cvar),
        ('marginal_sd', obs['marginal_sd'],
         sim_msd.mean(),  sim_msd.std(),  sim_msd),
        ('tail_share',  obs['tail_share'],
         sim_tail.mean(), sim_tail.std(), sim_tail),
    ]
    print(f"    {'stat':<14} {'observed':>10} {'sim_mean':>10} "
          f"{'sim_sd':>10} {'2-sided p':>12}")
    for name, ob, m, s, sim in rows:
        p = ppc_pvalue(ob, sim)
        print(f"    {name:<14} {ob:>10.4f} {m:>10.4f} {s:>10.4f} {p:>12.3f}")


def plot_lag1_ppc(component, obs_lag1, sim_lag1):
    out_dir = f"{OUT_DIR}/plots"
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(sim_lag1, bins=40, color=LIGHT_BLUE,
            edgecolor=NAVY, alpha=0.85,
            label='Posterior predictive distribution')
    ax.axvline(obs_lag1, color=ACCENT, ls='--', lw=2.0,
               label=f'Observed = {obs_lag1:.4f}')

    ax.set_xlabel('Lag-1 Pearson correlation of (prev, curr) residuals')
    ax.set_ylabel('Posterior predictive draws')
    ax.set_title(
        f'Posterior predictive check  |  Lag-1 correlation  |  '
        f'{DISPLAY_NAMES[component]}',
        loc='left',
    )
    ax.legend(loc='upper left')
    plt.savefig(f"{out_dir}/ppc_lag1_{component}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', nargs='+', default=['sg_ott'],
                        choices=COMPONENTS)
    parser.add_argument('--n_draws', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    t_total = time.time()
    for comp in args.components:
        run_ppc_component(comp, args.n_draws, rng)

    print(f"\nDone in {(time.time() - t_total)/60:.1f} min.")
    print(f"PPC arrays saved to ./{OUT_DIR}/")
    print(f"Plots saved to ./{OUT_DIR}/plots/")


if __name__ == '__main__':
    main()