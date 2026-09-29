"""
plot_diagnostics.py

Generate publication-quality diagnostic plots from momentum_pymc results.
One chart per PNG, clean matplotlib styling.

Per component:
    trace_<component>_<param>.png         Convergence trace (4 chains overlaid)
    posterior_rho_pop_<component>.png     Posterior of rho_pop with 95% HDI
    shrinkage_rho_<component>.png         Per-player rho_p shrinkage

Cross-component:
    forest_rho_pop.png                    rho_pop posteriors across components
    forest_tau_gamma.png                  tau_gamma posteriors across components
    compare_rho_pop_with_without_re.png   With vs without PT random effect

Run:
    python3 plot_diagnostics.py
    python3 plot_diagnostics.py --in_dir momentum_results
    python3 plot_diagnostics.py --components sg_ott sg_putt
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

IN_DIR     = "momentum_results_re"
COMPONENTS = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']
PARAMS     = ['rho_pop', 'tau_rho', 'tau_gamma', 'sigma']

DISPLAY_NAMES = {
    'sg_ott':   'Off-the-Tee',
    'sg_app':   'Approach',
    'sg_arg':   'Around-the-Green',
    'sg_putt':  'Putting',
    'sg_total': 'Total',
}

PARAM_DISPLAY = {
    'rho_pop':   r'$\rho_{\mathrm{pop}}$',
    'tau_rho':   r'$\tau_\rho$',
    'tau_gamma': r'$\tau_\gamma$',
    'sigma':     r'$\sigma$',
}

# Color palette: muted, scientific
NAVY       = '#1f3a5f'
LIGHT_BLUE = '#a6c8e0'
WARM_GRAY  = '#7a7a7a'
ACCENT     = '#c0392b'
CHAIN_COLORS = ['#1f3a5f', '#7b8fa1', '#5b8c5a', '#a8775c']

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_idata(in_dir, model, component):
    path = f"{in_dir}/idata_{model}_{component}.nc"
    if not os.path.exists(path):
        return None
    return az.from_netcdf(path)


def get_hdi(idata, param, prob=0.95):
    h = az.hdi(idata, var_names=[param], hdi_prob=prob)
    return float(h[param].sel(hdi='lower').values), \
           float(h[param].sel(hdi='higher').values)


# ---------------------------------------------------------------------------
# Plot: trace
# ---------------------------------------------------------------------------

def plot_trace(idata, component, param, out_path):
    """Caterpillar trace for a single scalar parameter (4 chains overlaid)."""
    if param not in idata.posterior.data_vars:
        return False
    samples = idata.posterior[param].values
    if samples.ndim != 2:
        return False
    n_chains, n_draws = samples.shape

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for c in range(n_chains):
        ax.plot(samples[c, :],
                color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                lw=0.7, alpha=0.75,
                label=f'Chain {c + 1}')
    ax.set_xlabel('Draw')
    ax.set_ylabel(PARAM_DISPLAY[param])
    ax.set_title(
        f'Trace  |  {PARAM_DISPLAY[param]}  |  {DISPLAY_NAMES[component]}',
        loc='left',
    )
    ax.legend(loc='upper right', ncol=4)
    plt.savefig(out_path)
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Plot: posterior of rho_pop
# ---------------------------------------------------------------------------

def plot_posterior_rho_pop(idata, component, out_path):
    samples = idata.posterior['rho_pop'].values.flatten()
    lo, hi = get_hdi(idata, 'rho_pop', prob=0.95)
    mean = float(samples.mean())

    pad = (samples.max() - samples.min()) * 0.15
    x = np.linspace(samples.min() - pad, samples.max() + pad, 500)
    y = gaussian_kde(samples)(x)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, y, color=NAVY, lw=2.2)
    in_hdi = (x >= lo) & (x <= hi)
    ax.fill_between(x[in_hdi], 0, y[in_hdi],
                    color=LIGHT_BLUE, alpha=0.65,
                    label=f'95% HDI  [{lo:.3f}, {hi:.3f}]')
    ax.axvline(0, color=ACCENT, ls='--', lw=1.5, alpha=0.85,
               label='Null (no momentum)')
    ax.axvline(mean, color=NAVY, ls=':', lw=1.5, alpha=0.9,
               label=f'Mean = {mean:.3f}')

    ax.set_xlabel(r'Within-tournament persistence  $\rho_{\mathrm{pop}}$')
    ax.set_ylabel('Posterior density')
    ax.set_title(
        f'Posterior of $\\rho_{{\\mathrm{{pop}}}}$  |  {DISPLAY_NAMES[component]}',
        loc='left',
    )
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Plot: forest plot across components
# ---------------------------------------------------------------------------

def plot_forest(in_dir, components, param, out_path,
                title=None, xlabel=None, show_zero=True):
    means, los, his, labels = [], [], [], []
    for comp in components:
        idata = load_idata(in_dir, 'B', comp)
        if idata is None:
            continue
        samples = idata.posterior[param].values.flatten()
        lo, hi = get_hdi(idata, param, prob=0.95)
        means.append(float(samples.mean()))
        los.append(lo)
        his.append(hi)
        labels.append(DISPLAY_NAMES[comp])

    order = np.argsort(means)[::-1]
    means  = [means[i]  for i in order]
    los    = [los[i]    for i in order]
    his    = [his[i]    for i in order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 0.85 * len(means) + 1.8))
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.plot([lo, hi], [i, i], color=NAVY, lw=2.6,
                solid_capstyle='round')
        ax.plot(m, i, 'o', color=NAVY, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'{m:.3f}', (m, i),
                    textcoords='offset points', xytext=(0, 14),
                    ha='center', fontsize=9.5, color=NAVY)

    if show_zero:
        ax.axvline(0, color=ACCENT, ls='--', lw=1.4, alpha=0.85)

    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel or PARAM_DISPLAY[param])
    if title:
        ax.set_title(title, loc='left')
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Plot: shrinkage of per-player rho
# ---------------------------------------------------------------------------

def plot_shrinkage(in_dir, component, out_path):
    idata = load_idata(in_dir, 'B', component)
    if idata is None or 'rho' not in idata.posterior.data_vars:
        return
    rho      = idata.posterior['rho'].values        # (chain, draw, player)
    rho_pop  = idata.posterior['rho_pop'].values.flatten()
    pop_mean = rho_pop.mean()

    means = rho.mean(axis=(0, 1))                   # (player,)
    hdi   = az.hdi(idata, var_names=['rho'],
                   hdi_prob=0.95)['rho'].values     # (player, 2)

    order = np.argsort(means)
    means_s = means[order]
    hdi_s   = hdi[order]
    n = len(means_s)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.vlines(x, hdi_s[:, 0], hdi_s[:, 1],
              color=LIGHT_BLUE, lw=0.6, alpha=0.55)
    ax.plot(x, means_s, 'o', color=NAVY, markersize=2.2, alpha=0.85)
    ax.axhline(pop_mean, color=ACCENT, ls='--', lw=1.7,
               label=fr'Population mean  $\rho_{{\mathrm{{pop}}}}$ = {pop_mean:.3f}')
    ax.axhline(0, color=WARM_GRAY, ls=':', lw=1.0, alpha=0.6)

    ax.set_xlabel('Player rank (by posterior mean)')
    ax.set_ylabel(r'Per-player persistence  $\rho_p$')
    ax.set_title(
        f'Per-player $\\rho_p$ posteriors  |  {DISPLAY_NAMES[component]}',
        loc='left',
    )
    ax.legend(loc='upper left')
    ax.set_xlim(-1, n)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Plot: with vs without random effect
# ---------------------------------------------------------------------------

def plot_compare_with_without_re(in_dir_re, in_dir_orig, components, out_path):
    rows = []
    for comp in components:
        a = load_idata(in_dir_re,   'B', comp)
        b = load_idata(in_dir_orig, 'B', comp)
        if a is None or b is None:
            continue
        s_re   = a.posterior['rho_pop'].values.flatten()
        s_orig = b.posterior['rho_pop'].values.flatten()
        lo_re, hi_re = get_hdi(a, 'rho_pop')
        lo_or, hi_or = get_hdi(b, 'rho_pop')
        rows.append({
            'label':     DISPLAY_NAMES[comp],
            'mean_re':   s_re.mean(),
            'lo_re':     lo_re,
            'hi_re':     hi_re,
            'mean_orig': s_orig.mean(),
            'lo_orig':   lo_or,
            'hi_orig':   hi_or,
        })

    if not rows:
        return

    rows.sort(key=lambda r: r['mean_re'], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 0.95 * len(rows) + 2))
    offset = 0.18
    for i, r in enumerate(rows):
        # Without RE (gray)
        ax.plot([r['lo_orig'], r['hi_orig']], [i + offset, i + offset],
                color=WARM_GRAY, lw=2.4, solid_capstyle='round')
        ax.plot(r['mean_orig'], i + offset, 'o',
                color=WARM_GRAY, markersize=9,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=5,
                label='Without random effect' if i == 0 else None)
        # With RE (navy)
        ax.plot([r['lo_re'], r['hi_re']], [i - offset, i - offset],
                color=NAVY, lw=2.4, solid_capstyle='round')
        ax.plot(r['mean_re'], i - offset, 'o',
                color=NAVY, markersize=9,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=5,
                label='With PT random effect' if i == 0 else None)

    ax.axvline(0, color=ACCENT, ls='--', lw=1.4, alpha=0.85)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r['label'] for r in rows])
    ax.set_xlabel(r'$\rho_{\mathrm{pop}}$  (95% HDI)')
    ax.set_title(
        r'Robustness:  $\rho_{\mathrm{pop}}$ with vs without '
        'player-tournament random effect',
        loc='left',
    )
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',     default=IN_DIR,
                        help='Directory containing idata_*.nc files')
    parser.add_argument('--out_dir',    default=None,
                        help='Output directory (default: <in_dir>/plots)')
    parser.add_argument('--orig_dir',   default='momentum_results',
                        help='Original (no-RE) results dir for comparison plot')
    parser.add_argument('--components', nargs='+', default=COMPONENTS,
                        choices=COMPONENTS)
    args = parser.parse_args()

    out_dir = args.out_dir or f"{args.in_dir}/plots"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading: {args.in_dir}/")
    print(f"Writing: {out_dir}/\n")

    print("Trace plots...")
    for comp in args.components:
        idata = load_idata(args.in_dir, 'B', comp)
        if idata is None:
            print(f"  {comp}: skipped (no .nc file)")
            continue
        for param in PARAMS:
            path = f"{out_dir}/trace_{comp}_{param}.png"
            plot_trace(idata, comp, param, path)
        print(f"  {comp}: 4 traces")

    print("\nPosterior densities (rho_pop)...")
    for comp in args.components:
        idata = load_idata(args.in_dir, 'B', comp)
        if idata is None:
            continue
        plot_posterior_rho_pop(idata, comp,
                               f"{out_dir}/posterior_rho_pop_{comp}.png")
        print(f"  {comp}")

    print("\nForest plots...")
    plot_forest(args.in_dir, args.components, 'rho_pop',
                f"{out_dir}/forest_rho_pop.png",
                title='Within-tournament momentum by SG component  '
                      '(95% HDI)',
                xlabel=r'$\rho_{\mathrm{pop}}$',
                show_zero=True)
    print("  forest_rho_pop")

    plot_forest(args.in_dir, args.components, 'tau_gamma',
                f"{out_dir}/forest_tau_gamma.png",
                title=r'Player-tournament effect magnitude $\tau_\gamma$  '
                      '(95% HDI)',
                xlabel=r'$\tau_\gamma$  (strokes)',
                show_zero=False)
    print("  forest_tau_gamma")

    if os.path.isdir(args.orig_dir):
        print("\nWith-vs-without-RE comparison...")
        plot_compare_with_without_re(
            args.in_dir, args.orig_dir, args.components,
            f"{out_dir}/compare_rho_pop_with_without_re.png",
        )
        print("  done")
    else:
        print(f"\nSkipping comparison plot ({args.orig_dir}/ not found)")

    print("\nShrinkage plots...")
    for comp in args.components:
        plot_shrinkage(args.in_dir, comp,
                       f"{out_dir}/shrinkage_rho_{comp}.png")
        print(f"  {comp}")

    print(f"\nAll plots in {out_dir}/")


if __name__ == '__main__':
    main()