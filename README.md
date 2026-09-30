# Golf Momentum

**Does a good round carry into the next one? A Bayesian hierarchical AR(1) test on ~134K professional rounds.**

Jake Kostoryz · Isaiah Nick

---

Every golf broadcast leans on the idea of momentum: a player gets hot, finds a rhythm,
rides a good Friday into the weekend. We wanted to know whether any of that survives
contact with the data once you control for two things that look like momentum but aren't:
how good the player already is, and how hard the course played that day.

We asked the narrowest version of the question we could actually answer. Within a single
tournament, does a player's residual performance in round *r* predict round *r+1*?

The answer feeds a bigger project: a live win-probability model that updates as rounds
are posted. If rounds are independent, that model can treat them as independent draws. If
they persist, it needs a Markov structure. We had to settle that before building anything.

## The short version

Yes, rounds persist, but the effect is small and it shrinks as you get closer to the hole.
Driving carries over the most; putting carries over the least. The part of the game
announcers talk about most when they say "hot" is the part with the least evidence.

![Posterior for population-level persistence in total strokes gained](momentum_results_re/plots/posterior_rho_pop_sg_total.png)

| Component | ρ (posterior mean) | 94% HDI |
|---|---|---|
| Off-the-Tee | 0.080 | [0.069, 0.091] |
| Total | 0.045 | [0.037, 0.052] |
| Approach | 0.031 | [0.023, 0.040] |
| Around-the-Green | 0.027 | [0.019, 0.036] |
| Putting | 0.018 | [0.009, 0.027] |

Read ρ = 0.08 as: roughly 8% of an above- or below-average driving round shows up again
the next day. Every interval clears zero, and the ordering is consistent across models,
priors and checks. The AR(1) model beats the independence model on out-of-sample fit
(PSIS-LOO) for every component.

## Method

### 1. Remove skill and course conditions first

Strokes gained is measured against the field, so a raw number blends the player's own
performance with the strength of that week's field and how the course set up. Neither is
momentum. `adjusted_sg.py` fits a walk-forward ridge regression that estimates each
player's skill as of the Monday before the event, plus a difficulty term for every round.
`build_residuals.py` subtracts both and keeps the leftover. Everything downstream runs on
that residual.

### 2. Compare two models per component

- **Model A (independence).** Each residual round is an iid draw.
- **Model B (AR(1)).** Each round is a partial echo of the previous one, with a
  player-specific persistence ρ_p that is partially pooled toward a population value
  ρ_pop.

Both are fit in PyMC, and the winner is picked by PSIS-LOO rather than by eyeballing
posteriors.

### 3. Control for the tournament itself

The first pass (`momentum_pymc.py`) found persistence everywhere, which was suspicious.
Anything constant across a player's four rounds in one week (a course that suits them, a
favorable weather draw, a skill estimate that's slightly stale) shifts all four rounds
together, and an AR(1) model will happily read that shared shift as momentum. So we added
a per-player-per-tournament random effect γ to absorb it.

### 4. Fix the sampler

Sampling γ directly (`momentum_pymc_explicit_gamma.py`) means ~33,000 extra latent
parameters and a textbook funnel: divergences, stalled chains, ESS in the double digits.
Since γ is Gaussian and enters additively, it can be integrated out in closed form. Within
each (player, tournament) cluster the likelihood becomes a multivariate normal with
compound-symmetric covariance σ²I + τ_γ²J. `momentum_pymc_re.py` implements that
marginalized model. Same posterior on the parameters we care about, no funnel, converges
in a few minutes with R-hat ≈ 1.00.

![Trace for τ_γ before and after marginalizing the random effect](momentum_results_re/plots/trace_sg_ott_tau_gamma.png)

### 5. Check that it holds up

- With the tournament control in place, ρ dropped by at most ~10% and the component
  ordering did not change.
- A posterior predictive check (`ppc.py`) reproduces the observed lag-1 autocorrelation:
  0.081 observed vs 0.079 simulated for off-the-tee.
- Prior sensitivity (`prior_sensitivity.py`) shows the estimate is stable across a range
  of priors on ρ_pop and τ_ρ.

![ρ_pop with and without the player-tournament random effect](momentum_results_re/plots/compare_rho_pop_with_without_re.png)

## What we'd do next

The win-probability model should carry round-to-round persistence. That's the decision
this analysis was meant to inform.

Two limitations are worth fixing before that model prices anything:

1. **Uncertainty in the inputs.** Player skill and round difficulty enter as point
   estimates, so their uncertainty doesn't propagate into the ρ intervals. A joint model
   would make the intervals more honest.
2. **Tails.** The Gaussian likelihood underestimates blow-up rounds by about a factor of
   three. A Student-t likelihood is the obvious replacement. It shouldn't move ρ much, but
   it matters for anything that has to price extreme outcomes.

## Repository layout

| Path | What it does |
|---|---|
| `adjusted_sg.py` | Walk-forward ridge regression for player skill and round difficulty |
| `build_residuals.py` | Builds the residual dataset the momentum models run on |
| `momentum_pymc.py` | Model A vs Model B, no tournament control (first pass) |
| `momentum_pymc_re.py` | Model A vs Model B with the marginalized player-tournament effect (headline result) |
| `momentum_pymc_explicit_gamma.py` | Same model with γ sampled explicitly; kept to document the funnel |
| `ppc.py` | Posterior predictive check on lag-1 autocorrelation |
| `prior_sensitivity.py` | Re-fits under alternative priors |
| `plot_diagnostics.py` | Generates every figure in the `momentum_results*` folders |
| `results.txt` | Full sampler output and LOO tables for every run |
| `momentum_results/` | Plots from the first pass |
| `momentum_results_re/` | Plots from the headline (marginalized) model |
| `momentum_results_explicit_gamma/` | Plots from the unmarginalized model |
| `momentum_results_ppc/` | Posterior predictive check plots |
| `report/momentum_presentation.pdf` | Slide deck |

## Data

Round-level strokes gained from the DataGolf API, covering the PGA Tour and LIV from 2019
through 2026 (~134K rounds, ~70K consecutive-round pairs per component, 674 players). The
source database is not included in this repo.

## Running it

Requires Python 3.10+ with `pymc`, `arviz`, `numpy`, `pandas`, `scipy`, `scikit-learn` and
`matplotlib`. With the database in place, run the scripts in the order listed above:
`adjusted_sg.py` → `build_residuals.py` → a `momentum_pymc*.py` model → `ppc.py` /
`prior_sensitivity.py` → `plot_diagnostics.py`.
