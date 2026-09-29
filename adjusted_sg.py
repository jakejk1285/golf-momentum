"""
adjusted_sg.py

Walk-forward two-way fixed effects ridge regression.

For every Monday in the data range, fit a separate ridge per component on
the rounds in the prior 5 years. Player coefficients (alpha_p) become that
player's adjusted skill at that Monday. Round coefficients (delta_r) absorb
course difficulty and field strength on that day.

Eight independent regressions per snapshot:
    sg_ott, sg_app, sg_arg, sg_putt, sg_total,
    driving_dist, prox_fw, prox_rgh

Setup:
    PGA + LIV tours.
    Players with 30+ rounds in the database (sg_total populated, any tour).
    150-day half-life, 5-year hard window.
    Major rounds (event_id IN 14, 26, 33, 100 on PGA tour) get a 1.2x weight
    multiplier.
    Non-SG components are weighted-mean-centered before fitting.
    Components with no data in window are skipped silently.

Outputs:
    player_skills_adjusted
        PK (snapshot_date, dg_id), one column per adj_<component>.
    round_fixed_effects
        PK (event_id, year, tour, round_num, component).
        delta from the snapshot at end of the round's week.

Run:
    cd ~/Desktop/golf-trading && python3 adjusted_sg.py

Resume:
    Snapshots already in player_skills_adjusted are skipped on rerun.
    For a clean rebuild, drop both tables before running.
"""

import sqlite3
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

DB_PATH         = "./golf.db"
TOURS           = ('pga', 'liv')
MIN_ROUNDS      = 30
ALPHA           = 2.5
HALF_LIFE_DAYS  = 150
WINDOW_DAYS     = 365 * 5
MAJOR_MULT      = 1.2
MAJOR_EVENT_IDS = (14, 26, 33, 100)
MAJOR_TOUR      = 'pga'

SG_COMPONENTS   = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']
TRAD_COMPONENTS = ['driving_dist', 'prox_fw', 'prox_rgh']
COMPONENTS      = SG_COMPONENTS + TRAD_COMPONENTS


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def init_tables(conn):
    cur = conn.cursor()
    skill_cols = ', '.join(f'adj_{c} REAL' for c in COMPONENTS)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS player_skills_adjusted (
            snapshot_date TEXT NOT NULL,
            dg_id         INTEGER NOT NULL,
            {skill_cols},
            PRIMARY KEY (snapshot_date, dg_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS round_fixed_effects (
            event_id      INTEGER NOT NULL,
            year          INTEGER NOT NULL,
            tour          TEXT NOT NULL,
            round_num     INTEGER NOT NULL,
            component     TEXT NOT NULL,
            delta         REAL NOT NULL,
            snapshot_date TEXT NOT NULL,
            PRIMARY KEY (event_id, year, tour, round_num, component)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_psa_dg "
                "ON player_skills_adjusted(dg_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rfe_event "
                "ON round_fixed_effects(event_id, year, tour, round_num)")
    conn.commit()


def get_done_snapshots(conn):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT snapshot_date FROM player_skills_adjusted"
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_rounds():
    conn = sqlite3.connect(DB_PATH)
    tours_in = ', '.join(f"'{t}'" for t in TOURS)
    df = pd.read_sql_query(f"""
        SELECT
            r.dg_id,
            r.event_id,
            r.year,
            r.tour,
            r.round_num,
            r.sg_ott,
            r.sg_app,
            r.sg_arg,
            r.sg_putt,
            r.sg_total,
            r.driving_dist,
            r.prox_fw,
            r.prox_rgh,
            e.date AS event_date
        FROM rounds r
        JOIN events e
          ON r.event_id = e.event_id
         AND r.year     = e.year
         AND r.tour     = e.tour
        WHERE r.tour IN ({tours_in})
    """, conn)
    conn.close()

    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df[df['event_date'].notna()].copy()
    df['round_date'] = (df['event_date']
                        + pd.to_timedelta(df['round_num'] - 1, unit='D'))
    return df


def filter_min_rounds(df, min_rounds=MIN_ROUNDS):
    counts = df[df['sg_total'].notna()].groupby('dg_id').size()
    keep = counts[counts >= min_rounds].index
    return df[df['dg_id'].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Snapshot scheduling
# ---------------------------------------------------------------------------

def monday_at_end_of_week(d):
    days_ahead = (7 - d.weekday()) % 7
    return d + pd.Timedelta(days=days_ahead)


def get_snapshots(df):
    first_mon = monday_at_end_of_week(df['round_date'].min())
    last_mon  = monday_at_end_of_week(df['round_date'].max())
    return pd.date_range(first_mon, last_mon, freq='W-MON')


# ---------------------------------------------------------------------------
# Per-component fit
# ---------------------------------------------------------------------------

def fit_component(df_window, comp, snapshot_date, center):
    """
    Returns (alpha_map: dg_id -> float, delta_map: round_key -> float).
    Skipped silently if no rows have this component populated.
    """
    sub = df_window[df_window[comp].notna()]
    if len(sub) == 0:
        return {}, {}

    sub = sub.copy()
    sub['round_key'] = (sub['event_id'].astype(str) + '_' +
                        sub['year'].astype(str)     + '_' +
                        sub['tour']                  + '_' +
                        sub['round_num'].astype(str))

    player_list = sorted(sub['dg_id'].unique())
    p_idx = {p: i for i, p in enumerate(player_list)}
    n_p = len(player_list)

    round_list = sorted(sub['round_key'].unique())
    r_idx = {r: i for i, r in enumerate(round_list)}
    n_r = len(round_list)

    n = len(sub)
    rows   = np.arange(n)
    p_cols = sub['dg_id'].map(p_idx).values
    r_cols = sub['round_key'].map(r_idx).values

    X_p = csr_matrix((np.ones(n), (rows, p_cols)), shape=(n, n_p))
    X_r = csr_matrix((np.ones(n), (rows, r_cols)), shape=(n, n_r))
    X   = hstack([X_p, X_r], format='csr')

    days_ago = (snapshot_date - sub['round_date']).dt.days.values
    w = np.power(0.5, days_ago / HALF_LIFE_DAYS)
    is_major = (sub['event_id'].isin(MAJOR_EVENT_IDS)
                & (sub['tour'] == MAJOR_TOUR)).values
    w = np.where(is_major, w * MAJOR_MULT, w)

    y = sub[comp].values.astype(float)
    if center:
        y = y - np.average(y, weights=w)

    model = Ridge(alpha=ALPHA, fit_intercept=False, solver='sparse_cg')
    model.fit(X, y, sample_weight=w)
    coef = model.coef_

    alpha_map = dict(zip(player_list, coef[:n_p].tolist()))
    delta_map = dict(zip(round_list,  coef[n_p:].tolist()))
    return alpha_map, delta_map


# ---------------------------------------------------------------------------
# Per-snapshot driver
# ---------------------------------------------------------------------------

def run_snapshot(df, snapshot_date, conn):
    window_start = snapshot_date - pd.Timedelta(days=WINDOW_DAYS)
    df_window = df[(df['round_date'] >  window_start) &
                   (df['round_date'] <= snapshot_date)]
    if len(df_window) == 0:
        return 0, 0

    snap_str = snapshot_date.strftime('%Y-%m-%d')

    claim_start = snapshot_date - pd.Timedelta(days=6)
    claimed = df_window[(df_window['round_date'] >= claim_start) &
                        (df_window['round_date'] <= snapshot_date)].copy()
    claimed['round_key'] = (claimed['event_id'].astype(str) + '_' +
                            claimed['year'].astype(str)     + '_' +
                            claimed['tour']                  + '_' +
                            claimed['round_num'].astype(str))
    claimed_keys = set(claimed['round_key'].unique())

    active_players = sorted(df_window['dg_id'].unique())
    skill_rows = {p: {f'adj_{c}': None for c in COMPONENTS}
                  for p in active_players}
    re_rows = []

    for comp in COMPONENTS:
        center = comp in TRAD_COMPONENTS
        alpha_map, delta_map = fit_component(
            df_window, comp, snapshot_date, center
        )
        for p, a in alpha_map.items():
            skill_rows[p][f'adj_{comp}'] = a
        for rk, d in delta_map.items():
            if rk in claimed_keys:
                ev, yr, t, rn = rk.split('_')
                re_rows.append(
                    (int(ev), int(yr), t, int(rn), comp,
                     float(d), snap_str)
                )

    cur = conn.cursor()
    cols = ['snapshot_date', 'dg_id'] + [f'adj_{c}' for c in COMPONENTS]
    placeholders = ', '.join(['?'] * len(cols))
    skill_data = [
        [snap_str, int(p)] + [skill_rows[p][f'adj_{c}'] for c in COMPONENTS]
        for p in active_players
    ]
    cur.executemany(
        f"INSERT OR REPLACE INTO player_skills_adjusted "
        f"({', '.join(cols)}) VALUES ({placeholders})",
        skill_data,
    )

    if re_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO round_fixed_effects "
            "(event_id, year, tour, round_num, component, "
            " delta, snapshot_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            re_rows,
        )

    conn.commit()
    return len(skill_data), len(re_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print(f"Loading {' + '.join(t.upper() for t in TOURS)} rounds...")
    df = load_rounds()
    print(f"  {len(df):,} rounds total.")

    print(f"Filtering to players with {MIN_ROUNDS}+ rounds "
          f"(sg_total populated)...")
    df = filter_min_rounds(df, MIN_ROUNDS)
    print(f"  {len(df):,} rounds across {df['dg_id'].nunique():,} players.")

    if len(df) == 0:
        print("No data after filtering. Exiting.")
        return

    conn = sqlite3.connect(DB_PATH)
    init_tables(conn)

    snapshots = get_snapshots(df)
    print(f"  {len(snapshots)} Monday snapshots from "
          f"{snapshots[0].date()} to {snapshots[-1].date()}.")

    done = get_done_snapshots(conn)
    pending = [s for s in snapshots if s.strftime('%Y-%m-%d') not in done]
    print(f"  {len(done)} already done, {len(pending)} pending.\n")

    total_skill = 0
    total_re    = 0
    for i, snap in enumerate(pending):
        t1 = time.time()
        n_skill, n_re = run_snapshot(df, snap, conn)
        total_skill += n_skill
        total_re    += n_re
        elapsed = time.time() - t0
        eta_h = (elapsed / max(i + 1, 1) * (len(pending) - i - 1)) / 3600
        print(f"  [{i+1}/{len(pending)}] {snap.date()} | "
              f"{n_skill:,} skill, {n_re:,} round delta | "
              f"{time.time()-t1:.1f}s | ETA {eta_h:.1f}h")

    conn.close()
    elapsed_h = (time.time() - t0) / 3600
    print(f"\nDone. {total_skill:,} skill rows, "
          f"{total_re:,} round delta rows in {elapsed_h:.2f}h.")


if __name__ == '__main__':
    main()