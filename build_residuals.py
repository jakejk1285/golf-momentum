"""
build_residuals.py

Construct per-round residual dataset for the within-tournament momentum
analysis (ST540 final project).

For each round on PGA+LIV from 2019-2026:
    residual_{p,r,k} = sg_{p,r,k}
                       - adj_skill_{p,k}(latest snapshot strictly before event_date)
                       - delta_{r,k}

Inputs (existing tables):
    rounds                  raw SG observations
    events                  event_date per tournament
    player_skills_adjusted  adj_<component> per (snapshot_date, dg_id)
    round_fixed_effects     delta per (event_id, year, tour, round_num, component)

Output:
    momentum_residuals      one row per (dg_id, event_id, year, tour, round_num)
                            with resid_<component> for the four SG components + sg_total

The snapshot used for each round is the most recent player_skills_adjusted
snapshot strictly before event_date for that player. The same snapshot is
used for all four rounds of a tournament, so within-tournament dynamics
are entirely in the residuals.

Run:
    cd ~/Desktop/golf-trading && python3 build_residuals.py
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH      = "./golf.db"
TOURS        = ('pga', 'liv')
START_YEAR   = 2019
MAX_LAG_DAYS = 90
COMPONENTS   = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']


def init_table(conn):
    cur = conn.cursor()
    resid_cols = ', '.join(f'resid_{c} REAL' for c in COMPONENTS)
    cur.execute("DROP TABLE IF EXISTS momentum_residuals")
    cur.execute(f"""
        CREATE TABLE momentum_residuals (
            dg_id         INTEGER NOT NULL,
            event_id      INTEGER NOT NULL,
            year          INTEGER NOT NULL,
            tour          TEXT NOT NULL,
            round_num     INTEGER NOT NULL,
            event_date    TEXT,
            snapshot_date TEXT,
            {resid_cols},
            PRIMARY KEY (dg_id, event_id, year, tour, round_num)
        )
    """)
    conn.commit()


def load_rounds(conn):
    tours_in = ', '.join(f"'{t}'" for t in TOURS)
    sg_cols = ', '.join(f'r.{c}' for c in COMPONENTS)
    df = pd.read_sql_query(f"""
        SELECT
            r.dg_id, r.event_id, r.year, r.tour, r.round_num,
            {sg_cols},
            e.date AS event_date
        FROM rounds r
        JOIN events e
          ON r.event_id = e.event_id
         AND r.year     = e.year
         AND r.tour     = e.tour
        WHERE r.tour IN ({tours_in})
          AND r.year  >= {START_YEAR}
    """, conn)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df[df['event_date'].notna()].copy()
    return df


def load_skills(conn):
    skill_cols = ', '.join(f'adj_{c}' for c in COMPONENTS)
    df = pd.read_sql_query(f"""
        SELECT snapshot_date, dg_id, {skill_cols}
        FROM player_skills_adjusted
    """, conn)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df


def load_round_effects(conn):
    df = pd.read_sql_query("""
        SELECT event_id, year, tour, round_num, component, delta
        FROM round_fixed_effects
    """, conn)
    df = df.pivot_table(
        index=['event_id', 'year', 'tour', 'round_num'],
        columns='component', values='delta'
    ).reset_index()
    df.columns.name = None
    rename = {c: f'delta_{c}' for c in COMPONENTS if c in df.columns}
    df = df.rename(columns=rename)
    return df


def attach_skills(rounds, skills):
    """For each round, find latest snapshot strictly before event_date per player."""
    rounds = rounds.sort_values('event_date').reset_index(drop=True)
    skills = skills.sort_values('snapshot_date').reset_index(drop=True)
    merged = pd.merge_asof(
        rounds,
        skills,
        left_on='event_date',
        right_on='snapshot_date',
        by='dg_id',
        direction='backward',
        allow_exact_matches=False,
    )
    return merged


def main():
    conn = sqlite3.connect(DB_PATH)

    print("Loading rounds...")
    rounds = load_rounds(conn)
    print(f"  {len(rounds):,} rounds across "
          f"{rounds['dg_id'].nunique():,} players.")

    print("Loading player skill snapshots...")
    skills = load_skills(conn)
    print(f"  {len(skills):,} (snapshot_date, dg_id) skill rows.")

    print("Loading round fixed effects...")
    rfe = load_round_effects(conn)
    print(f"  {len(rfe):,} round-effect rows.")

    print("Attaching most-recent prior skill snapshot to each round...")
    df = attach_skills(rounds, skills)
    matched = df['snapshot_date'].notna().sum()
    print(f"  {matched:,}/{len(df):,} rounds matched to a prior snapshot.")

    df = df[df['snapshot_date'].notna()].copy()
    lag_days = (df['event_date'] - df['snapshot_date']).dt.days
    keep = lag_days <= MAX_LAG_DAYS
    print(f"  Dropping {(~keep).sum():,} rounds with lag > {MAX_LAG_DAYS} days "
          f"(stale skill snapshot).")
    df = df[keep].copy()

    print("Attaching round fixed effects...")
    df = df.merge(rfe, on=['event_id', 'year', 'tour', 'round_num'], how='left')

    print("Computing residuals...")
    for c in COMPONENTS:
        skill_col = f'adj_{c}'
        delta_col = f'delta_{c}'
        if skill_col in df.columns and delta_col in df.columns:
            df[f'resid_{c}'] = df[c] - df[skill_col] - df[delta_col]
        else:
            df[f'resid_{c}'] = np.nan

    has_any_resid = df[[f'resid_{c}' for c in COMPONENTS]].notna().any(axis=1)
    df = df[has_any_resid].copy()

    print("Lag (event_date - snapshot_date) distribution:")
    lag_days = (df['event_date'] - df['snapshot_date']).dt.days
    print(f"  n={len(lag_days):,} | min={lag_days.min()} | "
          f"median={lag_days.median():.0f} | max={lag_days.max()}")

    print("Residual diagnostics by component:")
    for c in COMPONENTS:
        rc = f'resid_{c}'
        s = df[rc].dropna()
        if len(s) > 0:
            print(f"  {c}: n={len(s):,} | mean={s.mean():+.3f} | "
                  f"std={s.std():.3f}")

    df['event_date']    = df['event_date'].dt.strftime('%Y-%m-%d')
    df['snapshot_date'] = df['snapshot_date'].dt.strftime('%Y-%m-%d')

    out_cols = ['dg_id', 'event_id', 'year', 'tour', 'round_num',
                'event_date', 'snapshot_date'] + \
               [f'resid_{c}' for c in COMPONENTS]
    out = df[out_cols].copy()
    out['dg_id']     = out['dg_id'].astype(int)
    out['event_id']  = out['event_id'].astype(int)
    out['year']      = out['year'].astype(int)
    out['round_num'] = out['round_num'].astype(int)

    print("Writing momentum_residuals...")
    init_table(conn)
    out.to_sql('momentum_residuals', conn, if_exists='append', index=False)
    conn.close()
    print(f"Done. {len(out):,} rows written.")


if __name__ == '__main__':
    main()