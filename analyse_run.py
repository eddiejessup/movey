from sqlalchemy import select, desc
import pandas as pd
import numpy as np
import arrow

import db

import common as c

engine = db.get_engine()
conn = engine.connect()

(run_id, created_at, params) = conn.execute(
    (
        select(
            db.run_table.c.id,
                db.run_table.c.created_at,
            db.run_table.c.params,
        )
        .order_by(desc(db.run_table.c.id))
    ),
).first()

created_at_arr = arrow.get(created_at)
print(f"Analysing run {run_id}")
print(f"Created at {created_at_arr.format()} ({created_at_arr.humanize()})")

env_result = conn.execute(
    (
        select(
            db.env_table.c.id,
            db.env_table.c.step_view,
            db.env_table.c.step_sim,
            db.env_table.c.t_sim,
        )
        .where(db.env_table.c.run_id == run_id)
        .order_by(db.env_table.c.t_sim)
    ),
).all()
print(f"Read {len(env_result)} environment states")

d_env = (
    pd.DataFrame(
        env_result,
        columns=['env_id', 'step_view', 'step_sim', 't_sim']
    )
    # TODO: Fix properly, we write the same step_view twice if we resume from a
    # checkpoint, and have written outputs since the checkpoint.
    .drop_duplicates(subset=['step_view'], keep='first')
    .set_index('step_view', verify_integrity=True)
)
print(f"Environment states span {d_env.t_sim.min()} to {d_env.t_sim.max()}")
print(f"Latest environment state is at {d_env.step_sim.max()} steps")

print('Reading agents from database')
agent_result = conn.execute(
    (
        select(
            db.agent_table.c.agent_id,
            db.agent_table.c.env_id,
            db.agent_table.c.rx,
            db.agent_table.c.ry,
            db.agent_table.c.ux,
            db.agent_table.c.uy,
            db.agent_table.c.vx,
            db.agent_table.c.vy,
        )
        .join(db.env_table)
        .where(db.env_table.c.run_id == run_id)
    ),
).all()
print(f"Read {len(agent_result)} agent rows from database")

d_agent = pd.DataFrame(
    agent_result,
    columns=['agent_id', 'env_id', 'rx', 'ry', 'ux', 'uy', 'vx', 'vy'],
)
print(f"Read {d_agent.agent_id.nunique()} unique agents")

step_view_min, step_view_max = d_env.index.min(), d_env.index.max()

l_half = params.l * 0.5

for env_id in sorted(d_env.env_id.unique()):

    rs_list = []
    for row in d_agent.loc[lambda d: d.env_id == env_id].itertuples():
        rs_list.append([row.rx, row.ry])

    rs = np.array(rs_list)

    dr_norms = c.pairwise_norm_wrapped(rs, l_half)

    print(env_id)
    print(dr_norms.mean())
