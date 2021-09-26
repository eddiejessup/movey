import math

from sqlalchemy import select, desc
import arrow

import db
import run

def main():
    engine = db.get_engine()

    conn = engine.connect()

    (run_id, run_created_at, sim_params) = conn.execute(
        (
            select(
                db.run_table.c.id,
                db.run_table.c.created_at,
                db.run_table.c.params,
            )
            .order_by(desc(db.run_table.c.id))
        ),
    ).first()

    run_created_at_arr = arrow.get(run_created_at)
    print(f"Resuming run {run_id}")
    print(f"Created at {run_created_at_arr.format()} ({run_created_at_arr.humanize()})")

    (chk_id, chk_created_at, run_state, sim_state) = conn.execute(
        (
            select(
                db.chk_table.c.id,
                db.chk_table.c.created_at,
                db.chk_table.c.run_state,
                db.chk_table.c.sim_state,
            )
            .join(db.run_table)
            .where(db.run_table.c.id == run_id)
            .order_by(desc(db.chk_table.c.id))
        ),
    ).first()

    chk_created_at_arr = arrow.get(chk_created_at)
    print(f"Resuming from checkpoint {chk_id}")
    print(f"Created at {chk_created_at_arr.format()} ({chk_created_at_arr.humanize()})")

    dt_view = 0.02
    dstep_view = int(math.ceil(dt_view / sim_params.dt_sim))

    dt_chk = 2
    dstep_chk = int(math.ceil(dt_chk / sim_params.dt_sim))

    run_params = run.RunParams(
        t_sim_max=5,
        write_view=True,
        dstep_view=dstep_view,
        write_chk=True,
        dstep_chk=dstep_chk,
        run_id=run_id,
    )

    run.run(conn, run_params, run_state, sim_params, sim_state)


if __name__ == '__main__':
    main()
