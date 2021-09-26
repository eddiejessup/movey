import sys
import cProfile

import numpy as np

import common as c
import db
import run
# from cell_list import get_inters


def main():
    dt_sim = 0.02

    v_propulse = 0.1

    ag_repulse_v_0 = 1.0
    ag_repulse_d_0 = 0.01

    seg_repulse_v_0 = 1.0
    seg_repulse_d_0 = 0.01

    seg_align_omega_0 = 50.0
    seg_align_d_0 = 0.01

    d_rot_diff = 0.1
    len_rot_diff = np.sqrt(2 * d_rot_diff * dt_sim)

    d_trans_diff = 0.00002
    len_trans_diff = np.sqrt(2 * d_trans_diff * dt_sim)

    n = 2000

    l = 1.0

    # Initialize system state.

    # Initialize agents.

    r = np.zeros([n, 2], dtype=np.float64)
    r[:, 0] = np.random.uniform(-l * 0.5, l * 0.5, size=n)
    r[:, 1] = np.random.uniform(-l * 0.5, l * 0.5, size=n)

    u_p = np.zeros([n, 2])
    u_p[:, 0] = 1.0
    u_p = c.rotate_2d_vec(u_p, np.random.uniform(-np.pi, np.pi, size=n))

    # Initialize environment.

    segs = []

    sim_state = run.mk_sim_state(r, u_p)

    sim_params = run.SimParams(
        dt_sim=dt_sim,
        l=l,
        v_propulse=v_propulse,
        segs=segs,
        seg_repulse_v_0=seg_repulse_v_0,
        seg_repulse_d_0=seg_repulse_d_0,
        seg_align_omega_0=seg_align_omega_0,
        seg_align_d_0=seg_align_d_0,
        ag_repulse_d_0=ag_repulse_d_0,
        ag_repulse_v_0=ag_repulse_v_0,
        len_trans_diff=len_trans_diff,
        len_rot_diff=len_rot_diff,
        n=n,
    )

    engine = db.get_engine()

    conn = engine.connect()

    run_state = run.mk_run_state()

    dstep_view = sys.maxsize

    dstep_chk = sys.maxsize

    run_params = run.RunParams(
        t_sim_max=2,
        write_view=False,
        dstep_view=dstep_view,
        write_chk=False,
        dstep_chk=dstep_chk,
        run_id=-1,
    )

    run.run(conn, run_params, run_state, sim_params, sim_state)
    # cProfile.runctx(
    #     'run.run(conn, run_params, run_state, sim_params, sim_state)',
    #     globals=globals(), locals=locals(),
    #     filename='execute_profile',
    # )

    # get_inters(sim_state.r, sim_params.l, r_cut=4 * sim_params.ag_repulse_d_0)


if __name__ == '__main__':
    main()
