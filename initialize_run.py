import numpy as np

import db
import common as c
import run

seg_layouts = {
    'open_box': [
        [np.array([-0.25, 0.25]), np.array([0.25, 0.25])],
        [np.array([0.25, 0.25]), np.array([0.4, 0.0])],
        [np.array([-0.25, 0.25]), np.array([-0.4, 0.0])],
    ],
    'v_shape': [
        [np.array([-0.25, 0.25]), np.array([0, -0.25])],
        [np.array([0.25, 0.25]), np.array([0, -0.25])],
    ],
    'open_v_shape': [
        [np.array([-0.25, 0.25]), np.array([-0.05, -0.25])],
        [np.array([0.25, 0.25]), np.array([0.05, -0.25])],
    ],
    'none': [
    ],
}

def main():
    dt_sim = 0.02

    v_propulse = 0.1

    ag_repulse_v_0 = 1.0
    ag_repulse_d_0 = 0.002

    seg_repulse_v_0 = 1.0
    seg_repulse_d_0 = 0.01

    seg_align_omega_0 = 50.0
    seg_align_d_0 = 0.01

    d_rot_diff = 0.1
    # d_rot_diff = 0.0
    len_rot_diff = np.sqrt(2 * d_rot_diff * dt_sim)

    d_trans_diff = 0.00002
    # d_trans_diff = 0.0
    len_trans_diff = np.sqrt(2 * d_trans_diff * dt_sim)

    n = 2000

    l = 1.0

    # Initialize system state.

    # Initialize agents.

    position_alg = 'random'

    direction_alg = 'random'

    if position_alg == 'random':
        r = np.zeros([n, 2], dtype=np.float64)
        r[:, 0] = np.random.uniform(-l * 0.5, l * 0.5, size=n)
        r[:, 1] = np.random.uniform(-l * 0.5, l * 0.5, size=n)
    else:
        raise ValueError(position_alg)

    if direction_alg == 'random':
        u_p = np.zeros([n, 2])
        u_p[:, 0] = 1.0
        u_p = c.rotate_2d_vec(u_p, np.random.uniform(-np.pi, np.pi, size=n))
    elif direction_alg == 'all_east':
        u_p = np.zeros([n, 2])
    else:
        raise ValueError(direction_alg)

    # Initialize environment.

    segment_layout = 'open_v_shape'

    segs = seg_layouts[segment_layout]

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

    print('Creating run')
    run_id = run.initialize_run(conn, sim_params)
    run_state = run.mk_run_state()
    print(f"Created run with ID {run_id}")

    print("Making initial checkpoint")
    chk_id = run.write_checkpoint(conn, run_id, run_state, sim_state)
    print(f"Made initial checkpoint with ID {chk_id}")


if __name__ == '__main__':
    main()
