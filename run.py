from typing import List
from dataclasses import dataclass

import numpy as np
from sqlalchemy import insert

import db
import common as c
from cell_list import initialise_inters_structures, get_inters
import run_opt

def seg_align_omega(d, cos_th, omega_0, d_0):
    return run_opt.exp_repulse_v(d=d, v_0=cos_th * omega_0, d_0=d_0)


def wrap(d, l):
    l_half = l * 0.5

    d[:, 0] = np.where(d[:, 0] < -l_half, d[:, 0] + l, d[:, 0])
    d[:, 0] = np.where(d[:, 0] >  l_half, d[:, 0] - l, d[:, 0])

    d[:, 1] = np.where(d[:, 1] < -l_half, d[:, 1] + l, d[:, 1])
    d[:, 1] = np.where(d[:, 1] >  l_half, d[:, 1] - l, d[:, 1])


def write_checkpoint(conn, run_id, run_state, sim_state):
    [(chk_id,)] = conn.execute(
        (
            insert(db.chk_table)
            .returning(db.chk_table.c.id)
        ),
        {
            'run_id': run_id,
            'run_state': run_state,
            'sim_state': sim_state,
        },
    ).all()
    conn.commit()
    return chk_id


def write_view(conn, run_id, run_state, sim_state, v):
    [(env_id,)] = conn.execute(
        (
            insert(db.env_table)
            .returning(db.env_table.c.id)
        ),
        {
            'run_id': run_id,
            'step_sim': sim_state.step_sim,
            't_sim': sim_state.t_sim,
            'step_view': run_state.step_view,
        },
    ).all()

    conn.execute(
        insert(db.agent_table),
        [
            {
                'agent_id': i,
                'env_id': env_id,
                'rx': sim_state.r[i, 0],
                'ry': sim_state.r[i, 1],
                'ux': sim_state.u_p[i, 0],
                'uy': sim_state.u_p[i, 1],
                'vx': v[i, 0],
                'vy': v[i, 1],
            }
            for i in range(len(sim_state.r))
        ],
    )

    conn.commit()


@dataclass
class SimParams:
    dt_sim: float
    l: float
    v_propulse: float
    segs: List[np.ndarray]
    seg_repulse_v_0: float
    seg_repulse_d_0: float
    seg_align_omega_0: float
    seg_align_d_0: float
    ag_repulse_d_0: float
    ag_repulse_v_0: float
    len_trans_diff: float
    len_rot_diff: float
    n: int


@dataclass
class SimState:
    u_p: np.ndarray
    r: np.ndarray
    t_sim: float
    step_sim: int


@dataclass
class RunParams:
    t_sim_max: float
    write_view: bool
    dstep_view: int
    write_chk: bool
    dstep_chk: int
    run_id: int


@dataclass
class RunState:
    step_view: int


def mk_sim_state(r, u_p):
    return SimState(
        u_p=u_p,
        r=r,
        t_sim=0.0,
        step_sim=0,
    )


def mk_run_state():
    return RunState(step_view=0)


def run(conn, run_params: RunParams, run_state: RunState, sim_params: SimParams, sim_state: SimState):
    ag_ag_r_cut = 4 * sim_params.ag_repulse_d_0
    cl_ag_ag, cli_ag_ag, inters_ag_ag, intersi_ag_ag, inters_dr_ag_ag = initialise_inters_structures(
        l=sim_params.l, r_cut=ag_ag_r_cut, n=sim_params.n
    )
    while sim_state.t_sim < run_params.t_sim_max:
        # Compute environment and agent variables.

        # Agent propulsion.
        v_prop = np.multiply(sim_state.u_p, sim_params.v_propulse)

        # Agent-segment interaction.
        v_seg_repulse = np.zeros_like(sim_state.u_p)
        omega_seg_align = 0.0
        for seg in sim_params.segs:
            # Agent-segment repulsion.
            r_seg_near = c.point_line_segment(seg[0], seg[1], sim_state.r)
            dr_seg = sim_state.r - r_seg_near
            dr_seg_mag = c.norm(dr_seg)
            # u_seg: Normal pointing at the point.
            u_seg = dr_seg / dr_seg_mag[:, np.newaxis]
            v_seg_repulse += u_seg * run_opt.exp_repulse_v(dr_seg_mag, sim_params.seg_repulse_v_0, sim_params.seg_repulse_d_0)[:, np.newaxis]

            # Agent-segment alignment.
            cos_th = (sim_state.u_p * u_seg).sum(axis=1)
            u_p_par = sim_state.u_p - cos_th[:, np.newaxis] * u_seg
            th_change = np.arctan2(sim_state.u_p[:, 0] * u_p_par[:, 1] - sim_state.u_p[:, 1] * u_p_par[:, 0], sim_state.u_p[:, 0] * u_p_par[:, 0] + sim_state.u_p[:, 1] * u_p_par[:, 1] )
            th_change_sgn = np.sign(th_change)
            omega_seg_align += th_change_sgn * seg_align_omega(dr_seg_mag, np.abs(cos_th), sim_params.seg_align_omega_0, sim_params.seg_align_d_0)

        # Agent-agent interaction.
        get_inters(
            r=sim_state.r,
            l=sim_params.l,
            r_cut=ag_ag_r_cut,
            cl=cl_ag_ag,
            cli=cli_ag_ag,
            inters=inters_ag_ag,
            intersi=intersi_ag_ag,
            inters_dr=inters_dr_ag_ag,
        )
        v_ag_repulse = run_opt.ag_ag_repulse_v(
            sim_params.ag_repulse_v_0,
            sim_params.ag_repulse_d_0,
            inters_dr_ag_ag,
            intersi_ag_ag,
        )
        # v_ag_repulse = np.zeros_like(sim_state.u_p)
        # for i in range(sim_params.n):
        #     i_drs = inters_dr_ag_ag[i, :intersi_ag_ag[i]]
        #     i_urs = -c.unitize(i_drs)
        #     i_dr_norms = c.norm(i_drs)

        #     v_repulse_mags = run_opt.exp_repulse_v(i_dr_norms, sim_params.ag_repulse_v_0, sim_params.ag_repulse_d_0)

        #     v_ag_repulse[i] = (i_urs * v_repulse_mags[:, np.newaxis]).sum(axis=0)

        # Overall agent velocity.
        v = v_prop + v_seg_repulse + v_ag_repulse

        # Overall agent angular velocity.
        omega = omega_seg_align

        # Write environment and agent variables.
        if run_params.write_view and sim_state.step_sim % run_params.dstep_view == 0:
            print(f"Writing output")
            print(f"Sim time: {sim_state.t_sim} t")
            print(f"Sim step: {sim_state.step_sim} steps")

            write_view(conn, run_params.run_id, run_state, sim_state, v)
            run_state.step_view += 1

        if run_params.write_chk and sim_state.step_sim % run_params.dstep_chk == 0:
            print(f"Making checkpoint")
            write_checkpoint(conn, run_params.run_id, run_state, sim_state)

        # Update environment and agent state.

        # (A) Update agent position.

        # (A.a) Compute propulsion translation.
        dr = v * sim_params.dt_sim

        # (A.b) Compute translational diffusion translation.
        dr += np.random.normal(loc=0, scale=sim_params.len_trans_diff, size=(sim_params.n, 2))

        # (A.c) Perform the translation.
        sim_state.r += dr

        # (A.d) Apply periodic boundary condition.
        wrap(sim_state.r, sim_params.l)

        # (B.) Update agent direction.

        # (B.a) Compute rotational diffusion rotation.
        dth = np.random.normal(loc=0, scale=sim_params.len_rot_diff, size=sim_params.n)

        # (B.b) Compute torque rotation.
        dth += omega * sim_params.dt_sim

        # (B.c) Perform the rotation.
        sim_state.u_p = c.unitize(c.rotate_2d_vec(sim_state.u_p, dth))

        # (C). Update environment

        # (C.a) Upate time and step.
        sim_state.t_sim += sim_params.dt_sim
        sim_state.step_sim += 1

    if run_params.write_chk:
        print(f"Making final checkpoint")
        write_checkpoint(conn, run_params.run_id, run_state, sim_state)


def initialize_run(conn, sim_params):
    [(run_id,)] = conn.execute(
        (
            insert(db.run_table)
            .returning(db.run_table.c.id)
        ),
        {
            'params': sim_params,
        },
    ).all()

    if sim_params.segs:
        conn.execute(
            insert(db.seg_table),
            [
                {
                    'run_id': run_id,
                    'x1': seg[0][0],
                    'y1': seg[0][1],
                    'x2': seg[1][0],
                    'y2': seg[1][1],
                }
                for seg in sim_params.segs
            ],
        )
    return run_id
